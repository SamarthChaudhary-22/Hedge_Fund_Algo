import pandas as pd
import numpy as np
import os
import sys
import time
import re
from datetime import datetime, timedelta
from collections import Counter
from tqdm import tqdm  # Progress Bar
from numba import cuda
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
import nltk
from nltk.corpus import stopwords

# --- CONFIGURATION ---
POPULATION_SIZE = 4096
GENERATIONS = 50
MUTATION_RATE = 0.05
CUDA_THREADS = 256
VOCAB_SIZE = 500  # The engine will pick the top 500 most common words to test
YEARS_OF_DATA = 10

# API CONFIG
API_KEY = '*************'
SECRET_KEY = '*************'
BASE_URL = "https://paper-api.alpaca.markets"

# NLTK SETUP (For cleaning text)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))
# Add common financial "noise" words
STOP_WORDS.update(
    ['stock', 'market', 'shares', 'price', 'today', 'news', 'sp500', 'earnings', 'report', 'inc', 'corp', 'group'])

# SETUP ALPACA
if not API_KEY: sys.exit("❌ API Key Missing")
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')


# --- 1. DATA LOADING & LEARNING ---

def clean_and_tokenize(text):
    """Splits text into words, removes punctuation and stopwords."""
    # Remove non-alphabetic chars
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = text.split()
    # Filter stopwords and short words
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]


def fetch_and_process_data():
    """
    Safely fetches data with Rate Limiting to avoid 429 Errors.
    """
    print("📂 Reading 'sp500_regime.csv'...")
    try:
        df = pd.read_csv('sp500_regime.csv')
        df.columns = [c.strip().lower() for c in df.columns]
        symbols = df['symbol'].tolist()
    except Exception as e:
        sys.exit(f"❌ Error reading CSV: {e}")

    raw_news_data = []
    word_counter = Counter()

    print(f"🌍 Fetching Data for {len(symbols)} symbols...")
    print("   (Slowing down to respect API limits...)")

    # PROGRESS BAR
    for symbol in tqdm(symbols, desc="Downloading Data", unit="sym"):

        # --- 🛑 THE FIX: FORCED SLEEP ---
        # Sleep 0.65s guarantees max ~92 requests/min (Safe zone)
        time.sleep(0.65)

        try:
            # 1. Fetch Price History
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=365 * YEARS_OF_DATA)

            bars = api.get_bars(symbol, TimeFrame.Day, start=start_dt.strftime('%Y-%m-%d'), limit=10000).df
            if bars.empty: continue

            bars['date_str'] = bars.index.strftime('%Y-%m-%d')
            price_map = bars.set_index('date_str')[['open', 'close']].to_dict('index')
            trading_days = sorted(price_map.keys())

            # 2. Fetch News
            news_batch = api.get_news(symbol=symbol, start=start_dt.strftime('%Y-%m-%d'), limit=50)

            for article in news_batch:
                pub_date = article.created_at.strftime('%Y-%m-%d')

                # Find impact day
                future_days = [d for d in trading_days if d > pub_date]
                if not future_days: continue
                impact_day = future_days[0]

                # Calculate Return
                day_data = price_map[impact_day]
                if day_data['open'] == 0: continue
                ret = (day_data['close'] - day_data['open']) / day_data['open']

                tokens = clean_and_tokenize(article.headline)
                if not tokens: continue

                raw_news_data.append((tokens, ret))
                word_counter.update(tokens)

        except Exception as e:
            # If a specific symbol fails, print it but DON'T crash the whole script
            # tqdm.write(f"⚠️ Skipped {symbol}: {e}")
            continue

    if not raw_news_data:
        sys.exit("❌ No valid data found.")

    print(f"\n📚 Learning Vocabulary from {len(raw_news_data)} headlines...")

    # ... Rest of the function remains the same ...
    common_words = word_counter.most_common(VOCAB_SIZE)
    candidate_keywords = np.array([word for word, count in common_words])

    print(f"✅ Vocabulary Built. Top 5 words: {candidate_keywords[:5]}")
    print("⚙️  Compiling Matrices for GPU...")

    num_samples = len(raw_news_data)
    hits_matrix = np.zeros((num_samples, VOCAB_SIZE), dtype=np.int8)
    returns_array = np.zeros(num_samples, dtype=np.float32)

    vocab_map = {word: i for i, word in enumerate(candidate_keywords)}

    for i, (tokens, ret) in enumerate(raw_news_data):
        returns_array[i] = ret
        for token in tokens:
            if token in vocab_map:
                col_idx = vocab_map[token]
                hits_matrix[i, col_idx] = 1

    return candidate_keywords, hits_matrix, returns_array
# --- 2. CUDA KERNEL (Standard Genetic Logic) ---
@cuda.jit
def evaluate_population_kernel(population, hits_matrix, returns, fitness_scores):
    idx = cuda.grid(1)
    if idx < population.shape[0]:
        score = 0.0
        signal_count = 0

        # Loop through all news events
        for i in range(hits_matrix.shape[0]):
            is_flagged = False

            # Check if this event triggers the gene (keyword set)
            for k in range(hits_matrix.shape[1]):
                # If gene has keyword ON (1) AND article has keyword (1)
                if population[idx, k] == 1 and hits_matrix[i, k] == 1:
                    is_flagged = True
                    break

            if is_flagged:
                r = returns[i]
                # If price DROPPED (negative return), we want to reward this.
                if r < 0:
                    score += abs(r) * 2.0  # Big points for catching a crash
                else:
                    score -= r * 3.0  # Big penalty for flagging a gain (False Positive)
                signal_count += 1

        if signal_count > 0:
            fitness_scores[idx] = score
        else:
            fitness_scores[idx] = -999.0


# --- 3. GENETIC OPTIMIZER ---
def run_dynamic_optimizer():
    # 1. LOAD & LEARN
    candidate_keywords, hits_matrix, returns_array = fetch_and_process_data()
    num_keywords = len(candidate_keywords)

    # 2. INIT CUDA
    d_hits = cuda.to_device(hits_matrix)
    d_returns = cuda.to_device(returns_array)

    # Init Population (Randomly turn words ON/OFF)
    population = np.random.randint(0, 2, size=(POPULATION_SIZE, num_keywords)).astype(np.int32)

    print(f"\n🚀 STARTING GPU OPTIMIZATION on {len(hits_matrix)} events using {num_keywords} dynamic keywords...")

    # Progress bar for Generations
    for gen in tqdm(range(GENERATIONS), desc="Evolving Genes", unit="gen"):
        d_population = cuda.to_device(population)
        d_scores = cuda.device_array(POPULATION_SIZE, dtype=np.float32)
        blocks = (POPULATION_SIZE + CUDA_THREADS - 1) // CUDA_THREADS

        evaluate_population_kernel[blocks, CUDA_THREADS](d_population, d_hits, d_returns, d_scores)

        scores = d_scores.copy_to_host()
        best_idx = np.argmax(scores)

        # BREEDING LOGIC
        top_indices = np.argsort(scores)[-int(POPULATION_SIZE * 0.2):]
        parents = population[top_indices]
        new_population = np.zeros_like(population)
        new_population[0] = population[best_idx]  # Elitism

        for i in range(1, POPULATION_SIZE):
            p1 = parents[np.random.randint(len(parents))]
            p2 = parents[np.random.randint(len(parents))]
            cut = np.random.randint(num_keywords)
            child = np.concatenate((p1[:cut], p2[cut:]))
            if np.random.rand() < MUTATION_RATE:
                m_idx = np.random.randint(num_keywords)
                child[m_idx] = 1 - child[m_idx]
            new_population[i] = child
        population = new_population

    # --- FINAL OUTPUT ---
    best_idx = np.argmax(scores)
    best_mask = population[best_idx]

    # Filter the dynamic list based on the best mask
    final_red_flags = [candidate_keywords[i] for i in range(num_keywords) if best_mask[i] == 1]

    print("\n" + "=" * 50)
    print(f"🏆 OPTIMIZATION COMPLETE | Best Score: {scores[best_idx]:.4f}")
    print("=" * 50)
    print("Your Custom 'RED FLAG' Dictionary (Derived from your data):")
    print(final_red_flags)
    print("=" * 50)


if __name__ == "__main__":
    run_dynamic_optimizer()