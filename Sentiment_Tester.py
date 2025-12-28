import alpaca_trade_api as tradeapi
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# --- CONFIG ---
API_KEY = "PKZJ67SVI2FNLBATRWKWIVNABQ"
SECRET_KEY = "3e4wtYr1Qri2HKuPixNQT7TsQCQMBMtoQmvUzGGBvRAQ"
BASE_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# --- SETUP NLP ENGINE ---
print("--- Initializing the Shark's Brain (VADER) ---")
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading VADER dictionary...")
    nltk.download('vader_lexicon')

vader = SentimentIntensityAnalyzer()


def get_sentiment(symbol):
    print(f"\nChecking News for: {symbol}...")

    # 1. Get News from Alpaca (Benzinga Feed)
    # We look for news from the last 24 hours
    start_time = (datetime.now() - timedelta(hours=24)).strftime('%Y-%m-%dT%H:%M:%SZ')

    try:
        news_list = api.get_news(symbol=symbol, limit=5, start=start_time)
    except Exception as e:
        print(f"Error fetching news: {e}")
        return

    if len(news_list) == 0:
        print("No recent news found.")
        return

    total_score = 0

    # 2. Analyze Each Headline
    print(f"{'HEADLINE':<70} | {'SCORE':<5}")
    print("-" * 80)

    for article in news_list:
        headline = article.headline
        # Calculate Score (-1 to +1)
        score = vader.polarity_scores(headline)['compound']

        # Color code the output for easier reading
        if score > 0.2:
            label = "✅ POS"
        elif score < -0.2:
            label = "🛑 NEG"
        else:
            label = "⚪ NEU"

        print(f"{headline[:70]:<70} | {score:>5.2f} {label}")
        total_score += score

    # 3. Average Sentiment
    avg_score = total_score / len(news_list)
    print("-" * 80)
    print(f"AVERAGE SENTIMENT SCORE: {avg_score:.3f}")

    if avg_score > 0.2:
        print("🦈 SHARK DECISION: SMELLS BLOOD! (BUY SIGNAL)")
    elif avg_score < -0.2:
        print("🐻 SHARK DECISION: PANIC! (SHORT SIGNAL)")
    else:
        print("😴 SHARK DECISION: SLEEP (NO TRADE)")


if __name__ == "__main__":
    # Test on a few volatile stocks
    get_sentiment('TSLA')
    get_sentiment('NVDA')
    get_sentiment('AAPL')