import pandas as pd
import numpy as np
import os
import time
from numba import jit

# --- CONFIGURATION ---
DATA_DIR = "data_15min"
MIN_TRADES_REQUIRED = 50  # Ignore settings that rarely trade

# GRID SEARCH RANGES
# We will test every combination of these
ENTRY_ZS = np.arange(-3.0, -0.4, 0.1)  # -3.0 to -0.5
EXIT_ZS = np.arange(1.0, 4.1, 0.25)  # 1.0 to 4.0


# --- 1. THE NUMBA ENGINE (Compiles to C++ Speed) ---
@jit(nopython=True)
def backtest_kernel(prices, z_scores, entry_z, exit_z):
    """
    Simulates trading for ONE stock with ONE set of parameters.
    Returns: (Total Profit $, Trade Count, Win Count)
    """
    in_position = False
    entry_price = 0.0
    total_profit_pct = 0.0
    trades = 0
    wins = 0

    stop_price = 0.0

    n = len(prices)
    if n < 2: return 0.0, 0, 0

    for i in range(n):
        price = prices[i]
        z = z_scores[i]

        # Skip bad data
        if np.isnan(price) or np.isnan(z): continue

        # --- LOGIC ---
        if not in_position:
            # ENTRY SIGNAL
            if z < entry_z:
                in_position = True
                entry_price = price
                # Initial Hard Stop (-10%)
                stop_price = entry_price * 0.90

        else:
            # WE ARE IN A POSITION

            # 1. Calculate Profit Status
            current_pct = (price - entry_price) / entry_price

            # 2. RATCHET STOP LOSS (Update the Stop Price)
            # Tier 3: > 10% Profit -> Lock 5%
            if current_pct > 0.10:
                tier_stop = entry_price * 1.05
                if tier_stop > stop_price: stop_price = tier_stop

            # Tier 2: > 5% Profit -> Lock 2%
            elif current_pct > 0.05:
                tier_stop = entry_price * 1.02
                if tier_stop > stop_price: stop_price = tier_stop

            # Tier 1: > 2% Profit -> Breakeven
            elif current_pct > 0.02:
                tier_stop = entry_price * 1.00
                if tier_stop > stop_price: stop_price = tier_stop

            # 3. CHECK EXITS

            # A. Stop Loss Hit (Hard or Ratchet)
            if price < stop_price:
                # Execute Sell
                pnl = (stop_price - entry_price) / entry_price
                total_profit_pct += pnl
                trades += 1
                if pnl > 0: wins += 1
                in_position = False
                continue

            # B. Take Profit (Z-Score Target)
            if z > exit_z:
                # Execute Sell
                pnl = current_pct
                total_profit_pct += pnl
                trades += 1
                if pnl > 0: wins += 1
                in_position = False
                continue

    return total_profit_pct, trades, wins


# --- 2. DATA PREPROCESSOR ---
def load_and_prep_data():
    """
    Loads all CSVs, calculates Z-Scores, and stores them in RAM as Numpy Arrays.
    This runs ONCE so we don't reload files for every iteration.
    """
    print("--- 📂 Loading and Pre-processing Data ---")
    stock_data = []  # List of tuples: (symbol, price_array, z_array)

    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    print(f"Found {len(files)} files. Processing...")

    for idx, f in enumerate(files):
        try:
            path = os.path.join(DATA_DIR, f)
            df = pd.read_csv(path)

            if len(df) < 500: continue  # Skip tiny files

            # Ensure columns exist
            if 'close' not in df.columns: continue

            # Parse Dates
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

            # --- CALCULATE Z-SCORES (The tricky part) ---
            # 1. Resample to Daily to get the "Trend" (SMA 20)
            daily = df['close'].resample('D').last().dropna()

            # 2. Calculate Daily Stats
            daily_mean = daily.rolling(window=20).mean()
            daily_std = daily.rolling(window=20).std()

            # 3. Map back to 15-min (Forward Fill)
            # We reindex daily stats to match the 15-min index
            daily_stats = pd.DataFrame({'mean': daily_mean, 'std': daily_std})
            daily_stats = daily_stats.reindex(df.index, method='ffill')

            # 4. Compute Z-Score
            df['z_score'] = (df['close'] - daily_stats['mean']) / daily_stats['std']

            # 5. Convert to Numpy for Numba
            # Drop NaNs created by rolling window
            df.dropna(inplace=True)

            if len(df) > 0:
                price_arr = df['close'].to_numpy(dtype=np.float64)
                z_arr = df['z_score'].to_numpy(dtype=np.float64)
                symbol = f.replace('.csv', '')
                stock_data.append((symbol, price_arr, z_arr))

        except Exception as e:
            # print(f"Skipped {f}: {e}")
            pass

        if idx % 50 == 0: print(f"Processed {idx}/{len(files)}...", end='\r')

    print(f"\n✅ Successfully loaded {len(stock_data)} stocks into RAM.")
    return stock_data


# --- 3. THE OPTIMIZER LOOP ---
def run_optimization():
    # 1. Load Data
    data_cache = load_and_prep_data()
    if not data_cache: return

    print(f"\n--- 🚀 STARTING GRID SEARCH ({len(ENTRY_ZS) * len(EXIT_ZS)} Combinations) ---")
    print("Testing Global Strategy across ALL stocks...")

    best_score = -99999
    best_params = None
    results = []

    start_time = time.time()

    # Grid Search
    total_combos = len(ENTRY_ZS) * len(EXIT_ZS)
    counter = 0

    for entry_z in ENTRY_ZS:
        for exit_z in EXIT_ZS:
            counter += 1

            # Aggregators
            global_profit = 0.0
            global_trades = 0
            global_wins = 0

            # Run simulation on ALL stocks for this setting
            for symbol, prices, z_scores in data_cache:
                p, t, w = backtest_kernel(prices, z_scores, entry_z, exit_z)
                global_profit += p
                global_trades += t
                global_wins += w

            # Metrics
            if global_trades < MIN_TRADES_REQUIRED: continue

            avg_return = global_profit / global_trades if global_trades > 0 else 0
            win_rate = global_wins / global_trades if global_trades > 0 else 0

            # Scoring: We want high profit BUT also consistency (Win Rate)
            # Simple Score = Total Profit
            score = global_profit

            results.append({
                'Entry': entry_z,
                'Exit': exit_z,
                'Total_Profit_Units': global_profit,
                'Trades': global_trades,
                'Win_Rate': win_rate,
                'Avg_Return': avg_return
            })

            if score > best_score:
                best_score = score
                best_params = (entry_z, exit_z)
                print(
                    f"🔥 NEW BEST: Entry {entry_z:.1f} | Exit {exit_z:.2f} | Profit: {global_profit:.2f}u | WR: {win_rate:.1%} | Trades: {global_trades}")

            if counter % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Checked {counter}/{total_combos} | Time: {elapsed:.1f}s", end='\r')

    # --- 4. FINAL REPORT ---
    print("\n\n" + "=" * 40)
    print("🏆 OPTIMIZATION COMPLETE 🏆")
    print("=" * 40)

    # Sort by Profit
    results.sort(key=lambda x: x['Total_Profit_Units'], reverse=True)

    print(f"{'Rank':<5} {'Entry':<8} {'Exit':<8} {'Profit':<10} {'Win Rate':<10} {'Trades':<8}")
    print("-" * 60)

    for i in range(min(10, len(results))):
        r = results[i]
        print(
            f"#{i + 1:<4} {r['Entry']:<8.1f} {r['Exit']:<8.2f} {r['Total_Profit_Units']:<10.2f} {r['Win_Rate']:<10.1%} {r['Trades']:<8}")


if __name__ == "__main__":
    run_optimization()