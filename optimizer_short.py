import pandas as pd
import numpy as np
import os
import time
from numba import jit

# --- CONFIGURATION ---
DATA_DIR = "data_15min"
MIN_TRADES_REQUIRED = 50

# GRID SEARCH RANGES (Short Strategy)
# Entry: usually we short a "bounce" in a downtrend (e.g. Z > 0.5 or 1.0)
# Exit: we cover when it crashes back down (e.g. Z < -1.0)
ENTRY_ZS = np.arange(-2.0, 2.1, 0.25)  # -2.0 to 2.0
EXIT_ZS = np.arange(-3.0, 0.6, 0.25)  # -3.0 to 0.5


# --- 1. THE NUMBA ENGINE (Short Logic) ---
@jit(nopython=True)
def backtest_short_kernel(prices, z_scores, sma_50s, entry_z, exit_z):
    """
    Simulates SHORT trading.
    Entry: Price < SMA50 AND Z > entry_z
    Exit: Z < exit_z OR Stop Loss
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
        sma = sma_50s[i]

        if np.isnan(price) or np.isnan(z) or np.isnan(sma): continue

        if not in_position:
            # ENTRY SIGNAL (Trend Breakdown + Momentum Check)
            # We only short if Price is BELOW the 50-Day SMA
            if price < sma:
                if z > entry_z:
                    in_position = True
                    entry_price = price
                    # Initial Hard Stop (+10% for shorts means price goes UP)
                    stop_price = entry_price * 1.10

        else:
            # SHORT POSITION MANAGEMENT

            # 1. Calculate Profit (Short Profit = (Entry - Price) / Entry)
            current_pct = (entry_price - price) / entry_price

            # 2. RATCHET STOP LOSS (Inverted for Shorts)
            # If profit > 2%, move stop to Breakeven (Entry Price)
            # If profit > 5%, move stop to Entry * 0.98 (Lock 2%)

            # Tier 3: > 10% Profit -> Lock 5%
            if current_pct > 0.10:
                tier_stop = entry_price * 0.95
                if tier_stop < stop_price: stop_price = tier_stop  # We want stop price to go DOWN

            # Tier 2: > 5% Profit -> Lock 2%
            elif current_pct > 0.05:
                tier_stop = entry_price * 0.98
                if tier_stop < stop_price: stop_price = tier_stop

            # Tier 1: > 2% Profit -> Breakeven
            elif current_pct > 0.02:
                tier_stop = entry_price * 1.00
                if tier_stop < stop_price: stop_price = tier_stop

            # 3. CHECK EXITS

            # A. Stop Loss Hit (Price went ABOVE stop price)
            if price > stop_price:
                # Cover
                pnl = (entry_price - stop_price) / entry_price
                total_profit_pct += pnl
                trades += 1
                if pnl > 0: wins += 1
                in_position = False
                continue

            # B. Take Profit (Z-Score Target)
            if z < exit_z:
                # Cover
                pnl = current_pct
                total_profit_pct += pnl
                trades += 1
                if pnl > 0: wins += 1
                in_position = False
                continue

    return total_profit_pct, trades, wins


# --- 2. DATA PREPROCESSOR ---
def load_and_prep_data():
    print("--- 📂 Loading Data for Short Optimization ---")
    stock_data = []

    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

    for idx, f in enumerate(files):
        try:
            path = os.path.join(DATA_DIR, f)
            df = pd.read_csv(path)
            if len(df) < 500: continue

            if 'close' not in df.columns: continue
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

            # 1. Daily Resample
            daily = df['close'].resample('D').last().dropna()

            # 2. Stats (Mean 20, Std 20, SMA 50)
            daily_mean = daily.rolling(window=20).mean()
            daily_std = daily.rolling(window=20).std()
            daily_sma50 = daily.rolling(window=50).mean()  # Trend Filter

            # 3. Map to 15-min
            daily_stats = pd.DataFrame({
                'mean': daily_mean,
                'std': daily_std,
                'sma50': daily_sma50
            })
            daily_stats = daily_stats.reindex(df.index, method='ffill')

            # 4. Z-Score
            df['z_score'] = (df['close'] - daily_stats['mean']) / daily_stats['std']
            df['sma_50'] = daily_stats['sma50']

            df.dropna(inplace=True)

            if len(df) > 0:
                price_arr = df['close'].to_numpy(dtype=np.float64)
                z_arr = df['z_score'].to_numpy(dtype=np.float64)
                sma_arr = df['sma_50'].to_numpy(dtype=np.float64)

                stock_data.append((f.replace('.csv', ''), price_arr, z_arr, sma_arr))

        except:
            pass
        if idx % 50 == 0: print(f"Processed {idx}...", end='\r')

    print(f"\n✅ Loaded {len(stock_data)} stocks.")
    return stock_data


# --- 3. RUNNER ---
def run_optimization():
    data_cache = load_and_prep_data()
    if not data_cache: return

    print(f"\n--- 🐻 SHORT STRATEGY GRID SEARCH ---")

    best_score = -99999
    results = []
    start_time = time.time()
    counter = 0
    total_combos = len(ENTRY_ZS) * len(EXIT_ZS)

    for entry_z in ENTRY_ZS:
        for exit_z in EXIT_ZS:
            counter += 1
            if entry_z <= exit_z: continue  # Logic check: Entry must be > Exit for shorts

            global_profit = 0.0
            global_trades = 0
            global_wins = 0

            for symbol, prices, z_scores, sma_50s in data_cache:
                p, t, w = backtest_short_kernel(prices, z_scores, sma_50s, entry_z, exit_z)
                global_profit += p
                global_trades += t
                global_wins += w

            if global_trades < MIN_TRADES_REQUIRED: continue

            avg_return = global_profit / global_trades if global_trades > 0 else 0
            win_rate = global_wins / global_trades if global_trades > 0 else 0

            results.append({
                'Entry': entry_z,
                'Exit': exit_z,
                'Profit': global_profit,
                'Trades': global_trades,
                'WR': win_rate
            })

            if global_profit > best_score:
                best_score = global_profit
                print(
                    f"🔥 NEW BEST: Entry {entry_z:.2f} | Exit {exit_z:.2f} | Profit: {global_profit:.2f}u | WR: {win_rate:.1%}")

            if counter % 50 == 0:
                print(f"Checked {counter}/{total_combos}...", end='\r')

    # REPORT
    print("\n\n🏆 TOP SHORT SETTINGS 🏆")
    results.sort(key=lambda x: x['Profit'], reverse=True)
    print(f"{'Rank':<5} {'Entry':<8} {'Exit':<8} {'Profit':<10} {'WR':<10} {'Trades':<8}")
    print("-" * 60)
    for i in range(min(10, len(results))):
        r = results[i]
        print(
            f"#{i + 1:<4} {r['Entry']:<8.2f} {r['Exit']:<8.2f} {r['Profit']:<10.2f} {r['WR']:<10.1%} {r['Trades']:<8}")


if __name__ == "__main__":
    run_optimization()