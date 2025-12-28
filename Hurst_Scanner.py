import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import time
import requests
from io import StringIO
from datetime import datetime, timedelta
import os

# --- CONFIG ---
API_KEY = os.getenv('APCA_API_KEY_ID')
SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
BASE_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')


def get_sp500_tickers():
    print("--- Fetching S&P 500 List from Wikipedia ---")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        if response.status_code != 200: raise ValueError("Wiki connection failed")

        table = pd.read_html(StringIO(response.text))
        df = table[0]
        tickers = df['Symbol'].tolist()

        # FIX: Alpaca uses '.' (BRK.B), not '-' (BRK-B)
        tickers = [t.replace('-', '.') for t in tickers]

        print(f"Success! Found {len(tickers)} tickers.")
        return tickers
    except Exception as e:
        print(f"Wiki Error: {e}. Using Fallback.")
        return ['AAPL', 'MSFT', 'TSLA', 'SPY', 'QQQ', 'NVDA', 'AMD']


def get_hurst_exponent(time_series):
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0


def run_scan():
    tickers = get_sp500_tickers()
    results = []

    # Force 2 years of data to ensure we get 252 trading days
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

    print(f"\n--- CLASSIFYING {len(tickers)} STOCKS (RELAXED MODE) ---")

    for i, symbol in enumerate(tickers):
        try:
            bars = api.get_bars(
                symbol,
                tradeapi.rest.TimeFrame.Day,
                start=start_date,
                limit=None,
                feed='iex'
            ).df

            if bars.empty or len(bars) < 120: continue
            if 'close' in bars.columns:
                closes = bars['close'].values
            else:
                continue

            # Calculate Hurst
            H = get_hurst_exponent(closes)

            # --- RELAXED REGIME LOGIC ---
            # We tightened the "Random Walk" zone to just 0.48 - 0.52
            if H < 0.48:  # Was 0.46
                regime = 'MEAN_REVERSION'
            elif H > 0.52:  # Was 0.54
                regime = 'TRENDING'
            else:
                regime = 'RANDOM_WALK'

            volatility = bars['close'].pct_change().std() * 100

            results.append({
                'symbol': symbol,
                'hurst': round(H, 3),
                'regime': regime,
                'volatility': round(volatility, 2)
            })

            if i % 50 == 0 and i > 0:
                print(f"Processed {i} / {len(tickers)} stocks...")

        except KeyboardInterrupt:
            print("Stopped by User.")
            break
        except Exception as e:
            continue

    # --- SAVE AND CLEAN ---
    if not results:
        print("\nCRITICAL ERROR: No results found.")
    else:
        df_res = pd.DataFrame(results)

        # 1. REMOVE DUPLICATES (The Cleaner)
        # We drop the non-voting/secondary shares
        duplicates_to_drop = ['GOOG', 'FOX', 'NWS', 'UA']
        initial_count = len(df_res)
        df_res = df_res[~df_res['symbol'].isin(duplicates_to_drop)]
        dropped_count = initial_count - len(df_res)

        # 2. SAVE
        df_res.to_csv('sp500_regime.csv', index=False)

        print("\n" + "=" * 40)
        print(f"SCAN COMPLETE (Duplicates Removed: {dropped_count})")
        print(f"Mean Reverters (< 0.48): {len(df_res[df_res['regime'] == 'MEAN_REVERSION'])}")
        print(f"Trenders       (> 0.52): {len(df_res[df_res['regime'] == 'TRENDING'])}")
        print(f"Random Walk    (Noise):  {len(df_res[df_res['regime'] == 'RANDOM_WALK'])}")
        print("=" * 40)


if __name__ == "__main__":
    run_scan()