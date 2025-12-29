import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import os
import sys
import nltk
import time  # For Rate Limiting
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import pytz

# --- CONFIGURATION ---
API_KEY = os.getenv('APCA_API_KEY_ID')
SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
BASE_URL = "https://paper-api.alpaca.markets"

# RISK & STRATEGY PARAMETERS
MAX_POSITIONS = 20  # Optimized Rank 19
CASH_BUFFER = 5000  # Keep powder dry
HARD_STOP_PCT = -0.10  # Optimized Rank 19
NEWS_LIMIT = 50  # Intelligence
CONSENSUS_THRESHOLD = 0.70  # News Conviction

# --- HEDGING PARAMETERS ---
HEDGE_SYMBOL = 'GLD'  # Gold ETF
FEAR_SYMBOL = 'VIXY'  # VIX Proxy (Volatility ETF)
MARKET_SYMBOL = 'SPY'  # S&P 500
FEAR_THRESHOLD = 0.04  # If VIXY is up 4% today, it's a panic day

# --- OPTIMIZED ENTRY/EXIT ---
ENTRY_Z = -1.0
EXIT_Z = -1.25

# SETUP
if not API_KEY: sys.exit("API Key Missing")
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# INITIALIZE NLP
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
vader = SentimentIntensityAnalyzer()


def get_regime_map():
    """Loads the Strategy Map (Trend vs Mean Reversion)."""
    try:
        df = pd.read_csv('sp500_regime.csv')
        return dict(zip(df.symbol, df.regime))
    except:
        return {}


def get_sentiment_consensus(symbol):
    """
    Reads 50 articles. Returns a score ONLY if 70% agree.
    Filters out 'Baseless Rumors' by requiring a majority vote.
    """
    try:
        # Change 48 to 72 to cover weekends safely
        start_time = (datetime.now() - timedelta(hours=72)).strftime('%Y-%m-%dT%H:%M:%SZ')
        news_list = api.get_news(symbol=symbol, limit=NEWS_LIMIT, start=start_time)
        if not news_list: return 0.0, False

        pos_votes, neg_votes, total = 0, 0, 0
        for article in news_list:
            score = vader.polarity_scores(article.headline)['compound']
            if score > 0.2:
                pos_votes += 1
            elif score < -0.2:
                neg_votes += 1
            total += 1

        if total < 5: return 0.0, False

        pos_ratio = pos_votes / total
        neg_ratio = neg_votes / total

        if pos_ratio >= CONSENSUS_THRESHOLD:
            return 1.0, True
        elif neg_ratio >= CONSENSUS_THRESHOLD:
            return -1.0, True
        return 0.0, False
    except:
        return 0.0, False


def get_technical_data(symbol):
    """Gets Price, Z-Score, and SMA 50 for trend confirmation."""
    try:
        # Get enough data for 50 SMA
        bars = api.get_bars(symbol, tradeapi.rest.TimeFrame.Day, limit=60).df
        if bars.empty: return None, None, None

        closes = bars['close'].values
        current_price = closes[-1]

        # --- PENNY STOCK FILTER ---
        if current_price < 5.00:
            return None, None, None

        # Z-Score (20 Day)
        if len(closes) >= 20:
            mean_20 = np.mean(closes[-20:])
            std_20 = np.std(closes[-20:])
            z_score = (current_price - mean_20) / std_20 if std_20 > 0 else 0
        else:
            z_score = 0

        # Trend (50 Day SMA)
        if len(closes) >= 50:
            sma_50 = np.mean(closes[-50:])
        else:
            sma_50 = current_price

        return current_price, z_score, sma_50
    except:
        return None, None, None


def get_market_fear_index():
    """
    Checks if the market is in Panic Mode.
    Returns: True (Panic) / False (Normal)
    Logic: Checks overnight gap downs and intraday crashes.
    """
    try:
        # Check VIXY (Volatility)
        vix_bars = api.get_bars(FEAR_SYMBOL, tradeapi.rest.TimeFrame.Day, limit=2).df
        if len(vix_bars) < 2: return False

        # Compare Today's Close vs Yesterday's Close (Catch Gap Ups)
        vix_today = vix_bars.iloc[-1]['close']
        vix_yesterday = vix_bars.iloc[-2]['close']
        vix_change = (vix_today - vix_yesterday) / vix_yesterday

        # Check SPY (Market)
        spy_bars = api.get_bars(MARKET_SYMBOL, tradeapi.rest.TimeFrame.Day, limit=2).df
        spy_today = spy_bars.iloc[-1]['close']
        spy_yesterday = spy_bars.iloc[-2]['close']
        spy_change = (spy_today - spy_yesterday) / spy_yesterday

        # Panic Conditions
        if vix_change > FEAR_THRESHOLD:
            print(f"⚠️ HIGH VOLATILITY ALERT: {FEAR_SYMBOL} up {vix_change:.2%}")
            return True
        if spy_change < -0.02:
            print(f"⚠️ MARKET CRASH ALERT: {MARKET_SYMBOL} down {spy_change:.2%}")
            return True

        return False
    except:
        return False


def place_order(symbol, qty, side, current_price):
    """
    Smart Order Router:
    - Standard Hours: Market Orders (Fastest fill).
    - Extended Hours: Limit Orders (1% Margin).
    """
    try:
        clock = api.get_clock()
        is_regular_hours = clock.is_open

        if is_regular_hours:
            api.submit_order(symbol, qty, side, 'market', 'gtc')
            print(f"✅ MARKET ORDER: {side} {symbol} ({qty})")
        else:
            # Extended Hours Protection
            if side == 'buy':
                limit_price = round(current_price * 1.01, 2)
            else:
                limit_price = round(current_price * 0.99, 2)
            api.submit_order(symbol, qty, side, 'limit', limit_price=limit_price, time_in_force='day',
                             extended_hours=True)
            print(f"🌙 EXTENDED ORDER: {side} {symbol} @ {limit_price}")
    except Exception as e:
        print(f"❌ Order Failed: {e}")


def run_hedge_fund():
    print(f"--- 🐺 Hedge Fund vFinal (Verified): {datetime.now(pytz.timezone('US/Eastern'))} ---")

    regime_map = get_regime_map()
    account = api.get_account()
    equity = float(account.portfolio_value)
    cash = float(account.cash)

    # --- LOAD POSITIONS & PENDING ORDERS ---
    positions = api.list_positions()
    open_orders = api.list_orders(status='open')

    # We ignore stocks that we already own OR have pending orders for
    # This prevents "Double Tapping" a buy before the first fills
    held_symbols = {p.symbol for p in positions}
    held_symbols.update({o.symbol for o in open_orders})

    # --- 0. CHECK FOR PANIC (THE SHIELD) ---
    is_panic = get_market_fear_index()

    if is_panic:
        print("🚨 PANIC MODE ACTIVATED: MOVING TO GOLD.")
        if HEDGE_SYMBOL not in held_symbols:
            bars = api.get_bars(HEDGE_SYMBOL, tradeapi.rest.TimeFrame.Day, limit=1).df
            if not bars.empty:
                gld_price = bars.iloc[-1]['close']
                hedge_amt = cash * 0.20
                qty = int(hedge_amt / gld_price)
                if qty > 0:
                    print(f"🛡️ HEDGING: Buying {qty} shares of {HEDGE_SYMBOL}")
                    place_order(HEDGE_SYMBOL, qty, 'buy', gld_price)
    else:
        # --- RISK ON LOGIC (Sell Gold if Panic is Over) ---
        # FIX: Check if we already have a pending order for GLD to prevent double-selling (accidental short)
        has_pending_gld_order = any(o.symbol == HEDGE_SYMBOL for o in open_orders)

        for p in positions:
            if p.symbol == HEDGE_SYMBOL:
                if not has_pending_gld_order:
                    print("✅ PANIC OVER: Selling Gold Hedge to return to stocks.")
                    place_order(HEDGE_SYMBOL, p.qty, 'sell', float(p.current_price))
                else:
                    print("⏳ Gold Sell Order already pending. Waiting...")

    # --- 1. MANAGE POSITIONS ---
    for p in positions:
        symbol = p.symbol
        if symbol == HEDGE_SYMBOL: continue

        qty = float(p.qty)
        entry = float(p.avg_entry_price)
        current = float(p.current_price)

        # --- LONG POSITIONS ---
        if qty > 0:
            pct_profit = (current - entry) / entry

            # Ratchet Stop
            stop_thresh = HARD_STOP_PCT
            if pct_profit > 0.10:
                stop_thresh = 0.05
            elif pct_profit > 0.05:
                stop_thresh = 0.01

            if pct_profit < stop_thresh:
                print(f"🛑 STOP (LONG): {symbol} {pct_profit:.2%}")
                place_order(symbol, qty, 'sell', current)
                continue

            # Mean Reversion Profit Take
            regime = regime_map.get(symbol, 'MEAN_REVERSION')
            _, z, _ = get_technical_data(symbol)

            # --- PROFIT TAKE LOGIC MATCHING OPTIMIZER ---
            if regime == 'MEAN_REVERSION' and z > EXIT_Z and pct_profit > 0.01:
                print(f"💰 TAKE PROFIT: {symbol} (Z:{z:.2f} > {EXIT_Z})")
                place_order(symbol, qty, 'sell', current)

        # --- SHORT POSITIONS ---
        elif qty < 0:
            pct_profit = (entry - current) / entry

            # Ratchet Stop for Shorts
            stop_thresh = HARD_STOP_PCT
            if pct_profit > 0.10:
                stop_thresh = 0.05
            elif pct_profit > 0.05:
                stop_thresh = 0.01

            if pct_profit < stop_thresh:
                print(f"🛑 STOP (SHORT): {symbol} {pct_profit:.2%}")
                place_order(symbol, abs(qty), 'buy', current)
                continue

    # --- 2. HUNTING TRADES (FULL SCAN) ---
    if is_panic:
        print("⚠️ Market unsafe. Skipping stock hunting.")
        return

    if len(positions) >= MAX_POSITIONS:
        print("Portfolio Full. No new buys.")
        return

    # LOAD ALL TICKERS
    all_tickers = list(regime_map.keys())
    if not all_tickers:
        all_tickers = ['AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT', 'AMZN', 'GOOGL', 'META']

    # Shuffle to ensure fair scanning distribution
    np.random.shuffle(all_tickers)

    print(f"Scanning ALL {len(all_tickers)} candidates... (This will take time)")

    for i, symbol in enumerate(all_tickers):
        if cash < CASH_BUFFER:
            print("Cash Buffer Hit. Stopping Scan.")
            break

        if symbol in held_symbols: continue

        if i % 10 == 0: print(f"Scanned {i}/{len(all_tickers)}...", end='\r')

        # 1. Check Technicals (Fastest check first)
        price, z, sma_50 = get_technical_data(symbol)
        if price is None: continue

        # --- RATE LIMIT PROTECTION ---
        # Sleep for 0.2 seconds = max 300 requests/minute.
        # This keeps you safe from 429 errors.
        time.sleep(0.5)

        regime = regime_map.get(symbol, 'MEAN_REVERSION')
        sent_score, consensus = get_sentiment_consensus(symbol)

        signal = None
        reason = ""

        # A. MEAN REVERSION
        if regime == 'MEAN_REVERSION':
            if z < ENTRY_Z and sent_score > -0.6:
                signal = 'buy';
                reason = f"Oversold Z:{z:.2f}"

        # B. TREND FOLLOWING
        elif regime == 'TRENDING':
            if price > sma_50 and sent_score >= 0.7:
                signal = 'buy';
                reason = "Trend + News"

        # C. SHARK / DISASTER
        if sent_score == 1.0: signal = 'buy'; reason = "🦈 SHARK BUY"
        if sent_score == -1.0: signal = 'sell'; reason = "🐻 DISASTER SHORT"

        # EXECUTE
        if signal:
            target_pct = 0.95 / MAX_POSITIONS
            shares = int((equity * target_pct) / price)
            if shares > 0:
                print(f"\n🚀 {signal.upper()}: {symbol} | {reason}")
                place_order(symbol, shares, signal, price)
                cash -= (shares * price)
                held_symbols.add(symbol)

                if len(held_symbols) >= MAX_POSITIONS:
                    print("Max Positions Reached. Stopping Scan.")
                    break


if __name__ == "__main__":
    # Run for 5 hours and 45 minutes (leaving 15 min buffer before GitHub kills it)
    end_time = time.time() + (5.75 * 3600)

    print("--- 🟢 STARTING CONTINUOUS TRADING SESSION (5h 45m) ---")

    while time.time() < end_time:
        try:
            run_hedge_fund()
        except Exception as e:
            print(f"CRITICAL ERROR in loop: {e}")

        # Sleep for 60 seconds between full scans to prevent API bans
        print("Waiting 60 seconds before re-scanning...")
        time.sleep(60)

    print("--- 🔴 SESSION ENDING (Restarting via GitHub Actions) ---")