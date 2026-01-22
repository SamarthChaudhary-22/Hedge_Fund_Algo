import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import os
import sys
import nltk
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta, timezone
import pytz

# --- 🏆 FINAL CONFIGURATION (Rank #1: 71,680% Return) ---
API_KEY = os.getenv('APCA_API_KEY_ID')
SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
BASE_URL = "https://paper-api.alpaca.markets"

# RISK & STRATEGY
MAX_POSITIONS = 40
CASH_BUFFER = 2000
HARD_STOP_PCT = -0.10  # Tightened Stop Loss (Optimized)
DAILY_TARGET_PCT = 0.015 # If Portfolio is up 1.5% today, Close All.

# OPTIMIZED ENTRY/EXIT (The Sniper Setup)
ENTRY_Z = -0.5  # Buy the crash
EXIT_Z = 2.5  # "Always True" -> Relies purely on Profit Guard
PROFIT_GUARD = 0.03  # The 1% Sniper Rule

# INTELLIGENCE
NEWS_LIMIT = 30
CONSENSUS_THRESHOLD = 0.35

# HEDGING (The Shield)
HEDGE_SYMBOL = 'GLD'
FEAR_SYMBOL = 'VIXY'
MARKET_SYMBOL = 'SPY'
FEAR_THRESHOLD = 0.05  # Optimized: Only hedge if VIX spikes 5%

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
    current_day = datetime.now().weekday()
    lookback_hours = 72 if current_day == 0 else 24
    try:
        start_time = (datetime.now() - timedelta(hours=lookback_hours)).strftime('%Y-%m-%dT%H:%M:%SZ')
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

        if total < 3: return 0.0, False
        if (pos_votes / total) >= CONSENSUS_THRESHOLD:
            return 1.0, True
        elif (neg_votes / total) >= CONSENSUS_THRESHOLD:
            return -1.0, True
        return 0.0, False
    except:
        return 0.0, False


def get_technical_data(symbol):
    """
    FETCHES: Price, Z-Score (Blind), SMA50, CMF Slope
    LOGIC: Matches Backtest Engine vFinal
    """
    try:
        # 1. Fetch 60 days (Enough for 50SMA + 20CMF + Buffer)
        bars = api.get_bars(symbol, tradeapi.rest.TimeFrame.Day, limit=60).df

        if bars.empty or len(bars) < 50:
            return None, None, None, None

        # Data Prep
        closes = bars['close']
        highs = bars['high']
        lows = bars['low']
        volumes = bars['volume']
        current_price = closes.iloc[-1]

        # 2. BLIND Z-SCORE (Shifted by 1 Day)
        # We calculate mean/std on the HISTORY, excluding today to match backtest
        # We look at the last 20 days *before* today
        history = closes.iloc[:-1]
        if len(history) < 20: return None, None, None, None

        mean_20 = history.iloc[-20:].mean()
        std_20 = history.iloc[-20:].std()

        if std_20 == 0: return None, None, None, None
        z_score = (current_price - mean_20) / std_20

        # 3. SMA 50
        sma_50 = closes.iloc[-50:].mean()

        # 4. CMF (Chaikin Money Flow) & SLOPE
        # Money Flow Multiplier
        mfm = ((closes - lows) - (highs - closes)) / (highs - lows)
        mfm = mfm.fillna(0)
        mfv = mfm * volumes

        # 20-period CMF
        cmf = mfv.rolling(20).sum() / volumes.rolling(20).sum()

        # CMF Slope (Current - 10 days ago)
        # Ensure we have enough data for slope
        if len(cmf) < 12: return None, None, None, None

        cmf_current = cmf.iloc[-1]
        cmf_prev = cmf.iloc[-11]  # 10 bars ago
        cmf_slope = cmf_current - cmf_prev

        return current_price, z_score, sma_50, cmf_slope

    except Exception as e:
        # print(f"⚠️ Metrics Error {symbol}: {e}")
        return None, None, None, None

def get_market_fear_index():
    try:
        vix_bars = api.get_bars(FEAR_SYMBOL, tradeapi.rest.TimeFrame.Day, limit=2).df
        if len(vix_bars) < 2: return False

        vix_change = (vix_bars.iloc[-1]['close'] - vix_bars.iloc[-2]['close']) / vix_bars.iloc[-2]['close']
        if vix_change > FEAR_THRESHOLD:
            print(f"⚠️ HIGH VOLATILITY ALERT: {FEAR_SYMBOL} up {vix_change:.2%}")
            return True
        return False
    except:
        return False


def place_order(symbol, qty, side, current_price, order_type_label="manual"):
    try:
        clock = api.get_clock()
        # Unique ID for tagging (e.g., algo_stop_loss_AAPL_167823...)
        unique_id = f"algo_{order_type_label}_{symbol}_{int(time.time())}"

        if clock.is_open:
            api.submit_order(symbol, qty, side, 'market', 'gtc', client_order_id=unique_id)
            print(f"✅ MARKET ORDER ({order_type_label}): {side} {symbol} ({qty})")
        else:
            limit_price = round(current_price * 1.01, 2) if side == 'buy' else round(current_price * 0.99, 2)
            api.submit_order(symbol, qty, side, 'limit', limit_price=limit_price,
                             time_in_force='day', extended_hours=True, client_order_id=unique_id)
            print(f"🌙 EXTENDED ORDER ({order_type_label}): {side} {symbol} @ {limit_price}")
    except Exception as e:
        print(f"❌ Order Failed: {e}")


def get_cooldown_list():
    """
    OPTIMIZED: Fetches orders ONCE and returns a set of banned symbols.
    Returns: Set of symbols to skip.
    """
    banned_symbols = set()
    try:
        # Check last 24 hours of orders
        cutoff_time = (datetime.now(pytz.timezone('US/Eastern')) - timedelta(hours=24)).isoformat()
        # Limit 500 to capture high-frequency activity
        orders = api.list_orders(status='closed', limit=500, after=cutoff_time)

        for o in orders:
            if o.side == 'sell' and o.client_order_id:
                client_id = str(o.client_order_id)

                # 1. 24-HOUR BAN (Stop Loss)
                if 'stop_loss' in client_id:
                    banned_symbols.add(o.symbol)

                # 2. 30-MINUTE BAN (Take Profit)
                elif 'take_profit' in client_id:
                    filled_at = o.filled_at if o.filled_at else o.created_at
                    if filled_at:
                        filled_dt = datetime.fromisoformat(str(filled_at).replace('Z', '+00:00'))
                        time_since = datetime.now(pytz.timezone('UTC')) - filled_dt
                        # If sold less than 30 mins ago, ban it
                        if time_since < timedelta(minutes=30):
                            banned_symbols.add(o.symbol)

        return banned_symbols
    except Exception as e:
        print(f"⚠️ Cooldown Check Failed: {e}")
        return set()


def check_and_refresh_stale_orders():
    """
    Cancels any LIMIT order that has been open for >15 minutes.
    This frees up buying power and allows the bot to place a fresh order
    at the NEW price if the signal is still valid.
    """
    try:
        # Get all open orders
        orders = api.list_orders(status='open')

        # Current time in UTC (Alpaca uses UTC)
        now = datetime.now(timezone.utc)

        for o in orders:
            # Only check LIMIT orders (Market orders fill instantly)
            # We also ignore STOP orders (those are supposed to sit forever)
            if o.type != 'limit': continue

            # Parse submission time
            submitted_at = o.submitted_at
            if isinstance(submitted_at, str):
                # Convert string to datetime object
                submitted_at = datetime.fromisoformat(submitted_at.replace('Z', '+00:00'))

            # Calculate Age
            age = now - submitted_at

            if age > timedelta(minutes=15):
                print(f"♻️ REFRESH: {o.side.upper()} {o.symbol} order is {age.seconds // 60}m old. Canceling...")
                api.cancel_order(o.id)

    except Exception as e:
        print(f"⚠️ Stale Order Check Failed: {e}")


def run_hedge_fund():
    print(f"--- 🐺 Hedge Fund vFinal (Harvest Mode): {datetime.now(pytz.timezone('US/Eastern'))} ---")
    check_and_refresh_stale_orders()  #---Cleans up old limit orders
    regime_map = get_regime_map()
    account = api.get_account()
    equity = float(account.portfolio_value)
    cash = float(account.cash)

    # --- 🆕 HARVEST CHECK (Daily Goal) ---
    last_equity = float(account.last_equity)
    daily_gain_pct = (equity - last_equity) / last_equity

    harvest_mode = False
    if daily_gain_pct >= DAILY_TARGET_PCT:
        print(f"💰 DAILY GOAL HIT (+{daily_gain_pct:.2%})! Entering Harvest Mode.")
        harvest_mode = True

    positions = api.list_positions()
    open_orders = api.list_orders(status='open')
    held_symbols = {p.symbol for p in positions}
    held_symbols.update({o.symbol for o in open_orders})

    # --- 0. CHECK PANIC ---
    is_panic = get_market_fear_index()
    if is_panic:
        print("🚨 PANIC MODE ACTIVATED.")
        for p in positions:
            if p.symbol != HEDGE_SYMBOL and float(p.qty) > 0:
                print(f"😱 PANIC SELL: Liquidating {p.symbol}")
                place_order(p.symbol, p.qty, 'sell', float(p.current_price), 'panic_liquidate')

        if HEDGE_SYMBOL not in held_symbols:
            has_pending_gld = any(o.symbol == HEDGE_SYMBOL for o in open_orders)
            if not has_pending_gld:
                bars = api.get_bars(HEDGE_SYMBOL, tradeapi.rest.TimeFrame.Day, limit=1).df
                if not bars.empty:
                    gld_price = bars.iloc[-1]['close']
                    qty = int((float(account.cash) * 0.20) / gld_price)
                    if qty > 0:
                        print(f"🛡️ HEDGING: Buying {qty} shares of {HEDGE_SYMBOL}")
                        place_order(HEDGE_SYMBOL, qty, 'buy', gld_price, 'hedge_entry')
    else:
        for p in positions:
            if p.symbol == HEDGE_SYMBOL:
                has_pending = any(o.symbol == HEDGE_SYMBOL for o in open_orders)
                if not has_pending:
                    print("✅ PANIC OVER: Selling Gold Hedge.")
                    place_order(HEDGE_SYMBOL, p.qty, 'sell', float(p.current_price), 'hedge_exit')

    # --- 1. MANAGE POSITIONS (With Harvest & Ratchet) ---
    longs = [p for p in positions if float(p.qty) > 0]
    long_count = len(longs)
    print(f"📈 Current Long Positions: {long_count}/{MAX_POSITIONS}")

    for p in positions:
        symbol = p.symbol
        if symbol == HEDGE_SYMBOL: continue
        if float(p.qty) < 0: continue  # Skip Shorts

        qty = float(p.qty)
        entry = float(p.avg_entry_price)
        current = float(p.current_price)
        pct_profit = (current - entry) / entry

        # --- 🌾 HARVEST LOGIC ---
        # If we hit the daily goal, sell anything that is green.
        if harvest_mode and pct_profit > 0:
            print(f"✅ HARVEST: Closing {symbol} (+{pct_profit:.2%}) to bank Daily Goal.")
            place_order(symbol, qty, 'sell', current, 'harvest_win')
            long_count -= 1
            continue

        # --- 🛡️ THE RATCHET (Trailing Stop) ---
        # (Your Original Logic Preserved)
        stop_thresh = HARD_STOP_PCT  # Default -10%

        # Tier 1: Secure the small win
        if pct_profit > 0.02: stop_thresh = 0.00

        # Tier 2: Secure the medium win
        if pct_profit > 0.05: stop_thresh = 0.02

        # Tier 3: Secure the big win
        if pct_profit > 0.10: stop_thresh = 0.07

        if pct_profit < stop_thresh:
            print(f"🛑 TRAILING STOP HIT: {symbol} Profit:{pct_profit:.2%} < Threshold:{stop_thresh:.2%}")
            place_order(symbol, abs(qty), 'sell', current, 'trailing_stop')
            long_count -= 1
            continue

        # --- 💰 TAKE PROFIT ---
        # Only take standard profit if NOT in Harvest Mode (Harvest handles greens anyway)
        if not harvest_mode:
            _, z, _, _ = get_technical_data(symbol)
            if z is None: continue

            if regime_map.get(symbol,
                              'MEAN_REVERSION') == 'MEAN_REVERSION' and z > EXIT_Z and pct_profit > PROFIT_GUARD:
                print(f"💰 TAKE PROFIT: {symbol} (Z:{z:.2f} > {EXIT_Z})")
                place_order(symbol, qty, 'sell', current, 'take_profit')
                long_count -= 1

    # --- 2. HUNTING TRADES ---

    # 🛑 BLOCK NEW BUYS IF HARVEST MODE IS ON
    if harvest_mode:
        print("🛑 Harvest Mode Active: No new buys. Managing current positions only.")
        return

    if long_count >= MAX_POSITIONS:
        print("Portfolio Full (Longs). No new buys.")
        return

    print("Checking Cooldowns...")
    cooldown_blacklist = get_cooldown_list()
    print(f"Banned Symbols (Cooldown): {len(cooldown_blacklist)}")

    all_tickers = list(regime_map.keys())
    if not all_tickers: all_tickers = ['AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT']
    np.random.shuffle(all_tickers)

    print(f"Scanning {len(all_tickers)} candidates...")

    for i, symbol in enumerate(all_tickers):
        if cash < CASH_BUFFER:
            print("Cash Buffer Hit. Stopping Scan.")
            break
        if symbol in held_symbols: continue
        if symbol in cooldown_blacklist: continue

        # Double check harvest mode inside loop
        if harvest_mode: break

        if long_count >= MAX_POSITIONS:
            print("Portfolio Full (Longs). Ending Scan.")
            break

        if i % 10 == 0: print(f"Scanned {i}/{len(all_tickers)}...", end='\r')
        time.sleep(0.5)

        price, z, sma_50, cmf_slope = get_technical_data(symbol)
        if price is None: continue
        print(f"🔍Checking {symbol} | Price: {price} | Z: {z} | SMA: {sma_50} | CMF : {cmf_slope: .2f}")

        regime = regime_map.get(symbol, 'MEAN_REVERSION')
        signal = None
        reason = ""

        if is_panic and regime == 'TRENDING': continue

        if regime == 'MEAN_REVERSION':
            if z < ENTRY_Z and cmf_slope > 0 :
                sent_score, consensus = get_sentiment_consensus(symbol)
                if sent_score > -0.8:
                    signal = 'buy'
                    reason = f"Oversold Z:{z:.2f}"

        elif regime == 'TRENDING' and not is_panic:
            if price > sma_50:
                sent_score, consensus = get_sentiment_consensus(symbol)
                if sent_score >= 0.6:
                    signal = 'buy'
                    reason = "Trend + News"

        if signal:
            target_amount = (equity * 1.50) / MAX_POSITIONS
            shares = int(target_amount / price)
            if shares > 0:
                print(f"\n🚀 {signal.upper()}: {symbol} | {reason}")
                place_order(symbol, shares, signal, price, 'entry')
                cash -= (shares * price)
                held_symbols.add(symbol)
                long_count += 1
                if long_count >= MAX_POSITIONS: break


if __name__ == "__main__":
    end_time = time.time() + (5.75 * 3600)
    print("--- 🟢 STARTING CONTINUOUS TRADING SESSION (5h 45m) ---")
    while time.time() < end_time:
        try:
            run_hedge_fund()
        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
        print("Waiting 60 seconds...")
        time.sleep(60)
    print("--- 🔴 SESSION ENDING ---")



