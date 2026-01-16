import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import os
import sys
import nltk
import time
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import pytz

# --- 🐻 GRIZZLY SHORT ENGINE vFinal (Harvest Mode) ---
API_KEY = os.getenv('APCA_API_KEY_ID')
SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
BASE_URL = "https://paper-api.alpaca.markets"

# RISK MANAGEMENT
MAX_SHORT_POSITIONS = 30
POSITION_SIZE_NORMAL = 0.06  # Reduced to 6% (Aggressive but sustainable)
POSITION_SIZE_PANIC = 0.05  # 5% Defensive
PANIC_EXPOSURE_CAP = 0.30  # Max 30% account short
HARD_STOP_PCT = 0.10  # 10% Hard Stop
DAILY_TARGET_PCT = 0.015  # 1.5% Daily Goal (Matches Long Bot)

# OPTIMIZED STRATEGY (Rank #1: Momentum Breakdown)
ENTRY_Z_SHORT = -1.75  # Short if Z > -1.75 (and Price < SMA50)
EXIT_Z_SHORT = -2.00  # Cover when Z crashes below -2.00
NEG_CONSENSUS_REQ = 0.30
EARNINGS_LOOKAHEAD = 3

# KEYWORDS
RED_FLAGS = [
    "miss", "fail", "fall", "drop", "decline", "tumble", "plunge", "sink",
    "slip", "retreat", "crash", "dump", "loss", "lower", "weak",
    "warns", "cuts", "slashes", "downgrade", "investigation", "lawsuit",
    "scandal", "probe", "breach", "halt", "delay", "resign",
    "inflation", "tariffs", "yields", "fear", "uncertainty", "headwinds",
    "pressure", "volatility", "supply", "debt",
    "goldman", "jpmorgan", "scotiabank", "keefe", "bofa", "citigroup"
]

# SETUP
if not API_KEY: sys.exit("API Key Missing")
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
vader = SentimentIntensityAnalyzer()


def get_sp500_universe():
    try:
        df = pd.read_csv('sp500_regime.csv')
        return df['symbol'].tolist()
    except:
        return ['TSLA', 'NVDA', 'AMD', 'META', 'NFLX', 'BA', 'LULU', 'ROKU']


def get_technical_data(symbol):
    try:
        start_date = (datetime.now() - timedelta(days=300)).strftime('%Y-%m-%d')
        bars = api.get_bars(symbol, tradeapi.rest.TimeFrame.Day, start=start_date, limit=100, feed='iex').df
        if bars.empty: return None, None, None

        closes = bars['close'].values
        current_price = closes[-1]

        if current_price < 5.00: return None, None, None

        if len(closes) >= 20:
            mean_20 = np.mean(closes[-20:])
            std_20 = np.std(closes[-20:])
            z_score = (current_price - mean_20) / std_20 if std_20 > 0 else 0
        else:
            z_score = 0

        if len(closes) >= 50:
            sma_50 = np.mean(closes[-50:])
        else:
            sma_50 = current_price

        return current_price, z_score, sma_50
    except:
        return None, None, None


def get_earnings_status(symbol):
    try:
        ticker = yf.Ticker(symbol)
        cal = ticker.calendar

        if cal is None: return False, "N/A"

        if isinstance(cal, dict):
            if 'Earnings Date' in cal:
                dates = cal['Earnings Date']
                if len(dates) > 0:
                    earnings_date = dates[0]
                else:
                    return False, "N/A"
            else:
                vals = list(cal.values())
                if len(vals) > 0 and len(vals[0]) > 0:
                    earnings_date = vals[0][0]
                else:
                    return False, "N/A"
        elif not cal.empty:
            earnings_date = cal.iloc[0][0]
        else:
            return False, "N/A"

        if not isinstance(earnings_date, datetime):
            earnings_date = datetime.combine(earnings_date, datetime.min.time())

        days_until = (earnings_date - datetime.now()).days

        if 0 <= days_until <= EARNINGS_LOOKAHEAD:
            return True, earnings_date.strftime('%Y-%m-%d')
        return False, "Later"

    except Exception:
        return False, "Error"


def analyze_sentiment(symbol, mode='normal'):
    limit = 80 if mode == 'earnings' else 30
    current_day = datetime.now().weekday()
    lookback_hours = 72 if current_day == 0 else 24
    try:
        start_time = (datetime.now() - timedelta(hours=lookback_hours)).strftime('%Y-%m-%dT%H:%M:%SZ')
        news_list = api.get_news(symbol=symbol, limit=limit, start=start_time)
        if not news_list: return 0.0, 0.0, 0

        neg_articles = 0
        total_score = 0
        flag_count = 0
        total = 0

        for article in news_list:
            headline = article.headline.lower()
            score = vader.polarity_scores(article.headline)['compound']

            total_score += score
            total += 1

            if score < -0.05: neg_articles += 1
            if any(flag in headline for flag in RED_FLAGS): flag_count += 1

        if total == 0: return 0.0, 0.0, 0

        neg_ratio = neg_articles / total
        avg_score = total_score / total

        return neg_ratio, avg_score, flag_count
    except:
        return 0.0, 0.0, 0


def get_market_fear_index():
    try:
        vix_bars = api.get_bars("VIXY", tradeapi.rest.TimeFrame.Day, limit=2).df
        if len(vix_bars) < 2: return False
        change = (vix_bars.iloc[-1]['close'] - vix_bars.iloc[-2]['close']) / vix_bars.iloc[-2]['close']
        if change > 0.05:
            print(f"⚠️ MARKET PANIC DETECTED (VIXY +{change:.2%})")
            return True
        return False
    except:
        return False


def get_total_short_exposure(account):
    short_val = abs(float(account.short_market_value))
    equity = float(account.equity)
    if equity == 0: return 0
    return short_val / equity


def close_position(symbol, reason):
    try:
        clock = api.get_clock()
        qty = 0
        try:
            pos = api.get_position(symbol)
            qty = abs(int(float(pos.qty)))
            side = 'buy' if float(pos.qty) < 0 else 'sell'
        except:
            print(f"⚠️ Cannot close {symbol}: Position not found.")
            return

        orders = api.list_orders(status='open', symbols=[symbol])
        for o in orders:
            api.cancel_order(o.id)
            print(f"🗑️ Canceled Open Order for {symbol}")

        if clock.is_open:
            api.close_position(symbol)
            print(f"💰 MARKET CLOSE: {symbol} | {reason}")
        else:
            trade = api.get_latest_trade(symbol)
            current_price = trade.price
            limit_price = round(current_price * 1.01, 2) if side == 'buy' else round(current_price * 0.99, 2)

            api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='limit',
                time_in_force='day',
                limit_price=limit_price,
                extended_hours=True
            )
            print(f"🌙 EXTENDED CLOSE: {symbol} | {qty} @ {limit_price} | {reason}")

    except Exception as e:
        print(f"❌ Failed to Close {symbol}: {e}")


def place_short_order(symbol, qty, reason, stop_pct):
    try:
        current_price = api.get_latest_trade(symbol).price
        stop_price = round(current_price * stop_pct, 2)

        clock = api.get_clock()
        if clock.is_open:
            api.submit_order(symbol, qty, 'sell', 'market', 'gtc')
            print(f"✅ MARKET SHORT: {symbol} | {qty} | {reason}")
        else:
            limit_price = round(current_price * 0.99, 2)
            api.submit_order(symbol, qty, 'sell', 'limit', limit_price=limit_price,
                             time_in_force='day', extended_hours=True)
            print(f"🌙 EXTENDED SHORT: {symbol} | {qty} @ {limit_price} | {reason}")

        # Submitting Stop Loss
        api.submit_order(symbol, qty, 'buy', 'stop', stop_price=stop_price, time_in_force='gtc')
        print(f"🛡️ STOP LOSS SET: {symbol} @ ${stop_price}")

        return True  # <--- NEW: Signal Success

    except Exception as e:
        print(f"❌ Short Failed: {e}")
        return False  # <--- NEW: Signal Failure

def run_short_engine():
    print(f"--- 🐻 GRIZZLY SHORT ENGINE vFinal (Harvest Mode): {datetime.now(pytz.timezone('US/Eastern'))} ---")

    account = api.get_account()
    equity = float(account.equity)

    # --- 🆕 HARVEST CHECK (Daily Goal) ---
    last_equity = float(account.last_equity)
    daily_gain_pct = (equity - last_equity) / last_equity

    harvest_mode = False
    if daily_gain_pct >= DAILY_TARGET_PCT:
        print(f"💰 DAILY GOAL HIT (+{daily_gain_pct:.2%})! Short Engine entering Harvest Mode.")
        harvest_mode = True

    # 1. MANAGE POSITIONS
    positions = api.list_positions()
    shorts = [p for p in positions if int(p.qty) < 0]
    short_count = len(shorts)
    print(f"📉 Current Short Positions: {short_count}/{MAX_SHORT_POSITIONS}")

    for p in positions:
        if int(p.qty) >= 0: continue

        symbol = p.symbol
        entry_price = float(p.avg_entry_price)
        current_price = float(p.current_price)
        qty = abs(int(p.qty))

        # Calculate Short Profit (Inverse of price movement)
        pct_profit = (entry_price - current_price) / entry_price

        # --- 🌾 HARVEST LOGIC ---
        # If Daily Goal Hit -> Close Winners immediately
        if harvest_mode and pct_profit > 0:
            close_position(symbol, f"Harvest Win (+{pct_profit:.2%})")
            short_count -= 1
            continue

        # --- 🛡️ THE RATCHET (Trailing Stop for Shorts) ---
        stop_thresh = -HARD_STOP_PCT  # Default -10% (Price goes up 10%)

        # Tier 1: > 1.5% Profit -> Break Even
        if pct_profit > 0.015: stop_thresh = -0.005  # Price goes up 0.5%

        # Tier 2: > 5% Profit -> Lock 2%
        if pct_profit > 0.05: stop_thresh = 0.02  # Profit drops to 2%

        # Tier 3: > 10% Profit -> Lock 5%
        if pct_profit > 0.10: stop_thresh = 0.05

        # Check Stop Logic
        # Note: Stop threshold logic is tricky for shorts in terms of pct_profit.
        # If pct_profit drops below threshold, we exit.
        if pct_profit < stop_thresh:
            close_position(symbol, f"Trailing Stop Hit ({pct_profit:.2%} < {stop_thresh:.2%})")
            short_count -= 1
            continue

        # --- 💰 TAKE PROFIT (Normal Mode) ---
        # Optimized Strategy: Cover into the Crash (Z < -2.0)
        if not harvest_mode:
            _, z, _ = get_technical_data(symbol)
            if z is not None:
                if z < EXIT_Z_SHORT and pct_profit > 0.01:
                    close_position(symbol, f"Panic Cover (Z:{z:.2f} < {EXIT_Z_SHORT})")
                    short_count -= 1

    # 2. PANIC & CAP CHECK
    is_panic = get_market_fear_index()
    current_pos_size = POSITION_SIZE_NORMAL
    current_stop_buffer = 1.10

    if is_panic:
        current_pos_size = POSITION_SIZE_PANIC
        current_stop_buffer = 1.10
        if get_total_short_exposure(account) > PANIC_EXPOSURE_CAP:
            print("🛑 PANIC CAP HIT. No new shorts.")
            return

    # 3. SCANNING

    # 🛑 BLOCK NEW SHORTS IN HARVEST MODE
    if harvest_mode:
        print("🛑 Harvest Mode Active: No new shorts.")
        return

    if short_count >= MAX_SHORT_POSITIONS:
        print("🛑 Max Short Positions Reached. Scanning Paused.")
        return

    universe = get_sp500_universe()
    np.random.shuffle(universe)
    print(f"Scanning {len(universe)} stocks...")

    for symbol in universe:
        time.sleep(0.5)

        # Harvest Check Loop
        if harvest_mode: break

        if short_count >= MAX_SHORT_POSITIONS:
            print("🛑 Portfolio Full. Ending Scan.")
            break

        if any(p.symbol == symbol for p in positions): continue

        is_earnings, earn_date = get_earnings_status(symbol)
        price, z, sma_50 = get_technical_data(symbol)

        if price is None:
            continue

        print(f"🔍Check {symbol}: price: {price} | z: {z:.2f} | sma_50: {sma_50:.2f} | Earnings: {earn_date} ")

        order_placed = False

        if is_earnings:
            if earn_date != "Later" and earn_date != "Error":
                print(f"👀 EARNINGS WATCH: {symbol} on {earn_date}...")
                neg_ratio, avg_score, flags = analyze_sentiment(symbol, mode='earnings')

                if neg_ratio >= NEG_CONSENSUS_REQ or flags >= 2:
                    reason = f"Earnings Disaster (Neg: {neg_ratio:.0%})"
                    print(f"💀 EARNINGS SIGNAL: {symbol} | {reason}")

                    price = api.get_latest_trade(symbol).price
                    qty = int((equity * current_pos_size) / price)
                    if qty > 0:
                        if place_short_order(symbol, qty, reason, stop_pct=1.03):
                            order_placed = True

        else:
            # TECHNICAL SHORT (Optimized Momentum Logic)
            if price < sma_50:
                # ENTRY CONDITION: Trend Breakdown (Z > -1.75 is almost always true if price < SMA50)
                if z > ENTRY_Z_SHORT:
                    neg_ratio, avg_score, flags = analyze_sentiment(symbol, mode='normal')

                    if avg_score > 0.2:
                        print(f"✋ SKIP {symbol}: Breakdown but News is Good")
                    else:
                        reason = f"Breakdown Short (Price < SMA50 & Z:{z:.2f})"
                        print(f"📉 TECHNICAL SIGNAL: {symbol} | {reason}")

                        price = api.get_latest_trade(symbol).price
                        qty = int((equity * current_pos_size) / price)
                        if qty > 0:
                            if place_short_order(symbol, qty, reason, stop_pct=current_stop_buffer):
                                order_placed = True

        if order_placed:
            short_count += 1


if __name__ == "__main__":
    end_time = time.time() + (5.75 * 3600)
    print("--- 🟢 STARTING CONTINUOUS TRADING SESSION (5h 45m) ---")
    while time.time() < end_time:
        try:
            run_short_engine()
        except Exception as e:
            print(f"CRITICAL ERROR in loop: {e}")
        print("Waiting 60 seconds...")
        time.sleep(60)

    print("--- 🔴 SESSION ENDING ---")
