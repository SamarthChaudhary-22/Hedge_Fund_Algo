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

# --- 🐻 GRIZZLY SHORT ENGINE v7 (Trend Breakdown Logic) ---
API_KEY = os.getenv('APCA_API_KEY_ID')
SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
BASE_URL = "https://paper-api.alpaca.markets"

# RISK MANAGEMENT
MAX_SHORT_POSITIONS = 20
POSITION_SIZE_NORMAL = 0.05  # 3% per trade
POSITION_SIZE_PANIC = 0.03  # 1.5% (Defensive)
PANIC_EXPOSURE_CAP = 0.30  # Max 30% account short
HARD_STOP_PCT = 0.10  # 5% Hard Stop

# STRATEGY THRESHOLDS
ENTRY_Z_SHORT = -1.50  # Modified: Short if momentum is weak (Z > 1.0) AND Trend Broken
EXIT_Z_SHORT = -0.50  # Cover when price returns to Mean (Fair Value)
NEG_CONSENSUS_REQ = 0.30
EARNINGS_LOOKAHEAD = 3

# KEYWORDS
RED_FLAGS = [

    # 1. The Classics (confirmed by your AI)
    "miss", "fail", "fall", "drop", "decline", "tumble", "plunge", "sink",
    "slip", "retreat", "crash", "dump", "loss", "lower", "weak",

    # 2. The "Action" Verbs (found by your AI)
    "warns", "cuts", "slashes", "downgrade", "investigation", "lawsuit",
    "scandal", "probe", "breach", "halt", "delay", "resign",

    # 3. The "Macro" Fear words (found by your AI)
    "inflation", "tariffs", "yields", "fear", "uncertainty", "headwinds",
    "pressure", "volatility", "supply", "debt",

    # 4. The "Market Movers" (AI Discovery: When these firms speak, stocks move)
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
    """
    Returns: Price, Z-Score, and SMA_50 (Trend Line)
    """
    try:
        # Fetch 100 days to ensure we have enough for SMA 50
        start_date = (datetime.now() - timedelta(days=300)).strftime('%Y-%m-%d')
        bars = api.get_bars(symbol, tradeapi.rest.TimeFrame.Day, start=start_date, limit=100).df
        if bars.empty: return None, None, None

        closes = bars['close'].values
        current_price = closes[-1]

        if current_price < 5.00: return None, None, None

        # 1. Z-Score (20 Day)
        if len(closes) >= 20:
            mean_20 = np.mean(closes[-20:])
            std_20 = np.std(closes[-20:])
            z_score = (current_price - mean_20) / std_20 if std_20 > 0 else 0
        else:
            z_score = 0

        # 2. SMA 50 (The "Line in the Sand")
        if len(closes) >= 50:
            sma_50 = np.mean(closes[-50:])
        else:
            sma_50 = current_price  # Fallback

        return current_price, z_score, sma_50
    except:
        return None, None, None


def get_earnings_status(symbol):
    """
    Robust check for upcoming earnings using yfinance.
    Handles both Dict (new) and DataFrame (old) outputs.
    """
    try:
        ticker = yf.Ticker(symbol)
        cal = ticker.calendar

        if cal is None:
            return False, "N/A"

        # 1. Handle New yfinance (Returns Dictionary)
        if isinstance(cal, dict):
            # The dictionary usually looks like: {'Earnings Date': [datetime.date(2025, 1, 20)]}
            # We want the first date from the list
            if 'Earnings Date' in cal:
                dates = cal['Earnings Date']
                if len(dates) > 0:
                    earnings_date = dates[0]
                else:
                    return False, "N/A"
            else:
                # Fallback: Just grab the first available value
                vals = list(cal.values())
                if len(vals) > 0 and len(vals[0]) > 0:
                    earnings_date = vals[0][0]
                else:
                    return False, "N/A"

        # 2. Handle Old yfinance (Returns DataFrame)
        elif not cal.empty:
            earnings_date = cal.iloc[0][0]
        else:
            return False, "N/A"

        # 3. Process the Date
        # Ensure it's a datetime object (sometimes it's a datetime.date)
        if not isinstance(earnings_date, datetime):
            # Convert datetime.date to datetime
            earnings_date = datetime.combine(earnings_date, datetime.min.time())

        days_until = (earnings_date - datetime.now()).days

        if 0 <= days_until <= EARNINGS_LOOKAHEAD:
            return True, earnings_date.strftime('%Y-%m-%d')
        return False, "Later"

    except Exception as e:
        # DEBUG: Print the actual error so we can see it in logs
        # print(f"⚠️ Earn Error {symbol}: {e}")
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
    """
        Smart Close: Uses Market Orders during the day,
        and Limit Orders (Extended Hours) at night.
        """
    try:
        # 1. Get info to determine Order Type
        clock = api.get_clock()
        qty = 0
        try:
            pos = api.get_position(symbol)
            qty = abs(int(float(pos.qty)))
            side = 'buy' if float(pos.qty) < 0 else 'sell'  # Short -> Buy to Cover
        except:
            print(f"⚠️ Cannot close {symbol}: Position not found.")
            return

        # 2. CANCEL OPEN ORDERS (To clear the path)
        orders = api.list_orders(status='open', symbols=[symbol])
        for o in orders:
            api.cancel_order(o.id)
            print(f"🗑️ Canceled Open Order for {symbol}")

        # 3. EXECUTE CLOSE
        if clock.is_open:
            # NORMAL HOURS -> Market Order (Fastest)
            api.close_position(symbol)
            print(f"💰 MARKET CLOSE: {symbol} | {reason}")
        else:
            # EXTENDED HOURS -> Limit Order (Required)
            # Fetch current price to set a limit
            trade = api.get_latest_trade(symbol)
            current_price = trade.price

            # Set Limit Price with buffer (Buy high / Sell low to ensure fill)
            # If Buying to cover: Pay 1% MORE to ensure fill
            # If Selling to close: Accept 1% LESS to ensure fill
            limit_price = round(current_price * 1.01, 2) if side == 'buy' else round(current_price * 0.99, 2)

            api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='limit',
                time_in_force='day',
                limit_price=limit_price,
                extended_hours=True  # <--- THE MAGIC KEY
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

        api.submit_order(symbol, qty, 'buy', 'stop', stop_price=stop_price, time_in_force='gtc')
        print(f"🛡️ STOP LOSS SET: {symbol} @ ${stop_price}")

    except Exception as e:
        print(f"❌ Short Failed: {e}")


def run_short_engine():
    print(f"--- 🐻 GRIZZLY SHORT ENGINE v7 (Trend Breakdown): {datetime.now(pytz.timezone('US/Eastern'))} ---")

    account = api.get_account()
    equity = float(account.equity)

    # 1. MANAGE POSITIONS
    positions = api.list_positions()
    for p in positions:
        if int(p.qty) >= 0: continue

        symbol = p.symbol
        entry_price = float(p.avg_entry_price)
        current_price = float(p.current_price)
        qty = abs(int(p.qty))
        pct_profit = (entry_price - current_price) / entry_price

        if pct_profit < -HARD_STOP_PCT:
            close_position(symbol, f"Stop Loss Hit ({pct_profit:.2%})")
            continue

        _, z, _ = get_technical_data(symbol)
        if z is not None:
            if z < EXIT_Z_SHORT and pct_profit > 0.01:
                close_position(symbol, f"Take Profit (Z:{z:.2f} < {EXIT_Z_SHORT})")

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
    universe = get_sp500_universe()
    np.random.shuffle(universe)
    print(f"Scanning {len(universe)} stocks...")

    for symbol in universe:
        time.sleep(0.5)
        if any(p.symbol == symbol for p in positions): continue

        is_earnings, earn_date = get_earnings_status(symbol)
        price, z, sma_50 = get_technical_data(symbol)
        if price is None:
            print(f"Skipping {symbol}, price not available.")
            continue
        print(f"🔍Check {symbol}: price: {price} | z: {z} | sma_50: {sma_50} | Earnings: {earn_date} ")

        if is_earnings:
            print(f"👀 EARNINGS WATCH: {symbol} on {earn_date}...")
            neg_ratio, avg_score, flags = analyze_sentiment(symbol, mode='earnings')

            if neg_ratio >= NEG_CONSENSUS_REQ or flags >= 2:
                reason = f"Earnings Disaster (Neg: {neg_ratio:.0%})"
                print(f"💀 EARNINGS SIGNAL: {symbol} | {reason}")

                price = api.get_latest_trade(symbol).price
                qty = int((equity * current_pos_size) / price)
                if qty > 0: place_short_order(symbol, qty, reason, stop_pct=1.03)

        else:
            # TECHNICAL SHORT (Modified)
            price, z, sma_50 = get_technical_data(symbol)

            # REQUIREMENT: Price MUST be below 50-Day MA (Breakdown)
            if price < sma_50:
                if z > ENTRY_Z_SHORT:
                    neg_ratio, avg_score, flags = analyze_sentiment(symbol, mode='normal')

                    if avg_score > 0.2:
                        print(f"✋ SKIP {symbol}: Breakdown but News is Good")
                    else:
                        reason = f"Breakdown Short (Price < SMA50 & Z:{z:.2f})"
                        print(f"📉 TECHNICAL SIGNAL: {symbol} | {reason}")

                        price = api.get_latest_trade(symbol).price
                        qty = int((equity * current_pos_size) / price)
                        if qty > 0: place_short_order(symbol, qty, reason, stop_pct=current_stop_buffer)

        shorts = [p for p in api.list_positions() if int(p.qty) < 0]
        if len(shorts) >= MAX_SHORT_POSITIONS: break


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
