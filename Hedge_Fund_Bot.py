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
import yfinance as yf
import smart_hedge as Smart_Hedge
import json

# --- 🏆 FINAL CONFIGURATION (Rank #1: 71,680% Return) ---
API_KEY = os.getenv('APCA_API_KEY_ID')
SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
BASE_URL = "https://paper-api.alpaca.markets"

# RISK & STRATEGY
MAX_POSITIONS = 40
CASH_BUFFER = 2000
HARD_STOP_PCT = -0.10  # Tightened Stop Loss (Optimized)
DAILY_TARGET_PCT = 0.015 # If Portfolio is up 1.5% today, Close All.
HEDGE_RESERVE_PCT = 0.02

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
        start_date = (datetime.now() - timedelta(days=300)).strftime('%Y-%m-%d')
        bars = api.get_bars(symbol, tradeapi.rest.TimeFrame.Day, start=start_date, limit=300, feed='iex').df

        if bars.empty or len(bars) < 50:
            return None, None, None, None, None

        # Data Prep
        closes = bars['close']
        highs = bars['high']
        lows = bars['low']
        volumes = bars['volume']
        current_price = closes.iloc[-1]

        bars['pct_change'] = bars['close'].pct_change()
        current_daily_vol = bars['pct_change'].rolling(window=20).std().iloc[-1]

        # 2. BLIND Z-SCORE (Shifted by 1 Day)
        # We calculate mean/std on the HISTORY, excluding today to match backtest
        # We look at the last 20 days *before* today
        history = closes.iloc[:-1]
        if len(history) < 20: return None, None, None, None, None

        mean_20 = history.iloc[-20:].mean()
        std_20 = history.iloc[-20:].std()

        if std_20 == 0: return None, None, None, None, None
        z_score = (current_price - mean_20) / std_20

        # 3. SMA 50
        sma_50 = closes.iloc[-50:].mean()
        sma_distance = (current_price - sma_50) / sma_50

        # 4. CMF (Chaikin Money Flow) & SLOPE
        # Money Flow Multiplier
        mfm = ((closes - lows) - (highs - closes)) / (highs - lows)
        mfm = mfm.fillna(0)
        mfv = mfm * volumes

        # 20-period CMF
        cmf = mfv.rolling(20).sum() / volumes.rolling(20).sum()

        # CMF Slope (Current - 10 days ago)
        # Ensure we have enough data for slope
        if len(cmf) < 12: return None, None, None, None, None

        cmf_current = cmf.iloc[-1]
        cmf_prev = cmf.iloc[-11]  # 10 bars ago
        cmf_slope = cmf_current - cmf_prev

        return current_price, z_score, sma_distance, cmf_current, current_daily_vol

    except Exception as e:
        print(f"⚠️ Metrics Error {symbol}: {e}")
        return None, None, None, None, None

def get_market_fear_index():
    try:
        vix_start = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        vix_bars = api.get_bars(FEAR_SYMBOL, tradeapi.rest.TimeFrame.Day, start=vix_start, limit=10).df
        if len(vix_bars) < 2: return False
        current_bar = vix_bars.iloc[-1]
        prev_bar = vix_bars.iloc[-2]

        current = current_bar['close']
        open_price = current_bar['open']
        prev_close = prev_bar['close']
        daily_change = (current - prev_close) / prev_close
        intraday_change = (current - open_price) / open_price

        if daily_change > FEAR_THRESHOLD or intraday_change > FEAR_THRESHOLD:
            print(f"⚠️ Volatility Spike : Daily = {daily_change:.1%} | Intraday = {intraday_change:.1%}")
            return True
        return False
    except Exception as e:
        print(f"Error in get_market_fear_index: {e}")
        return False


class FactorOrthogonalizer:
    def __init__(self, weights_filepath = 'factor_weights.json'):
        try:
            with open(weights_filepath, 'r') as f:
                self.weights = json.load(f)
                print(f"factor_weights loaded from {weights_filepath}")
        except Exception as e:
            print(f"Error in factor_weights: {e}")
            self.weights = {}

    def clean_weights(self, symbol, raw_z, raw_c, raw_s):
        if symbol not in self.weights:
            return raw_z, raw_c, raw_s

        w = self.weights[symbol]
        orth_z = raw_z/w['sigma_Z']
        sigma_c = w.get('Sigma_C_res', w.get('sigma_C_res', 1.0))
        orth_c = (raw_c - w['alpha_C'] - (w['beta_C_Z'] * orth_z)) / sigma_c
        orth_s = (raw_s -w['alpha_S'] - (w['beta_S_Z'] * orth_z) - (w['beta_S_C'] * orth_c)) / w['sigma_S_res']

        return orth_z, orth_c, orth_s

orthoganolizer = FactorOrthogonalizer()



def place_order(symbol, qty, side, current_price, order_type_label="manual"):
    try:
        clock = api.get_clock()
        # Unique ID for tagging (e.g., algo_stop_loss_AAPL_167823...)
        unique_id = f"algo_{order_type_label}_{symbol}_{int(time.time())}"

        if clock.is_open:
            api.submit_order(symbol, qty, side, 'market', 'gtc', client_order_id=unique_id)
            print(f"✅ MARKET ORDER ({order_type_label}): {side} {symbol} ({qty})")
        else:
            yf_ticker = yf.Ticker(symbol)
            quote = api.get_latest_quote(symbol)

            try:
                df = yf_ticker.history(period='1d', interval='1m', prepost=True)

                if not df.empty:
                    yf_price = float(df.iloc[-1]['Close'])
                else:
                    yf_price = yf_ticker.fast_info['last_price']
            except:
                yf_price = 0

            if side == 'buy':
                current_price = yf_price
                if current_price is None or current_price == 0:
                    print(f"⚠️ Yahoo Failed/Zero for {symbol}. Using IEX.")
                    current_price = quote.ask_price
                if current_price == 0:
                    current_price = quote.bid_price
                limit_price = round(current_price * 1.01,2)

            else:
                current_price = yf_price
                if current_price is None or current_price == 0:
                    print(f"⚠️ Yahoo Failed/Zero for {symbol}. Using IEX.")
                    current_price = quote.bid_price
                if current_price == 0:
                    current_price = quote.ask_price
                limit_price = round(current_price * 0.99,2)

            if current_price == 0 or current_price is None:
                print(f"⚠️ DATA ERROR: Could not fetch valid quote for {symbol} from ANY source. Skipping.")
                return

            api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type = 'limit',
                time_in_force='day',
                limit_price=limit_price,
                extended_hours=True,
                client_order_id=unique_id
            )
            print(f"🌙 EXTENDED ORDER: {symbol} | {qty} @ {limit_price} (Ref: {current_price})")
    except Exception as e:
        error_msg = str(e).lower()
        if "insufficient buying power" in error_msg  or "buying power" in error_msg:
            print(f"⚠️ SKIP {symbol}: {error_msg}")
            return False
        else:
            print(f"❌❌ ERROR {side}ing {symbol}: {error_msg}")
            return False


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

def check_tape(symbol, lookback = 150):
    try:
        start_date = (datetime.now() - timedelta(days=lookback)).strftime('%Y-%m-%d')
        trades_resp = api.get_trades(symbol, start = start_date, limit=lookback, feed='iex')
        trades = list(trades_resp)
        if not trades or len(trades) < 30:
            return {'signal': 'neutral', 'reason': 'insufficient_data.'}

        buy_vol = 0
        sell_vol = 0
        last_aggressor = 0

        for i in range(1, len(trades)):
            raw_curr = trades[i]._raw
            raw_prev = trades[i - 1]._raw

            price = float(raw_curr.get('p', raw_curr.get('price', 0)))
            prev_price = float(raw_prev.get('p', raw_prev.get('price', 0)))
            size = float(raw_curr.get('s', raw_curr.get('size', 0)))

            if price > prev_price:
                buy_vol += size
                last_aggressor = 1
            elif price < prev_price:
                sell_vol += size
                last_aggressor = -1
            else:
                if last_aggressor == 1:
                    buy_vol += size
                elif last_aggressor == -1:
                    sell_vol += size

        total_vol = buy_vol + sell_vol
        if total_vol == 0:
            return {'signal': 'neutral', 'reason': 'insufficient_data.'}

        buy_ratio = buy_vol / total_vol

        start_price = float(trades[0]._raw.get('p', trades[0]._raw.get('price', 0)))
        end_price = float(trades[-1]._raw.get('p', trades[-1]._raw.get('price', 0)))
        price_delta = (end_price - start_price) / start_price
        print(f"Buys: {buy_ratio: .2%} | Price Move: {price_delta: .2%}")

        stagnation_threshold = 0.0005

        if buy_ratio > 0.70 and price_delta < stagnation_threshold:
            print(f"🧱 DETECTED HIDDEN SELL WALL! (Absorption at ${end_price})")
            return {'signal': 'sell_wall', 'reason': f"Absorption: {buy_ratio: .2%} Buys but price flat {price_delta: .2%}"}

        if buy_ratio < 0.30 and price_delta >= -stagnation_threshold:
            print(f"🧱 DETECTED HIDDEN BUY WALL! (Absorption at ${end_price})")
            return {'signal' : 'buy_wall', 'reason': f"Absorption: {1-buy_ratio: .2%} Sells but price flat {price_delta: .2%}"}

        return {'signal': 'clear', 'reason' : f'Normal Flow: {buy_ratio: .2%} Buy ratio, Move {price_delta: .2%}'}

    except Exception as e:
        print(f"⚠️ Error: {e}")
        return {'signal': 'neutral', 'reason' : f"Error: {e}"}
def check_oppossing_position(symbol, intended_direction):
    try:
        pos = api.get_position(symbol)
        qty = float(pos.qty)
        entry_price = float(pos.avg_entry_price)
        current_price = float(pos.current_price)

        is_long = qty < 0
        is_short = qty > 0

        if (intended_direction == 'long' and is_long) or (intended_direction == 'short' and is_short):
            return True

        if is_long:
            pct_profit = (current_price - entry_price) / entry_price * 100
        else:
            pct_profit = (entry_price - current_price) / entry_price * 100

        if pct_profit < -0.025:
            open_orders = api.list_orders(status='open', symbols=[symbol])
            for o in open_orders:
                api.cancel_order(o.id)

            api.close_position(symbol)
            time.sleep(2)
            return True
        else:
            return False
    except Exception as e:
        if 'position does not exist' in str(e).lower():
            return True
        else:
            print(f"Error Checking Position: {symbol}, {e}")
            return False


def calculate_dynamic_position(symbol, equity, price, daily_vol, pure_z, pure_c, held_symbols, max_positions=40):
    base_dollar_alloc = (equity * 1.50) / max_positions
    daily_target_vol = 0.015
    safe_vol = max(daily_vol, 0.005)
    vol_multiplier = daily_target_vol/daily_vol
    vol_multiplier = max(0.5, min(vol_multiplier, 2.0))

    conviction_mult = 1.0
    z_strength = abs(pure_z)

    if z_strength >= 3.5:
        conviction_mult = 1.50
    elif z_strength >= 2.5:
        conviction_mult = 1.25
    if abs(pure_c) > 2.0:
        conviction_mult += 0.20

    saturation_ratio = len(held_symbols) / max_positions
    saturation_penalty = 1.0
    if saturation_ratio >= 0.75:
        saturation_penalty = 0.80

    final_dollar_aloc = base_dollar_alloc * vol_multiplier * conviction_mult * saturation_penalty
    shares = int(final_dollar_aloc/ price)

    print(f"Size Calculated For {symbol}: Base: ${base_dollar_alloc: ,.0f} | Vol Mult: {vol_multiplier: .2f}x | Alpha Mult: {conviction_mult: .2f}x Final Allocation: ${final_dollar_aloc: ,.0f}")
    return shares


def run_hedge_fund():
    print(f"--- 🐺 Hedge Fund vFinal (Harvest Mode): {datetime.now(pytz.timezone('US/Eastern'))} ---")
    check_and_refresh_stale_orders()  #---Cleans up old limit orders
    regime_map = get_regime_map()
    account = api.get_account()
    equity = float(account.portfolio_value)
    cash = float(account.cash)
    buying_power = float(account.buying_power)
    hedge_reserve = equity * HEDGE_RESERVE_PCT
    print(f"Equity: ${equity} | BP: {buying_power: ,.2f} | Hedge Reserve: {hedge_reserve:,.2f}")
    insufficient_funds = buying_power < hedge_reserve

    if insufficient_funds:
        print(f"⛔ Reserving BP For Hedge: BP: ${buying_power: ,.2f} | Hedge Reserve: ${hedge_reserve: ,.2f}")

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
            try:
                quote = api.get_latest_quote(symbol)
                bid = float(quote.bp)
            except Exception as e:
                print(f"⚠️ Could Not Fetch Quote for {symbol}: {e}")
                bid = current
            real_pct_profit = (bid - entry) / entry
            if real_pct_profit > 0:
                print(f"✅ HARVEST: Closing {symbol} (+{pct_profit:.2%}) to bank Daily Goal.")
                place_order(symbol, qty, 'sell', current, 'harvest_win')
                long_count -= 1
            else:
                print(f"Last price for {symbol}: ${current:.2f} . However, Bid is: ${bid:.2f}.")
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


    # --- 🛡️ HEDGE FUND RISK MANAGER ---
    ny_time = datetime.now(pytz.timezone('US/Eastern'))
    # 1. MORNING & INTRADAY: Check for Vega Exit
    # If fear collapses (IV drops), we sell the hedge to save money.
    if ny_time.hour == 10 and 0 <= ny_time.minute <= 10:  # Test at 10:00 AM to 10:10 AM
        print("FORCING HEDGE TEST")
        Smart_Hedge.execute_omni_hedge()

    # 1. VEGA EXIT: Check all day if IV collapses
    if 9 <= ny_time.hour < 16:
        Smart_Hedge.check_vega_exit()

    # 2. MORNING HEDGE (9:30-9:50 AM): Check for gap/fragility
    if ny_time.hour == 9 and 30 <= ny_time.minute <= 50:
        Smart_Hedge.execute_omni_hedge()

    # 3. CLOSING HEDGE (3:45-3:59 PM): Deploy overnight protection
    if ny_time.hour == 15 and 45 <= ny_time.minute <= 59:
        # Check if we already have options (don't double buy)
        has_options = any(len(p.symbol) > 6 for p in positions)
        if not has_options:
            print("🛡️ CLOSING BELL: CALCULATING HEDGE.")
            Smart_Hedge.execute_omni_hedge()

    # 4. AFTER HOURS (4:10-4:15 PM): Close hedges and run attribution
    if ny_time.hour == 16 and 10 <= ny_time.minute <= 15:
        Smart_Hedge.close_all_hedges()

    # --- 2. HUNTING TRADES ---
    if insufficient_funds:
        print(f"👮🏻 Stopping Scan, Preserving BP For Hedging: BP:${buying_power: ,.2f} | Hedge Reserve: ${hedge_reserve: ,.2f}")
        return
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

        price, raw_z, raw_sma, raw_cmf, current_daily_vol = get_technical_data(symbol)
        if price is None: continue
        pure_z, pure_c, pure_s = orthoganolizer.clean_weights(symbol, raw_z, raw_cmf, raw_sma)
        print(f"🔍Checking {symbol} | Price: {price} | Z: {pure_z: .2f}σ | Vol: {pure_c:.2f}σ | Trend: {pure_s: .2f}σ ")

        signal = 'hold'
        reason = ""
        regime = regime_map.get(symbol, 'MEAN_REVERSION')
        if regime == 'MEAN_REVERSION':
            if pure_z < -2.0 and pure_c > 1.0 and pure_s > -1.0:
                signal = 'buy'
                reason = f"Oversold Z:{pure_z:.2f}"
        elif regime == 'TRENDING':
            if pure_s > 1.5 and pure_z < 0.0 and pure_c > 0.5:
                signal = 'buy'
                reason = f"Trend Dip: Trend: {pure_s: .2f} | Vol: {pure_c:.2f}"

        if signal == 'buy':
            print(f"Reading Tape for : {symbol}")
            tape_data = check_tape(symbol)

            if tape_data['signal'] == 'sell_wall':
                print(f"⛔ Abort Buying {symbol}: {tape_data['reason']}")
                continue
            can_proceed = check_oppossing_position(symbol, 'long')
            if not can_proceed:
                continue
            shares = calculate_dynamic_position(
                symbol=symbol,
                equity=equity,
                price=price,
                daily_vol=current_daily_vol,
                pure_z=pure_z,
                pure_c=pure_c,
                held_symbols=held_symbols,
                max_positions=MAX_POSITIONS
            )
            if shares > 0:
                print(f"\n🚀 {signal.upper()}: {symbol} | {reason}")
                success = place_order(symbol, shares, signal, price, 'entry')
                if success is False:
                    continue
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
        print("Waiting 30 seconds...")
        time.sleep(30)
    print("--- 🔴 SESSION ENDING ---")
