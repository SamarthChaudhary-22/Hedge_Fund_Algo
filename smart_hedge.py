import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import os
import sys
import math
import json
from datetime import datetime, timedelta, date
from scipy.stats import norm
import pytz
import yfinance as yf

API_KEY = os.getenv('APCA_API_KEY_ID')
SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
BASE_URL = "https://paper-api.alpaca.markets"

# ⚙️ FRAGILITY TRIGGERS (FIXED - More Realistic)
GAP_THRESHOLD = 0.002
ATR_THRESHOLD = 0.008

#  REGIME GATING
MAX_VOL_SPREAD = 0.05
SPREAD_LIMIT = 0.10

# BUDGET
BASE_DAILY_COST = 0.001  # 0.1% Peace
STRESS_DAILY_COST = 0.003  # 0.3% War

# TARGET EXPOSURE
LAYER_A_TARGET_RATIO = 0.40  # Neutralize 40% Delta
LAYER_B_TARGET_VEGA = 0.002  # Target 0.2% Vega Exposure

# ALLOCATION
LAYER_A_BUDGET_SPLIT = 0.70
LAYER_B_BUDGET_SPLIT = 0.30

# BETA PHYSICS
BASE_BETA = 1.0
MAX_STRESS_BETA = 1.6

# STATE
STATE_FILE = "hedge_state.json"

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')


# --- 📐 GREEK ENGINE ---
def calculate_greeks(S, K, T, r, sigma, type='put'):
    if T <= 0: T = 0.0001
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = (S * norm.pdf(d1) * math.sqrt(T)) / 100

    return {'delta': delta, 'gamma': gamma, 'vega': vega}


# --- 📊 MARKET INTERNALS ---
def get_market_internals():
    try:
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        daily = api.get_bars('SPY', tradeapi.rest.TimeFrame.Day, start=start_date, feed='iex').df
        daily = daily.tail(30)
        if len(daily) < 20:
            print("❌ Insufficient historical data")
            return None, 0, 0, 0, 0

        daily['returns'] = np.log(daily['close'] / daily['close'].shift(1))
        rv_20d = daily['returns'].tail(20).std() * math.sqrt(252)
        prev_close = daily.iloc[-1]['close']

        today_str = datetime.now().strftime('%Y-%m-%d')
        intraday = api.get_bars('SPY', tradeapi.rest.TimeFrame(15, tradeapi.rest.TimeFrameUnit.Minute),
                                start=today_str, limit=50, feed='iex').df

        if intraday.empty:
            # Fallback for very early morning
            current_price = prev_close
            atr_pct = 0.01
            gap_pct = 0
            print("⚠️ No intraday data yet, using previous close")
        else:
            current_price = intraday.iloc[-1]['close']
            high_low = (intraday['high'] - intraday['low']).mean()
            atr_pct = high_low / current_price
            gap_pct = abs(current_price - prev_close) / prev_close

        # 3. IV Check
        iv_est = get_real_iv_snapshot(current_price)

        return current_price, atr_pct, gap_pct, rv_20d, iv_est

    except Exception as e:
        print(f"❌ Market Data Error: {e}")
        return None, 0, 0, 0, 0


def fetch_option_contracts_manual(symbol, expiration_date=None):
    """
    SEARCHES for a list of option contracts (e.g., all SPY Puts for next week).
    Endpoint: /v2/options/contracts
    """
    params = {
        'underlying_symbols': symbol,
        'status': 'active',
        'limit': 1000
    }
    if expiration_date:
        params['expiration_date'] = expiration_date

    try:
        # 🚨 FIX: Hit the CONTRACTS endpoint, not snapshots
        response = api.get('/options/contracts', data=params)
        return response.get('option_contracts', [])
    except Exception as e:
        print(f"❌ Contract Fetch Failed: {e}")
        return []

def fetch_option_snapshot_manual(symbol):
    """
    GETS DATA for a single specific contract (e.g., Price, IV, Greeks).
    Endpoint: /v2/options/snapshots/{symbol}
    """
    try:
        # 🚨 FIX: Hit the SNAPSHOT endpoint
        response = api.get(f'/options/snapshots/{symbol}')
        return response
    except Exception as e:
        print(f"⚠️ Manual Snapshot Fetch Failed: {e}")
        return None


def get_real_iv_snapshot(spy_price):
    """
    Fetches ATM IV.
    Priority: Alpaca Snapshot (Loop) -> Yahoo Option Chain -> Fallback (0.18)
    """
    iv_result = 0.18
    try:
        today = date.today()
        target_date = today + timedelta(days=5)
        exp_str = target_date.strftime('%Y-%m-%d')

        # ---------------------------------------------------------
        # 🟢 ATTEMPT 1: ALPACA (Try Top 5 Candidates)
        # ---------------------------------------------------------
        try:
            contracts_data = fetch_option_contracts_manual('SPY', expiration_date=exp_str)

            if contracts_data:
                def get_strike(c):
                    return float(c['strike_price']) if isinstance(c, dict) else float(c.strike_price)

                sorted_contracts = sorted(contracts_data, key=lambda x: abs(get_strike(x) - spy_price))

                # Loop through top 50 to find ONE valid snapshot
                for candidate in sorted_contracts[:50]:
                    symbol = candidate['symbol'] if isinstance(candidate, dict) else candidate.symbol
                    snap = fetch_option_snapshot_manual(symbol)

                    if snap:
                        iv = snap.get('implied_volatility')
                        if iv is None and 'greeks' in snap: iv = snap['greeks'].get('implied_volatility')

                        if iv and float(iv) > 0.05:
                            print(f"⚡ IV Source: Alpaca ({float(iv):.1%}) via {symbol}")
                            return float(iv)
                            break
        except Exception as e:
            # Don't print full stack trace for Alpaca, it's expected to fail on Paper
            pass

            # ---------------------------------------------------------
        # 🟡 ATTEMPT 2: YAHOO FINANCE (Bulletproof Version)
        # ---------------------------------------------------------
        if iv_result == 0.18:  # If Alpaca didn't give us a result different from default
            print("🔄 Switching to Yahoo Finance for IV...")
            try:
                spy = yf.Ticker("SPY")
                exps = spy.options
                if not exps: raise Exception("No Yahoo Expirations")

                closest_exp = min(exps, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d').date() - target_date).days))
                print(f"   (Using Yahoo Chain: {closest_exp})")

                chain = spy.option_chain(closest_exp)
                puts = chain.puts

                if not puts.empty:
                    # Filter for valid IVs (> 1%)
                    valid_puts = puts[puts['impliedVolatility'] > 0.01]
                    if not valid_puts.empty:
                        valid_puts['dist'] = abs(valid_puts['strike'] - spy_price)
                        atm_row = valid_puts.sort_values('dist').iloc[0]
                        yf_iv = atm_row['impliedVolatility']

                        print(f"⚡ IV Source: Yahoo Chain ({yf_iv:.1%})")
                        iv_result = yf_iv

            except Exception as e:
                print(f"⚠️ Yahoo IV Failed: {e}")
    except Exception as e:
        print(f"❌ Critical IV Failure: {e}")

        # ---------------------------------------------------------
        # 🛡️ SANITY CHECK (The Fix)
        # ---------------------------------------------------------
        # If IV is suspiciously low (e.g. 1.6%), assume data error and clamp it.
    if iv_result < 0.08:
        print(f"⚠️ Detected Garbage Data (IV={iv_result:.1%}). Clamping to min 10%.")
        return 0.10

    return iv_result

def find_real_quote(symbol):
    """
    Robust Quote Fetcher: Alpaca (Primary) -> Yahoo (Fallback)
    """
    # 1. Try Alpaca First
    try:
        snap = fetch_option_snapshot_manual(symbol)
        if snap and 'latest_quote' in snap:
            quote = snap['latest_quote']
            bid = float(quote.get('bid_price', 0))
            ask = float(quote.get('ask_price', 0))

            if ask > 0 and bid > 0:
                mid = (bid + ask) / 2
                spread_pct = (ask - bid) / mid
                if spread_pct < SPREAD_LIMIT:
                    print(f"✅ Alpaca quote: ${ask} (spread: {spread_pct:.1%})")
                    return ask
                else:
                    print(f"⚠️ Alpaca spread too wide ({spread_pct:.1%}), trying Yahoo")
    except Exception as e:
        print(f"⚠️ Alpaca snapshot failed: {e}")

    # 2. Yahoo Finance Fallback
    try:
        yf_opt = yf.Ticker(symbol)
        yf_price = yf_opt.fast_info.get('last_price')

        if yf_price and yf_price > 0:
            print(f"✅ Yahoo Price: ${yf_price}")
            return yf_price
    except Exception as e:
        print(f"❌ Yahoo failed for {symbol}: {e}")

    print(f"❌ No valid quote found for {symbol}")
    return 0


# --- CONTRACT SELECTION ---
def scan_and_select_contract(spy_price, days_out, iv_est, goal='delta', target_val=0.50, option_type='put'):
    today = date.today()
    exp_date = (today + timedelta(days=days_out)).strftime('%Y-%m-%d')
    T = days_out / 365.0

    print(f"🔍 Scanning {option_type.upper()} contracts for {exp_date}")

    try:
        contracts_data = fetch_option_contracts_manual('SPY', expiration_date=exp_date)

        if not contracts_data:
            print(f"No contract data for {exp_date}. Trying next week")
            exp_date = (today + timedelta(days=days_out + 1)).strftime('%Y-%m-%d')
            contracts_data = fetch_option_contracts_manual('SPY', expiration_date=exp_date)

        if not contracts_data: return None, 0, {}

        candidates = []
        for c in contracts_data:
            strike = float(c['strike_price']) if isinstance(c, dict) else float(c.strike_price)
            if option_type == 'put':
                if not (0.95 * spy_price < strike < 1.00 * spy_price): continue
            else:
                if not (1.0 * spy_price < strike < 1.05 * spy_price): continue

            # Pass option_type to greek calculator
            greeks = calculate_greeks(spy_price, strike, T, 0.05, iv_est, option_type)
            abs_delta = abs(greeks['delta'])

            score = 0
            # --- SCORING ENGINE ---
            if goal == 'delta':
                delta_match = -abs(abs_delta - target_val)
                gamma_boost = greeks['gamma'] * 100
                score = delta_match + (gamma_boost * 0.5)

            elif goal == 'vega':
                if abs_delta < 0.05 or abs_delta > 0.40: continue
                score = greeks['vega'] / (abs_delta + 0.1)

            candidates.append({'contract': c, 'greeks': greeks, 'score': score})

        candidates.sort(key=lambda x: x['score'], reverse=True)
        top_candidates = candidates[:50]

        best_c = None
        best_real_score = -999
        best_price = 0
        best_greeks = {}

        for cand in top_candidates:
            symbol = cand['contract']['symbol'] if isinstance(cand['contract'], dict) else cand['contract'].symbol
            real_price = find_real_quote(symbol)

            if real_price <= 0: continue

            if goal == 'delta':
                final_score = cand['score']
            elif goal == 'vega':
                final_score = cand['score'] / real_price

            if final_score > best_real_score:
                best_real_score = final_score
                best_c = cand['contract']
                best_price = real_price
                best_greeks = cand['greeks']

        if best_c:
            sym = best_c['symbol'] if isinstance(best_c, dict) else best_c.symbol
            print(f"Found Contract at: {sym} at ${best_price:.2f}")
            return best_c, best_price, best_greeks
        else:
            print(f"No contract data found.")
            return None, 0, {}

    except Exception as e:
        print(f"❌ Contract selection error: {e}")
        return None, 0, {}


# --- 📝 ATTRIBUTION ---
def save_hedge_state(equity, hedge_cost):
    state = {
        "timestamp": datetime.now().isoformat(),
        "entry_equity": equity,
        "hedge_cost": hedge_cost
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)
    print(f"💾 Saved hedge state: Cost ${hedge_cost:.2f}")


def load_hedge_state():
    if not os.path.exists(STATE_FILE):
        return None
    with open(STATE_FILE, 'r') as f:
        return json.load(f)


def generate_attribution_report(current_equity, hedge_pnl):
    state = load_hedge_state()
    if not state:
        print("⚠️ No hedge state found for attribution")
        return

    entry_equity = state['entry_equity']
    hedge_cost = state.get('hedge_cost', 0)

    real_pnl = current_equity - entry_equity
    unhedged_pnl = real_pnl - hedge_pnl

    # Drawdown Efficiency
    drawdown_saved = max(0, hedge_pnl)
    efficiency_ratio = 0
    if hedge_cost > 0:
        efficiency_ratio = drawdown_saved / hedge_cost

    print("\n📊 --- ATTRIBUTION REPORT (OMNI) ---")
    print(f"📉 Unhedged PnL:  ${unhedged_pnl:,.2f}")
    print(f"🛡️ Hedge PnL:     ${hedge_pnl:,.2f}")
    print(f"✅ REAL PnL:      ${real_pnl:,.2f}")
    print(f"💸 Cost Basis:    ${hedge_cost:,.2f}")

    if efficiency_ratio > 1.0:
        print(f"🏆 ELITE EFFICIENCY: Saved ${drawdown_saved:,.2f} (Ratio: {efficiency_ratio:.2f}x)")
    else:
        print(f"⚠️ DRAG: Efficiency {efficiency_ratio:.2f}x")
    print("-----------------------------------")

    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)


# --- 🚀 EXECUTION ENGINE ---
def execute_omni_hedge():
    ny_time = datetime.now(pytz.timezone('US/Eastern'))
    print(f"\n🧱 SMART HEDGE CHECK: {ny_time.strftime('%H:%M:%S')}")

    # SAFETY: Wait for liquidity at market open
    if ny_time.hour == 9 and ny_time.minute < 35:
        print("⏳ Market Opening... Waiting for spreads to normalize (9:35 AM).")
        return
    check_vega_exit()

    account = api.get_account()
    equity = float(account.portfolio_value)

    # 1. DATA & REGIME
    spy_price, atr, gap, rv, iv = get_market_internals()
    if not spy_price:
        print("❌ Cannot get market data, skipping hedge")
        return

    print(f"📊 SPY: ${spy_price:.2f} | Gap: {gap:.2%} | ATR: {atr:.2%} | IV: {iv:.1%}")

    # FRAGILITY DETECTION (FIXED)
    is_fragile = False
    reason = "Stable"

    if gap > GAP_THRESHOLD:
        is_fragile = True
        reason = f"Gap {gap:.2%} > {GAP_THRESHOLD:.2%}"
    elif atr > ATR_THRESHOLD:
        is_fragile = True
        reason = f"ATR {atr:.2%} > {ATR_THRESHOLD:.2%}"

    # IV-RV Filter
    vol_spread = iv - rv
    is_vol_cheap = vol_spread < MAX_VOL_SPREAD
    print(f"📊 VOL: IV={iv:.1%} | RV={rv:.1%} | Spread={vol_spread:.1%} | Cheap: {is_vol_cheap}")

    # CLOSING WINDOW (FIXED LOGIC)
    is_closing_window = (ny_time.hour == 15 and ny_time.minute >= 45)

    # DECISION LOGIC (FIXED)
    if not is_fragile and not is_closing_window:
        print(f"✅ Market Stable ({reason}). No Hedge Needed.")
        return

    if is_fragile:
        print(f"⚠️ FRAGILE MARKET: {reason}")
    if is_closing_window:
        print(f"🌙 CLOSING WINDOW: Deploying overnight protection")

    # 2. BUDGET & BETA
    excess_vol = max(0, atr - 0.005)
    stress_beta = BASE_BETA + min(0.6, math.sqrt(excess_vol) * 2.5)

    target_budget_pct = STRESS_DAILY_COST if is_fragile else BASE_DAILY_COST
    total_budget = equity * target_budget_pct

    print(f"💰 Hedge Budget: ${total_budget:.2f} ({target_budget_pct:.2%}) | StressBeta: {stress_beta:.2f}")

    total_cost_incurred = 0

    # --- LAYER A: PUTS (Downside Protection) ---
    if is_closing_window or is_fragile:
        print("\n🛡️ HEDGE SIDE 1: PUTS (Crash Guard)...")
        contract_p, price_p, greeks_p = scan_and_select_contract(
            spy_price, 1, iv, goal='delta', target_val=0.40, option_type='put'
        )

        if contract_p and price_p > 0:
            budget_p = total_budget * 0.50
            qty_p = int(budget_p / (price_p * 100))

            if qty_p > 0:
                cost = submit_order(contract_p, qty_p, price_p, "Hedge PUTS")
                total_cost_incurred += cost
            else:
                print("⚠️ PUT quantity = 0 (price too high or budget too small)")
        else:
            print("❌ No Puts Found.")

    # --- LAYER B: CALLS (Upside Melt-Up Protection) ---
    if is_closing_window or is_fragile:
        print("\n🛡️ HEDGE SIDE 2: CALLS (Melt-Up Guard)...")
        contract_c, price_c, greeks_c = scan_and_select_contract(
            spy_price, 1, iv, goal='delta', target_val=0.40, option_type='call'
        )

        if contract_c and price_c > 0:
            budget_c = total_budget * 0.50
            qty_c = int(budget_c / (price_c * 100))

            if qty_c > 0:
                cost = submit_order(contract_c, qty_c, price_c, "Hedge CALLS")
                total_cost_incurred += cost
            else:
                print("⚠️ CALL quantity = 0 (price too high or budget too small)")
        else:
            print("❌ No Calls Found.")

    if total_cost_incurred > 0:
        save_hedge_state(equity, total_cost_incurred)
    else:
        print("⚠️ No hedges placed (all contracts failed or qty=0)")


def check_vega_exit():
    """
    Exit hedges if IV collapses
    """
    try:
        spy_price, _, _, _, current_iv = get_market_internals()
        if spy_price and current_iv < 0.12:
            print(f"📉 IV COLLAPSE ({current_iv:.1%}). Exiting Hedges.")
            close_all_hedges()
    except:
        pass


def submit_order(contract, qty, price, label):
    symbol = contract['symbol'] if isinstance(contract, dict) else contract.symbol
    limit = round(price * 1.01, 2)

    print(f"👉 SUBMITTING {label}: {qty}x {symbol} @ ${limit}")

    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side='buy',
            type='limit',
            limit_price=limit,
            time_in_force='day'
        )
        cost = qty * price * 100
        print(f"✅ Order submitted! Cost: ${cost:.2f}")
        return cost
    except Exception as e:
        print(f"❌ Order Error: {e}")
        return 0


def close_all_hedges():
    print("\n⏰ CLOSING HEDGES & ATTRIBUTION...")
    total_hedge_pnl = 0

    try:
        positions = api.list_positions()
        hedge_count = 0

        for p in positions:
            if len(p.symbol) > 6:
                hedge_count += 1
                pnl = float(p.unrealized_pl)
                total_hedge_pnl += pnl
                print(f"🛡️ Closing {p.symbol}: PnL ${pnl:.2f}")
                api.close_position(p.symbol)

        if hedge_count == 0:
            print("ℹ️ No hedge positions to close")
            return

        account = api.get_account()
        generate_attribution_report(float(account.equity), total_hedge_pnl)

    except Exception as e:
        print(f"❌ Close hedges error: {e}")


if __name__ == "__main__":
    execute_omni_hedge()
