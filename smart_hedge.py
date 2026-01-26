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

# --- 🧠 SMART HEDGE: OMNI EDITION (INSTITUTIONAL GRADE) ---
API_KEY = os.getenv('APCA_API_KEY_ID')
SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
BASE_URL = "https://paper-api.alpaca.markets"

# ⚙️ FRAGILITY TRIGGERS
GAP_THRESHOLD = 0.004  # > 0.4% Overnight Gap
ATR_THRESHOLD = 0.012  # > 1.2% Daily Range

# 📉 REGIME GATING
MAX_VOL_SPREAD = 0.05  # IV-RV Spread Cap
SPREAD_LIMIT = 0.10  # Max Bid-Ask Spread allowed (Liquidity Guard)

# 💰 BUDGET
BASE_DAILY_COST = 0.001  # 0.1% Peace
STRESS_DAILY_COST = 0.003  # 0.3% War

# 📐 TARGET EXPOSURE
LAYER_A_TARGET_RATIO = 0.40  # Neutralize 40% Delta
LAYER_B_TARGET_VEGA = 0.002  # Target 0.2% Vega Exposure

# 🧱 ALLOCATION
LAYER_A_BUDGET_SPLIT = 0.70
LAYER_B_BUDGET_SPLIT = 0.30

# 📊 BETA PHYSICS
BASE_BETA = 1.0
MAX_STRESS_BETA = 1.6

# 💾 STATE
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
        daily = api.get_bars('SPY', tradeapi.rest.TimeFrame.Day, limit=30).df
        if len(daily) < 20: return None, 0, 0, 0, 0

        daily['returns'] = np.log(daily['close'] / daily['close'].shift(1))
        rv_20d = daily['returns'].tail(20).std() * math.sqrt(252)
        prev_close = daily.iloc[-1]['close']

        intraday = api.get_bars('SPY', tradeapi.rest.TimeFrame(15, tradeapi.rest.TimeFrameUnit.Minute), limit=20).df
        current_price = intraday.iloc[-1]['close']
        high_low = (intraday['high'] - intraday['low']).mean()
        atr_pct = high_low / current_price
        gap_pct = abs(current_price - prev_close) / prev_close

        # 🚨 FIX 1: REAL IMPLIED VOLATILITY (No more ATR guessing)
        # We fetch a quick snapshot of ATM puts to get the market's true IV
        iv_est = get_real_iv_snapshot(current_price)

        return current_price, atr_pct, gap_pct, rv_20d, iv_est
    except:
        return None, 0, 0, 0, 0


def get_real_iv_snapshot(spy_price):
    """Fetches ATM option snapshot to get Market Maker IV"""
    try:
        today = date.today()
        exp = (today + timedelta(days=5)).strftime('%Y-%m-%d')  # Look at weekly
        contracts = api.get_option_contracts('SPY', expiration_date=exp, option_type='put').option_contracts
        # Find ATM
        atm_c = min(contracts, key=lambda x: abs(float(x.strike_price) - spy_price))
        snap = api.get_option_snapshot(atm_c.symbol)
        # Fallback to calculated if None
        return snap.implied_volatility if snap.implied_volatility else 0.15
    except:
        return 0.15  # Fallback


def find_real_quote(symbol):
    try:
        snap = api.get_option_snapshot(symbol)
        bid = snap.latest_quote.bid_price
        ask = snap.latest_quote.ask_price
        if ask <= 0: return snap.latest_trade.price

        # 🚨 FIX 4: LIQUIDITY GUARD
        # If spread is too wide, return 0 (Skip this contract)
        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid
        if spread_pct > SPREAD_LIMIT:
            # print(f"⚠️ Liquidity Skip: {symbol} Spread {spread_pct:.1%}")
            return 0

        return ask
    except:
        return 0


# --- 🔍 OMNI-TIER CONTRACT SELECTION ---
def scan_and_select_contract(spy_price, days_out, iv_est, goal='delta', target_val=0.50):
    today = date.today()
    exp_date = (today + timedelta(days=days_out)).strftime('%Y-%m-%d')
    T = days_out / 365.0

    try:
        contracts = api.get_option_contracts(
            'SPY', status='active', expiration_date=exp_date, option_type='put'
        ).option_contracts

        if not contracts:
            exp_date = (today + timedelta(days=days_out + 1)).strftime('%Y-%m-%d')
            contracts = api.get_option_contracts('SPY', status='active', expiration_date=exp_date,
                                                 option_type='put').option_contracts

        if not contracts: return None, 0, {}

        candidates = []
        for c in contracts:
            strike = float(c.strike_price)
            if not (0.75 * spy_price < strike < 1.05 * spy_price): continue

            greeks = calculate_greeks(spy_price, strike, T, 0.05, iv_est, 'put')
            abs_delta = abs(greeks['delta'])

            score = 0

            # --- GOAL: LAYER A (Gamma/Delta) ---
            if goal == 'delta':
                # 🚨 FIX 3: GAMMA AWARENESS
                # We want Delta match, BUT we boost score for high Gamma (Fast reaction)
                delta_match = -abs(abs_delta - target_val)
                gamma_boost = greeks['gamma'] * 100  # Normalize
                score = delta_match + (gamma_boost * 0.5)

                # --- GOAL: LAYER B (Skew-Aware Convexity) ---
            elif goal == 'vega':
                if abs_delta < 0.05 or abs_delta > 0.40: continue
                skew_boost = 1.3 if 0.15 <= abs_delta <= 0.25 else 1.0
                base_efficiency = greeks['vega'] / (abs_delta + 0.1)
                score = base_efficiency * skew_boost

            candidates.append({'contract': c, 'greeks': greeks, 'score': score})

        # Rank and Price Check
        candidates.sort(key=lambda x: x['score'], reverse=True)
        top_candidates = candidates[:5]

        best_c = None
        best_real_score = -999
        best_price = 0
        best_greeks = {}

        for cand in top_candidates:
            real_price = find_real_quote(cand['contract'].symbol)
            if real_price <= 0: continue

            # Greek-Per-Dollar Optimization
            if goal == 'delta':
                # For Layer A, we prioritize the Gamma/Delta score we calculated
                final_score = cand['score']

            elif goal == 'vega':
                # For Layer B, pure Vega Bang-for-Buck
                final_score = (cand['greeks']['vega'] / real_price) * (
                    1.3 if 0.15 <= abs(cand['greeks']['delta']) <= 0.25 else 1.0)

            if final_score > best_real_score:
                best_real_score = final_score
                best_c = cand['contract']
                best_price = real_price
                best_greeks = cand['greeks']

        return best_c, best_price, best_greeks

    except:
        return None, 0, {}


# --- 📝 ATTRIBUTION ---
def save_hedge_state(equity, hedge_cost):
    state = {
        "timestamp": datetime.now().isoformat(),
        "entry_equity": equity,
        "hedge_cost": hedge_cost
    }
    with open(STATE_FILE, 'w') as f: json.dump(state, f)


def load_hedge_state():
    if not os.path.exists(STATE_FILE): return None
    with open(STATE_FILE, 'r') as f: return json.load(f)


def generate_attribution_report(current_equity, hedge_pnl):
    state = load_hedge_state()
    if not state: return
    entry_equity = state['entry_equity']
    hedge_cost = state.get('hedge_cost', 0)

    real_pnl = current_equity - entry_equity
    unhedged_pnl = real_pnl - hedge_pnl

    # 🚨 FIX 5: KPI (Drawdown Efficiency)
    drawdown_saved = max(0, hedge_pnl)  # Only count if positive
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
    if os.path.exists(STATE_FILE): os.remove(STATE_FILE)


# --- 🚀 EXECUTION ENGINE ---
def execute_omni_hedge():
    print(f"\n🧱 SMART HEDGE (OMNI: GAMMA+LIQUIDITY AWARE) {datetime.now().strftime('%H:%M')}")
    # 🚨 SAFETY FIX: LIQUIDITY WAKE-UP
    # Market Makers widen spreads at 9:30. We wait for 9:35 AM to get real IV data.
    ny_time = datetime.now(pytz.timezone('US/Eastern'))
    if ny_time.hour == 9 and ny_time.minute < 35:
        print("⏳ Market Opening... Waiting for spreads to normalize (9:35 AM).")
        return  # Skip this loop, wait for next heartbeat
    account = api.get_account()
    equity = float(account.portfolio_value)

    # 1. DATA & REGIME
    spy_price, atr, gap, rv, iv = get_market_internals()
    if not spy_price: return

    is_fragile = False
    reason = "Stable"
    if gap > GAP_THRESHOLD:
        is_fragile = True; reason = f"Gap {gap:.2%}"
    elif atr > 0.008:
        is_fragile = True; reason = f"HiVol {atr:.2%}"

    # IV-RV Filter
    vol_spread = iv - rv
    is_vol_cheap = vol_spread < MAX_VOL_SPREAD
    print(f"📊 VOL: IV={iv:.1%} | RV={rv:.1%} | Spread={vol_spread:.1%}")

    ny_time = datetime.now(pytz.timezone('US/Eastern'))
    is_closing_window = (ny_time.hour == 15 and ny_time.minute >= 50)

    if not is_fragile and not is_closing_window:
        print(f"✅ Market Stable ({reason}). No Hedge.")
        return

    # 2. BUDGET & BETA
    excess_vol = max(0, atr - 0.005)
    stress_beta = BASE_BETA + min(0.6, math.sqrt(excess_vol) * 2.5)

    target_budget_pct = STRESS_DAILY_COST if is_fragile else BASE_DAILY_COST
    total_budget = equity * target_budget_pct

    print(f"⚠️ Hedge Active: {reason} | StressBeta: {stress_beta:.2f} | Budget: ${total_budget:.2f}")

    total_cost_incurred = 0

    # --- LAYER A: GAMMA (Open/Close Shock) ---
    if is_closing_window or is_fragile:
        print("🛡️ Optimizing Layer A (Gamma)...")
        contract_a, price_a, greeks_a = scan_and_select_contract(spy_price, 1, iv, goal='delta', target_val=0.40)

        if contract_a and price_a > 0:
            actual_delta = abs(greeks_a['delta'])
            spy_deltas_at_risk = (equity * stress_beta) / spy_price
            target_hedge_deltas = spy_deltas_at_risk * LAYER_A_TARGET_RATIO

            qty_a = int(target_hedge_deltas / (actual_delta * 100))
            budget_a = total_budget * LAYER_A_BUDGET_SPLIT
            if (qty_a * price_a * 100) > budget_a:
                qty_a = int(budget_a / (price_a * 100))

            if qty_a > 0:
                submit_order(contract_a, qty_a, price_a, "Layer A")
                total_cost_incurred += (qty_a * price_a * 100)

    # --- LAYER B: VEGA (Skew-Aware Convexity) ---
    if is_vol_cheap or is_fragile:
        print("🚀 Optimizing Layer B (Vega)...")
        contract_b, price_b, greeks_b = scan_and_select_contract(spy_price, 4, iv, goal='vega')

        if contract_b and price_b > 0:
            actual_vega = greeks_b['vega']
            target_vega_exposure = equity * LAYER_B_TARGET_VEGA
            qty_b = int(target_vega_exposure / (actual_vega * 100))

            budget_b = total_budget * LAYER_B_BUDGET_SPLIT
            if (qty_b * price_b * 100) > budget_b:
                qty_b = int(budget_b / (price_b * 100))

            if qty_b > 0:
                submit_order(contract_b, qty_b, price_b, "Layer B")
                total_cost_incurred += (qty_b * price_b * 100)
    else:
        print("⏸️ Layer B Skipped (Vol Expensive)")

    save_hedge_state(equity, total_cost_incurred)


def check_vega_exit():
    """
    🚨 FIX 2: VEGA STOP LOSS
    If IV crashes, we exit Layer B to stop the bleed.
    """
    try:
        spy_price, _, _, _, current_iv = get_market_internals()
        # Heuristic: If IV drops below 12% (Calm), kill hedges
        if current_iv < 0.12:
            print(f"📉 IV COLLAPSE ({current_iv:.1%}). Exiting Hedges.")
            close_all_hedges()
    except:
        pass


def submit_order(contract, qty, price, label):
    limit = round(price * 1.01, 2)
    print(f"👉 EXEC {label}: {qty}x {contract.symbol} @ ${limit}")
    try:
        api.submit_order(symbol=contract.symbol, qty=qty, side='buy', type='limit', limit_price=limit,
                         time_in_force='day')
    except Exception as e:
        print(f"Order Error: {e}")


def close_all_hedges():
    print("\n⏰ CLOSING & ATTRIBUTION...")
    total_hedge_pnl = 0
    try:
        positions = api.list_positions()
        for p in positions:
            if len(p.symbol) > 6:
                total_hedge_pnl += float(p.unrealized_pl)
                api.close_position(p.symbol)

        account = api.get_account()
        generate_attribution_report(float(account.equity), total_hedge_pnl)
    except:
        pass


if __name__ == "__main__":
    execute_omni_hedge()