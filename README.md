#  Grizzly & Bull: ML-Powered Quantitative Hedge Fund

**A sophisticated, dual-engine automated trading system designed for Alpaca Markets.**

This project implements a professional-grade **Long/Short Equity Strategy** that operates 24/7. It utilizes advanced statistical arbitrage (Z-Scores), regime detection, NLP-based sentiment analysis, and a proprietary "Ratchet" risk management system to consistently harvest alpha while strictly controlling downside risk.

----------

##  Key Features

### 1. Dual-Engine Architecture

The system is split into two specialized autonomous bots that run in parallel:

-   **The Long Bot (`Hedge_Fund_Bot.py`):**
    
    -   **Strategy:** Mean Reversion & Dip Buying.
        
    -   **Logic:** Identifies high-quality stocks that are statistically oversold (Z-Score < -0.5) during uptrends.
        
    -   **Optimization:** Tuned to capture "singles and doubles" (consistent small gains).
        
-   **The Short Bot (`Short_Engine.py`):**
    
    -   **Strategy:** Momentum Breakdown.
        
    -   **Logic:** Identifies stocks breaking down below key trendlines (Price < SMA50) with accelerating negative momentum (Z-Score > -1.75).
        
    -   **Exit:** Covers positions into panic flushes (Z-Score < -2.00).
        

### 2. NLP & Machine Learning (Genetic Algorithms)

Our sentiment analysis is not based on generic dictionaries. We utilize a custom **Genetic Algorithm Optimizer (`Sentiment_Optimizer.py`)** to mathematically derive our trading signals.

-   **The Process:** The script mines 10 years of financial news headlines and aligns them with next-day price drops.
    
-   **The Evolution:** It uses **Numba (CUDA)** to simulate millions of keyword combinations, evolving a "Red Flag" dictionary that specifically predicts market crashes for our S&P 500 universe.
    
-   **The Result:** A scientifically backed list of trigger words (e.g., "miss", "guidance", "inventory") that powers the Short Bot's decision-making.
    

### 3. "Ratchet" Risk Management

Unlike standard trailing stops, our proprietary **Ratchet System** locks in profits dynamically based on performance tiers:

-   **Tier 1:** If Profit > 1.5% → Stop moves to **Break Even**.
    
-   **Tier 2:** If Profit > 5% → Stop moves to **+2% Profit**.
    
-   **Tier 3:** If Profit > 10% → Stop moves to **+7% Profit**.
    
-   **Hard Stop:** All positions have a catastrophic hard stop at **-10%**.
    

### 4. Harvest Mode (Global Take Profit)

To prevent "giving back" daily gains, both bots communicate via a global equity check:

-   **The Trigger:** If the Total Portfolio Equity hits **+1.5%** for the day.
    
-   **The Action:** The system enters **Harvest Mode**.
    
    -   **Liquidation:** Immediately closes all **profitable** positions to lock in cash.
        
    -   **Defense:** Keeps losing positions open but protected by the Ratchet Stop.
        
    -   **Freeze:** Stops all new buying/shorting for the remainder of the session.
        

### 5. Intelligent Execution

-   **Stale Order Sweeper:** Automatically detects and cancels Limit Orders that have sat unfilled for >15 minutes to free up Buying Power and prevent "Zombie Orders."
    
-   **Smart Counting:** Proprietary logic ensures the bot tracks its true position count, preventing "Phantom Fills" even when API calls fail due to insufficient buying power.
    
-   **Discord Notifications:** Real-time alerts for every Fill, Harvest, and Stop Loss delivered instantly to your phone via `Discord_Listener.py`.
    

----------

##  Technical Architecture

### Tech Stack

-   **Language:** Python 3.9+
    
-   **Broker:** Alpaca Trade API (Paper/Live)
    
-   **Data Science:** Pandas, NumPy, Numba (High-Performance Computing)
    
-   **NLP:** NLTK (Tokenization), Custom Genetic Algorithm
    
-   **CI/CD:** GitHub Actions (Continuous 24/7 Operation)
    

### Optimization (Numba & CUDA)

We utilized custom-built **High-Frequency Grid Search Optimizers** (`Optimizer_Long.py`, `Optimizer_Short.py`, & `Sentiment_Optimizer.py`) using **Numba** to compile Python code into machine code. This allowed us to backtest **millions of data points** across thousands of parameter combinations to scientifically derive our Entry/Exit Z-Scores and Sentiment Keywords.

----------

## File Structure

**File**

**Description**

`Hedge_Fund_Bot.py`

**The Long Engine.** Scans for buys, manages long positions, handles hedging.

`Short_Engine.py`

**The Short Engine.** Scans for breakdowns, manages short positions.

Sentiment_Optimizer.py

**ML Trainer.** Uses Genetic Algorithms to learn which news keywords predict crashes.

`discord_listener.py`

**Notification Service.** Listens to the Alpaca trade stream and pushes alerts to Discord.

`optimizer_long.py`

**Backtester.** Numba-optimized script to find best Long Z-Score settings.

`optimizer_short.py`

**Backtester.** Numba-optimized script to find best Short Z-Score settings.

`diagnose.py`

**Utility.** Checks account health, Buying Power, and margin usage.

`sp500_regime.csv`

**Data.** Maps tickers to their market regime (Trend vs. Mean Reversion).

`.github/workflows/main.yml`

**Deployment.** Configures the 24/7 cloud runner.

----------

## Installation & Setup

### 1. Prerequisites

-   An Alpaca Markets Account (Paper or Live).
    
-   A Discord Webhook URL (for notifications).
    

### 2. Environment Variables

If running locally, set these in your `.env` file. If running on GitHub, add them to **Secrets**.

Bash

```
APCA_API_KEY_ID="your_alpaca_key"
APCA_API_SECRET_KEY="your_alpaca_secret"
DISCORD_WEBHOOK_URL="your_discord_webhook"

```

### 3. Install Dependencies

Bash

```
pip install -r requirements.txt

```

### 4. Run Locally

You can run the bots in separate terminals for full parallel execution:

Bash

```
# Terminal 1
python Hedge_Fund_Bot.py

# Terminal 2
python Short_Selling_Engine.py

# Terminal 3
python Discord_Listener.py

```

----------

## Deployment (GitHub Actions)

This project is configured for 24/7 Continuous Trading using GitHub Actions.

The workflow file .github/workflows/main.yml defines three parallel jobs:

1.  **Long Strategy** (Runs for 6 hours, restarts automatically).
    
2.  **Short Strategy** (Runs for 6 hours, restarts automatically).
    
3.  **Discord Listener** (Runs for 6 hours, restarts automatically).
    

**Schedule:** CRON job triggers every 6 hours (`0 */6 * * *`) to ensure zero downtime.
