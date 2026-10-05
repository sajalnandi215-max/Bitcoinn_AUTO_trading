# Bitcoin Auto Trading (5-Min Dynamic Momentum Trader)

An automated Bitcoin algorithmic paper-trading and ML forecasting system built in Python. It fetches live and historical crypto market data using CCXT (Binance), engineers momentum, volatility, and volume indicators, and continuously retrains an ensemble classifier with live feedback.

## Features

- **Live Market Data Feed**: Fetches real-time 5-minute OHLCV candles from Binance via `ccxt`.
- **Feature Engineering**: Calculates 1-minute and 5-minute returns, rolling volatility, volume moving averages, and liquidity ratios.
- **Machine Learning Engine**: Trains and updates a Random Forest Classifier using long-term historical data combined with live feedback scorecards.
- **Hourly Retraining**: Automatically adapts the ML model to evolving market conditions.
- **Paper Trading Execution**: Simulates trade entry, position holding, PnL calculation, and performance tracking without risking real capital.
- **Live Scorecard Logging**: Keeps a record of model predictions vs. actual market outcomes to refine future training cycles.

## Project Structure

```text
├── data_loader.py          # Binance data fetching utilities
├── feature_engine.py       # Indicator and feature engineering logic
├── historical_loader.py    # Long-term historical candle downloader
├── historical_btc_10yr.csv # Historical baseline OHLCV dataset
├── model.py                # Model training, feature pipeline, and retrain loop
├── paper_trader.py         # Main execution loop for live paper trading
├── paper_scorecard.csv     # Historical scorecard log of prediction vs actuals
├── virtual_wallet.csv      # Balance progression log
├── requirements.txt        # Python library dependencies
└── .gitignore              # Ignores __pycache__, virtual environments, and large .pkl files
```

## Getting Started

### 1. Prerequisites

Ensure Python 3.9+ is installed on your system.

### 2. Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/sajalnandi215-max/Bitcoinn_AUTO_trading.git
cd Bitcoinn_AUTO_trading
pip install -r requirements.txt
```

### 3. Running the Paper Trader

To launch the paper trading bot:

```bash
python paper_trader.py
```

The bot will:
1. Initialize the paper scorecard log (`paper_scorecard.csv`).
2. Load or train the initial machine learning model.
3. Fetch live Binance 5-minute candles.
4. Execute trades, evaluate positions every 5 minutes, track PnL, and log prediction accuracy.
5. Trigger retraining every hour using historical data and live scorecard results.

## Disclaimer

This software is for educational, research, and paper-trading purposes only. Cryptocurrency trading involves substantial risk of loss. Do not risk money you cannot afford to lose.
