import os
import time
import ccxt
import pandas as pd
from datetime import datetime
from model import load_or_initialize_model, train_or_update_model, engineer_features
 
SCORECARD_FILE = "paper_scorecard.csv"

def initialize_scorecard():
    if not os.path.exists(SCORECARD_FILE):
        df_init = pd.DataFrame(columns=["timestamp", "Prediction", "Actual", "Score"])
        df_init.to_csv(SCORECARD_FILE, index=False)

def log_scorecard(prediction, actual, score):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = pd.DataFrame([[timestamp, prediction, actual, score]], 
                           columns=["timestamp", "Prediction", "Actual", "Score"])
    new_row.to_csv(SCORECARD_FILE, mode='a', header=False, index=False)

def fetch_live_candles():
    """Fetches recent 5-minute BTC/USDT candles via CCXT for live feature engineering."""
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='5m', limit=50)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df

def run_dynamic_momentum_trader():
    initialize_scorecard()
    
    balance = 84.33
    print(f"🚀 Starting 5-Min Dynamic Momentum Trader | Initial Balance: ${balance:.2f} USDT")
    
    # Load or initialize model on startup
    model = load_or_initialize_model()
    
    current_position = None  # 'BULLISH' or 'BEARISH'
    entry_price = 0.0
    entry_time = None
    last_retrain_hour = datetime.now().hour

    while True:
        try:
            current_time = datetime.now()
            
            # 1. Hourly Retraining Check
            if current_time.hour != last_retrain_hour:
                print(f"\n🔄 Triggering scheduled model retraining at {current_time.strftime('%H:%M:%S')}...")
                model = train_or_update_model(None)
                last_retrain_hour = current_time.hour

            # 2. Fetch live data and engineer features
            df_raw = fetch_live_candles()
            df = engineer_features(df_raw)
            
            if df.empty:
                print("⚠️ Warning: Empty dataset returned from live fetch. Retrying in 30 seconds...")
                time.sleep(30)
                continue

            latest_row = df.iloc[-1]
            current_price = latest_row['close']
            
            feature_cols = ["return_1m", "return_5m", "volatility", "volume_ratio"]
            X_latest = df[feature_cols].tail(1)
            
            prediction_encoded = model.predict(X_latest)[0]
            prediction_label = "BULLISH" if prediction_encoded == 1 else "BEARISH"
            
            proba = model.predict_proba(X_latest)[0]
            confidence = max(proba) * 100

            # 3. Handle Active Position & 5-Minute Evaluation
            if current_position is not None:
                elapsed_seconds = (current_time - entry_time).total_seconds()
                
                # If 5 minutes (300 seconds) have passed, close and switch position
                if elapsed_seconds >= 300:
                    if current_position == 'BULLISH':
                        pnl_pct = (current_price - entry_price) / entry_price
                    else:  # BEARISH
                        pnl_pct = (entry_price - current_price) / entry_price

                    pnl_dollars = balance * pnl_pct
                    balance += pnl_dollars

                    actual_label = "BULLISH" if current_price > entry_price else "BEARISH"
                    score = 1 if current_position == actual_label else 0

                    log_scorecard(current_position, actual_label, score)
                    print(f"📊 5-Min Scorecard Logged -> Prediction: {current_position} | Actual: {actual_label} | Score: [{score}]")
                    print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] 🟢 Closed [{current_position}] at ${current_price:,.2f} | PnL: ${pnl_dollars:+.2f} | Balance: ${balance:.2f}\n")
                    
                    # Open new position
                    current_position = prediction_label
                    entry_price = current_price
                    entry_time = current_time
                    print(f"🟢 Opened New Position: {current_position} at ${entry_price:,.2f}\n")
                else:
                    # Print minute-by-minute status while holding the 5-min trade
                    print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] 🤖 AI Signal: {prediction_label} (Conf: {confidence:.1f}%) at BTC ${current_price:,.2f} | Status: Holding [{current_position}]")
            else:
                # Initial opening on first run
                current_position = prediction_label
                entry_price = current_price
                entry_time = current_time
                print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] 🤖 AI Signal: {prediction_label} (Conf: {confidence:.1f}%) at BTC ${current_price:,.2f}")
                print(f"🟢 Opened New Position: {current_position} at ${entry_price:,.2f}\n")

            # Check every 60 seconds to provide minute-by-minute visibility
            time.sleep(60)

        except KeyboardInterrupt:
            print(f"\n⚠️ Stopping dynamic momentum trader safely. Final Balance: ${balance:.2f}")
            break
        except Exception as e:
            print(f"⚠️ Error encountered in execution loop: {e}")
            time.sleep(10)

if __name__ == "__main__":
    run_dynamic_momentum_trader()