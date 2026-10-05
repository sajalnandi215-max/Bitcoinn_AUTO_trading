import os
import time
import ccxt
import pandas as pd
from datetime import datetime

HISTORICAL_FILE = "historical_btc_10yr.csv"
SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"

def fetch_massive_historical_data():
    if os.path.exists(HISTORICAL_FILE):
        print(f"Loading existing historical dataset from {HISTORICAL_FILE}...")
        return pd.read_csv(HISTORICAL_FILE, parse_dates=["timestamp"])

    print("Downloading long-term historical data from Binance (this may take a minute)...")
    exchange = ccxt.binance({'enableRateLimit': True})
    
    since = exchange.parse8601('2021-01-01T00:00:00Z')
    all_bars = []
    now = exchange.milliseconds()
    
    while since < now:
        try:
            bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=since, limit=1000)
            if not bars:
                break
            since = bars[-1][0] + 1
            all_bars.extend(bars)
            print(f"Fetched up to timestamp: {datetime.utcfromtimestamp(bars[-1][0]/1000).strftime('%Y-%m-%d %H:%M')}")
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Pagination error: {e}")
            time.sleep(5)
            
    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df.to_csv(HISTORICAL_FILE, index=False)
    print(f"Saved {len(df)} historical candles to {HISTORICAL_FILE}")
    return df

if __name__ == "__main__":
    fetch_massive_historical_data()