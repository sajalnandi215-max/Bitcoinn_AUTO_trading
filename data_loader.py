import ccxt
import pandas as pd

def fetch_binance_data(symbol="BTC/USDT", timeframe="5m", limit=1000):
    """Fetches clean OHLCV historical and live candle data from Binance via CCXT."""
    try:
        exchange = ccxt.binance()
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(
            bars, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        print(f"Data fetch error from Binance: {e}")
        return None