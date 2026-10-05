import pandas as pd
import numpy as np

def engineer_features(df):
    """Calculates crypto-native indicators: momentum, volatility, and liquidity."""
    df = df.copy()
    
    # 1. Momentum & Returns
    df["return_1m"] = df["close"].pct_change(1) * 100
    df["return_5m"] = df["close"].pct_change(5) * 100
    
    # 2. Volatility (Rolling standard deviation of returns)
    df["volatility"] = df["return_1m"].rolling(window=10).std()
    
    # 3. Liquidity & Volume Profile
    df["volume_sma"] = df["volume"].rolling(window=10).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_sma"] + 1e-8)
    
    # 4. Supervised Target Label: Did the next candle close higher? (1 = Yes/Bullish, 0 = No/Bearish)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    
    # Drop initial rolling NaN rows
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df