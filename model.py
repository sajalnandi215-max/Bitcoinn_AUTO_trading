import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

MODEL_FILE = "btc_model.pkl"
HISTORICAL_FILE = "historical_btc_10yr.csv"
SCORECARD_FILE = "paper_scorecard.csv"

def engineer_features(df):
    df = df.copy()
    
    # Force all price and volume columns to be numeric floats (drops bad rows/headers)
    num_cols = ["open", "high", "low", "close", "volume"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=num_cols, inplace=True)
    
    # Calculate features
    df["return_1m"] = df["close"].pct_change(1) * 100
    df["return_5m"] = df["close"].pct_change(5) * 100
    df["volatility"] = df["return_1m"].rolling(window=10).std()
    df["volume_sma"] = df["volume"].rolling(window=10).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_sma"] + 1e-8)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def load_or_initialize_model(df=None):
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        if df is not None and not df.empty:
            features = ["return_1m", "return_5m", "volatility", "volume_ratio"]
            X = df[features]
            y = df["target"].astype(int)
            model.fit(X, y)
            joblib.dump(model, MODEL_FILE)
        return model

def train_or_update_model(current_df):
    feature_cols = ["return_1m", "return_5m", "volatility", "volume_ratio"]
    
    # 1. Load and Engineer Historical Baseline Data
    if os.path.exists(HISTORICAL_FILE):
        print("Loading and engineering 10-year historical baseline dataset...")
        df_hist_raw = pd.read_csv(
            HISTORICAL_FILE, 
            header=None, 
            names=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df_hist = engineer_features(df_hist_raw)
        df_hist_subset = df_hist[feature_cols + ["target"]].copy()
    else:
        df_hist_subset = pd.DataFrame(columns=feature_cols + ["target"])

    # 2. Load Live Scorecard Feedback
    if os.path.exists(SCORECARD_FILE):
        df_scorecard = pd.read_csv(SCORECARD_FILE)
        if not df_scorecard.empty and "Score" in df_scorecard.columns:
            print(f"[{len(df_scorecard)} scorecard entries found] Merging live feedback into training set...")
            df_scorecard["target"] = pd.to_numeric(df_scorecard["Score"], errors="coerce")
            df_scorecard_subset = df_scorecard[feature_cols + ["target"]].copy()
            df_combined = pd.concat([df_hist_subset, df_scorecard_subset], ignore_index=True)
        else:
            df_combined = df_hist_subset
    else:
        df_combined = df_hist_subset

    df_combined = df_combined.dropna()
    for col in feature_cols + ["target"]:
        df_combined[col] = pd.to_numeric(df_combined[col], errors="coerce")
    df_combined = df_combined.dropna()
    
    X = df_combined[feature_cols]
    y = df_combined["target"].astype(int)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, MODEL_FILE)
    print("✅ Model successfully retrained on engineered historical + live scorecard data and saved!")
    return model