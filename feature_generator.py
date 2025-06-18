# feature_generator.py

import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta
from data_handler import DataHandler
import os

class FeatureGenerator:
    """
    Generates a feature set from raw stock price data for ML model training.
    Uses the pandas-ta library for a richer set of features.
    """
    def __init__(self, data_df: pd.DataFrame):
        if data_df.empty: raise ValueError("Input DataFrame cannot be empty.")
        self.data = data_df.copy()
        print("FeatureGenerator initialized.")

    def generate_features(self):
        """Runs all feature generation methods and returns the final DataFrame."""
        print("Generating a rich set of features using pandas-ta...")

        custom_strategy = ta.Strategy(
            name="Universal_Strategy",
            description="A combination of momentum, trend, volatility, and volume indicators",
            ta=[
                {"kind": "rsi", "length": 14},
                {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
                {"kind": "bbands", "length": 20, "std": 2.0},
                {"kind": "atr", "length": 14},
                {"kind": "mfi", "length": 14},
                {"kind": "adx", "length": 14},
                {"kind": "stoch", "k": 14, "d": 3},
                {"kind": "sma", "length": 50, "col_names": "SMA_50"},
            ]
        )
        
        self.data.ta.strategy(custom_strategy)

        # Manually calculate OBV to control dtype and avoid warnings
        obv_values = (self.data['Volume'] * np.sign(self.data['Close'].diff())).cumsum()
        self.data['OBV'] = obv_values.fillna(0)

        # Create "meta-features" like price vs moving average
        if 'SMA_50' in self.data.columns:
            self.data['price_vs_sma50'] = (self.data['Close'] / self.data['SMA_50']) - 1
        
        # Keep other price features
        self.data['price_change_1d'] = self.data['Close'].pct_change(periods=1)
        self.data['price_change_5d'] = self.data['Close'].pct_change(periods=5)
        self.data['volatility_21d'] = self.data['Close'].pct_change(periods=1).rolling(window=21).std()
        
        self.data.dropna(inplace=True)
        print(f"Feature generation complete. Final dataset has {self.data.shape[0]} rows and {self.data.shape[1]} columns.")
        return self.data

def run_feature_generation_pipeline(ticker: str, output_dir: str = '.'):
    """
    Orchestrates fetching data and generating features for a given ticker.
    """
    print(f"\n--- Running Feature Generation for {ticker} ---")
    today = datetime.now()
    start_date = (today - timedelta(days=5*365)).strftime('%Y-%m-%d')
    handler = DataHandler(ticker=ticker)
    historical_data = handler.fetch_historical_data(start_date=start_date)

    if not historical_data.empty:
        feature_gen = FeatureGenerator(data_df=historical_data)
        feature_dataset = feature_gen.generate_features()

        if feature_dataset.empty:
            print(f"--- WARNING: No data remained for {ticker} after feature generation. Skipping. ---")
            return False

        feature_dataset['label'] = -1
        
        output_filename = os.path.join(output_dir, f"{ticker}_features.csv")
        feature_dataset.to_csv(output_filename, index=True)
        return True
    else:
        print(f"Could not fetch data for {ticker}, aborting feature generation.")
        return False

if __name__ == '__main__':
    stock_ticker = 'RELIANCE.NS'
    output_directory = 'training_dataset'
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    run_feature_generation_pipeline(stock_ticker, output_dir=output_directory)
