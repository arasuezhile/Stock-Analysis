# feature_generator.py

import pandas as pd
from datetime import datetime, timedelta
# Import our existing DataHandler class
from data_handler import DataHandler

class FeatureGenerator:
    """
    Generates a feature set from raw stock price data for ML model training.
    This version calculates indicators manually to avoid external dependencies.
    """
    def __init__(self, data_df: pd.DataFrame):
        if data_df.empty: raise ValueError("Input DataFrame cannot be empty.")
        self.data = data_df.copy()
        print("FeatureGenerator initialized.")

    def add_technical_indicators(self):
        """Calculates and adds standard technical indicators manually."""
        print("Adding standard technical indicators (RSI, MACD, Bollinger Bands)...")
        close = self.data['Close']
        
        # Bollinger Bands
        sma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        self.data['BBM_20_2.0'] = sma_20
        self.data['BBU_20_2.0'] = sma_20 + (std_20 * 2)
        self.data['BBL_20_2.0'] = sma_20 - (std_20 * 2)

        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        self.data['MACD_12_26_9'] = ema_12 - ema_26
        self.data['MACDs_12_26_9'] = self.data['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
        self.data['MACDh_12_26_9'] = self.data['MACD_12_26_9'] - self.data['MACDs_12_26_9']

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        self.data['RSI_14'] = 100 - (100 / (1 + rs))

        print("Technical indicators added.")
        return self

    def add_price_features(self):
        """Adds features based on price and volume changes."""
        print("Adding price and volume features...")
        self.data['price_change_1d'] = self.data['Close'].pct_change(periods=1)
        self.data['price_change_5d'] = self.data['Close'].pct_change(periods=5)
        self.data['volatility_21d'] = self.data['price_change_1d'].rolling(window=21).std()
        print("Price features added.")
        return self

    def generate_features(self):
        """Runs all feature generation methods and returns the final DataFrame."""
        self.add_technical_indicators()
        self.add_price_features()
        self.data.dropna(inplace=True)
        print(f"Feature generation complete. Final dataset has {self.data.shape[0]} rows.")
        return self.data

# This block allows you to test the script directly from your terminal
if __name__ == '__main__':
    stock_ticker = 'RELIANCE.NS'
    print(f"\n--- Generating feature set for {stock_ticker} ---")

    # 1. Fetch data
    today = datetime.now()
    start_date = (today - timedelta(days=5*365)).strftime('%Y-%m-%d')
    handler = DataHandler(ticker=stock_ticker)
    historical_data = handler.fetch_historical_data(start_date=start_date)

    if not historical_data.empty:
        # 2. Generate features
        feature_gen = FeatureGenerator(data_df=historical_data)
        feature_dataset = feature_gen.generate_features()
        
        # 3. Add a placeholder 'label' column for our future work
        feature_dataset['label'] = 'NotLabeled'
        
        # 4. Save the dataset to a CSV file for inspection
        output_filename = f"{stock_ticker}_features.csv"
        feature_dataset.to_csv(output_filename)
        
        print(f"\nSuccessfully created feature dataset.")
        print(f"Saved to '{output_filename}'")
    else:
        print("Could not fetch data, aborting feature generation.")
