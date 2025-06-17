# data_handler.py

import pandas as pd
import yfinance as yf
from datetime import datetime

class DataHandler:
    """
    Handles fetching and preparing historical stock data from Yahoo Finance.
    """
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.stock = yf.Ticker(self.ticker)
        print(f"DataHandler initialized for ticker: '{self.ticker}'")

    def fetch_historical_data(self, start_date: str, end_date: str = None) -> pd.DataFrame:
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        print(f"Fetching data for '{self.ticker}' from {start_date} to {end_date}...")
        try:
            history_df = self.stock.history(start=start_date, end=end_date, auto_adjust=True)
            if history_df.empty:
                print(f"--- WARNING ---: No data found for ticker '{self.ticker}'.")
                return pd.DataFrame()
            history_df.drop(columns=['Dividends', 'Stock Splits'], inplace=True, errors='ignore')
            print("Data fetched successfully.")
            return history_df
        except Exception as e:
            print(f"--- ERROR ---: An unexpected error occurred: {e}")
            return pd.DataFrame()

if __name__ == '__main__':
    print("\n--- Running DataHandler Test ---")
    print("\n--- Test Case 1: Valid Ticker (TCS.NS) ---")
    tcs_handler = DataHandler(ticker='TCS.NS')
    tcs_data = tcs_handler.fetch_historical_data(start_date='2024-01-01')
    if not tcs_data.empty:
        print("\nSample of fetched data for TCS.NS:")
        print(tcs_data.head())
