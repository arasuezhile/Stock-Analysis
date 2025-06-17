# data_handler.py
import pandas as pd
import yfinance as yf
from datetime import datetime

class DataHandler:
    """Handles fetching and preparing historical stock data from Yahoo Finance."""
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.stock = yf.Ticker(self.ticker)
    
    def fetch_historical_data(self, start_date: str, end_date: str = None, interval: str = '1d') -> pd.DataFrame:
        if end_date is None: end_date = datetime.now().strftime('%Y-%m-%d')
        print(f"Fetching {interval} data for '{self.ticker}' from {start_date} to {end_date}...")
        try:
            history_df = self.stock.history(start=start_date, end=end_date, auto_adjust=True, interval=interval)
            if history_df.empty: 
                print(f"--- WARNING ---: No {interval} data found for the specified range.")
                return pd.DataFrame()
            history_df.drop(columns=['Dividends', 'Stock Splits'], inplace=True, errors='ignore')
            return history_df
        except Exception as e: 
            print(f"--- ERROR ---: An error occurred: {e}")
            return pd.DataFrame()
