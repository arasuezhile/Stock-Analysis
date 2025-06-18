# labeling_script.py

import pandas as pd
import numpy as np
import os

class DataLabeler:
    """
    Labels a feature dataset based on the actual future price movement
    using dynamic thresholds based on the stock's volatility (ATR).
    """
    def __init__(self, feature_file_path: str):
        if not os.path.exists(feature_file_path):
            raise FileNotFoundError(f"Feature file not found at: {feature_file_path}")
        
        self.feature_df = pd.read_csv(feature_file_path, index_col='Date', parse_dates=True)
        print("DataLabeler initialized with feature dataset.")

    def add_future_trend_labels(self, forecast_days: int = 5, atr_multiplier: float = 0.75):
        """
        Calculates the future price change and assigns a label using a dynamic threshold
        based on the Average True Range (ATR).
        """
        print(f"Generating labels based on dynamic ATR threshold...")
        
        future_close = self.feature_df['Close'].shift(-forecast_days)
        self.feature_df['future_pct_change'] = ((future_close - self.feature_df['Close']) / self.feature_df['Close']) * 100

        # --- THIS BLOCK CONTAINS THE FINAL FIX ---
        # Find the ATR column, which we know from debugging is named 'ATRr_14'
        atr_col_name = 'ATRr_14' 
        
        if atr_col_name not in self.feature_df.columns:
            # Fallback for any other naming conventions, though unlikely now
            for col in self.feature_df.columns:
                if 'ATR' in col.upper() and str(14) in col:
                    atr_col_name = col
                    break
        
        if atr_col_name not in self.feature_df.columns:
            raise ValueError("ATR feature column could not be found. Please check the feature generator output.")
        print(f"Found and using ATR column: '{atr_col_name}'")
        
        # Calculate the dynamic threshold for each day based on its ATR
        threshold = (self.feature_df[atr_col_name] / self.feature_df['Close']) * 100 * atr_multiplier
        # --- END OF FIX ---

        # Define the labeling function based on the dynamic threshold
        def apply_label(row):
            pct_change = row['future_pct_change']
            t = threshold.loc[row.name]
            if pct_change > t:
                return 1  # UPTREND
            elif pct_change < -t:
                return 2  # DOWNTREND
            else:
                return 0  # SIDEWAYS

        self.feature_df['label'] = self.feature_df.apply(apply_label, axis=1)
        
        self.feature_df.dropna(subset=['label', 'future_pct_change'], inplace=True)
        self.feature_df['label'] = self.feature_df['label'].astype(int)

        print("Labeling complete.")
        return self.feature_df

def run_labeling_pipeline(ticker: str, feature_file_path: str, output_dir: str = '.'):
    """
    Orchestrates data labeling and saves the labeled dataset to a CSV.
    """
    print(f"\n--- Running Data Labeling for {ticker} ---")
    try:
        labeler = DataLabeler(feature_file_path=feature_file_path)
        labeled_df = labeler.add_future_trend_labels()
        
        output_filename = os.path.join(output_dir, f"{ticker}_labeled_dataset.csv")
        labeled_df.to_csv(output_filename, index=True)
        print(f"Successfully created labeled dataset and saved to '{output_filename}'.")
        print("\n--- Label Distribution Summary ---")
        print("0=SIDEWAYS, 1=UPTREND, 2=DOWNTREND")
        print(labeled_df['label'].value_counts(normalize=True).sort_index())
        return True
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return False
    except Exception as e:
        print(f"--- ERROR ---: An error occurred during labeling: {e}")
        return False

if __name__ == '__main__':
    stock_ticker = 'RELIANCE.NS'
    output_directory = 'training_dataset'
    feature_filename = os.path.join(output_directory, f"{stock_ticker}_features.csv")

    print(f"\n--- Creating labeled dataset for {stock_ticker} ---")
    if not os.path.exists(feature_filename):
        run_feature_generation_pipeline(stock_ticker, output_dir=output_directory)
    
    run_labeling_pipeline(stock_ticker, feature_filename, output_dir=output_directory)
