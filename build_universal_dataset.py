# build_universal_dataset.py

import pandas as pd
import os

# Import the pipeline functions from your existing scripts
from feature_generator import run_feature_generation_pipeline
from labeling_script import run_labeling_pipeline

def create_universal_labeled_dataset(tickers: list, output_dir: str = 'training_dataset'):
    """
    Runs the feature generation and labeling pipeline for a list of tickers,
    then combines all the labeled data into a single CSV file.
    All artifacts are saved in the output_dir.
    """
    print(f"--- Starting Universal Dataset Creation (Output Directory: {output_dir}) ---")
    
    # --- CHANGE ---
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # --- END CHANGE ---

    all_labeled_data = []

    for ticker in tickers:
        print(f"\n--- Processing Ticker: {ticker} ---")
        
        # --- CHANGE ---
        # Define file paths inside the output directory
        feature_file = os.path.join(output_dir, f"{ticker}_features.csv")
        labeled_file = os.path.join(output_dir, f"{ticker}_labeled_dataset.csv")
        # --- END CHANGE ---

        if not os.path.exists(feature_file):
            if not run_feature_generation_pipeline(ticker, output_dir=output_dir): # --- CHANGE ---
                print(f"--- WARNING: Failed to generate features for {ticker}. Skipping. ---")
                continue
        else:
            print(f"Feature file for {ticker} already exists. Skipping generation.")
        
        if not os.path.exists(labeled_file):
            if not run_labeling_pipeline(ticker, feature_file, output_dir=output_dir): # --- CHANGE ---
                print(f"--- WARNING: Failed to label data for {ticker}. Skipping. ---")
                continue
        else:
            print(f"Labeled file for {ticker} already exists. Skipping labeling.")

        try:
            df = pd.read_csv(labeled_file)
            df['Ticker'] = ticker 
            all_labeled_data.append(df)
            print(f"Successfully processed and added data for {ticker}.")
        except FileNotFoundError:
            print(f"--- WARNING: Could not find labeled file for {ticker} after processing. Skipping. ---")

    if not all_labeled_data:
        print("--- ERROR: No data was processed. Universal dataset not created. ---")
        return

    # --- CHANGE ---
    # Define the final output file path
    universal_output_filename = os.path.join(output_dir, 'universal_labeled_dataset.csv')
    universal_df = pd.concat(all_labeled_data, ignore_index=True)
    universal_df.to_csv(universal_output_filename, index=False)
    # --- END CHANGE ---
    
    print(f"\n--- Success! Universal dataset created at '{universal_output_filename}' ---")
    print(f"Total rows in dataset: {len(universal_df)}")

if __name__ == '__main__':
    stock_list = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 
        'HINDUNILVR.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS',
        'BAJFINANCE.NS', 'BHARTIARTL.NS', 'ITC.NS',
        'MARUTI.NS', 'ASIANPAINT.NS', 'LT.NS','SAIL.NS','HINDCOPPER.NS'
    ]
    
    create_universal_labeled_dataset(tickers=stock_list)
