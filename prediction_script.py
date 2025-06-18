# prediction_script.py

import pandas as pd
import joblib
from datetime import datetime, timedelta
import argparse
import os

from data_handler import DataHandler
from feature_generator import FeatureGenerator

def generate_prediction_report(ticker: str):
    """
    Generates a full prediction and back-testing report for a given stock ticker
    using the pre-trained UNIVERSAL machine learning model.
    """
    # --- CHANGE ---
    # Load the model from the training_dataset directory
    model_dir = 'training_dataset'
    universal_model_filename = os.path.join(model_dir, 'universal_elliott_wave_model.pkl')
    # --- END CHANGE ---
    
    print(f"--- Loading Universal Model: {universal_model_filename} ---")
    try:
        model = joblib.load(universal_model_filename)
    except FileNotFoundError:
        print(f"ERROR: Universal model file not found: '{universal_model_filename}'")
        print("Please run 'build_universal_dataset.py' and then 'training_script.py' to create it.")
        return
        
    print(f"\n--- Preparing Latest Data for {ticker} ---")
    handler = DataHandler(ticker=ticker)
    start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
    price_data = handler.fetch_historical_data(start_date=start_date)
    
    if price_data.empty:
        print(f"Could not fetch price data for {ticker}. Aborting.")
        return
        
    feature_gen = FeatureGenerator(data_df=price_data)
    feature_dataset = feature_gen.generate_features()

    print("\n--- Generating Predictions ---")
    X_predict = feature_dataset.drop(columns=['label'], errors='ignore')
    non_numeric_cols = X_predict.select_dtypes(include=['object']).columns
    if not non_numeric_cols.empty:
        X_predict = X_predict.drop(columns=non_numeric_cols)
    
    predictions = model.predict(X_predict)
    feature_dataset['predicted_label'] = predictions
    
    print("\n" + "="*25 + " 4-Week Forecast " + "="*25)
    last_day_prediction = feature_dataset['predicted_label'].iloc[-1]
    
    label_to_state = {
        0: "No Clear Pattern", 1: "Impulse Wave 1", 2: "Corrective Wave 2",
        3: "Impulse Wave 3", 4: "Corrective Wave 4", 5: "Impulse Wave 5",
        6: "Corrective Wave A", 7: "Corrective Wave B", 8: "Corrective Wave C"
    }
    label_to_trend = {
        0: "NEUTRAL / SIDEWAYS", 1: "UPTREND", 2: "DOWNTREND (Correction)",
        3: "UPTREND", 4: "DOWNTREND (Correction)", 5: "UPTREND",
        6: "DOWNTREND", 7: "UPTREND (Corrective Bounce)", 8: "DOWNTREND"
    }
    
    current_state = label_to_state.get(last_day_prediction, "Unknown")
    current_trend = label_to_trend.get(last_day_prediction, "Unknown")
    
    print(f"Current Model-Predicted State: {current_state}")
    print(f"Implied Near-Term Trend: {current_trend}\n")
    
    forecast_data = []
    for i in range(1, 5):
        forecast_data.append({'Week': f'Week {i}', 'Predicted Trend': current_trend})
    forecast_df = pd.DataFrame(forecast_data)
    print(forecast_df.to_string(index=False))

    # (Your back-testing logic can be pasted here without changes)
    print("\n\n" + "="*25 + " 4-Week Back-testing Report " + "="*25)
    print("Back-testing logic to be implemented here...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate stock prediction report using a universal Elliott Wave model.")
    parser.add_argument('ticker', type=str, help='The stock ticker symbol (e.g., RELIANCE.NS)')
    args = parser.parse_args()
    generate_prediction_report(args.ticker)
