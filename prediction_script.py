# prediction_script.py

import pandas as pd
import joblib
from datetime import datetime, timedelta
import argparse
import os

# Import callable functions from other scripts
from data_handler import DataHandler # Used directly in this script
from feature_generator import run_feature_generation_pipeline, FeatureGenerator # Added FeatureGenerator import
from labeling_script import run_labeling_pipeline
from training_script import run_training_pipeline

# Define paths for generated files (centralized helper functions for consistency)
# These paths must match how the other scripts save their files.
def get_feature_file_path(ticker, output_dir='.'):
    return os.path.join(output_dir, f"{ticker}_features.csv")

def get_labeled_dataset_path(ticker, output_dir='.'):
    return os.path.join(output_dir, f"{ticker}_labeled_dataset.csv")

def get_model_filename(ticker, output_dir='.'):
    return os.path.join(output_dir, f"{ticker}_elliott_wave_model.pkl")

def generate_prediction_report(ticker: str):
    """
    Generates a full prediction and back-testing report for a given stock ticker
    using the pre-trained machine learning model.
    This function also orchestrates the data processing and model training
    if the trained model for the given ticker does not exist.
    """
    model_filename = get_model_filename(ticker)
    labeled_dataset_path = get_labeled_dataset_path(ticker)
    feature_file_path = get_feature_file_path(ticker)

    # --- 1. Orchestration: Check if model exists, if not, run pipeline ---
    if not os.path.exists(model_filename):
        print(f"\nModel for {ticker} not found at '{model_filename}'. Initiating full data processing and training pipeline...")

        # Step 1: Feature Generation
        if not os.path.exists(feature_file_path):
            if not run_feature_generation_pipeline(ticker):
                print("Pipeline aborted: Feature Generation failed.")
                return
        else:
            print(f"Feature file '{feature_file_path}' already exists. Skipping feature generation.")

        # Step 2: Data Labeling
        if not os.path.exists(labeled_dataset_path):
            if not run_labeling_pipeline(ticker, feature_file_path):
                print("Pipeline aborted: Data Labeling failed.")
                return
        else:
            print(f"Labeled dataset '{labeled_dataset_path}' already exists. Skipping labeling.")

        # Step 3: Model Training
        # Check again if model exists. It shouldn't if the first check failed,
        # but subsequent steps might have created it if they were fixed mid-run.
        if not os.path.exists(model_filename):
            if not run_training_pipeline(ticker, labeled_dataset_path):
                print("Pipeline aborted: Model Training failed.")
                return
        else:
            print(f"Model '{model_filename}' already exists. Skipping training.")

        print(f"\nPipeline successfully completed. Model '{model_filename}' is now available.")
    else:
        print(f"\nModel for {ticker} found at '{model_filename}'. Proceeding with prediction.")

    # --- 2. Load the Trained Model ---
    print(f"\n--- Loading Trained Model: {model_filename} ---")
    try:
        model = joblib.load(model_filename)
    except FileNotFoundError: # Should theoretically not happen now with orchestration
        print(f"ERROR: Model file not found: '{model_filename}'. Please ensure it was trained.")
        return

    # --- 3. Fetch and Prepare Latest Data for Prediction ---
    print(f"\n--- Preparing Latest Data for {ticker} ---")
    handler = DataHandler(ticker=ticker)
    # Fetch data only for the last few years for prediction features, no need for full history
    start_date_data_fetch = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    price_data = handler.fetch_historical_data(start_date=start_date_data_fetch)

    if price_data.empty:
        print("Could not fetch price data. Aborting prediction.")
        return

    feature_gen = FeatureGenerator(data_df=price_data)
    feature_dataset = feature_gen.generate_features()

    # --- 4. Make Predictions on the Full Dataset ---
    print("\n--- Generating Predictions ---")
    # Drop the 'label' column if it exists in the raw feature_dataset from feature_generator
    X = feature_dataset.drop(columns=['label'], errors='ignore')

    # Ensure X has the same columns as during training
    # For simplicity, we're relying on consistent feature generation.
    # In a robust system, you'd load feature names from training and align X.

    predictions = model.predict(X)
    feature_dataset['predicted_label'] = predictions

    # --- 5. Generate Forward-Looking Forecast ---
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

    last_close_price = feature_dataset['Close'].iloc[-1]
    print(f"Last Closing Price (as of today): ₹{last_close_price:.2f}\n")

    forecast_data = []

    # Initialize cumulative factors for price range
    cumulative_min_factor = 1.0
    cumulative_max_factor = 1.0

    # Get the current date for forecast week ending calculation
    current_date_for_forecast = datetime.now()

    for i in range(1, 5):
        forecast_week_ending_date = current_date_for_forecast + timedelta(weeks=i)

        expected_pct_movement_str = "N/A"
        weekly_min_pct_change_decimal = 0.0
        weekly_max_pct_change_decimal = 0.0

        # Define illustrative weekly expected movement ranges based on the predicted trend
        # These are examples; actual model would need to predict magnitude for precise values
        if current_trend == "NEUTRAL / SIDEWAYS":
            weekly_min_pct_change_decimal = -0.010 # -1.0% per week
            weekly_max_pct_change_decimal = 0.010  # +1.0% per week
            expected_pct_movement_str = f"{weekly_min_pct_change_decimal*100:.1f}% to {weekly_max_pct_change_decimal*100:.1f}% (weekly)"
        elif current_trend == "UPTREND":
            weekly_min_pct_change_decimal = 0.015 # +1.5% per week
            weekly_max_pct_change_decimal = 0.035 # +3.5% per week
            expected_pct_movement_str = f"{weekly_min_pct_change_decimal*100:.1f}% to {weekly_max_pct_change_decimal*100:.1f}% (weekly expected increase)"
        elif current_trend.startswith("DOWNTREND"): # Handles DOWNTREND and DOWNTREND (Correction)
            weekly_min_pct_change_decimal = -0.035 # -3.5% per week
            weekly_max_pct_change_decimal = -0.015 # -1.5% per week
            expected_pct_movement_str = f"{weekly_min_pct_change_decimal*100:.1f}% to {weekly_max_pct_change_decimal*100:.1f}% (weekly expected decrease)"
        elif current_trend == "UPTREND (Corrective Bounce)":
            weekly_min_pct_change_decimal = 0.005 # +0.5% per week
            weekly_max_pct_change_decimal = 0.020 # +2.0% per week
            expected_pct_movement_str = f"{weekly_min_pct_change_decimal*100:.1f}% to {weekly_max_pct_change_decimal*100:.1f}% (weekly temporary increase)"
        else: # Unknown or No Clear Pattern
            expected_pct_movement_str = "Magnitude not explicitly predicted (weekly)"
            # For unpredicted trends, cumulative factors won't change here, reflecting uncertainty.

        # Update cumulative factors for the possible price range by the end of this week
        cumulative_min_factor *= (1 + weekly_min_pct_change_decimal)
        cumulative_max_factor *= (1 + weekly_max_pct_change_decimal)

        # Calculate possible price range for this week (cumulative from initial last_close_price)
        possible_min_price = last_close_price * cumulative_min_factor
        possible_max_price = last_close_price * cumulative_max_factor
        possible_price_range_str = f"₹{possible_min_price:.2f} - ₹{possible_max_price:.2f}"

        forecast_data.append({
            'Week': f'Week {i}',
            'Week Ending Date': forecast_week_ending_date.strftime('%Y-%m-%d'),
            'Predicted Trend': current_trend,
            'Expected % Movement': expected_pct_movement_str,
            'Possible Price Range': possible_price_range_str
        })

    forecast_df = pd.DataFrame(forecast_data)
    print(forecast_df.to_string(index=False))

    # --- General Interpretation of Forecasted Price Movement ---
    print(f"\n--- General Interpretation of Forecasted Price Movement for {ticker} ---")
    if current_trend == "NEUTRAL / SIDEWAYS":
        print(f"The 'Expected % Movement' (e.g., ±1.0% weekly) refers to the potential fluctuation within *each individual week*.")
        print("The 'Possible Price Range' shows the *cumulative* potential price spread by the end of that specific week, starting from the last closing price.")
        print("Therefore, even with a neutral weekly prediction, the cumulative range widens over time as the price could drift to either the upper or lower bound each week.")
        print("For instance, if the price consistently moved down by 1% each week for 4 weeks, the total decrease would be more than 4% (due to compounding).")
        print("Conversely, if it moved up by 1% each week, the total increase would also compound.")
        print("This widening range reflects the increasing uncertainty and potential dispersion of outcomes further into the future, even within a generally neutral market condition.")
    elif current_trend == "UPTREND":
        print(f"The 'Possible Price Range' shows the cumulative expected upward movement based on the weekly trend. Prices are expected to generally rise week over week.")
    elif current_trend.startswith("DOWNTREND"):
        print(f"The 'Possible Price Range' shows the cumulative expected downward movement based on the weekly trend. Prices are expected to generally fall week over week.")
    elif current_trend == "UPTREND (Corrective Bounce)":
        print(f"The 'Possible Price Range' reflects a temporary upward movement from the current price, which could lead to prices slightly above current levels, but typically within a larger corrective pattern.")
    else:
        print("General price movement interpretation for this trend is not specifically defined by the model's illustrative ranges.")


    # --- 6. Generate Back-testing Report ---
    print("\n\n" + "="*25 + " 4-Week Back-testing Report " + "="*25)

    backtest_data = []

    # Make the 'today' variable timezone-aware by matching the data's timezone
    today_for_backtest = pd.Timestamp.now(tz=feature_dataset.index.tz)

    for i in range(4, 0, -1):
        end_of_week = today_for_backtest - timedelta(weeks=i-1)
        start_of_week_full_timestamp = today_for_backtest - timedelta(weeks=i)

        # Convert start_of_week to just a date (midnight) to match index granularity
        start_of_week_for_loc = start_of_week_full_timestamp.normalize()

        week_data = feature_dataset[(feature_dataset.index >= start_of_week_for_loc) & (feature_dataset.index < end_of_week)]

        if week_data.empty:
            continue

        # Using searchsorted to find the index for forward-fill (ffill)
        temp_idx = feature_dataset.index.searchsorted(start_of_week_for_loc, side='right')

        # Handle cases where start_of_week_for_loc is before the first element
        if temp_idx == 0:
            prediction_date_index = 0
        else:
            prediction_date_index = temp_idx - 1

        # Original check: Ensure we don't go out of bounds (still relevant)
        if prediction_date_index == 0: continue

        predicted_label = feature_dataset['predicted_label'].iloc[prediction_date_index - 1]
        predicted_trend = label_to_trend.get(predicted_label, "Unknown")

        actual_start_price = week_data['Close'].iloc[0]
        actual_end_price = week_data['Close'].iloc[-1]
        actual_change_pct = ((actual_end_price - actual_start_price) / actual_start_price) * 100
        actual_trend = "UPTREND" if actual_change_pct > 0 else "DOWNTREND"

        backtest_data.append({
            'Week Ending': end_of_week.strftime('%Y-%m-%d'),
            'Closing Price': f"₹{actual_end_price:.2f}",
            'Predicted Trend': predicted_trend,
            'Actual Trend': actual_trend,
            'Actual Change %': f"{actual_change_pct:.2f}%"
        })

    if backtest_data:
        backtest_df = pd.DataFrame(backtest_data)
        print(backtest_df.to_string(index=False))
    else:
        print("Not enough historical data to generate a full 4-week back-test.")

# --- Main execution block for command line ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate stock prediction report based on Elliott Wave analysis.")
    parser.add_argument('ticker', type=str, help='The stock ticker symbol (e.g., RELIANCE.NS)')

    args = parser.parse_args()

    generate_prediction_report(args.ticker)
