# generate_report.py

import pandas as pd
import joblib
from datetime import datetime, timedelta
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import our existing helper classes
from data_handler import DataHandler
from feature_generator import FeatureGenerator

def run_analysis_report(ticker: str, output_dir: str = 'training_dataset'):
    """
    Generates a complete analysis report for a given stock ticker, including:
    1. Current trend and 3-5 week forecast.
    2. A 4-week back-testing validation report.
    3. A price chart with the current trend annotated.
    """
    print(f"--- Generating Full Analysis Report for: {ticker} ---")

    # --- 1. Load the Universal Model ---
    model_path = os.path.join(output_dir, 'universal_elliott_wave_model.pkl')
    print(f"Loading universal model from: '{model_path}'")
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"--- ERROR ---: Universal model not found.")
        print("Please ensure you have run 'build_universal_dataset.py' and 'training_script.py' first.")
        return

    # --- 2. Fetch and Prepare Data ---
    print(f"Fetching latest data for {ticker}...")
    handler = DataHandler(ticker=ticker)
    # Fetch data for the last 2 years for context and feature calculation
    start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
    price_data = handler.fetch_historical_data(start_date=start_date)
    
    if price_data.empty:
        print(f"Could not fetch price data for {ticker}. Aborting.")
        return

    # Generate the same features the model was trained on
    feature_gen = FeatureGenerator(data_df=price_data)
    feature_dataset = feature_gen.generate_features()

    # --- 3. Make Predictions ---
    print("Running model predictions...")
    X_predict = feature_dataset.drop(columns=['label'], errors='ignore')
    predictions = model.predict(X_predict)
    feature_dataset['predicted_label'] = predictions

    # --- 4. Define Mappings and Current Status ---
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
    
    last_day_prediction = feature_dataset['predicted_label'].iloc[-1]
    current_state = label_to_state.get(last_day_prediction, "Unknown")
    current_trend = label_to_trend.get(last_day_prediction, "Unknown")

    # --- 5. Display Text Report: Forecast ---
    print("\n" + "="*20 + f" Analysis for {ticker} " + "="*20)
    print(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Current Model-Predicted State: {current_state}")
    print(f"Implied Near-Term Trend: {current_trend}")
    
    print("\n--- 3-5 Week Trend Forecast ---")
    forecast_data = []
    for i in range(1, 6):
        forecast_data.append({'Forecast Week': f'Week {i}', 'Predicted Trend': current_trend})
    forecast_df = pd.DataFrame(forecast_data)
    print(forecast_df.to_string(index=False))

    # --- 6. Display Text Report: Back-testing Validation ---
    print("\n--- 4-Week Back-testing Report (Validation) ---")
    backtest_data = []
    
    # Make the 'today' variable timezone-aware by matching the data's timezone
    today = pd.Timestamp.now(tz=feature_dataset.index.tz)

    for i in range(4, 0, -1):
        end_of_week = today - timedelta(weeks=i-1)
        start_of_week = today - timedelta(weeks=i)

        try:
            # Find the insertion point for the start_of_week date using a compatible method
            temp_idx = feature_dataset.index.searchsorted(start_of_week, side='right')
            if temp_idx == 0:
                continue
            # The prediction date is the one right before this insertion point
            prediction_date_index = temp_idx - 1

            predicted_label = feature_dataset['predicted_label'].iloc[prediction_date_index]
            predicted_trend = label_to_trend.get(predicted_label, "Unknown")
        except (KeyError, IndexError):
            predicted_trend = "No Data"

        # --- THIS LINE CONTAINS THE FIX ---
        # Compare aware-to-aware by removing the unnecessary tz_convert(None)
        week_data = price_data[(price_data.index >= start_of_week) & (price_data.index < end_of_week)]
        
        if not week_data.empty:
            actual_start_price = week_data['Close'].iloc[0]
            actual_end_price = week_data['Close'].iloc[-1]
            actual_change_pct = ((actual_end_price - actual_start_price) / actual_start_price) * 100
            actual_trend = "UPTREND" if actual_change_pct > 0 else "DOWNTREND"
            
            backtest_data.append({
                'Week Ending': end_of_week.strftime('%Y-%m-%d'),
                'Predicted Trend': predicted_trend,
                'Actual Trend': actual_trend,
                'Actual Change': f"{actual_change_pct:.2f}%"
            })

    if backtest_data:
        backtest_df = pd.DataFrame(backtest_data)
        print(backtest_df.to_string(index=False))
    else:
        print("Not enough recent data to generate a full back-testing report.")

    # --- 7. Generate and Save Chart ---
    print("\n--- Generating Chart ---")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot the last 9 months of price data for context
    chart_data = price_data.last('9M')
    ax.plot(chart_data.index, chart_data['Close'], label='Close Price', color='dodgerblue')

    # Add a title and labels
    ax.set_title(f'9-Month Price Chart for {ticker}\nReport Date: {datetime.now().strftime("%Y-%m-%d")}', fontsize=16)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)

    # Add an annotation for the current predicted trend
    trend_color = 'green' if 'UPTREND' in current_trend else 'red' if 'DOWNTREND' in current_trend else 'orange'
    ax.text(0.02, 0.95, f'Current Predicted Trend: {current_trend}', 
            transform=ax.transAxes, fontsize=14, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=trend_color, alpha=0.7))
            
    # Format the date axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    
    plt.legend()
    plt.grid(True)
    
    # Save the chart to the output directory
    chart_filename = os.path.join(output_dir, f"{ticker}_analysis_chart.png")
    plt.savefig(chart_filename)
    print(f"Chart successfully saved to '{chart_filename}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a complete analysis report for a stock.")
    parser.add_argument('ticker', type=str, help='The stock ticker symbol (e.g., RELIANCE.NS)')
    args = parser.parse_args()
    
    run_analysis_report(args.ticker)
