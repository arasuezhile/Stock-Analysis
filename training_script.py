# training_script.py

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib # Used for saving the trained model
import os # Added for path operations

# Helper function for consistent file naming
def get_model_filename(ticker, output_dir='.'):
    return os.path.join(output_dir, f"{ticker}_elliott_wave_model.pkl")

# --- NEW FUNCTION FOR ORCHESTRATION ---
def run_training_pipeline(ticker: str, labeled_dataset_path: str, output_dir: str = '.'):
    """
    Loads the labeled dataset, trains an XGBoost classifier, evaluates it,
    and saves the final model to a file, using a ticker for filename convention.

    Args:
        ticker (str): The stock ticker symbol.
        labeled_dataset_path (str): The path to the labeled CSV file.
        output_dir (str): Directory to save the model.
    Returns True on success, False otherwise.
    """
    print(f"--- Starting Model Training Process for {ticker} ---")

    # 1. Load the Labeled Data
    try:
        df = pd.read_csv(labeled_dataset_path, index_col='Date', parse_dates=True)
        print(f"Successfully loaded labeled dataset: '{labeled_dataset_path}'")
    except FileNotFoundError:
        print(f"ERROR: The dataset file was not found: '{labeled_dataset_path}'")
        print("Please run the feature generation and labeling scripts first.")
        return False
    except Exception as e:
        print(f"--- ERROR ---: Could not load dataset: {e}")
        return False

    # 2. Prepare Data: Separate Features (X) and Target (y)
    print("Preparing data for training...")
    # The 'label' column is our target variable (what we want to predict)
    y = df['label']
    # All other columns (except date and ticker info) are our input features
    # Drop the index 'Date' as well, which might have been loaded as a column if index_col was not set
    X = df.drop(columns=['label'], errors='ignore') # 'Date' is now the index, so no need to drop as a column

    # Identify non-numeric columns that might be present (e.g., 'level_0' if reset_index was used implicitly)
    non_numeric_cols = X.select_dtypes(include=['object']).columns
    if not non_numeric_cols.empty:
        print(f"Warning: Dropping non-numeric columns {list(non_numeric_cols)} from features.")
        X = X.drop(columns=non_numeric_cols)

    # 3. Split Data into Training and Testing Sets
    # We split the data to train the model on one part and test it on another, unseen part.
    # shuffle=False is important for time-series data to prevent looking into the future.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"Data split into training set ({len(X_train)} rows) and testing set ({len(X_test)} rows).")

    # 4. Initialize and Train the XGBoost Model
    print("\n--- Training XGBoost Model ---")
    # We use XGBClassifier for multi-class classification problems like ours.
    # Dynamically set num_class based on unique labels found in y.
    model = xgb.XGBClassifier(
        objective='multi:softmax', # Specifies this is a multi-class problem
        num_class=len(y.unique()), # Dynamically set num_class based on unique labels (0-8 will give 9 classes)
        use_label_encoder=False,   # Modern XGBoost requirement
        eval_metric='mlogloss'     # Logarithmic loss metric for evaluation
    )

    model.fit(X_train, y_train)
    print("Model training complete.")

    # 5. Evaluate the Model's Performance
    print("\n--- Evaluating Model Performance on Unseen Test Data ---")
    # Use the trained model to make predictions on the test set
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    # Ensure all labels from the full dataset are covered, not just test set unique ones
    all_possible_labels = sorted(df['label'].unique())
    target_names = [str(label) for label in all_possible_labels]

    # --- FIX START ---
    # Pass the 'labels' parameter to ensure classification_report knows all possible classes
    print(classification_report(y_test, y_pred, labels=all_possible_labels, target_names=target_names, zero_division=0))
    # --- FIX END ---

    # 6. Save the Trained Model
    model_filename = get_model_filename(ticker, output_dir)
    joblib.dump(model, model_filename)
    print(f"\n--- Model Saved ---")
    print(f"The trained model has been saved to '{model_filename}'.")
    print("This file can now be used for future predictions.")
    return True

# This block allows you to run the training directly from your terminal
if __name__ == '__main__':
    stock_ticker = 'RELIANCE.NS' # Default ticker for direct execution
    # Define the path to the dataset we created in the last step
    dataset_path = f"{stock_ticker}_labeled_dataset.csv"

    print(f"\n--- Running Model Training for {stock_ticker} (Direct Run) ---")
    run_training_pipeline(stock_ticker, dataset_path) # Calls the new function for direct execution
