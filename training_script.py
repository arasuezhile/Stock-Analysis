# training_script.py

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def run_universal_training(dataset_path: str, output_dir: str = 'training_dataset'):
    """
    Trains a universal model on the aggregated dataset from multiple stocks.
    Saves the model in the output_dir.
    """
    print(f"--- Starting Universal Model Training using '{dataset_path}' ---")

    try:
        df = pd.read_csv(dataset_path)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"Successfully loaded universal dataset with {len(df)} rows.")
    except FileNotFoundError:
        print(f"ERROR: Universal dataset not found at '{dataset_path}'")
        print("Please run 'build_universal_dataset.py' first.")
        return False

    print("Preparing data for training...")
    y = df['label']
    X = df.drop(columns=['label', 'Ticker', 'Date'])
    non_numeric_cols = X.select_dtypes(include=['object']).columns
    if not non_numeric_cols.empty:
        print(f"Warning: Dropping non-numeric columns from features: {list(non_numeric_cols)}")
        X = X.drop(columns=non_numeric_cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"Data split into training set ({len(X_train)} rows) and testing set ({len(X_test)} rows).")

    print("\n--- Training Universal XGBoost Model ---")
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(y.unique()),
        use_label_encoder=False,
        eval_metric='mlogloss',
    )
    model.fit(X_train, y_train)
    print("Model training complete.")

    print("\n--- Evaluating Model Performance on Unseen Test Data ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Universal Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # --- CHANGE ---
    # Save the model to the specified output directory
    model_filename = os.path.join(output_dir, "universal_elliott_wave_model.pkl")
    # --- END CHANGE ---
    
    joblib.dump(model, model_filename)
    print(f"\n--- Model Saved ---")
    print(f"The universal trained model has been saved to '{model_filename}'.")
    return True

if __name__ == '__main__':
    # --- CHANGE ---
    # Define paths relative to the new directory structure
    output_directory = 'training_dataset'
    universal_dataset_path = os.path.join(output_directory, 'universal_labeled_dataset.csv')
    # --- END CHANGE ---
    run_universal_training(dataset_path=universal_dataset_path, output_dir=output_directory)
