# training_script.py

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
# --- CHANGE: Import the utility for calculating class weights ---
from sklearn.utils.class_weight import compute_sample_weight
# --- END CHANGE ---
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def run_universal_training(dataset_path: str, output_dir: str = 'training_dataset'):
    """
    Trains a universal model on the aggregated dataset.
    Includes RandomizedSearchCV and sample weighting to handle class imbalance.
    """
    print(f"--- Starting Universal Model Training with Hyperparameter Tuning & Sample Weighting ---")

    try:
        df = pd.read_csv(dataset_path)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"Successfully loaded universal dataset with {len(df)} rows.")
    except FileNotFoundError:
        print(f"ERROR: Universal dataset not found at '{dataset_path}'")
        return False

    print("Preparing data for training...")
    y = df['label']
    X = df.drop(columns=['label', 'Ticker', 'Date', 'future_pct_change'])

    non_numeric_cols = X.select_dtypes(include=['object']).columns
    if not non_numeric_cols.empty:
        X = X.drop(columns=non_numeric_cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"Data split into training set ({len(X_train)} rows) and testing set ({len(X_test)} rows).")

    # --- HYPERPARAMETER TUNING BLOCK (WITH NEW WEIGHTING) ---
    print("\n--- Searching for Best Hyperparameters with Class Weighting ---")
    
    # --- CHANGE: Calculate sample weights to penalize errors on minority classes more heavily ---
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    # --- END CHANGE ---

    param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    }

    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(y.unique()),
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=25,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    # --- CHANGE: Pass the sample weights to the .fit() method ---
    # This tells the search to prioritize models that perform well on the under-represented classes.
    random_search.fit(X_train, y_train, sample_weight=sample_weights)
    # --- END CHANGE ---
    
    print("\nBest Hyperparameters found:")
    print(random_search.best_params_)
    
    best_model = random_search.best_estimator_

    print("\n--- Evaluating Best Model's Performance on Unseen Test Data ---")
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Tuned Universal Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    model_filename = os.path.join(output_dir, "universal_elliott_wave_model.pkl")
    joblib.dump(best_model, model_filename)
    print(f"\n--- Best Model Saved to '{model_filename}' ---")
    return True

if __name__ == '__main__':
    output_directory = 'training_dataset'
    universal_dataset_path = os.path.join(output_directory, 'universal_labeled_dataset.csv')
    run_universal_training(dataset_path=universal_dataset_path, output_dir=output_directory)
