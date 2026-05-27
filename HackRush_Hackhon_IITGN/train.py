import sys
import os
import joblib
import pandas as pd
import numpy as np
from collections import Counter

# Add the 'src' folder to the Python search path to import our modules
sys.path.append(os.path.abspath('src'))

from data_loader import load_data_chunked
from preprocessor import preprocess_pipeline
from model import split_data, train_lightgbm, evaluate_model

# Local system dataset files path
raw_csv_path = r"C:\Users\siddp\Downloads\Dataset for default loan prediction\loan.csv"
output_dir = "models"
output_assets_path = os.path.join(output_dir, "model_assets.joblib")

def extract_categorical_frequency_mappings(raw_data_path, feature_columns):
    """
    Extracts the normalized frequency mappings of all categorical columns
    directly from the raw CSV in a single pass. This allows us to convert
    text categories into exact float frequencies inside Streamlit.
    """
    print("[Trainer] Extracting category frequency mappings from raw dataset...")
    
    # Read the first 5 rows of raw CSV to check which columns are text/objects
    df_first = pd.read_csv(raw_data_path, nrows=5)
    
    cat_columns_to_extract = []
    for col in feature_columns:
        if col in df_first.columns:
            if df_first[col].dtype == object or str(df_first[col].dtype) == 'category':
                cat_columns_to_extract.append(col)
                
    print(f"[Trainer] Categorical features identified: {cat_columns_to_extract}")
    
    # Initialize frequency Counters
    counts = {col: Counter() for col in cat_columns_to_extract}
    
    # Read raw CSV in chunks of 500,000, loading ONLY the categorical columns to save RAM
    chunk_size = 500000
    for chunk in pd.read_csv(raw_data_path, usecols=cat_columns_to_extract, chunksize=chunk_size, low_memory=False):
        for col in cat_columns_to_extract:
            # Map NaNs to __MISSING__ so we can count them
            val_counts = chunk[col].fillna('__MISSING__').value_counts().to_dict()
            for val, count in val_counts.items():
                counts[col][val] += count
                
    # Normalize the counts to get probabilities
    mappings = {}
    for col in cat_columns_to_extract:
        col_counts = counts[col]
        
        # Identify the mode (most frequent non-missing category)
        non_missing = {k: v for k, v in col_counts.items() if k != '__MISSING__'}
        mode_val = max(non_missing, key=non_missing.get) if non_missing else 'Unknown'
        
        # Add missing count to the mode category (imputation matching)
        missing_count = col_counts.get('__MISSING__', 0)
        if missing_count > 0:
            col_counts[mode_val] = col_counts.get(mode_val, 0) + missing_count
            del col_counts['__MISSING__']
            
        # Normalize
        total = sum(col_counts.values())
        freq_map = {str(k): float(v / total) for k, v in col_counts.items()}
        
        mappings[col] = {
            'freq_map': freq_map,
            'default_mode': str(mode_val)
        }
        
    print("[Trainer] Category frequency mappings extracted successfully!")
    return mappings

def main():
    print("=== STARTING SMARTLEND TRAINING PIPELINE ===")
    
    # 1. Load Raw CSV Data
    if not os.path.exists(raw_csv_path):
        print(f"Error: Raw CSV dataset not found at: {raw_csv_path}")
        return
        
    # We load raw data using our modular chunk loader
    df_raw = load_data_chunked(raw_csv_path)
    
    # 2. Run Preprocessing Pipeline
    df_processed = preprocess_pipeline(df_raw)
    
    # 3. Stratified Data Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_processed)
    
    # 4. Train LightGBM Model
    model = train_lightgbm(X_train, y_train, X_val, y_val)
    
    # 5. Evaluate Model
    metrics = evaluate_model(model, X_test, y_test)
    
    # 6. Extract Feature Medians (for filling advanced inputs in UI)
    feature_medians = X_train.median().to_dict()
    feature_medians = {k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in feature_medians.items()}
    
    # 7. Extract Winsorization Limits (for clipping outliers in UI)
    outlier_cols = ['annual_inc', 'loan_amnt', 'dti']
    winsorize_limits = {}
    for col in outlier_cols:
        if col in X_train.columns:
            winsorize_limits[col] = {
                'lower': float(X_train[col].quantile(0.01)),
                'upper': float(X_train[col].quantile(0.99))
            }
            
    # 8. Extract Categorical Frequency Mappings
    feature_columns = list(X_train.columns)
    cat_mappings = extract_categorical_frequency_mappings(raw_csv_path, feature_columns)
    
    # 9. Save all assets together
    os.makedirs(output_dir, exist_ok=True)
    assets = {
        'model': model,
        'features': feature_columns,
        'medians': feature_medians,
        'winsorize_limits': winsorize_limits,
        'cat_mappings': cat_mappings,
        'test_metrics': metrics
    }
    
    print(f"[Trainer] Saving model assets to: {output_assets_path}...")
    joblib.dump(assets, output_assets_path)
    print("=== MODEL TRAINING AND SERIALIZATION PIPELINE COMPLETE ===")

if __name__ == "__main__":
    main()
