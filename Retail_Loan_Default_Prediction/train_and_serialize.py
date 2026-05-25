import sys
import os
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
dataset_path = r"C:\Users\siddp\Downloads\Dataset for default loan prediction\loan.csv"
parquet_path = r"C:\Users\siddp\Downloads\HackRush_2026\Retail_Loan_Default_Prediction\data\processed_loan_data.parquet"
output_dir = r"C:\Users\siddp\Downloads\HackRush_2026\Retail_Loan_Default_Prediction\models"
output_model_path = os.path.join(output_dir, "model_assets.joblib")

# Add src to python path to use modeling.py
sys.path.append(os.path.abspath('src'))
from modeling import split_data, evaluate_model

def train_lgb(X_train, y_train, X_val, y_val):
    logging.info("Training LightGBM model...")
    # Calculate scale_pos_weight
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    logging.info(f"Scale pos weight: {scale_pos_weight:.4f}")
    
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )
    
    lgb_clf.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30)]
    )
    
    return lgb_clf

def extract_cat_mappings(raw_csv_path, final_columns):
    logging.info("Extracting categorical frequency maps in a single pass...")
    
    # 1. Identify which columns in final_columns are categorical in raw csv
    df_first = pd.read_csv(raw_csv_path, nrows=5)
    
    cat_cols = []
    for col in final_columns:
        if col in df_first.columns:
            if df_first[col].dtype == object or str(df_first[col].dtype) == 'category':
                cat_cols.append(col)
                
    logging.info(f"Categorical columns found: {cat_cols}")
    
    # Initialize Counters
    counts = {col: Counter() for col in cat_cols}
    
    # Read raw CSV in chunks, loading only the categorical columns to save memory and time
    chunk_size = 500000
    logging.info("Reading raw CSV in chunks of 500k...")
    for chunk in pd.read_csv(raw_csv_path, usecols=cat_cols, chunksize=chunk_size, low_memory=False):
        for col in cat_cols:
            val_counts = chunk[col].fillna('__MISSING__').value_counts().to_dict()
            for val, count in val_counts.items():
                counts[col][val] += count
                
    # Now build the final frequency maps
    mappings = {}
    for col in cat_cols:
        col_counts = counts[col]
        
        # Find mode (most frequent non-missing value)
        non_missing_counts = {k: v for k, v in col_counts.items() if k != '__MISSING__'}
        if non_missing_counts:
            mode_val = max(non_missing_counts, key=non_missing_counts.get)
        else:
            mode_val = 'Unknown'
            
        # Re-attribute missing values to the mode
        missing_count = col_counts.get('__MISSING__', 0)
        if missing_count > 0:
            col_counts[mode_val] = col_counts.get(mode_val, 0) + missing_count
            del col_counts['__MISSING__']
            
        # Normalize counts to get probabilities
        total_count = sum(col_counts.values())
        freq_map = {str(k): float(v / total_count) for k, v in col_counts.items()}
        
        mappings[col] = {
            'freq_map': freq_map,
            'default_mode': str(mode_val)
        }
        
    logging.info("Categorical frequency maps extraction completed.")
    return mappings

def main():
    logging.info("Starting optimized model training and serialization pipeline...")
    
    # 1. Load processed Parquet
    if not os.path.exists(parquet_path):
        logging.error(f"Parquet file not found at {parquet_path}")
        return
        
    logging.info(f"Loading preprocessed data from {parquet_path}")
    df_processed = pd.read_parquet(parquet_path)
    
    # 2. Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_processed)
    
    # 3. Train LightGBM model
    lgb_model = train_lgb(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    y_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
    y_pred = lgb_model.predict(X_test)
    eval_results = evaluate_model("LightGBM Test Set", y_test, y_pred_proba, y_pred)
    logging.info(f"Evaluation results: {eval_results}")
    
    # 4. Get medians for all features in X_train (for filling Streamlit form non-inputs)
    feature_medians = X_train.median().to_dict()
    feature_medians = {k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in feature_medians.items()}
    
    # 5. Extract winsorization thresholds (percentiles) for outlier columns
    outlier_cols = ['annual_inc', 'loan_amnt', 'dti']
    winsorize_limits = {}
    for col in outlier_cols:
        if col in X_train.columns:
            winsorize_limits[col] = {
                'lower': float(X_train[col].quantile(0.01)),
                'upper': float(X_train[col].quantile(0.99))
            }
    logging.info(f"Winsorization limits: {winsorize_limits}")
    
    # 6. Extract categorical mappings from raw loan.csv (single pass)
    if not os.path.exists(dataset_path):
        logging.error(f"Raw CSV not found at {dataset_path}, cannot extract frequency mappings!")
        return
        
    final_features = list(X_train.columns)
    cat_mappings = extract_cat_mappings(dataset_path, final_features)
    
    # 7. Save model assets
    os.makedirs(output_dir, exist_ok=True)
    assets = {
        'model': lgb_model,
        'features': final_features,
        'medians': feature_medians,
        'winsorize_limits': winsorize_limits,
        'cat_mappings': cat_mappings,
        'test_metrics': eval_results
    }
    
    logging.info(f"Saving assets to {output_model_path}")
    joblib.dump(assets, output_model_path)
    logging.info("Model training and serialization completed successfully!")

if __name__ == "__main__":
    main()
