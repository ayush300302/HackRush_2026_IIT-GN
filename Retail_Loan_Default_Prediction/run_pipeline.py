import sys
import os
import pandas as pd

# Add src directory to path
sys.path.append(os.path.abspath('src'))

from data_loader import load_data_chunked
from preprocessing import preprocess_pipeline
from modeling import split_data, train_baselines, train_advanced_models

dataset_path = r"C:\Users\siddp\Downloads\Dataset for default loan prediction\loan.csv"

def main():
    print("Loading data...")
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return
        
    df_raw = load_data_chunked(dataset_path)

    if df_raw is not None:
        print("Preprocessing data...")
        df_processed = preprocess_pipeline(df_raw)
        
        print("Saving processed data to Parquet...")
        output_dir = r"C:\Users\siddp\Downloads\HackRush_2026\Retail_Loan_Default_Prediction\data"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "processed_loan_data.parquet")
        
        try:
            df_processed.to_parquet(output_path, index=False)
            print(f"Saved successfully to {output_path}")
        except Exception as e:
            print(f"Failed to save parquet (might need to 'pip install pyarrow'): {e}")
        
        print("Splitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_processed)
        
        print("Training Baselines...")
        baseline_results, baseline_models = train_baselines(X_train, y_train, X_val, y_val, sample_size=200000)
        print(baseline_results)
        
        print("Training Advanced Models...")
        adv_results, adv_models = train_advanced_models(X_train, y_train, X_val, y_val)
        print(adv_results)
        
        print("\n--- Final Model Comparison ---")
        final_comparison = pd.concat([baseline_results, adv_results], ignore_index=True)
        final_comparison = final_comparison.sort_values(by='PR-AUC', ascending=False)
        print(final_comparison)
        
if __name__ == "__main__":
    main()
