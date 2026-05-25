import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def drop_high_missing_cols(df: pd.DataFrame, threshold=0.5):
    """Drop columns with missing values greater than the threshold."""
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index
    logging.info(f"Dropping {len(cols_to_drop)} columns with > {threshold*100}% missing values.")
    return df.drop(columns=cols_to_drop)

def impute_missing_values(df: pd.DataFrame):
    """Impute numerical features with median and categorical with mode."""
    logging.info("Imputing missing values...")
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        if df[col].isnull().any():
            # Mode can be empty, so we handle it gracefully
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna('Unknown')
    return df

def winsorize_features(df: pd.DataFrame, columns, lower=0.01, upper=0.99):
    """Clip values at the lower and upper percentiles to handle outliers."""
    logging.info(f"Winsorizing features: {columns}")
    for col in columns:
        if col in df.columns:
            l_val = df[col].quantile(lower)
            u_val = df[col].quantile(upper)
            df[col] = df[col].clip(lower=l_val, upper=u_val)
    return df

def feature_engineering(df: pd.DataFrame):
    """Create domain-specific features."""
    logging.info("Creating domain-specific features...")
    if 'annual_inc' in df.columns and 'loan_amnt' in df.columns:
        # Avoid division by zero
        df['income_to_loan_ratio'] = df['annual_inc'] / (df['loan_amnt'] + 1e-5)
    
    # We can create a simple binary target if not already present
    if 'loan_status' in df.columns and 'is_default' not in df.columns:
        default_categories = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'Late (31-120 days)']
        df['is_default'] = df['loan_status'].isin(default_categories).astype(int)

    return df

def encode_categoricals(df: pd.DataFrame):
    """Encode categorical variables."""
    logging.info("Encoding categorical variables...")
    
    # Ordinal encoding for grade and sub_grade
    if 'grade' in df.columns:
        grade_mapping = {g: i for i, g in enumerate(sorted(df['grade'].dropna().unique()))}
        df['grade'] = df['grade'].map(grade_mapping)
        
    if 'sub_grade' in df.columns:
        sub_grade_mapping = {g: i for i, g in enumerate(sorted(df['sub_grade'].dropna().unique()))}
        df['sub_grade'] = df['sub_grade'].map(sub_grade_mapping)

    # Frequency encoding for high cardinality/other nominals to save memory
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        # Avoid target leakage, exclude target columns
        if col not in ['loan_status', 'is_default']:
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq)
            
    return df

def apply_variance_threshold(df: pd.DataFrame, threshold=0.01):
    """Remove features with very low variance."""
    logging.info("Applying variance threshold...")
    num_cols = df.select_dtypes(include=[np.number]).columns
    # Exclude target from feature selection
    features = [c for c in num_cols if c not in ['is_default', 'loan_status']]
    
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df[features])
    
    cols_to_keep = np.array(features)[selector.get_support()]
    cols_to_drop = set(features) - set(cols_to_keep)
    
    logging.info(f"Dropping {len(cols_to_drop)} low variance features.")
    return df.drop(columns=list(cols_to_drop))

def drop_leakage_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop post-origination columns that cause target leakage."""
    leakage_cols = [
        'collection_recovery_fee',
        'debt_settlement_flag',
        'last_credit_pull_d',
        'last_pymnt_amnt',
        'last_pymnt_d',
        'out_prncp',
        'out_prncp_inv',
        'recoveries',
        'total_pymnt',
        'total_pymnt_inv',
        'total_rec_int',
        'total_rec_late_fee',
        'total_rec_prncp',
        'loan_status'
    ]
    cols_to_drop = [col for col in leakage_cols if col in df.columns]
    logging.info(f"Dropping {len(cols_to_drop)} post-origination columns to prevent target leakage: {cols_to_drop}")
    return df.drop(columns=cols_to_drop)

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing pipeline."""
    logging.info("Starting preprocessing pipeline...")
    
    # 1. Generate target label first (since it depends on loan_status)
    df = feature_engineering(df)
    
    # 2. Drop leakage columns (including loan_status)
    df = drop_leakage_cols(df)
    
    # 3. Apply general cleaning and profiling
    df = drop_high_missing_cols(df, threshold=0.5)
    df = impute_missing_values(df)
    
    outlier_cols = ['annual_inc', 'loan_amnt', 'dti']
    df = winsorize_features(df, outlier_cols)
    
    df = encode_categoricals(df)
    df = apply_variance_threshold(df)
    
    logging.info("Preprocessing pipeline completed.")
    return df

