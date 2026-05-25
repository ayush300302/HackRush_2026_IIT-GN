import pandas as pd
import numpy as np

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates custom calculated columns.
    1. income_to_loan_ratio: How many times their salary is relative to the loan.
    2. is_default: Creates a binary target (1 for default/charged-off, 0 for paid back).
    """
    print("[Preprocessor] Generating engineered features...")
    
    # 1. Income-to-loan ratio
    if 'annual_inc' in df.columns and 'loan_amnt' in df.columns:
        df['income_to_loan_ratio'] = df['annual_inc'] / (df['loan_amnt'] + 1e-5)
        
    # 2. Target mapping (is_default)
    if 'loan_status' in df.columns and 'is_default' not in df.columns:
        default_categories = [
            'Charged Off', 
            'Default', 
            'Does not meet the credit policy. Status:Charged Off', 
            'Late (31-120 days)'
        ]
        # Set to 1 if the status matches default indicators, else 0
        df['is_default'] = df['loan_status'].isin(default_categories).astype(int)
        
    return df

def drop_leakage_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns that are generated AFTER the loan is funded (e.g. total recovery fees, 
    last payment amount). Using these during prediction causes target leakage (cheating).
    """
    leakage_cols = [
        'collection_recovery_fee', 'debt_settlement_flag', 'last_credit_pull_d',
        'last_pymnt_amnt', 'last_pymnt_d', 'out_prncp', 'out_prncp_inv',
        'recoveries', 'total_pymnt', 'total_pymnt_inv', 'total_rec_int',
        'total_rec_late_fee', 'total_rec_prncp', 'loan_status'
    ]
    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    print(f"[Preprocessor] Dropping {len(cols_to_drop)} post-origination leakage columns...")
    return df.drop(columns=cols_to_drop)

def drop_high_missing_cols(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Identifies and drops columns where more than 50% of the values are missing.
    """
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index
    print(f"[Preprocessor] Dropping {len(cols_to_drop)} columns with >{threshold*100}% missing values.")
    return df.drop(columns=cols_to_drop)

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values:
    - Numerical columns are filled with their median value.
    - Categorical columns are filled with their most frequent value (mode).
    """
    print("[Preprocessor] Imputing missing values...")
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Fill numeric NaNs
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
            
    # Fill categorical NaNs
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna('Unknown')
                
    return df

def winsorize_features(df: pd.DataFrame, columns: list, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """
    Clips extreme values at the 1st and 99th percentiles to handle outliers
    without losing the predictive signal.
    """
    print(f"[Preprocessor] Winsorizing numerical outliers for: {columns}...")
    for col in columns:
        if col in df.columns:
            l_val = df[col].quantile(lower)
            u_val = df[col].quantile(upper)
            df[col] = df[col].clip(lower=l_val, upper=u_val)
    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms text/string category columns into numeric numbers.
    It replaces each text category with its frequency ratio in the dataset.
    """
    print("[Preprocessor] Mapping categorical columns to frequency ratios...")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in cat_cols:
        if col not in ['is_default']:
            # Calculate counts frequency
            freq = df[col].value_counts(normalize=True)
            # Replaces the text key with the float frequency value
            df[col] = df[col].map(freq).astype(float)
            
    return df

def apply_variance_threshold(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """
    Filters out numeric features with almost zero variance (columns where values 
    are virtually constant and contain no predictive variance).
    """
    print("[Preprocessor] Filtering low-variance features...")
    num_cols = df.select_dtypes(include=[np.number]).columns
    features = [c for c in num_cols if c not in ['is_default']]
    
    # Critical: Do NOT run variance threshold check on categorical features 
    # that have been frequency-encoded (since their frequency variance is naturally very small)
    protected_cat_cols = [
        'term', 'grade', 'sub_grade', 'emp_length', 'home_ownership', 
        'verification_status', 'purpose', 'title', 'initial_list_status', 
        'application_type', 'disbursement_method'
    ]
    features = [c for c in features if c not in protected_cat_cols]
    
    # Calculate variance
    variances = df[features].var()
    cols_to_drop = variances[variances < threshold].index
    print(f"[Preprocessor] Dropping {len(cols_to_drop)} features with variance < {threshold}...")
    return df.drop(columns=list(cols_to_drop))

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    The orchestrator function. Executes all data cleaning steps in a sequence.
    """
    print("[Preprocessor] Executing full data preprocessing pipeline...")
    
    # 1. Target engineering
    df = feature_engineering(df)
    
    # 2. Target leakage filter
    df = drop_leakage_cols(df)
    
    # 3. Missing column filter
    df = drop_high_missing_cols(df, threshold=0.5)
    
    # 4. Imputation
    df = impute_missing_values(df)
    
    # 5. Outlier winsorization
    outlier_cols = ['annual_inc', 'loan_amnt', 'dti']
    df = winsorize_features(df, outlier_cols)
    
    # 6. Frequency encoding
    df = encode_categoricals(df)
    
    # 7. Low variance filter
    df = apply_variance_threshold(df, threshold=0.01)
    
    print("[Preprocessor] Data preprocessing pipeline complete!")
    return df
