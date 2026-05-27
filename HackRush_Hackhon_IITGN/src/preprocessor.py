import pandas as pd
import numpy as np

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates custom calculated columns.
    1. income_to_loan_ratio: Borrower annual salary relative to requested loan amount.
    2. installment_to_income_ratio: Percentage of monthly income spent on loan installment.
    3. is_default: Binary target flag indicating default/risk status based on loan_status.
    """
    print("[Preprocessor] Generating engineered features...")
    
    # 1. Income-to-loan ratio
    if 'annual_inc' in df.columns and 'loan_amnt' in df.columns:
        df['income_to_loan_ratio'] = df['annual_inc'] / (df['loan_amnt'] + 1e-5)
        
    # 2. Installment-to-income ratio (Monthly installment / Monthly income)
    if 'installment' in df.columns and 'annual_inc' in df.columns:
        monthly_income = df['annual_inc'] / 12.0
        df['installment_to_income_ratio'] = df['installment'] / (monthly_income + 1e-5)
        
    # 3. Target mapping (is_default)
    if 'loan_status' in df.columns and 'is_default' not in df.columns:
        default_categories = [
            'Charged Off', 
            'Default', 
            'Does not meet the credit policy. Status:Charged Off', 
            'Late (31-120 days)'
        ]
        df['is_default'] = df['loan_status'].isin(default_categories).astype(int)
        
    return df

def drop_leakage_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops target leakage columns that are only generated post-origination.
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
    Drops columns with more than the threshold percentage of missing values.
    """
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index
    print(f"[Preprocessor] Dropping {len(cols_to_drop)} columns with >{threshold*100}% missing values.")
    return df.drop(columns=cols_to_drop)

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values:
    - Numerical columns are filled with their median value.
    - Categorical columns are filled with their mode (most frequent value).
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
    Winsorizes extreme numerical features by capping them at specified percentiles.
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
    Encodes text categorical columns using smart ordinal mapping for ordered fields,
    and frequency ratios for unordered fields.
    """
    print("[Preprocessor] Mapping categorical columns using smart encoding...")
    
    # Define ordinal maps
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    
    sub_grade_map = {}
    grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    for i, g in enumerate(grades):
        for num in ['1', '2', '3', '4', '5']:
            sub_grade_map[g + num] = i * 5 + int(num)
            
    emp_length_map = {
        '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
        '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
        '10+ years': 10, 'Unknown': 0
    }
    
    # Apply ordinal maps if columns are present
    if 'grade' in df.columns:
        df['grade'] = df['grade'].map(grade_map).fillna(4.0).astype(float)
        
    if 'sub_grade' in df.columns:
        df['sub_grade'] = df['sub_grade'].map(sub_grade_map).fillna(17.0).astype(float)
        
    if 'emp_length' in df.columns:
        df['emp_length'] = df['emp_length'].map(emp_length_map).fillna(5.0).astype(float)
        
    # For remaining object/categorical columns, apply standard frequency encoding
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if col not in ['is_default', 'grade', 'sub_grade', 'emp_length']:
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq).astype(float)
            
    return df

def apply_variance_threshold(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """
    Filters out numerical columns with near-zero variance.
    """
    print("[Preprocessor] Filtering low-variance features...")
    num_cols = df.select_dtypes(include=[np.number]).columns
    features = [c for c in num_cols if c not in ['is_default']]
    
    # We do not drop protected frequency-encoded categorical columns
    protected_cat_cols = [
        'term', 'grade', 'sub_grade', 'emp_length', 'home_ownership', 
        'verification_status', 'purpose', 'title', 'initial_list_status', 
        'application_type', 'disbursement_method'
    ]
    features = [c for c in features if c not in protected_cat_cols]
    
    variances = df[features].var()
    cols_to_drop = variances[variances < threshold].index
    print(f"[Preprocessor] Dropping {len(cols_to_drop)} features with variance < {threshold}...")
    return df.drop(columns=list(cols_to_drop))

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sequentially executes all preprocessing stages.
    """
    print("[Preprocessor] Executing data preprocessing pipeline...")
    
    df = feature_engineering(df)
    df = drop_leakage_cols(df)
    df = drop_high_missing_cols(df, threshold=0.5)
    df = impute_missing_values(df)
    
    outlier_cols = ['annual_inc', 'loan_amnt', 'dti']
    df = winsorize_features(df, outlier_cols)
    # 6. Frequency encoding
    df = encode_categoricals(df)
    
    # [Tuned Improvement]: We do not run the manual low-variance filter. 
    # LightGBM handles feature selection automatically and performs better with the full feature set.
    # df = apply_variance_threshold(df, threshold=0.01)
    
    print("[Preprocessor] Preprocessing pipeline complete!")
    return df
