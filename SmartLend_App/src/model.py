import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

def split_data(df: pd.DataFrame, target_col: str = 'is_default', test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42):
    """
    Splits the preprocessed DataFrame into Train, Validation, and Test partitions.
    Uses 'stratify=y' to make sure that the default rate percentage is identical in all splits.
    """
    print("[Model] Splitting dataset into train, val, and test splits...")
    
    # Separate input features (X) from the default label (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Select only numerical columns (sklearn classifiers require numbers)
    X = X.select_dtypes(exclude=['object', 'category'])
    
    # 1. Split off the test partition (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # 2. Split the remaining 80% into training (60%) and validation (20%)
    val_ratio_temp = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_temp, stratify=y_temp, random_state=random_state
    )
    
    print(f"[Model] Split completed. Train: {X_train.shape[0]} rows, Val: {X_val.shape[0]} rows, Test: {X_test.shape[0]} rows")
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> lgb.LGBMClassifier:
    """
    Trains a LightGBM classifier.
    Calculates scale_pos_weight dynamically to handle the high default class imbalance.
    """
    print("[Model] Initializing LightGBM classifier...")
    
    # Calculate scale weight: (number of non-defaults) / (number of defaults)
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"[Model] Class balance ratio (scale_pos_weight): {scale_pos_weight:.4f}")
    
    # Define classifier
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit model with validation early stopping to prevent overfitting
    print("[Model] Training model (fitting trees)...")
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30)]
    )
    
    return clf

def evaluate_model(model: lgb.LGBMClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluates the model on the unseen test set.
    Calculates ROC-AUC (sorting probability ranking) and PR-AUC (useful for rare defaults).
    """
    print("[Model] Running evaluation on test partition...")
    
    # Predict probabilities (column 1 represents probability of default)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print(f"[Model] Test ROC-AUC: {roc_auc:.4f}")
    print(f"[Model] Test PR-AUC:  {pr_auc:.4f}")
    
    return {
        'ROC-AUC': float(roc_auc),
        'PR-AUC': float(pr_auc)
    }
