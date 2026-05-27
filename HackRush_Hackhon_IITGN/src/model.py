import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

def split_data(df: pd.DataFrame, target_col: str = 'is_default', test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42):
    """
    Splits the preprocessed dataset into train, validation, and test sets.
    Uses stratification to maintain identical target default rates in all splits.
    """
    print("[Model] Splitting dataset into train, validation, and test splits...")
    
    # Target and predictor variables separation
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Exclude remaining text-based variables
    X = X.select_dtypes(exclude=['object', 'category'])
    
    # 1. Separate test partition (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # 2. Separate training (60%) and validation (20%) partitions
    val_ratio_temp = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_temp, stratify=y_temp, random_state=random_state
    )
    
    print(f"[Model] Splitting done. Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> lgb.LGBMClassifier:
    """
    Trains a simple LightGBM classifier, adjusting for class imbalance.
    """
    print("[Model] Creating and fitting LightGBM classifier...")
    
    # Calculate scale weight for class imbalance mitigation
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"[Model] Class balance ratio scale_pos_weight: {scale_pos_weight:.4f}")
    
    # Initialize LightGBM classifier with simple default parameters
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )
    
    # Train the model with early stopping on validation partition
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30)]
    )
    
    return clf

def evaluate_model(model: lgb.LGBMClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluates model performance on the test set.
    """
    print("[Model] Running model evaluation on unseen test partition...")
    
    # Compute class probability predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC-AUC and PR-AUC scores
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print(f"[Model] Evaluation: ROC-AUC = {roc_auc:.4f}, PR-AUC = {pr_auc:.4f}")
    
    return {
        'ROC-AUC': float(roc_auc),
        'PR-AUC': float(pr_auc)
    }
