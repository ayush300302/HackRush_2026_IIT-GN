import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report
import lightgbm as lgb
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def split_data(df: pd.DataFrame, target_col='is_default', test_size=0.2, val_size=0.2, random_state=42):
    """Split data into train, validation, and test sets using stratified sampling."""
    logging.info("Splitting data into Train/Val/Test sets...")
    
    # Check if target exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Ensure no categorical object cols are passed to sklearn directly
    X = X.select_dtypes(exclude=['object', 'category'])
    
    # 1. Split into train_val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # 2. Split train_val into train and validation
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state
    )
    
    logging.info(f"Train size: {X_train.shape}, Val size: {X_val.shape}, Test size: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_model(model_name: str, y_true, y_pred_proba, y_pred):
    """Custom evaluation metrics for imbalanced classification."""
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)
    
    logging.info(f"--- {model_name} Evaluation ---")
    logging.info(f"ROC-AUC: {roc_auc:.4f}")
    logging.info(f"PR-AUC: {pr_auc:.4f}")
    logging.info(f"F1-Score: {f1:.4f}")
    logging.info(f"Classification Report:\n{classification_report(y_true, y_pred)}")
    
    return {
        'Model': model_name,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'F1-Score': f1
    }

def train_baselines(X_train, y_train, X_val, y_val, sample_size=200000):
    """Train Baseline models (Logistic Regression, Decision Tree)."""
    logging.info(f"Training Baselines on a subsample of {sample_size} rows...")
    
    # Subsample to speed up baseline training if dataset is too large
    if len(X_train) > sample_size:
        # We need stratified subsample
        X_sub, _, y_sub, _ = train_test_split(X_train, y_train, train_size=sample_size, stratify=y_train, random_state=42)
    else:
        X_sub, y_sub = X_train, y_train

    results = []
    
    # Logistic Regression
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_sub, y_sub)
    y_pred_proba_lr = lr.predict_proba(X_val)[:, 1]
    y_pred_lr = lr.predict(X_val)
    results.append(evaluate_model("Logistic Regression", y_val, y_pred_proba_lr, y_pred_lr))
    
    # Decision Tree
    dt = DecisionTreeClassifier(class_weight='balanced', max_depth=10, random_state=42)
    dt.fit(X_sub, y_sub)
    y_pred_proba_dt = dt.predict_proba(X_val)[:, 1]
    y_pred_dt = dt.predict(X_val)
    results.append(evaluate_model("Decision Tree", y_val, y_pred_proba_dt, y_pred_dt))
    
    return pd.DataFrame(results), (lr, dt)

def train_advanced_models(X_train, y_train, X_val, y_val):
    """Train Advanced models (LightGBM, XGBoost) natively handling class imbalance."""
    logging.info("Training Advanced Models on full training set...")
    
    results = []
    
    # Calculate scale_pos_weight
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    # LightGBM
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )
    # Using early stopping callback
    lgb_clf.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    y_pred_proba_lgb = lgb_clf.predict_proba(X_val)[:, 1]
    y_pred_lgb = lgb_clf.predict(X_val)
    results.append(evaluate_model("LightGBM", y_val, y_pred_proba_lgb, y_pred_lgb))
    
    # XGBoost
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30
    )
    xgb_clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    y_pred_proba_xgb = xgb_clf.predict_proba(X_val)[:, 1]
    y_pred_xgb = xgb_clf.predict(X_val)
    results.append(evaluate_model("XGBoost", y_val, y_pred_proba_xgb, y_pred_xgb))
    
    return pd.DataFrame(results), (lgb_clf, xgb_clf)
