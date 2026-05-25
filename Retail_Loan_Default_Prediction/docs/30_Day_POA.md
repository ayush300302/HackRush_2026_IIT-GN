# Retail Loan Default Prediction: 30-Day Plan of Action (POA)

This document outlines the step-by-step roadmap for converting our initial loan default prediction pipeline into a robust, production-grade 30-day Machine Learning project.

---

## Phase 1: Project Setup & Environment (Days 1-3)
**Goal:** Establish a solid foundation to handle 2.26 million rows of data efficiently.
*   **Day 1:** Initialize Git repository, define directory structure (`data/`, `notebooks/`, `src/`, `models/`), and set up the virtual environment (`requirements.txt`).
*   **Day 2:** Configure memory-efficient data loading strategies (e.g., Pandas chunking and downcasting datatypes) for the massive dataset.
*   **Day 3:** Set up experiment tracking (e.g., MLflow) and code formatting tools (Black, Ruff).

## Phase 2: Deep Exploratory Data Analysis (Days 4-8)
**Goal:** Understand the nuances, anomalies, and patterns within our 145 features.
*   **Day 4:** High-level data profiling (shape, data types, missing value heatmaps, memory footprint).
*   **Day 5:** Univariate analysis: Visualizing distributions of key features and the target variable.
*   **Day 6:** Bivariate analysis: Analyzing the relationship between features and the target variable (`loan_status`).
*   **Day 7:** Correlation analysis: Identifying multi-collinearity and plotting correlation matrices.
*   **Day 8:** Outlier detection and deep dive into the 12x default rate gap across specific groups.

## Phase 3: Data Cleaning & Preprocessing (Days 9-12)
**Goal:** Prepare the raw data for machine learning algorithms.
*   **Day 9:** Missing value imputation strategies (mean, median, mode, or predictive imputation).
*   **Day 10:** Handling outliers (clipping, robust scaling) without losing critical predictive signals.
*   **Day 11:** Encoding categorical variables (Target Encoding, One-Hot Encoding).
*   **Day 12:** Scaling and transforming numerical features (Standardization, Log Transformation).

## Phase 4: Advanced Feature Engineering & Selection (Days 13-17)
**Goal:** Create highly predictive features and remove noise.
*   **Day 13:** Domain-specific feature creation (e.g., `income_to_loan_ratio`, `debt_to_income`).
*   **Day 14:** Additional aggregation features based on employment and credit history.
*   **Day 15:** Feature Selection techniques (Variance Thresholding, Mutual Information).
*   **Day 16:** Recursive Feature Elimination (RFE) or L1-based selection to isolate the strongest predictors.
*   **Day 17:** Finalize dataset schema and save preprocessed data (Parquet format) for fast training.

## Phase 5: Baseline Modeling & Evaluation Strategy (Days 18-20)
**Goal:** Establish baseline performance metrics to benchmark advanced models.
*   **Day 18:** Train/Validation/Test splitting strategy (ensuring no data leakage).
*   **Day 19:** Defining evaluation metrics specifically for imbalanced datasets (Precision-Recall AUC, F1-Score, ROC-AUC).
*   **Day 20:** Train baseline models (Logistic Regression, Decision Trees) and record metrics.

## Phase 6: Advanced Modeling & Hyperparameter Tuning (Days 21-24)
**Goal:** Build the final predictive models focused on high accuracy and recall.
*   **Day 21:** Implement powerful ensemble models (Random Forest, LightGBM, XGBoost).
*   **Day 22:** Address class imbalance natively (Scale Pos Weight, Focal Loss algorithms).
*   **Day 23:** Advanced hyperparameter tuning for LightGBM/XGBoost using Optuna.
*   **Day 24:** Calculate Expected Loss (EL) using Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD).

## Phase 7: Model Interpretability & Business Logic (Days 25-26)
**Goal:** Ensure the model's decisions can be explained to business stakeholders and regulators.
*   **Day 25:** Global interpretability (Feature Importance scores).
*   **Day 26:** Local explainable AI with SHAP (explaining individual loan approval/rejection decisions).

## Phase 8: Production Pipeline & API Engineering (Days 27-28)
**Goal:** Wrap the trained model in a scalable backend API.
*   **Day 27:** Refactor notebook code into modular, object-oriented Python scripts (`src/`).
*   **Day 28:** Develop a REST API using FastAPI to serve real-time predictions.

## Phase 9: Deployment & Documentation (Days 29-30)
**Goal:** Deploy the application for end-users and document the system.
*   **Day 29:** Build an interactive frontend dashboard using Streamlit to visualize predictions and SHAP values.
*   **Day 30:** Dockerize the entire application (Frontend + Backend) and finalize the `README.md` and PDF Project Report.
