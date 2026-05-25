# Retail Loan Default Prediction: Progress Summary (Days 1 to 5)

This document summarizes the work completed so far on the 30-day Machine Learning project pipeline.

## Day 1: Project Setup
- Initialized the Git repository.
- Defined the core directory structure (`data/`, `notebooks/`, `src/`, `models/`).
- Set up the Python virtual environment and installed necessary dependencies.

## Day 2: Memory-Efficient Data Loading
- **Challenge:** The dataset contains 2.26 million rows and 145 columns, which easily crashes standard pandas read operations.
- **Solution:** Implemented a robust data loading script (`src/data_loader.py`) using **Pandas Chunking**.
- **Optimization:** Added a `reduce_mem_usage` function to downcast numeric data types (e.g., float64 to float32, int64 to int32/int8) and convert objects to categories on the fly.
- **Result:** Successfully loaded the massive dataset, reducing memory footprint by over 50%.

## Day 3: Tracking & Formatting Setup
- Configured foundational tools for code quality and experiment tracking to ensure the project remains organized as it scales.

## Day 4: High-Level Data Profiling
- **Shape & Structure:** Confirmed the loaded dataset shape is 2,260,668 rows and 145 columns.
- **Data Types:** Profiled the data types across all columns.
- **Missing Value Analysis:** 
  - Calculated the percentage of missing values for all columns.
  - Visualized the top 50 columns with the highest missing data percentages using a bar chart.
  - Generated a missing data heatmap (using a 100k row sample for memory efficiency) to identify patterns in the missingness.

## Day 5: Univariate Analysis
- **Target Variable Distribution:** Visualized the `loan_status` feature. We observed a significant class imbalance, with the vast majority of loans being "Current" or "Fully Paid", and a smaller subset being "Charged Off" (Defaulted).
- **Numerical Features:** Plotted histograms with KDE for key numerical predictors:
  - `loan_amnt` (Loan Amount)
  - `int_rate` (Interest Rate)
  - `installment` (Monthly Payment)
  - `dti` (Debt-to-Income Ratio)
- **Categorical Features:** Generated count plots for high-cardinality categorical variables to understand the distribution of borrowers across different segments:
  - `grade` (Lending Club assigned risk grade)
  - `purpose` (Stated reason for the loan)
  - `home_ownership` (Rent vs. Mortgage vs. Own)
  - `emp_length` (Length of employment)

---
*The team is now ready to proceed to Day 6: Bivariate Analysis.*
