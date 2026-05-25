# Day 7: Correlation Analysis Summary

## Overview
Today, we focused on identifying multi-collinearity within the numerical features of our 2.26 million row loan dataset. High correlation between predictor variables can inflate standard errors in linear models and make feature importance difficult to interpret in ensemble models. We generated a Pearson correlation matrix and visualized it via a heatmap to isolate these redundant feature clusters.

## Key Insights

We successfully computed correlations across the numerical subset of our 145 features. Below are the most critical sets of highly correlated (redundant) features where the absolute correlation coefficient exceeds 0.8:

### 1. Loan Origination Amounts
*   `loan_amnt`, `funded_amnt`, and `funded_amnt_inv` share a correlation near **1.0**. These features are virtually identical, representing the amount applied for, funded by the lender, and funded by investors, respectively. 

### 2. Payment and Principal Metrics
*   `total_pymnt` and `total_pymnt_inv` (Correlation: **1.0**)
*   `total_pymnt` and `total_rec_prncp` (Correlation: **0.99**)
*   `out_prncp` and `out_prncp_inv` (Correlation: **1.0**)
These features essentially measure the exact same aspect of the loan's current payoff status.

### 3. Installment and Loan Amount
*   `loan_amnt` and `installment` (Correlation: **0.95**). Naturally, higher loan amounts dictate higher monthly installments.

### 4. Credit Line Accounts
*   `open_acc` (Number of open credit lines) and `num_sats` (Number of satisfactory accounts) share a correlation of **0.999**.
*   `tot_cur_bal` and `tot_hi_cred_lim` (Correlation: **0.97**). The total current balance closely tracks the total high credit limit.

## Next Steps
In **Phase 4 (Days 13-17: Advanced Feature Engineering & Selection)**, we will systematically drop the redundant counterparts identified today. For example, we will likely keep `loan_amnt` and drop `funded_amnt` and `funded_amnt_inv`. By pruning these highly correlated features, we will reduce the dataset's dimensionality, decrease training time, and prevent our models from overfitting to duplicated signals.

Tomorrow, we proceed to **Day 8: Outlier detection and deep dive into the 12x default rate gap across specific groups.**
