# Day 6: Bivariate Analysis Summary

## Overview
Today, we conducted a Bivariate Analysis on the 2.26 million row dataset to uncover the relationships between various predictors and the `loan_status` (target variable). 

To ensure memory efficiency and clear visualizations, we engineered a binary `Default` variable, where `1` indicates a bad loan status (e.g., Charged Off, Default, Late) and `0` indicates a good loan status (Fully Paid, Current).

## Key Insights

### 1. Categorical Features vs. Default Rate
*   **Loan Grade:** There is a clear linear progression of risk. Borrowers assigned an 'A' grade have the lowest default rates, whereas 'F' and 'G' grades exhibit exponentially higher default rates, confirming that the initial risk assessment by the lender is highly predictive.
*   **Loan Term:** Loans with a 60-month term have significantly higher default rates compared to 36-month loans. The longer exposure period combined with generally riskier profiles for longer terms drives this trend.
*   **Home Ownership:** Borrowers who 'Rent' have slightly higher default probabilities compared to those with a 'Mortgage' or who 'Own' their homes outright, likely serving as a proxy for financial stability and wealth.
*   **Loan Purpose:** Loans taken out for 'Small Business' ventures show the highest risk of default, whereas traditional reasons like 'Credit Card' refinancing or 'Car' loans perform relatively better.

### 2. Numerical Features vs. Default Status
*   **Interest Rate (`int_rate`):** Expectedly, the interest rate is a strong discriminator. Defaulted loans have a notably higher median interest rate than non-defaulted loans, reflecting the lender's risk premium.
*   **Debt-to-Income (`dti`):** Borrowers who default tend to have higher Debt-to-Income ratios. A higher DTI suggests that the borrower is more heavily burdened by existing debt and more vulnerable to financial shocks.
*   **Loan Amount (`loan_amnt`):** The distributions for loan amounts are fairly similar between the two groups, but defaulted loans do lean slightly toward higher amounts. 
*   **Annual Income (`annual_inc`):** Lower-income borrowers are slightly more prevalent in the default group, although the difference isn't as stark as with `int_rate` or `dti`.

## Next Steps
Having established these strong bivariate relationships, we are ready to move on to **Day 7: Correlation Analysis**. We will look into multi-collinearity to ensure we don't feed highly correlated redundant features (like `loan_amnt` and `installment`) into our predictive models.
