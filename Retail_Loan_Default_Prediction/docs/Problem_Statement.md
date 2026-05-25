# Retail Loan Default Prediction: Problem Statement

## The Scenario
Imagine you run a bank or a lending platform. Every day, thousands of people come to you asking for loans to buy cars, consolidate debt, or start a business. Your goal is to lend them the money and earn a profit through interest.

However, lending money carries a major risk. 

## The Core Problem
When a customer asks for a loan, you must answer one critical question:
**"Will this person successfully pay the money back, or will they default (fail to pay)?"**

- If you **give** a loan to someone who defaults, your bank loses a significant amount of money.
- If you **reject** a loan for a trustworthy person, you lose out on potential profit and frustrate a good customer.

In our dataset, the default rate shows a massive gap between different grades of loans (some are 12x more likely to default!). Manually reviewing every application to catch these risky borrowers is slow, biased, and prone to human error.

## The Solution We Are Building
We are building a **Smart Machine Learning System** to predict whether an applicant will default on their loan *before* the money is handed out. 

Our AI assistant will analyze the applicant's financial history, including:
1. **Annual Income & Debt:** How much do they earn versus how much they already owe?
2. **Loan Details:** How much are they asking for, and at what interest rate?
3. **Employment History:** How long have they held their current job?
4. **Credit History:** Do they have a history of missing payments or bankruptcy?

## Why It Matters
By accurately predicting defaults, our model will:
1. **Save Millions:** Minimize financial losses from bad loans.
2. **Improve Speed:** Automate the approval process so good customers get their money faster.
3. **Enhance Fairness:** Remove human bias from the lending decision, relying purely on data.
4. **Explain Decisions:** Using tools like SHAP, our model will explain *why* someone was rejected (e.g., "Debt-to-Income ratio is too high"), ensuring transparency for regulators and customers.
