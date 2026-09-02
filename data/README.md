# Dataset documentation

## Source

This repository uses the public [Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) for educational analysis. The source dataset's original terms continue to apply.

## Files

| File | Rows | Columns | Description |
|---|---:|---:|---|
| `raw/loan_approval_train.csv` | 614 | 13 | Original project input, preserved unchanged |
| `processed/loan_prediction_cleaned.csv` | 614 | 14 | Imputed, renamed and feature-engineered analytical data |
| `processed/loan_prediction_ml_ready.csv` | 614 | 13 | Encoded, scaled, numeric demonstration dataset with one redundant feature removed |

## Raw-data fields

| Field | Meaning |
|---|---|
| `Loan_ID` | Application identifier |
| `Gender` | Recorded applicant gender category |
| `Married` | Recorded marital-status category |
| `Dependents` | Number of dependants, with `3+` as the highest category |
| `Education` | Graduate or not-graduate category |
| `Self_Employed` | Recorded self-employment category |
| `ApplicantIncome` | Applicant income in the source dataset's stated units |
| `CoapplicantIncome` | Co-applicant income in the source dataset's stated units |
| `LoanAmount` | Requested loan amount in the source dataset's stated units |
| `Loan_Amount_Term` | Loan term in months |
| `Credit_History` | Source credit-history flag |
| `Property_Area` | Rural, semi-urban or urban category |
| `Loan_Status` | Historical outcome: `Y` or `N` |

## Verified source-data quality

- 614 rows and 13 columns
- 0 exact duplicate rows
- 149 missing values across seven fields
- 422 `Y` outcomes and 192 `N` outcomes
- 614 unique application identifiers

## Processing summary

1. Impute categorical fields using their modes.
2. Impute `LoanAmount` using its median.
3. Impute `Loan_Amount_Term` and `Credit_History` using their modes.
4. Remove `Loan_ID` from the analytical features.
5. Create `total_income` and `loan_income_ratio`.
6. Convert column names to `snake_case`.
7. Encode binary and ordinal categories and one-hot encode `property_area`.
8. Standardise selected numerical variables.
9. Remove `applicant_income` from the demonstration ML-ready export because of its high correlation with engineered `total_income`.

## Important modelling caution

The published ML-ready file records the preprocessing exercise performed on the full dataset. Because its scaler was fitted before a final model-evaluation split, it must not be used to claim unbiased predictive performance. In future modelling work, split the data first and fit imputers, encoders, scalers and feature-selection decisions on the training set only, preferably inside a scikit-learn pipeline.

This educational dataset and its historical outcome must not be used to make real lending decisions.

