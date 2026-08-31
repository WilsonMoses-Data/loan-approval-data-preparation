# Loan Approval: Data Preparation for Machine Learning

> Cleaning, feature engineering, encoding, scaling, outlier assessment, and feature selection for a 614-record loan-approval dataset.

**Programme:** AnalystLab Africa Data Science Internship — Week 2  
**Project phase:** Completed  
**Author:** [Wilson Moses](https://github.com/WilsonMoses-Data)

## Project overview

This project transforms raw loan-application data into clean analytical and machine-learning-ready datasets. The work documents each preprocessing choice so later modelling can begin from a consistent, numeric, and interpretable foundation.

> **Goal:** Prepare the data responsibly for a future model that estimates historical loan-approval outcomes; this repository does not claim to deliver or deploy a production lending model.

## Dataset

- **Source:** [Loan Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- **Records:** 614 applications
- **Original columns:** 13
- **Target:** `Loan_Status` (`Y` = approved, `N` = not approved)

## Business and technical questions

1. Which fields contain missing values?
2. Which variables require encoding or scaling?
3. Which features are redundant or strongly correlated?
4. Do apparent outliers represent errors or plausible applicants?
5. Which features provide the strongest early evidence for later modelling?
6. Is the final dataset complete, numeric, and ready for the next analytical phase?

## Workflow

### 1. Inspection

- Reviewed shape, data types, summary statistics, and duplicate records.
- Identified missing values across `Gender`, `Married`, `Dependents`, `Self_Employed`, `LoanAmount`, `Loan_Amount_Term`, and `Credit_History`.
- Confirmed that no exact duplicate rows were present.

### 2. Missing-value treatment

| Variable type or field | Treatment | Rationale |
|---|---|---|
| Gender, married, dependents, self-employed | Mode | Low missingness and no established unknown category |
| Loan amount | Median | More resistant to extreme values than the mean |
| Loan amount term | Mode | Discrete repeated term values, with 360 months dominant |
| Credit history | Mode | Binary categorical flag |

The cleaned dataset was verified to contain no missing values or duplicates.

### 3. Feature engineering

- Removed `Loan_ID` because it is a unique identifier rather than an explanatory feature.
- Created `total_income = applicant_income + coapplicant_income`.
- Created `loan_income_ratio = loan_amount / total_income` as a simple affordability indicator.
- Renamed columns to `snake_case`.

### 4. Encoding

| Variable | Treatment |
|---|---|
| `gender`, `married`, `education`, `self_employed` | Binary encoding |
| `credit_history` | Binary type correction |
| `dependents` | Custom numerical encoding, with `3+` represented as `3` |
| `property_area` | One-hot encoding with a dropped reference category |
| `loan_status` | `Y → 1`, `N → 0` |

### 5. Scaling

`StandardScaler` was applied to continuous fields whose different units and ranges can affect scale-sensitive models:

- `applicant_income`
- `coapplicant_income`
- `loan_amount`
- `loan_amount_term`
- `total_income`
- `loan_income_ratio`

### 6. Outlier assessment

Boxplots and the IQR rule identified potential outliers. They were retained because the extreme income and loan values appeared plausible rather than obvious data-entry errors.

| Feature | IQR outliers detected |
|---|---:|
| `applicant_income` | 50 |
| `total_income` | 50 |
| `loan_amount` | 41 |
| `loan_income_ratio` | 25 |
| `coapplicant_income` | 18 |

### 7. Feature selection

- `applicant_income` and engineered `total_income` had a correlation of approximately `0.89`.
- `applicant_income` was removed from the published ML-ready dataset to reduce redundancy.
- `credit_history` showed the strongest relationship with the target in the initial evidence.
- Random Forest feature importance was used as an exploratory screening tool, not as final causal or policy evidence.

## Outputs

| File | Rows | Columns | Purpose |
|---|---:|---:|---|
| `loan_prediction_cleaned.csv` | 614 | 14 | Cleaned and feature-engineered analytical data |
| `loan_prediction_ml_ready.csv` | 614 | 13 | Numeric, encoded, scaled, reduced-redundancy data |

Both published outputs contain zero missing values and zero exact duplicates.

## Repository contents

```text
loan-prediction-feature-engineering/
├── README.md
├── LICENSE
├── Load_Prediction_Data_Inspection.ipynb
├── Prediction_Loan_Train.csv
├── loan_prediction_cleaned.csv
├── loan_prediction_ml_ready.csv
├── Business Understanding Report - Loan Prediction.pdf
└── Data Preprocessing Report.pdf
```

### Key files

- [`Load_Prediction_Data_Inspection.ipynb`](Load_Prediction_Data_Inspection.ipynb) — executed preprocessing and feature-selection notebook.
- [`Business Understanding Report - Loan Prediction.pdf`](Business%20Understanding%20Report%20-%20Loan%20Prediction.pdf) — business framing and analytical objectives.
- [`Data Preprocessing Report.pdf`](Data%20Preprocessing%20Report.pdf) — documented preparation decisions and outputs.

## Tools used

- Python and Jupyter Notebook
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn: `StandardScaler`, `train_test_split`, and `RandomForestClassifier`

## Reproducing the analysis

1. Clone the repository.
2. Create a Python environment.
3. Install Jupyter, pandas, NumPy, Matplotlib, Seaborn, and scikit-learn.
4. Open `Load_Prediction_Data_Inspection.ipynb`.
5. Confirm the raw CSV path used in the loading cell.
6. Run the notebook from top to bottom.

## Limitations

- Mode imputation can reinforce the most common category and reduce variation.
- Scaling was performed within this data-preparation phase; future predictive work should fit transformations on training data only.
- Random Forest importance can be unstable and does not establish causation.
- Historical approval decisions may reflect policy or social bias.
- The affordability ratio omits interest, existing debt, expenses, and other obligations.

## Next steps

- Rename the notebook to `Loan_Prediction_Data_Preparation.ipynb`.
- Add a pinned dependency file.
- Move raw and processed datasets into separate folders.
- Continue with statistical analysis and leakage-safe train/test preprocessing.
- Consolidate this work with the Week 3 repository when programme reporting permits.

## Licence and data source

Original code and documentation are released under the repository’s [MIT Licence](LICENSE). The dataset remains subject to the terms of its [original Kaggle source](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset).

## Contact

**Wilson Moses** — Data Scientist × AI Engineer in development  
[LinkedIn](https://www.linkedin.com/in/wilson-moses-9207b22bb) · [GitHub](https://github.com/WilsonMoses-Data) · [Moses Learns Data](https://www.tiktok.com/@moses.learnsdata)
