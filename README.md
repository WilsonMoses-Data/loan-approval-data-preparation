# Loan Approval: Advanced Analysis and Feature Engineering

> Statistical analysis, feature engineering, responsible feature selection, and leakage-safe machine-learning preparation for historical loan-approval data.

**Programme:** AnalystLab Africa Data Science Internship — Week 3  
**Project phase:** Completed  
**Author:** [Wilson Moses](https://github.com/WilsonMoses-Data)

## Project overview

This project investigates 614 historical loan applications to identify the characteristics associated with loan approval and prepare a documented dataset for later predictive modelling.

Building on the cleaning completed during Week 2, the work moves beyond descriptive summaries into statistical inference, multivariate exploration, engineered financial indicators, responsible feature decisions, and a preprocessing workflow designed to limit data leakage.

> **Business question:** Which applicant, household, financial, and credit-related characteristics are associated with historical loan decisions, and how can those characteristics be prepared responsibly for predictive modelling?

## Objectives

1. Validate the quality and consistency of the cleaned dataset.
2. Examine distributions, outliers, approval patterns, and relationships.
3. Test whether observed relationships are statistically supported.
4. Create interpretable household, income, credit, and repayment features.
5. Select useful modelling features while controlling redundancy and fairness risks.
6. Prepare training and testing data without fitting transformations on held-out observations.
7. Translate technical evidence into business recommendations.

## Dataset summary

| Measure | Result |
|---|---:|
| Applications analysed | 614 |
| Approved applications | 422 |
| Rejected applications | 192 |
| Overall approval rate | 68.73% |
| Engineered features created | 12 |
| Final analytical columns | 26 |
| Selected source modelling features | 12 |
| Encoded modelling features | 13 |
| Training observations | 491 |
| Held-out testing observations | 123 |

The target is `loan_status`, represented as `Y`/`N` in the source and `1`/`0` for numerical analysis.

## Tools and technologies

- Python and Jupyter Notebook
- pandas and NumPy
- Matplotlib and Seaborn
- SciPy
- scikit-learn: `mutual_info_classif`, `train_test_split`, `ColumnTransformer`, `OneHotEncoder`, and `StandardScaler`

## Analytical workflow

### 1. Advanced data-quality assessment

The notebook checks dimensions, data types, missing values, duplicates, categorical consistency, invalid numerical values, engineered-feature calculations, IQR outliers, and target balance.

### 2. Exploratory analysis

The analysis includes:

- financial distributions and log transformations;
- categorical frequencies and approval rates;
- cross-tabulations and row percentages;
- numerical comparisons by outcome;
- Pearson and Spearman correlations; and
- combined credit-history and property-area analysis.

### 3. Statistical analysis

| Question | Method | Result | Interpretation |
|---|---|---|---|
| Credit history vs. approval | Chi-square | `p < 0.001`; Cramér’s V = `0.536` | Strong association |
| Property area vs. approval | Chi-square | `p = 0.002`; Cramér’s V = `0.142` | Significant but comparatively weak association |
| Total income by outcome | Mann–Whitney U | `p = 0.713` | No significant independent difference |
| Loan amount by outcome | Mann–Whitney U | `p = 0.398` | No significant independent difference |
| Total income across property areas | Kruskal–Wallis | `p = 0.140` | No significant difference |
| Total income vs. loan amount | Spearman correlation | `rho = 0.688`; `p < 0.001` | Strong positive relationship |

Nonparametric methods were used for right-skewed financial variables. Statistical significance was assessed at `alpha = 0.05`.

### 4. Feature engineering

Twelve features were created:

| Feature | Purpose |
|---|---|
| `dependents_numeric` | Numerical representation of dependents |
| `family_size` | Estimated household size |
| `family_size_group` | Interpretable household-size band |
| `income_band` | Descriptive household-income range |
| `has_coapplicant_income` | Indicates a coapplicant income contribution |
| `coapplicant_income_share` | Share of household income from the coapplicant |
| `term_years` | Repayment term in years |
| `estimated_monthly_principal` | Approximate monthly principal payment |
| `payment_income_ratio` | Approximate repayment burden relative to income |
| `credit_risk_category` | Business-readable credit-history category |
| `log_total_income` | Reduced-skew household income |
| `log_loan_amount` | Reduced-skew requested amount |

The affordability measures are proxies. Interest, existing debt, expenses, insurance, and verified monthly obligations are unavailable.

### 5. Feature selection

Final source features were selected using statistical evidence, mutual information, interpretability, redundancy, and responsible-use considerations.

```python
final_selected_features = [
    "married",
    "dependents_numeric",
    "education",
    "self_employed",
    "property_area",
    "credit_history",
    "log_total_income",
    "log_loan_amount",
    "term_years",
    "has_coapplicant_income",
    "coapplicant_income_share",
    "payment_income_ratio",
]
```

`gender` was excluded from predictive inputs because it can function as a protected characteristic. It remains relevant for fairness monitoring and subgroup evaluation.

### 6. Leakage-safe ML preparation

1. Separate predictors and target.
2. Create a stratified 80/20 split using `random_state=42`.
3. Fit categorical encoding and continuous scaling on training data only.
4. Apply the fitted transformations to held-out data.
5. Export analytical and model-ready datasets.

## Key findings

- Credit history is the dominant observed factor associated with approval.
- Applicants with positive credit history had a 79.05% approval rate, compared with 7.87% for those without positive credit history.
- Property area has a statistically significant but much weaker relationship with approval.
- Total income and loan amount are strongly related to each other, but neither shows a significant independent difference between approval outcomes in the selected tests.
- The engineered repayment-burden indicator improves interpretability but cannot substitute for a complete affordability assessment.
- Statistical association does not establish causation or justify automated lending decisions.

## Repository contents

```text
Loan-Prediction-Advanced-Data-Exploration-Statistical-Analysis-Feature-Engineering/
├── README.md
├── LICENSE
├── Advance_Analysis_Loan_Predicition.ipynb
├── Business_Insights_Report.pdf
├── Statistical_Analysis_Report.pdf
├── Feature_Engineering_Documentation.pdf
├── loan_prediction_week3_final_cleaned.csv
├── loan_prediction_week3_ml_ready.csv
├── loan_prediction_week3_train.csv
└── week3_updated_data_dictionary.csv
```

### Key deliverables

- [`Advance_Analysis_Loan_Predicition.ipynb`](Advance_Analysis_Loan_Predicition.ipynb) — complete executed analysis.
- [`Business_Insights_Report.pdf`](Business_Insights_Report.pdf) — decision-oriented findings and recommendations.
- [`Statistical_Analysis_Report.pdf`](Statistical_Analysis_Report.pdf) — statistical methods and results.
- [`Feature_Engineering_Documentation.pdf`](Feature_Engineering_Documentation.pdf) — definitions and rationale for engineered features.
- [`week3_updated_data_dictionary.csv`](week3_updated_data_dictionary.csv) — updated field documentation.

## Reproducing the analysis

1. Clone the repository.
2. Create a Python environment.
3. Install Jupyter, pandas, NumPy, Matplotlib, Seaborn, SciPy, and scikit-learn.
4. Open `Advance_Analysis_Loan_Predicition.ipynb`.
5. Confirm the data path used in the loading cell.
6. Run the notebook from top to bottom.

A pinned dependency file should be added to make future reproduction more reliable.

## Limitations

- The dataset is small and represents historical decisions, not objective creditworthiness.
- Historical approval patterns may contain policy or social bias.
- Important affordability factors are unavailable.
- The work prepares data for modelling but does not present a validated production model.
- Findings describe association and should not be interpreted as causal effects.

## Next steps

- Correct the notebook filename from `Predicition` to `Prediction`.
- Add a dependency file and reproducibility instructions tied to a Python version.
- Train and compare interpretable classification baselines.
- Evaluate calibration, decision thresholds, subgroup performance, and error costs.
- Consolidate this phase with the Week 2 preparation work into one end-to-end case study when programme reporting permits.

## Licence and data source

Original code and documentation are released under the repository’s [MIT Licence](LICENSE). The loan dataset remains subject to the terms of its [original Kaggle source](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset).

## Contact

**Wilson Moses** — Data Scientist × AI Engineer in development  
[LinkedIn](https://www.linkedin.com/in/wilson-moses-9207b22bb) · [GitHub](https://github.com/WilsonMoses-Data) · [Moses Learns Data](https://www.tiktok.com/@moses.learnsdata)
