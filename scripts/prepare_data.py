"""Reproduce the cleaned and numeric loan-approval data exports."""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "loan_approval_train.csv"
PROCESSED_DIR = ROOT / "data" / "processed"


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Impute missing data and create the documented analytical features."""
    data = raw.copy()

    for column in ["Gender", "Married", "Dependents", "Self_Employed"]:
        data[column] = data[column].fillna(data[column].mode()[0])

    data["LoanAmount"] = data["LoanAmount"].fillna(data["LoanAmount"].median())
    data["Loan_Amount_Term"] = data["Loan_Amount_Term"].fillna(
        data["Loan_Amount_Term"].mode()[0]
    )
    data["Credit_History"] = data["Credit_History"].fillna(
        data["Credit_History"].mode()[0]
    )

    data = data.drop(columns=["Loan_ID"])
    data["TotalIncome"] = data["ApplicantIncome"] + data["CoapplicantIncome"]
    if data["TotalIncome"].eq(0).any():
        raise ValueError("TotalIncome contains zero; LoanIncomeRatio would be undefined.")
    data["LoanIncomeRatio"] = data["LoanAmount"] / data["TotalIncome"]

    return data.rename(
        columns={
            "Gender": "gender",
            "Married": "married",
            "Dependents": "dependents",
            "Education": "education",
            "Self_Employed": "self_employed",
            "ApplicantIncome": "applicant_income",
            "CoapplicantIncome": "coapplicant_income",
            "LoanAmount": "loan_amount",
            "Loan_Amount_Term": "loan_amount_term",
            "Credit_History": "credit_history",
            "Property_Area": "property_area",
            "Loan_Status": "loan_status",
            "TotalIncome": "total_income",
            "LoanIncomeRatio": "loan_income_ratio",
        }
    )


def create_numeric_export(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Encode, scale and reduce the cleaned full-sample demonstration data."""
    data = cleaned.copy()
    binary_maps = {
        "gender": {"Male": 1, "Female": 0},
        "married": {"Yes": 1, "No": 0},
        "education": {"Graduate": 1, "Not Graduate": 0},
        "self_employed": {"Yes": 1, "No": 0},
        "loan_status": {"Y": 1, "N": 0},
    }
    for column, mapping in binary_maps.items():
        data[column] = data[column].map(mapping)

    data["credit_history"] = data["credit_history"].astype(int)
    data["dependents"] = data["dependents"].replace({"3+": "3"}).astype(int)
    data = pd.get_dummies(data, columns=["property_area"], drop_first=True, dtype=int)

    numerical_features = [
        "applicant_income",
        "coapplicant_income",
        "loan_amount",
        "loan_amount_term",
        "total_income",
        "loan_income_ratio",
    ]
    data[numerical_features] = StandardScaler().fit_transform(data[numerical_features])
    return data.drop(columns=["applicant_income"])


def validate(cleaned: pd.DataFrame, numeric: pd.DataFrame) -> None:
    """Fail fast when the published outputs violate their documented contract."""
    if cleaned.shape != (614, 14):
        raise ValueError(f"Unexpected cleaned shape: {cleaned.shape}")
    if numeric.shape != (614, 14):
        raise ValueError(f"Unexpected numeric shape: {numeric.shape}")
    if cleaned.isna().any().any() or numeric.isna().any().any():
        raise ValueError("Published outputs contain missing values.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in numeric.dtypes):
        raise TypeError("The ML-ready output is not fully numeric.")


def main() -> None:
    """Create and verify the two processed datasets."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_PATH)
    cleaned = clean_data(raw)
    numeric = create_numeric_export(cleaned)
    validate(cleaned, numeric)
    cleaned.to_csv(PROCESSED_DIR / "loan_prediction_cleaned.csv", index=False)
    numeric.to_csv(PROCESSED_DIR / "loan_prediction_ml_ready.csv", index=False)
    print("Created verified cleaned and numeric datasets in data/processed/.")


if __name__ == "__main__":
    main()

