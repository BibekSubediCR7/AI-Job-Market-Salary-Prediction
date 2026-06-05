import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

ORDINAL_MAPS = {
    "experience_level": {
        "Entry (0-2 yrs)": 1, "Mid (3-5 yrs)": 2,
        "Senior (6-9 yrs)": 3, "Lead (10+ yrs)": 4
    },
    "education_required": {
        "Bootcamp/Self-taught": 1, "Associate's": 2,
        "Bachelor's": 3, "Master's": 4, "PhD": 5
    },
    "company_size": {
        "Startup (1-50)": 1, "SME (51-500)": 2, "Mid-size (501-5000)": 3,
        "Enterprise (5000+)": 4, "Big Tech (FAANG+)": 5
    }
}

OHE_COLS = ["job_title", "country", "remote_work", "industry"]
SCALE_COLS = ["years_of_experience", "experience_encoded", "education_encoded", "company_size_encoded"]


def apply_ordinal_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Map ordinal string columns to integers."""
    df = df.copy()
    for col, mapping in ORDINAL_MAPS.items():
        df[col.replace("experience_level", "experience_encoded")
             .replace("education_required", "education_encoded")
             .replace("company_size", "company_size_encoded")] = df[col].map(mapping)

    # Cleaner explicit version
    df["experience_encoded"] = df["experience_level"].map(ORDINAL_MAPS["experience_level"])
    df["education_encoded"] = df["education_required"].map(ORDINAL_MAPS["education_required"])
    df["company_size_encoded"] = df["company_size"].map(ORDINAL_MAPS["company_size"])

    df.drop(columns=["experience_level", "education_required", "company_size"], inplace=True)
    return df


def apply_ohe(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode nominal columns, cast bool to int."""
    df = pd.get_dummies(df, columns=OHE_COLS, drop_first=True)
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def apply_log_target(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform the target column."""
    df = df.copy()
    df["log_salary"] = np.log(df["annual_salary_usd"])
    df.drop(columns=["annual_salary_usd"], inplace=True)
    return df


def fit_scaler(df: pd.DataFrame, save_path: str = "models/scaler.pkl") -> StandardScaler:
    """Fit StandardScaler on numeric/ordinal columns and save it."""
    scaler = StandardScaler()
    df[SCALE_COLS] = scaler.fit_transform(df[SCALE_COLS])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {save_path}")
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply a pre-fitted scaler (used at inference time)."""
    df = df.copy()
    df[SCALE_COLS] = scaler.transform(df[SCALE_COLS])
    return df


def run_full_pipeline(df: pd.DataFrame, save_scaler_path: str = "models/scaler.pkl") -> pd.DataFrame:
    """Run all preprocessing steps end to end. Used in training pipeline."""
    df = apply_log_target(df)
    df = apply_ordinal_encoding(df)
    df = apply_ohe(df)
    scaler = fit_scaler(df, save_path=save_scaler_path)
    return df