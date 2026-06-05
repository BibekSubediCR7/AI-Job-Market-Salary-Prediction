import pandas as pd
import os

EXPECTED_COLUMNS = [
    "job_title", "experience_level", "years_of_experience",
    "education_required", "annual_salary_usd", "country",
    "remote_work", "company_size", "industry"
]

COLS_TO_DROP = [
    "job_category", "city", "required_skills",
    "ai_salary_premium_pct", "demand_score", "demand_growth_yoy_pct",
    "benefits_score_10", "posting_year", "posting_month",
    "is_senior", "is_remote_friendly", "is_llm_role"
]

VALID_VALUES = {
    "experience_level": ["Entry (0-2 yrs)", "Mid (3-5 yrs)", "Senior (6-9 yrs)", "Lead (10+ yrs)"],
    "education_required": ["Bootcamp/Self-taught", "Associate's", "Bachelor's", "Master's", "PhD"],
    "company_size": ["Startup (1-50)", "SME (51-500)", "Mid-size (501-5000)", "Enterprise (5000+)", "Big Tech (FAANG+)"],
    "remote_work": ["Fully Remote", "Hybrid", "On-site"],
    "country": ["Australia", "Canada", "China", "France", "Germany", "Global",
                 "India", "Japan", "Netherlands", "Singapore", "Switzerland", "UAE", "UK", "USA"],
    "industry": ["Automotive", "Consulting", "Education", "Energy", "Finance",
                 "Government", "Healthcare", "Manufacturing", "Media", "Research", "Retail", "Technology"],
    "job_title": [
        "AI Agent Developer", "AI Business Analyst", "AI Compliance Manager",
        "AI Engineer", "AI Ethics Officer", "AI Infrastructure Eng",
        "AI Product Manager", "AI Research Scientist", "AI Security Engineer",
        "AI Solutions Architect", "Computer Vision Engineer", "Data Engineer (AI)",
        "Data Scientist", "Deep Learning Engineer", "Generative AI Engineer",
        "LLM Engineer", "ML Engineer", "MLOps Engineer", "Multimodal AI Engineer",
        "NLP Engineer", "Prompt Engineer", "RAG Engineer", "Robotics Engineer (AI)",
        "Senior Data Scientist", "Senior ML Engineer"
    ]
}


def load_raw(path: str) -> pd.DataFrame:
    """Load raw CSV, drop irrelevant columns, validate schema."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    # Drop columns that exist in the raw file but are not needed
    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    df.drop(columns=cols_present, inplace=True)

    # Check all expected columns are present
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    return df[EXPECTED_COLUMNS]


def validate(df: pd.DataFrame) -> None:
    """Check nulls and categorical values are within known set."""
    # Null check
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        raise ValueError(f"Null values found:\n{nulls}")

    # Categorical value check
    for col, valid in VALID_VALUES.items():
        bad = df[~df[col].isin(valid)][col].unique()
        if len(bad) > 0:
            raise ValueError(f"Unexpected values in '{col}': {bad}")

    print("Validation passed. Shape:", df.shape)