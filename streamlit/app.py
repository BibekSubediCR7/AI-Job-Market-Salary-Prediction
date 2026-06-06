import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import os
import base64
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Job Salary Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "linear_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
DATA_PATH   = os.path.join(BASE_DIR, "data", "raw", "ai_jobs_market.csv")
PHOTO_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photo.jpg")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Header fix ── */
header[data-testid="stHeader"] {
    background-color: #0d0d0d !important;
    box-shadow: none !important;
    border-bottom: 1px solid #191919 !important;
}
header[data-testid="stHeader"]::before { background-image: none !important; }

/* ── Sidebar collapsed toggle ── */
[data-testid="stSidebarCollapsedControl"] {
    background-color: #1a1a1a !important;
    border: 1px solid #e74c3c !important;
    border-left: none !important;
    border-radius: 0 8px 8px 0 !important;
    z-index: 999 !important;
    padding: 0.4rem 0.3rem !important;
}
[data-testid="stSidebarCollapsedControl"] button {
    color: #e74c3c !important;
    background: transparent !important;
}
[data-testid="stSidebarCollapsedControl"] svg {
    fill: #e74c3c !important;
    stroke: #e74c3c !important;
    width: 1.1rem !important;
    height: 1.1rem !important;
}
/* ── Reduce gap before tabs ── */
.stTabs { margin-top: 0.25rem !important; }

/* ── Developer card ── */
.dev-card {
    background: #111111;
    border: 1px solid #1c1c1c;
    border-radius: 12px;
    padding: 1.2rem 1rem 1rem;
    text-align: center;
    position: sticky;
    top: 1rem;
}

/* ── Compact info card ── */
.info-card { padding: 1rem 1.25rem; margin-bottom: 0.5rem; }
.info-card p { font-size: 0.8rem; margin: 0; line-height: 1.7; }
/* ── Base ── */
.stApp { background-color: #0d0d0d; }
[data-testid="stSidebar"] {
    background-color: #0f0f0f;
    border-right: 1px solid #191919;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #111111;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1c1c1c;
}
.stTabs [data-baseweb="tab"] {
    color: #666 !important;
    background: transparent;
    border-radius: 7px;
    padding: 0.45rem 1.4rem;
    font-weight: 500;
    font-size: 0.87rem;
    letter-spacing: 0.02em;
}
.stTabs [aria-selected="true"] { background: #e74c3c !important; color: #fff !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }

/* ── Button ── */
.stButton > button {
    background: #e74c3c !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    padding: 0.65rem 2rem !important; font-weight: 600 !important;
    font-size: 0.88rem !important; letter-spacing: 0.05em !important;
    width: 100% !important;
}
.stButton > button:hover { background: #c0392b !important; }

/* ── Labels ── */
.stSelectbox label, .stSlider label {
    color: #888 !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Select input ── */
div[data-baseweb="select"] > div {
    background: #141414 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    color: #d0d0d0 !important;
}

/* ── Radio ── */
.stRadio [data-testid="stMarkdownContainer"] p { font-size: 0.83rem; color: #999; }
[data-testid="stRadio"] label span { color: #999 !important; }

/* ── Section header ── */
.sec-hdr {
    font-size: 0.65rem; font-weight: 600; color: #666;
    text-transform: uppercase; letter-spacing: 0.16em;
    margin: 1.4rem 0 0.8rem; padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e1e1e;
}

/* ── Info card (main page description) ── */
.info-card {
    background: #111111; border: 1px solid #1c1c1c;
    border-radius: 10px; padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
}
.info-card p { font-size: 0.83rem; color: #888; line-height: 1.75; margin: 0; }
.info-pill {
    display: inline-block; background: #161616; border: 1px solid #222;
    border-radius: 20px; padding: 0.2rem 0.85rem;
    font-size: 0.68rem; color: #e74c3c; margin: 0.25rem 0.4rem 0.25rem 0;
    letter-spacing: 0.04em; white-space: nowrap;
}

/* ── Metric cards ── */
.metric-card {
    background: #111111; border: 1px solid #1c1c1c;
    border-radius: 10px; padding: 1.2rem 1rem; text-align: center;
}
.metric-val { font-size: 1.7rem; font-weight: 700; color: #e74c3c; letter-spacing: -0.02em; }
.metric-lbl { font-size: 0.65rem; color: #666; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.3rem; }

/* ── Salary display ── */
.salary-card {
    background: #111111; border: 1px solid #1c1c1c;
    border-radius: 12px; padding: 2rem 1.5rem 1.5rem; text-align: center;
}
.salary-lbl { font-size: 0.65rem; color: #555; text-transform: uppercase; letter-spacing: 0.16em; margin-bottom: 0.5rem; }
.salary-amt { font-size: 2.9rem; font-weight: 800; color: #ececec; letter-spacing: -0.03em; line-height: 1; }
.salary-range { font-size: 0.78rem; color: #666; margin-top: 0.5rem; }
.pct-badge {
    display: inline-block; background: #161616; border: 1px solid #242424;
    border-radius: 20px; padding: 0.2rem 0.9rem;
    font-size: 0.72rem; color: #e74c3c; margin-top: 0.75rem;
}

/* ── Contribution bars ── */
.contrib-group-lbl { font-size: 0.62rem; color: #555; text-transform: uppercase; letter-spacing: 0.14em; margin: 0.9rem 0 0.4rem; }
.contrib-item { display: flex; align-items: center; gap: 0.65rem; margin: 0.4rem 0; }
.contrib-lbl { font-size: 0.74rem; color: #999; min-width: 160px; max-width: 160px; text-align: right; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.contrib-track { flex: 1; height: 5px; background: #1a1a1a; border-radius: 3px; overflow: hidden; }
.contrib-fill-pos { height: 100%; background: #27ae60; border-radius: 3px; }
.contrib-fill-neg { height: 100%; background: #e74c3c; border-radius: 3px; }
.contrib-pct { font-size: 0.68rem; min-width: 40px; }

/* ── Placeholder ── */
.placeholder {
    background: #0f0f0f; border: 1px dashed #1c1c1c; border-radius: 12px;
    padding: 4rem 2rem; text-align: center; color: #444; font-size: 0.85rem; line-height: 1.9;
}

/* ── Data table ── */
.data-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.data-table th {
    background: #111111; color: #666; font-size: 0.65rem;
    text-transform: uppercase; letter-spacing: 0.1em;
    padding: 0.6rem 1rem; text-align: left; border-bottom: 1px solid #1e1e1e;
}
.data-table td { padding: 0.52rem 1rem; color: #bbb; border-bottom: 1px solid #111111; }
.data-table tr:last-child td { border-bottom: none; }

/* ── Developer card ── */
.dev-name { font-size: 0.98rem; font-weight: 700; color: #ddd; text-align: center; margin-top: 0.8rem; }
.dev-role { font-size: 0.7rem; color: #666; text-align: center; margin-top: 0.25rem; line-height: 1.6; }
.sidebar-label { font-size: 0.6rem; color: #333; text-transform: uppercase; letter-spacing: 0.16em; text-align: center; margin-bottom: 1rem; }
.social-wrap { text-align: center; margin-top: 0.8rem; display: flex; gap: 0.4rem; flex-wrap: wrap; justify-content: center; }
.social-link {
    padding: 0.22rem 0.65rem; background: #141414; border: 1px solid #222;
    border-radius: 20px; color: #777 !important; text-decoration: none !important;
    font-size: 0.68rem; font-weight: 500;
}
.social-link:hover { border-color: #e74c3c; color: #e74c3c !important; }

/* ── App heading ── */
.app-title { font-size: 1.55rem; font-weight: 800; color: #e8e8e8; letter-spacing: -0.03em; margin-bottom: 0.15rem; }
.app-sub { font-size: 0.8rem; color: #777; margin-bottom: 1rem; }

/* ── Disclaimer ── */
.disclaimer { font-size: 0.65rem; color: #555; margin-top: 1.2rem; line-height: 1.7; }

/* ── Footer ── */
.footer {
    text-align: center; padding: 2.5rem 0 1.5rem; color: #444;
    font-size: 0.72rem; border-top: 1px solid #131313; margin-top: 3rem; line-height: 2.2;
}
.footer a { color: #e74c3c !important; text-decoration: none; }

/* ── Streamlit overrides ── */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
.block-container { padding-top: 2.5rem; padding-bottom: 0; max-width: 1180px; }
</style>
""", unsafe_allow_html=True)
# ── Constants ──────────────────────────────────────────────────────────────────
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

SCALE_COLS = [
    "years_of_experience", "experience_encoded",
    "education_encoded", "company_size_encoded"
]

FEATURE_COLUMNS = [
    "years_of_experience", "experience_encoded", "education_encoded", "company_size_encoded",
    "job_title_AI Business Analyst", "job_title_AI Compliance Manager", "job_title_AI Engineer",
    "job_title_AI Ethics Officer", "job_title_AI Infrastructure Eng", "job_title_AI Product Manager",
    "job_title_AI Research Scientist", "job_title_AI Security Engineer", "job_title_AI Solutions Architect",
    "job_title_Computer Vision Engineer", "job_title_Data Engineer (AI)", "job_title_Data Scientist",
    "job_title_Deep Learning Engineer", "job_title_Generative AI Engineer", "job_title_LLM Engineer",
    "job_title_ML Engineer", "job_title_MLOps Engineer", "job_title_Multimodal AI Engineer",
    "job_title_NLP Engineer", "job_title_Prompt Engineer", "job_title_RAG Engineer",
    "job_title_Robotics Engineer (AI)", "job_title_Senior Data Scientist", "job_title_Senior ML Engineer",
    "country_Canada", "country_China", "country_France", "country_Germany", "country_Global",
    "country_India", "country_Japan", "country_Netherlands", "country_Singapore", "country_Switzerland",
    "country_UAE", "country_UK", "country_USA",
    "remote_work_Hybrid", "remote_work_On-site",
    "industry_Consulting", "industry_Education", "industry_Energy", "industry_Finance",
    "industry_Government", "industry_Healthcare", "industry_Manufacturing", "industry_Media",
    "industry_Research", "industry_Retail", "industry_Technology"
]

JOB_TITLES = [
    "AI Agent Developer", "AI Business Analyst", "AI Compliance Manager", "AI Engineer",
    "AI Ethics Officer", "AI Infrastructure Eng", "AI Product Manager", "AI Research Scientist",
    "AI Security Engineer", "AI Solutions Architect", "Computer Vision Engineer",
    "Data Engineer (AI)", "Data Scientist", "Deep Learning Engineer", "Generative AI Engineer",
    "LLM Engineer", "ML Engineer", "MLOps Engineer", "Multimodal AI Engineer",
    "NLP Engineer", "Prompt Engineer", "RAG Engineer", "Robotics Engineer (AI)",
    "Senior Data Scientist", "Senior ML Engineer"
]
COUNTRIES   = ["Australia", "Canada", "China", "France", "Germany", "Global", "India",
               "Japan", "Netherlands", "Singapore", "Switzerland", "UAE", "UK", "USA"]
REMOTE_OPTS = ["Fully Remote", "Hybrid", "On-site"]
INDUSTRIES  = ["Automotive", "Consulting", "Education", "Energy", "Finance", "Government",
               "Healthcare", "Manufacturing", "Media", "Research", "Retail", "Technology"]
EXP_LEVELS  = ["Entry (0-2 yrs)", "Mid (3-5 yrs)", "Senior (6-9 yrs)", "Lead (10+ yrs)"]
EDUCATIONS  = ["Bootcamp/Self-taught", "Associate's", "Bachelor's", "Master's", "PhD"]
COMP_SIZES  = ["Startup (1-50)", "SME (51-500)", "Mid-size (501-5000)", "Enterprise (5000+)", "Big Tech (FAANG+)"]

MAE_USD = 18824


# ── Loaders ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


@st.cache_data
def load_raw_data():
    df = pd.read_csv(DATA_PATH)
    drop = ["job_category", "city", "required_skills", "ai_salary_premium_pct",
            "demand_score", "demand_growth_yoy_pct", "benefits_score_10",
            "posting_year", "posting_month", "is_senior", "is_remote_friendly", "is_llm_role"]
    df.drop(columns=[c for c in drop if c in df.columns], inplace=True)
    return df


@st.cache_data
def get_test_results():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        df = pd.read_csv(DATA_PATH)
        drop = ["job_category", "city", "required_skills", "ai_salary_premium_pct",
                "demand_score", "demand_growth_yoy_pct", "benefits_score_10",
                "posting_year", "posting_month", "is_senior", "is_remote_friendly", "is_llm_role"]
        df.drop(columns=[c for c in drop if c in df.columns], inplace=True)

        df["log_salary"] = np.log(df["annual_salary_usd"])
        df.drop(columns=["annual_salary_usd"], inplace=True)

        df["experience_encoded"]   = df["experience_level"].map(ORDINAL_MAPS["experience_level"])
        df["education_encoded"]    = df["education_required"].map(ORDINAL_MAPS["education_required"])
        df["company_size_encoded"] = df["company_size"].map(ORDINAL_MAPS["company_size"])
        df.drop(columns=["experience_level", "education_required", "company_size"], inplace=True)

        df = pd.get_dummies(df, columns=["job_title", "country", "remote_work", "industry"], drop_first=True)
        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype(int)

        df[SCALE_COLS] = scaler.transform(df[SCALE_COLS])

        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0

        X = df[FEATURE_COLUMNS]
        y = df["log_salary"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return y_test.values, model.predict(X_test)

    except Exception:
        return None, None


# ── Inference ──────────────────────────────────────────────────────────────────
def clean_label(name):
    mapping = {
        "experience_encoded":   "Experience Level",
        "education_encoded":    "Education",
        "company_size_encoded": "Company Size",
        "years_of_experience":  "Years of Experience"
    }
    if name in mapping:
        return mapping[name]
    if name.startswith("job_title_"):  return "Title: "   + name[10:]
    if name.startswith("country_"):    return "Country: " + name[8:]
    if name.startswith("remote_work_"):return "Remote: "  + name[12:]
    if name.startswith("industry_"):   return "Industry: "+ name[9:]
    return name


def predict_salary(inputs, model, scaler):
    fd = {col: 0.0 for col in FEATURE_COLUMNS}

    fd["experience_encoded"]   = float(ORDINAL_MAPS["experience_level"][inputs["experience_level"]])
    fd["education_encoded"]    = float(ORDINAL_MAPS["education_required"][inputs["education_required"]])
    fd["company_size_encoded"] = float(ORDINAL_MAPS["company_size"][inputs["company_size"]])
    fd["years_of_experience"]  = float(inputs["years_of_experience"])

    # Apply saved scaler to ordinal + numeric cols
    scale_vals = np.array([[fd[c] for c in SCALE_COLS]])
    scaled = scaler.transform(scale_vals)[0]
    for i, c in enumerate(SCALE_COLS):
        fd[c] = float(scaled[i])

    # OHE flags
    for prefix, key in [
        ("job_title_",   "job_title"),
        ("country_",     "country"),
        ("remote_work_", "remote_work"),
        ("industry_",    "industry")
    ]:
        col_key = prefix + inputs[key]
        if col_key in fd:
            fd[col_key] = 1.0

    X = np.array([[fd[col] for col in FEATURE_COLUMNS]])
    log_pred = model.predict(X)[0]
    salary = np.exp(log_pred)

    contributions = {
        col: model.coef_[i] * fd[col]
        for i, col in enumerate(FEATURE_COLUMNS)
        if fd[col] != 0.0
    }
    return salary, contributions


# ── Chart helpers ──────────────────────────────────────────────────────────────
def dark_fig(figsize=(9, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")
    ax.tick_params(colors="#444", labelsize=8)
    ax.xaxis.label.set_color("#444")
    ax.yaxis.label.set_color("#444")
    ax.title.set_color("#bbb")
    ax.title.set_fontsize(10)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color("#1e1e1e")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#181818", linewidth=0.5)
    return fig, ax


def chart_category(df, col, col_label, order=None):
    grouped = df.groupby(col)["annual_salary_usd"].median().sort_values(ascending=True)
    if order:
        grouped = grouped.reindex([x for x in order if x in grouped.index])
    height = min(max(4.0, len(grouped) * 0.42), 11.0)
    fig, ax = dark_fig((8.5, height))
    bars = ax.barh(grouped.index, grouped.values / 1000,
                   color="#e74c3c", alpha=0.82, height=0.62)
    ax.set_xlabel("Median Salary ($K)")
    ax.set_title(f"Median Salary by {col_label}", pad=12)
    for bar, val in zip(bars, grouped.values):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"${val/1000:.0f}K", va="center", fontsize=7.5, color="#555")
    ax.set_xlim(0, grouped.values.max() / 1000 * 1.2)
    plt.tight_layout()
    return fig


def chart_actual_vs_pred(y_test, y_pred):
    actual = np.exp(y_test) / 1000
    pred   = np.exp(y_pred) / 1000
    mn = min(actual.min(), pred.min())
    mx = max(actual.max(), pred.max())
    fig, ax = dark_fig((5.8, 5.5))
    ax.scatter(actual, pred, alpha=0.35, color="#e74c3c", s=14, zorder=3)
    ax.plot([mn, mx], [mn, mx], "--", color="#333", linewidth=1.0)
    ax.set_xlabel("Actual Salary ($K)")
    ax.set_ylabel("Predicted Salary ($K)")
    ax.set_title("Actual vs Predicted", pad=10)
    ax.grid(axis="both", color="#181818", linewidth=0.5)
    plt.tight_layout()
    return fig


def chart_residual_dist(y_test, y_pred):
    resid = y_test - y_pred
    fig, ax = dark_fig((5.8, 4.0))
    ax.hist(resid, bins=35, color="#e74c3c", alpha=0.7, edgecolor="none")
    ax.axvline(0, color="#444", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution", pad=10)
    plt.tight_layout()
    return fig


def chart_coefficients(model):
    coef_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "coef": model.coef_})
    coef_df["abs"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs", ascending=False)
    top = pd.concat([
        coef_df[coef_df["coef"] > 0].head(8),
        coef_df[coef_df["coef"] < 0].head(8)
    ]).sort_values("coef")
    labels = [clean_label(f) for f in top["feature"]]
    colors = ["#e74c3c" if v < 0 else "#27ae60" for v in top["coef"]]
    fig, ax = dark_fig((8.0, 7.2))
    ax.barh(labels, top["coef"].values, color=colors, alpha=0.82, height=0.65)
    ax.axvline(0, color="#2a2a2a", linewidth=0.8)
    ax.set_xlabel("Coefficient (log scale)")
    ax.set_title("Top Feature Coefficients", pad=12)
    ax.grid(axis="x", color="#181818", linewidth=0.5)
    plt.tight_layout()
    return fig




# ── Main header ────────────────────────────────────────────────────────────────
hdr_left, hdr_right = st.columns([2.8, 1], gap="large")

with hdr_left:
    st.markdown("<div class='app-title'>AI Job Market Salary Predictor</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='app-sub'>Linear regression &nbsp;·&nbsp; 1,500 records "
        "&nbsp;·&nbsp; 14 countries &nbsp;·&nbsp; 25 job titles</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='info-card'><p>"
        "Input a job profile and get an estimated annual salary in USD, "
        "with a breakdown showing which factors are pushing the number up or down. "
        "Built with linear regression on 54 encoded features from 9 raw columns."
        "</p><div style='margin-top:0.75rem;'>"
        "<span class='info-pill'>Test R\u00b2 0.84</span>"
        "<span class='info-pill'>MAE $18,824</span>"
        "<span class='info-pill'>5-Fold CV 0.85</span>"
        "</div></div>",
        unsafe_allow_html=True
    )

with hdr_right:
    if os.path.exists(PHOTO_PATH):
        with open(PHOTO_PATH, "rb") as _f:
            _b64 = base64.b64encode(_f.read()).decode()
        _photo = (
            "<img src=\"data:image/jpeg;base64," + _b64 + "\""
            " style=\"width:72px;height:72px;border-radius:50%;"
            "border:2px solid #e74c3c;object-fit:cover;"
            "display:block;margin:0 auto 0.75rem;\"/>"
        )
    else:
        _photo = (
            "<div style=\"width:72px;height:72px;border-radius:50%;"
            "background:#1a1a1a;border:2px solid #e74c3c;"
            "margin:0 auto 0.75rem;display:flex;align-items:center;"
            "justify-content:center;font-size:1.3rem;color:#333;\">B</div>"
        )

    st.markdown(
        "<div class=\"dev-card\">"
        + _photo
        + "<div class=\"dev-name\">Bibek Subedi</div>"
        "<div class=\"dev-role\">Aspiring ML Engineer<br>AI &amp; Data Science</div>"
        "<div class=\"social-wrap\" style=\"margin-top:0.75rem;\">"
        "<a class=\"social-link\" href=\"https://github.com/BibekSubediCR7\""
        " target=\"_blank\">GitHub</a>"
        "<a class=\"social-link\" href=\"https://www.linkedin.com/in/bibeksubedicr7/\""
        " target=\"_blank\">LinkedIn</a>"
        "<a class=\"social-link\" href=\"https://instagram.com/bibek_subedi_cr7\""
        " target=\"_blank\">Instagram</a>"
        "</div></div>",
        unsafe_allow_html=True
    )

model, scaler = load_artifacts()

if model is None:
    st.markdown("""
        <div style='background:#111; border:1px solid #e74c3c; border-radius:10px;
                    padding:1.5rem; color:#e74c3c; font-size:0.85rem; margin-top:1rem;'>
            Model not found. Run <code style='background:#1a1a1a; padding:0.1rem 0.4rem;
            border-radius:4px;'>python main.py</code> from the project root first.
        </div>""", unsafe_allow_html=True)
    st.stop()

tab1, tab2, tab3 = st.tabs(["Predict", "Explore the Market", "Model Performance"])


# ── Tab 1: Predict ─────────────────────────────────────────────────────────────
with tab1:
    col_form, col_result = st.columns([1, 1.55], gap="large")

    with col_form:
        st.markdown("<div class='sec-hdr'>Your Profile</div>", unsafe_allow_html=True)

        job_title        = st.selectbox("Job Title",        JOB_TITLES,  index=JOB_TITLES.index("ML Engineer"))
        experience_level = st.selectbox("Experience Level", EXP_LEVELS,  index=1)
        education        = st.selectbox("Education",        EDUCATIONS,  index=2)

        # Auto-derive years from selected experience level (mid-point of each band)
        _exp_to_years = {
            "Entry (0-2 yrs)": 1,
            "Mid (3-5 yrs)":   4,
            "Senior (6-9 yrs)": 7,
            "Lead (10+ yrs)":  12
        }
        years_exp = _exp_to_years[experience_level]
        col_a, col_b     = st.columns(2)
        with col_a:
            country      = st.selectbox("Country",      COUNTRIES,   index=COUNTRIES.index("USA"))
            company_size = st.selectbox("Company Size", COMP_SIZES,  index=2)
        with col_b:
            industry     = st.selectbox("Industry",     INDUSTRIES,  index=INDUSTRIES.index("Technology"))
            remote_work  = st.selectbox("Remote Work",  REMOTE_OPTS, index=0)

        st.markdown("<div style='margin-top:1.4rem;'></div>", unsafe_allow_html=True)
        predict_btn = st.button("Predict Salary")

    with col_result:
        if predict_btn:
            inputs = {
                "job_title":        job_title,
                "experience_level": experience_level,
                "years_of_experience": years_exp,
                "education_required":  education,
                "country":          country,
                "company_size":     company_size,
                "industry":         industry,
                "remote_work":      remote_work
            }
            salary, contributions = predict_salary(inputs, model, scaler)

            raw_df     = load_raw_data()
            percentile = (raw_df["annual_salary_usd"] < salary).mean() * 100
            low_est    = max(salary - MAE_USD, 0)
            high_est   = salary + MAE_USD

            st.markdown(f"""
                <div class='salary-card'>
                    <div class='salary-lbl'>Estimated Annual Salary</div>
                    <div class='salary-amt'>${salary:,.0f}</div>
                    <div class='salary-range'>Likely range &nbsp; ${low_est:,.0f} &ndash; ${high_est:,.0f}</div>
                    <div>
                        <span class='pct-badge'>Top {100 - percentile:.0f}% of the dataset</span>
                    </div>
                </div>""", unsafe_allow_html=True)

            # Contribution breakdown
            pos = sorted(
                [(k, v) for k, v in contributions.items() if v > 0],
                key=lambda x: x[1], reverse=True
            )[:4]
            neg = sorted(
                [(k, v) for k, v in contributions.items() if v < 0],
                key=lambda x: x[1]
            )[:4]

            all_abs = [abs(v) for _, v in pos + neg]
            max_abs = max(all_abs) if all_abs else 1.0

            st.markdown("<div class='sec-hdr' style='margin-top:1.3rem;'>What is driving this</div>",
                        unsafe_allow_html=True)

            if pos:
                st.markdown("<div class='contrib-group-lbl'>Boosting</div>", unsafe_allow_html=True)
                html = ""
                for name, val in pos:
                    w   = abs(val) / max_abs * 100
                    pct = (np.exp(abs(val)) - 1) * 100
                    html += f"""
                    <div class='contrib-item'>
                        <span class='contrib-lbl'>{clean_label(name)}</span>
                        <div class='contrib-track'>
                            <div class='contrib-fill-pos' style='width:{w:.0f}%'></div>
                        </div>
                        <span class='contrib-pct' style='color:#27ae60;'>+{pct:.0f}%</span>
                    </div>"""
                st.markdown(html, unsafe_allow_html=True)

            if neg:
                st.markdown("<div class='contrib-group-lbl' style='margin-top:0.8rem;'>Reducing</div>",
                            unsafe_allow_html=True)
                html = ""
                for name, val in neg:
                    w   = abs(val) / max_abs * 100
                    pct = (np.exp(abs(val)) - 1) * 100
                    html += f"""
                    <div class='contrib-item'>
                        <span class='contrib-lbl'>{clean_label(name)}</span>
                        <div class='contrib-track'>
                            <div class='contrib-fill-neg' style='width:{w:.0f}%'></div>
                        </div>
                        <span class='contrib-pct' style='color:#e74c3c;'>-{pct:.0f}%</span>
                    </div>"""
                st.markdown(html, unsafe_allow_html=True)

            st.markdown("""
                <div class='disclaimer'>
                    Percentages reflect the model's coefficient relative to the reference baseline
                    (AI Agent Developer, Australia, Automotive, Fully Remote). Ordinal feature
                    contributions reflect scaled values. Range uses training MAE of $18,824.
                </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
                <div class='placeholder'>
                    Fill in your profile on the left<br>
                    and click Predict Salary.
                    <br><br>
                    <span style='font-size:0.72rem; color:#1e1e1e;'>
                        Try switching countries or company sizes<br>
                        to see how much each factor moves the number.
                    </span>
                </div>""", unsafe_allow_html=True)


# ── Tab 2: Explore the Market ──────────────────────────────────────────────────
with tab2:
    raw_df = load_raw_data()

    dimension_options = {
        "Experience Level": ("experience_level",   EXP_LEVELS),
        "Country":          ("country",            None),
        "Company Size":     ("company_size",        COMP_SIZES),
        "Job Title":        ("job_title",           None),
        "Industry":         ("industry",            None),
        "Education":        ("education_required",  EDUCATIONS),
        "Remote Work":      ("remote_work",         REMOTE_OPTS),
    }

    col_sel, col_chart = st.columns([1, 2.2], gap="large")

    with col_sel:
        st.markdown("<div class='sec-hdr'>Choose Dimension</div>", unsafe_allow_html=True)
        dim_label = st.radio(
            "", list(dimension_options.keys()),
            label_visibility="collapsed"
        )
        col_name, order = dimension_options[dim_label]

        st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='sec-hdr'>Breakdown</div>", unsafe_allow_html=True)

        grouped = (
            raw_df.groupby(col_name)["annual_salary_usd"]
            .agg(["median", "count"])
            .sort_values("median", ascending=False)
            .round(0)
        )
        grouped.columns = ["Median", "Count"]

        rows = ""
        for idx, row in grouped.iterrows():
            rows += f"<tr><td>{idx}</td><td>${row['Median']:,.0f}</td><td>{int(row['Count'])}</td></tr>"

        st.markdown(f"""
            <table class='data-table'>
                <thead><tr>
                    <th>{dim_label}</th><th>Median</th><th>Count</th>
                </tr></thead>
                <tbody>{rows}</tbody>
            </table>""", unsafe_allow_html=True)

    with col_chart:
        st.markdown(
            f"<div class='sec-hdr'>Median Annual Salary by {dim_label}</div>",
            unsafe_allow_html=True
        )
        fig = chart_category(raw_df, col_name, dim_label, order)
        st.pyplot(fig)
        plt.close(fig)


# ── Tab 3: Model Performance ───────────────────────────────────────────────────
with tab3:
    y_test, y_pred = get_test_results()

    if y_test is None:
        st.markdown(
            "<div style='color:#e74c3c; font-size:0.85rem; margin-top:1rem;'>"
            "Could not evaluate model. Ensure main.py has been run.</div>",
            unsafe_allow_html=True
        )
    else:
        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(np.exp(y_test), np.exp(y_pred))
        rmse = np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred)))

        # Metric cards
        m1, m2, m3, m4 = st.columns(4)
        for col, val, lbl in [
            (m1, f"{r2:.4f}",    "Test R²"),
            (m2, f"${mae:,.0f}", "MAE (USD)"),
            (m3, f"${rmse:,.0f}","RMSE (USD)"),
            (m4, "0.8514",       "5-Fold CV R²")
        ]:
            with col:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-val'>{val}</div>
                        <div class='metric-lbl'>{lbl}</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.6rem;'></div>", unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1.7], gap="large")

        with col_left:
            # Prediction accuracy table
            st.markdown("<div class='sec-hdr'>Prediction Accuracy — 300 Test Samples</div>",
                        unsafe_allow_html=True)
            errors = np.abs(np.exp(y_test) - np.exp(y_pred))
            rows = ""
            for t in [10_000, 20_000, 30_000, 50_000]:
                count = (errors <= t).sum()
                pct   = count / len(errors) * 100
                rows += f"<tr><td>Within ${t:,}</td><td>{count} / 300</td><td>{pct:.1f}%</td></tr>"
            st.markdown(f"""
                <table class='data-table'>
                    <thead><tr><th>Threshold</th><th>Count</th><th>Share</th></tr></thead>
                    <tbody>{rows}</tbody>
                </table>""", unsafe_allow_html=True)

            # CV fold breakdown
            st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
            st.markdown("<div class='sec-hdr'>5-Fold Cross-Validation Breakdown</div>",
                        unsafe_allow_html=True)
            cv_data = [
                ("Fold 1", "0.8573"), ("Fold 2", "0.8670"),
                ("Fold 3", "0.8738"), ("Fold 4", "0.8551"), ("Fold 5", "0.8037"),
            ]
            cv_rows = "".join(
                f"<tr><td>{f}</td><td>{s}</td></tr>" for f, s in cv_data
            )
            cv_rows += "<tr style='border-top:1px solid #222;'><td><b style='color:#ccc;'>Mean</b></td><td><b style='color:#e74c3c;'>0.8514</b></td></tr>"
            st.markdown(f"""
                <table class='data-table'>
                    <thead><tr><th>Fold</th><th>R²</th></tr></thead>
                    <tbody>{cv_rows}</tbody>
                </table>""", unsafe_allow_html=True)

            # Sample predictions
            st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
            st.markdown("<div class='sec-hdr'>Sample Predictions vs Actual</div>",
                        unsafe_allow_html=True)
            sample_actual = np.exp(y_test[:5])
            sample_pred   = np.exp(y_pred[:5])
            sample_rows = ""
            for a, p in zip(sample_actual, sample_pred):
                err   = a - p
                sign  = "+" if err >= 0 else ""
                color = "#27ae60" if abs(err) < 20000 else "#e74c3c"
                sample_rows += (
                    f"<tr><td>${a:,.0f}</td><td>${p:,.0f}</td>"
                    f"<td style='color:{color};'>{sign}${err:,.0f}</td></tr>"
                )
            st.markdown(f"""
                <table class='data-table'>
                    <thead><tr><th>Actual</th><th>Predicted</th><th>Error</th></tr></thead>
                    <tbody>{sample_rows}</tbody>
                </table>""", unsafe_allow_html=True)

            # Residual distribution
            st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
            st.markdown("<div class='sec-hdr'>Residual Distribution</div>", unsafe_allow_html=True)
            fig_rd = chart_residual_dist(y_test, y_pred)
            st.pyplot(fig_rd)
            plt.close(fig_rd)

        with col_right:
            st.markdown("<div class='sec-hdr'>Actual vs Predicted Salary</div>",
                        unsafe_allow_html=True)
            fig_avp = chart_actual_vs_pred(y_test, y_pred)
            st.pyplot(fig_avp)
            plt.close(fig_avp)

            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
            st.markdown("<div class='sec-hdr'>Top Feature Coefficients</div>",
                        unsafe_allow_html=True)
            fig_coef = chart_coefficients(model)
            st.pyplot(fig_coef)
            plt.close(fig_coef)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
    <div class='footer'>
        <a href='https://github.com/BibekSubediCR7/AI-Job-Market-Salary-Prediction'
           target='_blank'>GitHub Repository</a>
        <br>
        &copy; 2026 Bibek Subedi. Made with love.
    </div>""", unsafe_allow_html=True)