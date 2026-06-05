import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import os

sns.set_theme(style="whitegrid")
PLOT_DIR = "reports/model_plots"


def compute_metrics(y_true, y_pred) -> dict:
    """Compute R², MAE, RMSE in both log and dollar scale."""
    r2 = r2_score(y_true, y_pred)
    mae_log = mean_absolute_error(y_true, y_pred)
    rmse_log = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_usd = mean_absolute_error(np.exp(y_true), np.exp(y_pred))
    rmse_usd = np.sqrt(mean_squared_error(np.exp(y_true), np.exp(y_pred)))

    metrics = {
        "r2": round(r2, 4),
        "mae_log": round(mae_log, 4),
        "rmse_log": round(rmse_log, 4),
        "mae_usd": round(mae_usd, 0),
        "rmse_usd": round(rmse_usd, 0)
    }

    print(f"R²        : {metrics['r2']}")
    print(f"MAE (log) : {metrics['mae_log']}")
    print(f"RMSE (log): {metrics['rmse_log']}")
    print(f"MAE (USD) : ${metrics['mae_usd']:,.0f}")
    print(f"RMSE (USD): ${metrics['rmse_usd']:,.0f}")
    return metrics


def error_distribution(y_true, y_pred) -> None:
    """Print count and % of predictions within common dollar thresholds."""
    errors = np.abs(np.exp(y_true) - np.exp(y_pred))
    n = len(errors)
    print("\nPrediction accuracy (dollar thresholds):")
    for t in [10_000, 20_000, 30_000, 50_000]:
        count = (errors <= t).sum()
        print(f"  Within ${t:,}: {count}/{n} ({count/n*100:.1f}%)")


def plot_actual_vs_predicted(y_true, y_pred) -> None:
    """Scatter plot of actual vs predicted salary in dollars."""
    actual = np.exp(y_true) / 1000
    predicted = np.exp(y_pred) / 1000
    min_val, max_val = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())

    plt.figure(figsize=(7, 6))
    plt.scatter(actual, predicted, alpha=0.4, color="steelblue", s=20, label="Predictions")
    plt.plot([min_val, max_val], [min_val, max_val],
             color="red", linewidth=1.2, linestyle="--", label="Perfect fit")
    plt.xlabel("Actual Salary ($K)")
    plt.ylabel("Predicted Salary ($K)")
    plt.title("Actual vs Predicted Salary")
    plt.legend()
    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.savefig(f"{PLOT_DIR}/actual_vs_predicted.png", dpi=150)
    plt.show()


def plot_residuals(y_true, y_pred) -> None:
    """Residuals vs predicted values plot."""
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.4, color="coral", s=20)
    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Predicted Log Salary")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residuals vs Predicted Values")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/residuals_vs_predicted.png", dpi=150)
    plt.show()


def plot_residual_distribution(y_true, y_pred) -> None:
    """Histogram of residuals."""
    residuals = y_true - y_pred

    plt.figure(figsize=(7, 4))
    plt.hist(residuals, bins=40, color="steelblue", edgecolor="white")
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Distribution of Residuals")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/residual_distribution.png", dpi=150)
    plt.show()


def plot_coefficients(model: LinearRegression, feature_names: list) -> None:
    """Bar chart of top positive and negative feature coefficients."""
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": model.coef_})
    coef_df["abs_coef"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)

    top_pos = coef_df[coef_df["coefficient"] > 0].head(8)
    top_neg = coef_df[coef_df["coefficient"] < 0].head(8)
    plot_df = pd.concat([top_pos, top_neg]).sort_values("coefficient")

    colors = ["coral" if c < 0 else "steelblue" for c in plot_df["coefficient"]]
    plt.figure(figsize=(10, 7))
    plt.barh(plot_df["feature"], plot_df["coefficient"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Coefficient Value")
    plt.title("Top Feature Coefficients (Log Salary Scale)")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/feature_coefficients.png", dpi=150)
    plt.show()