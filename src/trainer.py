import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
import pickle
import os


def split_features_target(df: pd.DataFrame, target_col: str = "log_salary"):
    """Separate X and y from the model-ready dataframe."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def train_test_split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Wrapper around sklearn train_test_split for consistency."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train(X_train, y_train) -> LinearRegression:
    """Fit a LinearRegression model and return it."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"Model trained. Intercept: {model.intercept_:.4f}")
    print(f"Number of coefficients: {len(model.coef_)}")
    return model


def cross_validate(model: LinearRegression, X, y, cv: int = 5) -> dict:
    """Run k-fold cross-validation and return scores."""
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    results = {
        "scores": scores.tolist(),
        "mean": round(scores.mean(), 4),
        "std": round(scores.std(), 4)
    }
    print(f"CV R² scores: {[round(s, 4) for s in scores]}")
    print(f"Mean: {results['mean']} | Std: {results['std']}")
    return results


def save_model(model: LinearRegression, path: str = "models/linear_model.pkl") -> None:
    """Serialize and save the trained model."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def load_model(path: str = "models/linear_model.pkl") -> LinearRegression:
    """Load a saved model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at {path}. Run main.py first.")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model