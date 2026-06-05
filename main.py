from src.data_loader import load_raw, validate
from src.preprocessor import run_full_pipeline
from src.trainer import split_features_target, train_test_split_data, train, cross_validate, save_model
from src.evaluator import compute_metrics, error_distribution, plot_actual_vs_predicted, plot_residuals, plot_residual_distribution, plot_coefficients

RAW_PATH = "data/raw/ai_jobs_market.csv"
MODEL_PATH = "models/linear_model.pkl"
SCALER_PATH = "models/scaler.pkl"


def main():
    # 1. Load and validate
    print("=== Loading data ===")
    df = load_raw(RAW_PATH)
    validate(df)

    # 2. Preprocess
    print("\n=== Preprocessing ===")
    df_ready = run_full_pipeline(df, save_scaler_path=SCALER_PATH)

    # 3. Split
    print("\n=== Splitting ===")
    X, y = split_features_target(df_ready)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # 4. Train
    print("\n=== Training ===")
    model = train(X_train, y_train)

    # 5. Cross-validate
    print("\n=== Cross-validation ===")
    cross_validate(model, X, y, cv=5)

    # 6. Evaluate
    print("\n=== Evaluation (test set) ===")
    y_pred = model.predict(X_test)
    compute_metrics(y_test, y_pred)
    error_distribution(y_test, y_pred)

    # 7. Plots
    print("\n=== Generating plots ===")
    plot_actual_vs_predicted(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    plot_residual_distribution(y_test, y_pred)
    plot_coefficients(model, X.columns.tolist())

    # 8. Save model
    print("\n=== Saving model ===")
    save_model(model, path=MODEL_PATH)
    print("\nDone.")


if __name__ == "__main__":
    main()