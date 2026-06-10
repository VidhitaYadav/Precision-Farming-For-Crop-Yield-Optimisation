"""
Module 6: Model Evaluation & Diagnostics
==========================================
Computes standard regression metrics (R2, RMSE, MAE, MAPE),
generates diagnostics plots (Actual vs Predicted, Residuals, Errors Histogram),
and compares Random Forest performance against baseline models.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Global style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURE_DIR = "outputs/figures"


def _save_fig(name: str):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate R2, RMSE, MAE, and MAPE.
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Avoid division by zero in MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    metrics = {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape
    }

    return metrics


def actual_vs_predicted_plot(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Scatter plot of actual vs predicted values with a 45-degree reference line.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.4, color="#4C72B0", edgecolors='w', s=20)

    # 45-degree diagonal line
    max_val = max(max(y_true), max(y_pred))
    min_val = min(min(y_true), min(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')

    ax.set_title("Actual vs. Predicted Crop Yield", fontsize=14, fontweight="bold")
    ax.set_xlabel("Actual Yield (tonnes/ha)", fontsize=12)
    ax.set_ylabel("Predicted Yield (tonnes/ha)", fontsize=12)
    ax.legend()
    _save_fig("eval_actual_vs_predicted.png")


def residual_plot(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Plot residuals (errors) vs predicted values.
    """
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.4, color="#C44E52", edgecolors='w', s=20)
    ax.axhline(y=0, color='black', linestyle='--', lw=2)

    ax.set_title("Residuals vs. Predicted Yield", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Yield (tonnes/ha)", fontsize=12)
    ax.set_ylabel("Residuals (Actual - Predicted)", fontsize=12)
    _save_fig("eval_residuals_vs_predicted.png")


def residual_histogram(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Plot histogram distribution of residuals.
    """
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(residuals, kde=True, color="#55A868", bins=50, ax=ax, edgecolor='white')

    ax.set_title("Distribution of Residuals (Errors)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Residual (tonnes/ha)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    _save_fig("eval_residual_histogram.png")


def baseline_comparison(X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: pd.Series, y_test: pd.Series,
                        rf_metrics: dict) -> pd.DataFrame:
    """
    Train baseline models (Linear Regression and Decision Tree) and
    compare performance metrics against Random Forest.
    """
    print("[INFO] Training baseline models for comparison...")

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_metrics = compute_metrics(y_test, lr_pred)

    # 2. Decision Tree
    dt = DecisionTreeRegressor(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_metrics = compute_metrics(y_test, dt_pred)

    # Compile table
    comparison_data = {
        "Model": ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"],
        "R2": [lr_metrics["R2"], dt_metrics["R2"], rf_metrics["R2"]],
        "RMSE": [lr_metrics["RMSE"], dt_metrics["RMSE"], rf_metrics["RMSE"]],
        "MAE": [lr_metrics["MAE"], dt_metrics["MAE"], rf_metrics["MAE"]],
        "MAPE (%)": [lr_metrics["MAPE (%)"], dt_metrics["MAPE (%)"], rf_metrics["MAPE (%)"]]
    }

    comparison_df = pd.DataFrame(comparison_data)

    print("\nBaseline Model Comparison:")
    print(comparison_df.to_string(index=False))

    # Save comparison table
    os.makedirs(FIGURE_DIR, exist_ok=True)
    comparison_df.to_csv(os.path.join(FIGURE_DIR, "baseline_comparison.csv"), index=False)
    print(f"  [SAVED] baseline_comparison.csv")

    return comparison_df


def run_evaluation_pipeline(X_train: pd.DataFrame, X_test: pd.DataFrame,
                            y_train: pd.Series, y_test: pd.Series,
                            model, feature_names: list) -> dict:
    """
    Run complete evaluation pipeline, generate diagnostic plots, and compare with baselines.
    """
    print("=" * 60)
    print(" MODEL EVALUATION & DIAGNOSTICS")
    print("=" * 60)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    rf_metrics = compute_metrics(y_test, y_pred)
    print("\nRandom Forest Test Metrics:")
    for metric_name, val in rf_metrics.items():
        print(f"  {metric_name:<10}: {val:.4f}")

    # Generate plots
    print("\nGenerating Diagnostic Plots...")
    actual_vs_predicted_plot(y_test.values, y_pred)
    residual_plot(y_test.values, y_pred)
    residual_histogram(y_test.values, y_pred)

    # Compare with baselines
    baseline_comparison(X_train, X_test, y_train, y_test, rf_metrics)

    return rf_metrics
