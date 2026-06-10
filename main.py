"""
Precision Farming for Crop Yield Optimisation — Main Pipeline Runner
====================================================================
Orchestrates the entire machine learning pipeline end-to-end:
1. Preprocess raw datasets (crop_yield.csv & crop_recommendation.csv)
2. Run Exploratory Data Analysis (EDA)
3. Run K-Means soil profiling (clustering)
4. Split data, train, and tune Random Forest regressor
5. Evaluate performance diagnostics against baseline models
6. Generate output summary charts and save model package
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure src modules are discoverable
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_preprocessing import run_preprocessing_pipeline
from src.eda import run_eda
from src.clustering import run_clustering
from src.model import prepare_features, split_data, tune_hyperparameters, train_random_forest, get_feature_importance, save_model
from src.evaluation import run_evaluation_pipeline
from src.visualization import run_visualization_pipeline

# Configurable options
RUN_EDA = True
TUNE_HYPERPARAMS = True  # Set to True to perform GridSearchCV hyperparameter tuning
CLUSTERING_K = None      # Set to None to automatically find optimal K using Silhouette score


def run_pipeline():
    print("=" * 70)
    print(" PRECISION FARMING FOR CROP YIELD OPTIMISATION — PIPELINE RUNNER")
    print("=" * 70)

    # 1. Load and preprocess datasets
    master_df, label_encoders = run_preprocessing_pipeline("data/raw")

    # 2. Run EDA (Exploratory Data Analysis)
    if RUN_EDA:
        run_eda(master_df)

    # 3. K-Means Soil Profiling (Clustering)
    clustered_df = run_clustering(master_df, k=CLUSTERING_K)

    # Save final clustered master dataset
    os.makedirs("data/processed", exist_ok=True)
    clustered_df.to_csv("data/processed/clustered_master_dataset.csv", index=False)
    print(f"\n[INFO] Saved clustered_master_dataset.csv -> data/processed/")

    # 4. Supervised Random Forest Modelling
    X, y = prepare_features(clustered_df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    if TUNE_HYPERPARAMS:
        model = tune_hyperparameters(X_train, y_train, random_state=42)
    else:
        model = train_random_forest(X_train, y_train, random_state=42)

    # Extract feature importances
    importance_df = get_feature_importance(model, X.columns.tolist())

    # 5. Model Evaluation and Diagnostics
    metrics = run_evaluation_pipeline(X_train, X_test, y_train, y_test, model, X.columns.tolist())

    # 6. Generate final summary visualizations
    # Use the test split records from the original df for state/crop yield grouped comparisons
    test_orig_df = clustered_df.loc[y_test.index]
    y_pred = model.predict(X_test)
    run_visualization_pipeline(test_orig_df, importance_df, y_test.values, y_pred)

    # Save the trained model package (model, scaler, encoders)
    save_model(model, scaler=None, encoders=label_encoders)

    # Check if target metric is achieved (R2 >= 0.75)
    r2 = metrics["R2"]
    print("\n" + "=" * 70)
    print(" PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"  Final Random Forest Test R2 Score: {r2:.4f}")
    print(f"  Test RMSE                        : {metrics['RMSE']:.4f}")
    print(f"  Test MAE                         : {metrics['MAE']:.4f}")
    print(f"  Test MAPE                        : {metrics['MAPE (%)']:.2f}%")
    
    if r2 >= 0.75:
        print("  [SUCCESS] Target metric R2 >= 0.75 was ACHIEVED!")
    else:
        print("  [WARNING] Target metric R2 >= 0.75 was not achieved. Consider additional feature engineering.")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
