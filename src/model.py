"""
Module 5: Random Forest Regression (Yield Prediction)
======================================================
Prepares features, splits data, trains a Random Forest Regressor,
performs hyperparameter tuning, and saves the trained model.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Define default features to use for predicting Yield
FEATURE_COLS = [
    "Soil_Cluster",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "N_mean",
    "P_mean",
    "K_mean",
    "ph_mean",
    "temperature_mean",
    "humidity_mean",
    "NPK_Index",
    "Rainfall_Fert_Ratio",
    "Log_Area",
    "Season_encoded",
    "Crop_encoded",
    "State_encoded"
]

TARGET_COL = "Yield"
MODEL_DIR = "outputs/models"


def prepare_features(df: pd.DataFrame, feature_cols: list = None) -> tuple:
    """
    Separate features (X) and target (y) from the master DataFrame.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    # Ensure all specified features are present in the dataframe
    available_features = [col for col in feature_cols if col in df.columns]
    missing_features = [col for col in feature_cols if col not in df.columns]

    if missing_features:
        print(f"[WARNING] Some requested features were missing: {missing_features}")
        print(f"[INFO] Proceeding with available features: {available_features}")

    X = df[available_features].copy()
    y = df[TARGET_COL].copy()

    print(f"[INFO] Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Split the dataset into training (80%) and testing (20%) sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[INFO] Split data into Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                        n_estimators: int = 100, max_depth: int = 15,
                        min_samples_split: int = 5, min_samples_leaf: int = 2,
                        random_state: int = 42) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor with specified hyperparameters.
    """
    print(f"[INFO] Training Random Forest Regressor (n_estimators={n_estimators}, max_depth={max_depth})...")
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("[INFO] Model training complete.")
    return rf


def tune_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> RandomForestRegressor:
    """
    Perform hyperparameter tuning using GridSearchCV on a subset of data (for speed).
    """
    print("[INFO] Starting hyperparameter tuning using GridSearchCV...")

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4]
    }

    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)

    # If dataset is very large, subset it for Grid Search to prevent timeout/slowness
    if len(X_train) > 5000:
        print(f"[INFO] Subsetting training data for Grid Search from {len(X_train)} to 5000 rows.")
        X_sub = X_train.sample(5000, random_state=random_state)
        y_sub = y_train.loc[X_sub.index]
    else:
        X_sub = X_train
        y_sub = y_train

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_sub, y_sub)

    print(f"[INFO] Best parameters found: {grid_search.best_params_}")
    print(f"[INFO] Best CV R2 score: {grid_search.best_score_:.4f}")

    # Retrain on the FULL training set using the best hyperparameters
    best_params = grid_search.best_params_
    best_rf = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=random_state,
        n_jobs=-1
    )
    print("[INFO] Retraining best model on full training set...")
    best_rf.fit(X_train, y_train)
    print("[INFO] Retraining complete.")

    return best_rf


def get_feature_importance(model: RandomForestRegressor, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importances and return them sorted in a DataFrame.
    """
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    print("\nFeature Importances:")
    print(importance_df.to_string())
    return importance_df


def save_model(model, scaler=None, encoders=None, path: str = None):
    """
    Serialise and save the trained model (and scaling/encoding artifacts) to disk.
    """
    if path is None:
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, "rf_model.pkl")

    model_package = {
        'model': model,
        'scaler': scaler,
        'encoders': encoders
    }

    joblib.dump(model_package, path)
    print(f"[INFO] Saved trained model package to {path}")


def load_model(path: str = None):
    """
    Load the serialised model package from disk.
    """
    if path is None:
        path = os.path.join(MODEL_DIR, "rf_model.pkl")

    if not os.path.exists(path):
        raise FileNotFoundError(f"No serialised model found at {path}")

    model_package = joblib.load(path)
    print(f"[INFO] Loaded model package from {path}")
    return model_package


if __name__ == "__main__":
    # Test script standalone
    df = pd.read_csv("data/processed/master_dataset.csv")
    if 'Soil_Cluster' not in df.columns:
        # Dummy cluster labels if clustering hasn't run
        df['Soil_Cluster'] = np.random.randint(0, 3, size=len(df))

    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_random_forest(X_train, y_train)
    importances = get_feature_importance(model, X.columns.tolist())
    save_model(model)
