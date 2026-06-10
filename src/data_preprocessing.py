"""
Module 1 & 2: Data Collection, Cleaning, Merging, and Feature Engineering
==========================================================================
Loads the two raw Kaggle datasets (crop_yield.csv & crop_recommendation.csv),
cleans them, merges soil/weather features into the crop production data,
and engineers new features for downstream modelling.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ──────────────────────────────────────────────
#  1. LOAD RAW DATASETS
# ──────────────────────────────────────────────
def load_datasets(data_dir: str = "data/raw"):
    """Load crop_yield.csv and crop_recommendation.csv from *data_dir*."""
    crop_path = os.path.join(data_dir, "crop_yield.csv")
    soil_path = os.path.join(data_dir, "crop_recommendation.csv")

    crop_df = pd.read_csv(crop_path)
    soil_df = pd.read_csv(soil_path)

    print(f"[INFO] Loaded crop_yield.csv         -> {crop_df.shape}")
    print(f"[INFO] Loaded crop_recommendation.csv -> {soil_df.shape}")
    return crop_df, soil_df


# ──────────────────────────────────────────────
#  2. CLEAN INDIVIDUAL DATASETS
# ──────────────────────────────────────────────
def clean_crop_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the crop yield dataset:
    - Strip whitespace from string columns
    - Handle missing values (median for numeric, mode for categorical)
    - Remove exact duplicate rows
    """
    df = df.copy()

    # Strip whitespace from object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Drop exact duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    if before != after:
        print(f"[INFO] Removed {before - after} duplicate rows from crop data")

    # Impute missing numeric values with median
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"[INFO] Imputed {col} nulls with median = {median_val:.2f}")

    # Impute missing categorical values with mode
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"[INFO] Imputed {col} nulls with mode = {mode_val}")

    return df


def clean_soil_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the crop recommendation (soil) dataset:
    - Strip whitespace from label column
    - Handle missing values
    - Remove duplicates
    """
    df = df.copy()

    if "label" in df.columns:
        df["label"] = df["label"].str.strip()

    df.drop_duplicates(inplace=True)

    # Impute numeric nulls
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    return df


# ──────────────────────────────────────────────
#  3. OUTLIER TREATMENT (IQR-based capping)
# ──────────────────────────────────────────────
def treat_outliers_iqr(df: pd.DataFrame, columns: list = None,
                       factor: float = 1.5) -> pd.DataFrame:
    """
    Cap outliers in *columns* using the IQR method.
    Values below Q1 - factor*IQR are set to that lower bound;
    values above Q3 + factor*IQR are set to that upper bound.
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        outliers_count = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers_count > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            print(f"[INFO] Capped {outliers_count} outliers in '{col}' "
                  f"[{lower:.2f}, {upper:.2f}]")

    return df


# ──────────────────────────────────────────────
#  4. MERGE DATASETS
# ──────────────────────────────────────────────
def merge_datasets(crop_df: pd.DataFrame,
                   soil_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge soil-level features into the crop production dataset.

    Strategy:
    - The soil dataset has per-crop soil/weather readings (N, P, K, ph,
      temperature, humidity, rainfall, label). We aggregate these to
      crop-level averages and merge on the 'Crop' <-> 'label' key.
    - This enriches each crop-production row with mean soil nutrient
      values (N, P, K, ph) and weather proxies (temperature, humidity).
    """
    # Normalise join keys
    crop_df = crop_df.copy()
    soil_df = soil_df.copy()

    # Map crop names to match labels in recommendation dataset
    crop_mapping = {
        "arhar/tur": "pigeonpeas",
        "gram": "chickpea",
        "urad": "blackgram",
        "moong(green gram)": "mungbean",
        "masoor": "lentil",
        "cotton(lint)": "cotton",
        "moth": "mothbeans"
    }
    crop_df["Crop_lower"] = crop_df["Crop"].str.lower().str.strip().replace(crop_mapping)
    soil_df["Crop_lower"] = soil_df["label"].str.lower().str.strip()

    # Aggregate soil features per crop
    soil_agg = (
        soil_df
        .groupby("Crop_lower")
        .agg(
            N_mean=("N", "mean"),
            P_mean=("P", "mean"),
            K_mean=("K", "mean"),
            ph_mean=("ph", "mean"),
            temperature_mean=("temperature", "mean"),
            humidity_mean=("humidity", "mean"),
            soil_rainfall_mean=("rainfall", "mean"),
        )
        .reset_index()
    )

    # Merge
    merged = crop_df.merge(soil_agg, on="Crop_lower", how="left")
    merged.drop(columns=["Crop_lower"], inplace=True)

    matched = merged["N_mean"].notna().sum()
    total = len(merged)
    print(f"[INFO] Merged soil features: {matched}/{total} rows matched "
          f"({matched/total*100:.1f}%)")

    # Fill unmatched soil features with overall median
    soil_cols = ["N_mean", "P_mean", "K_mean", "ph_mean",
                 "temperature_mean", "humidity_mean", "soil_rainfall_mean"]
    for col in soil_cols:
        if merged[col].isnull().sum() > 0:
            merged[col].fillna(merged[col].median(), inplace=True)

    return merged


# ──────────────────────────────────────────────
#  5. FEATURE ENGINEERING
# ──────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features:
    - NPK_Index          : combined soil nutrient score
    - Rainfall_Fert_Ratio: annual rainfall / (fertilizer + 1)
    - Yield_per_Fert     : yield efficiency relative to fertilizer
    - Log_Area           : log-transform of area for normality
    - Log_Production     : log-transform of production
    - Season_encoded     : label-encoded season
    - Crop_encoded       : label-encoded crop
    - State_encoded      : label-encoded state
    """
    df = df.copy()

    # NPK Index (geometric-mean style)
    df["NPK_Index"] = (df["N_mean"] + df["P_mean"] + df["K_mean"]) / 3.0

    # Rainfall to Fertilizer ratio
    df["Rainfall_Fert_Ratio"] = df["Annual_Rainfall"] / (df["Fertilizer"] + 1)

    # Log transforms (add 1 to avoid log(0))
    df["Log_Area"] = np.log1p(df["Area"])
    df["Log_Production"] = np.log1p(df["Production"])

    # Label-encode categorical columns
    label_encoders = {}
    for col in ["Season", "Crop", "State"]:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col])
        label_encoders[col] = le

    print(f"[INFO] Engineered {6} new features. Final shape: {df.shape}")
    return df, label_encoders


# ──────────────────────────────────────────────
#  6. FEATURE SCALING
# ──────────────────────────────────────────────
def scale_features(df: pd.DataFrame, columns: list) -> tuple:
    """
    Apply StandardScaler to selected numeric *columns*.
    Returns (df_scaled, fitted_scaler).
    """
    df = df.copy()
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    print(f"[INFO] Scaled {len(columns)} features with StandardScaler")
    return df, scaler


# ──────────────────────────────────────────────
#  7. FULL PREPROCESSING PIPELINE
# ──────────────────────────────────────────────
def run_preprocessing_pipeline(data_dir: str = "data/raw"):
    """
    Execute the complete preprocessing pipeline end-to-end.

    Returns:
        master_df        : cleaned, merged, feature-engineered DataFrame
        label_encoders   : dict of fitted LabelEncoders
    """
    print("=" * 60)
    print(" STEP 1/5: Loading Datasets")
    print("=" * 60)
    crop_df, soil_df = load_datasets(data_dir)

    print("\n" + "=" * 60)
    print(" STEP 2/5: Cleaning Data")
    print("=" * 60)
    crop_df = clean_crop_data(crop_df)
    soil_df = clean_soil_data(soil_df)

    print("\n" + "=" * 60)
    print(" STEP 3/5: Treating Outliers (IQR)")
    print("=" * 60)
    # Treat outliers on key numeric columns
    outlier_cols = ["Area", "Production", "Annual_Rainfall",
                    "Fertilizer", "Pesticide", "Yield"]
    crop_df = treat_outliers_iqr(crop_df, columns=outlier_cols)

    print("\n" + "=" * 60)
    print(" STEP 4/5: Merging Datasets")
    print("=" * 60)
    master_df = merge_datasets(crop_df, soil_df)

    print("\n" + "=" * 60)
    print(" STEP 5/5: Feature Engineering")
    print("=" * 60)
    master_df, label_encoders = engineer_features(master_df)

    # Save processed dataset
    os.makedirs("data/processed", exist_ok=True)
    master_df.to_csv("data/processed/master_dataset.csv", index=False)
    print(f"\n[INFO] Saved master_dataset.csv -> data/processed/ "
          f"({master_df.shape[0]} rows x {master_df.shape[1]} cols)")

    return master_df, label_encoders


# ──────────────────────────────────────────────
if __name__ == "__main__":
    df, encoders = run_preprocessing_pipeline()
    print("\nFinal columns:", df.columns.tolist())
    print(df.head())
