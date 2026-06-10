"""
Module 3: Exploratory Data Analysis (EDA)
==========================================
Generates descriptive statistics and publication-quality visualisations
to explore distributions, correlations, and trends in the agricultural data.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


# Global style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURE_DIR = "outputs/figures"


def _save_fig(name: str):
    """Save current figure to the output directory."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path}")


# ──────────────────────────────────────────────
#  1. DESCRIPTIVE STATISTICS
# ──────────────────────────────────────────────
def descriptive_statistics(df: pd.DataFrame):
    """Print and save descriptive statistics for all numeric columns."""
    stats = df.describe().T
    stats["median"] = df.median(numeric_only=True)
    stats["skew"] = df.skew(numeric_only=True)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    stats.to_csv(os.path.join(FIGURE_DIR, "descriptive_statistics.csv"))
    print("  [SAVED] descriptive_statistics.csv")
    print(stats.to_string())
    return stats


# ──────────────────────────────────────────────
#  2. CORRELATION HEATMAP
# ──────────────────────────────────────────────
def correlation_heatmap(df: pd.DataFrame):
    """Generate a correlation heatmap of all numeric features."""
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(16, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                annot=True, fmt=".2f", linewidths=0.5,
                square=True, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation Heatmap of Agricultural Features",
                 fontsize=16, fontweight="bold", pad=20)
    _save_fig("eda_correlation_heatmap.png")


# ──────────────────────────────────────────────
#  3. YIELD DISTRIBUTION
# ──────────────────────────────────────────────
def yield_distribution(df: pd.DataFrame):
    """Histogram of overall yield distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw yield
    axes[0].hist(df["Yield"], bins=50, color="#4C72B0", edgecolor="white",
                 alpha=0.8)
    axes[0].set_title("Yield Distribution (Raw)", fontweight="bold")
    axes[0].set_xlabel("Yield (tonnes/hectare)")
    axes[0].set_ylabel("Frequency")

    # Log-transformed yield
    if "Log_Production" in df.columns:
        log_yield = np.log1p(df["Yield"])
        axes[1].hist(log_yield, bins=50, color="#DD8452", edgecolor="white",
                     alpha=0.8)
        axes[1].set_title("Yield Distribution (Log-transformed)",
                          fontweight="bold")
        axes[1].set_xlabel("log(1 + Yield)")
        axes[1].set_ylabel("Frequency")

    _save_fig("eda_yield_distribution.png")


# ──────────────────────────────────────────────
#  4. YIELD BY STATE — BOX PLOT
# ──────────────────────────────────────────────
def yield_by_state_boxplot(df: pd.DataFrame, top_n: int = 15):
    """Box plot of yield across the top-N states by median yield."""
    state_medians = df.groupby("State")["Yield"].median().nlargest(top_n)
    top_states = state_medians.index.tolist()
    subset = df[df["State"].isin(top_states)]

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(data=subset, x="State", y="Yield", order=top_states,
                palette="viridis", ax=ax)
    ax.set_title(f"Yield Distribution — Top {top_n} States",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("State")
    ax.set_ylabel("Yield (tonnes/ha)")
    ax.tick_params(axis="x", rotation=45)
    _save_fig("eda_yield_by_state_boxplot.png")


# ──────────────────────────────────────────────
#  5. YIELD BY CROP — BAR CHART
# ──────────────────────────────────────────────
def yield_by_crop_bar(df: pd.DataFrame, top_n: int = 15):
    """Bar chart of average yield for top-N crops."""
    crop_yield = (
        df.groupby("Crop")["Yield"]
        .mean()
        .nlargest(top_n)
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    crop_yield.plot(kind="barh", color=sns.color_palette("coolwarm", top_n),
                    edgecolor="white", ax=ax)
    ax.set_title(f"Average Yield — Top {top_n} Crops",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean Yield (tonnes/ha)")
    ax.set_ylabel("Crop")
    _save_fig("eda_yield_by_crop_bar.png")


# ──────────────────────────────────────────────
#  6. RAINFALL vs YIELD SCATTER
# ──────────────────────────────────────────────
def rainfall_yield_scatter(df: pd.DataFrame):
    """Scatter plot of Annual Rainfall vs Yield."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["Annual_Rainfall"], df["Yield"],
               alpha=0.3, s=10, c="#4C72B0")
    ax.set_title("Annual Rainfall vs Crop Yield",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Annual Rainfall (mm)")
    ax.set_ylabel("Yield (tonnes/ha)")
    _save_fig("eda_rainfall_vs_yield.png")


# ──────────────────────────────────────────────
#  7. TIME-SERIES TRENDS
# ──────────────────────────────────────────────
def time_series_trends(df: pd.DataFrame):
    """Plot year-wise trends for mean yield, rainfall, and production."""
    yearly = df.groupby("Crop_Year").agg(
        Mean_Yield=("Yield", "mean"),
        Mean_Rainfall=("Annual_Rainfall", "mean"),
        Total_Production=("Production", "sum"),
    ).reset_index()

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(yearly["Crop_Year"], yearly["Mean_Yield"],
                 marker="o", color="#4C72B0", linewidth=2)
    axes[0].set_title("Year-wise Mean Yield Trend",
                      fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Mean Yield (t/ha)")

    axes[1].bar(yearly["Crop_Year"], yearly["Mean_Rainfall"],
                color="#55A868", alpha=0.8)
    axes[1].set_title("Year-wise Mean Annual Rainfall",
                      fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Rainfall (mm)")

    axes[2].plot(yearly["Crop_Year"], yearly["Total_Production"],
                 marker="s", color="#C44E52", linewidth=2)
    axes[2].set_title("Year-wise Total Production",
                      fontsize=13, fontweight="bold")
    axes[2].set_ylabel("Total Production")
    axes[2].set_xlabel("Year")

    _save_fig("eda_time_series_trends.png")


# ──────────────────────────────────────────────
#  8. SEASON DISTRIBUTION
# ──────────────────────────────────────────────
def season_distribution(df: pd.DataFrame):
    """Pie chart showing the distribution of crop records by season."""
    season_counts = df["Season"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = sns.color_palette("Set2", len(season_counts))
    ax.pie(season_counts, labels=season_counts.index, autopct="%1.1f%%",
           colors=colors, startangle=140, textprops={"fontsize": 11})
    ax.set_title("Crop Records by Season", fontsize=14, fontweight="bold")
    _save_fig("eda_season_distribution.png")


# ──────────────────────────────────────────────
#  9. SOIL FEATURE DISTRIBUTIONS
# ──────────────────────────────────────────────
def soil_feature_distributions(df: pd.DataFrame):
    """Box plots for soil nutrient features (N, P, K, pH)."""
    soil_cols = [c for c in ["N_mean", "P_mean", "K_mean", "ph_mean"]
                 if c in df.columns]
    if not soil_cols:
        print("  [SKIP] No soil columns found for distribution plot")
        return

    fig, axes = plt.subplots(1, len(soil_cols), figsize=(4 * len(soil_cols), 6))
    if len(soil_cols) == 1:
        axes = [axes]

    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for i, col in enumerate(soil_cols):
        sns.boxplot(y=df[col], ax=axes[i], color=palette[i % len(palette)])
        axes[i].set_title(col.replace("_mean", "").upper(),
                          fontweight="bold")

    plt.suptitle("Soil Feature Distributions",
                 fontsize=15, fontweight="bold", y=1.02)
    _save_fig("eda_soil_distributions.png")


# ──────────────────────────────────────────────
#  MAIN EDA RUNNER
# ──────────────────────────────────────────────
def run_eda(df: pd.DataFrame):
    """Execute the full EDA pipeline and save all figures."""
    print("=" * 60)
    print(" EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\n1. Descriptive Statistics")
    descriptive_statistics(df)

    print("\n2. Correlation Heatmap")
    correlation_heatmap(df)

    print("\n3. Yield Distribution")
    yield_distribution(df)

    print("\n4. Yield by State (Box Plot)")
    yield_by_state_boxplot(df)

    print("\n5. Yield by Crop (Bar Chart)")
    yield_by_crop_bar(df)

    print("\n6. Rainfall vs Yield (Scatter)")
    rainfall_yield_scatter(df)

    print("\n7. Time-Series Trends")
    time_series_trends(df)

    print("\n8. Season Distribution")
    season_distribution(df)

    print("\n9. Soil Feature Distributions")
    soil_feature_distributions(df)

    print(f"\n[INFO] All EDA figures saved to {FIGURE_DIR}/")


if __name__ == "__main__":
    df = pd.read_csv("data/processed/master_dataset.csv")
    run_eda(df)
