"""
Module 7: Output Visualisation & Reporting
===========================================
Generates summary charts including feature importance, soil cluster
distribution, and actual vs. predicted yield comparisons for districts.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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


def feature_importance_chart(importance_df: pd.DataFrame, top_n: int = 15):
    """
    Generate horizontal bar chart of features sorted by importance.
    """
    # Select top N features
    top_features = importance_df.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Gradient of blues/greens for feature importance
    palette = sns.color_palette("viridis", len(top_features))

    sns.barplot(
        x="Importance",
        y="Feature",
        data=top_features,
        palette=palette,
        ax=ax,
        edgecolor='white'
    )

    ax.set_title(f"Random Forest Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    ax.set_xlabel("Relative Importance Score", fontsize=12)
    ax.set_ylabel("Agricultural & Environmental Feature", fontsize=12)
    _save_fig("viz_feature_importance.png")


def cluster_distribution_chart(df: pd.DataFrame):
    """
    Produce a pie chart showing district records distribution by soil cluster.
    """
    if "Soil_Cluster" not in df.columns:
        print("[WARNING] 'Soil_Cluster' not in DataFrame. Skipping cluster distribution chart.")
        return

    cluster_counts = df["Soil_Cluster"].value_counts().sort_index()
    labels = [f"Cluster {i}" for i in cluster_counts.index]

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = sns.color_palette("pastel", len(cluster_counts))

    ax.pie(
        cluster_counts,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        textprops={"fontsize": 12, "fontweight": "bold"},
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'antialiased': True}
    )

    ax.set_title("Distribution of Records across Soil Clusters", fontsize=14, fontweight="bold")
    _save_fig("viz_cluster_distribution.png")


def yield_comparison_chart(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, top_n: int = 20):
    """
    Produce a grouped bar chart of actual vs predicted yield by Crop/State (top N records).
    """
    # Create a temporary dataframe of true and predicted yields with State and Crop
    compare_df = pd.DataFrame({
        "State": df["State"],
        "Crop": df["Crop"],
        "Actual Yield": y_true,
        "Predicted Yield": y_pred
    })

    # Group by State and Crop and average
    grouped = compare_df.groupby(["State", "Crop"]).mean().reset_index()

    # Sort by actual yield and take top N
    grouped = grouped.sort_values(by="Actual Yield", ascending=False).head(top_n)

    # Label for x-axis
    grouped["Label"] = grouped["Crop"] + " (" + grouped["State"] + ")"

    # Melt to long format for Seaborn grouped bar plot
    melted = pd.melt(
        grouped,
        id_vars=["Label"],
        value_vars=["Actual Yield", "Predicted Yield"],
        var_name="Type",
        value_name="Yield (tonnes/ha)"
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    sns.barplot(
        x="Label",
        y="Yield (tonnes/ha)",
        hue="Type",
        data=melted,
        palette=["#4C72B0", "#DD8452"],
        ax=ax,
        edgecolor='white'
    )

    ax.set_title(f"Yield Comparison: Actual vs. Predicted (Top {top_n} Crops/States)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Crop (State)", fontsize=12)
    ax.set_ylabel("Yield (tonnes/hectare)", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Legend")
    _save_fig("viz_yield_comparison.png")


def run_visualization_pipeline(df: pd.DataFrame, importance_df: pd.DataFrame,
                               y_true: np.ndarray, y_pred: np.ndarray):
    """
    Run the full visualization pipeline.
    """
    print("=" * 60)
    print(" GENERATING FINAL VISUALISATIONS")
    print("=" * 60)

    feature_importance_chart(importance_df)
    cluster_distribution_chart(df)

    # Use the test subset of the original dataframe (last len(y_true) rows if not shuffled/split matched)
    # Since we know split_data keeps indexes, we can index df using the test target indexes!
    # Let's ensure this is passed properly from main.py
    yield_comparison_chart(df, y_true, y_pred)
