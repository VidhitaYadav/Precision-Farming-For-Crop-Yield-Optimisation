"""
Module 4: K-Means Soil Profiling (Clustering)
===============================================
Groups districts/crop-records into distinct soil profile categories
using K-Means clustering on soil nutrient and weather features.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURE_DIR = "outputs/figures"


def _save_fig(name: str):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path}")


# ──────────────────────────────────────────────
#  1. SELECT SOIL FEATURES FOR CLUSTERING
# ──────────────────────────────────────────────
SOIL_FEATURES = ["N_mean", "P_mean", "K_mean", "ph_mean",
                  "temperature_mean", "humidity_mean"]


def select_soil_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract soil-related features for clustering."""
    available = [c for c in SOIL_FEATURES if c in df.columns]
    print(f"  Using {len(available)} soil features: {available}")
    return df[available].copy()


# ──────────────────────────────────────────────
#  2. FIND OPTIMAL K (Elbow + Silhouette)
# ──────────────────────────────────────────────
def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 11)):
    """
    Determine the optimal number of clusters using:
    - Elbow method (inertia / WCSS)
    - Silhouette score
    """
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "#4C72B0"
    ax1.plot(list(k_range), inertias, "o-", color=color1, linewidth=2,
             label="Inertia (WCSS)")
    ax1.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax1.set_ylabel("Inertia (WCSS)", color=color1, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color1)

    color2 = "#C44E52"
    ax2 = ax1.twinx()
    ax2.plot(list(k_range), silhouettes, "s--", color=color2, linewidth=2,
             label="Silhouette Score")
    ax2.set_ylabel("Silhouette Score", color=color2, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle("Optimal K: Elbow Method & Silhouette Score",
                 fontsize=14, fontweight="bold")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    _save_fig("cluster_elbow_silhouette.png")

    # Choose k with highest silhouette
    best_k = list(k_range)[np.argmax(silhouettes)]
    best_sil = max(silhouettes)
    print(f"  Best k = {best_k} (Silhouette = {best_sil:.4f})")

    return best_k


# ──────────────────────────────────────────────
#  3. FIT K-MEANS
# ──────────────────────────────────────────────
def fit_kmeans(X_scaled: np.ndarray, k: int):
    """Train K-Means with *k* clusters; return the fitted model."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    km.fit(X_scaled)
    print(f"  K-Means fitted with k={k}, inertia={km.inertia_:.2f}")
    return km


# ──────────────────────────────────────────────
#  4. VISUALISE CLUSTERS (PCA)
# ──────────────────────────────────────────────
def visualize_clusters_pca(X_scaled: np.ndarray, labels: np.ndarray,
                           k: int):
    """
    Reduce features to 2D with PCA and scatter-plot the clusters.
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))
    palette = sns.color_palette("bright", k)

    for cluster_id in range(k):
        mask = labels == cluster_id
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[palette[cluster_id]], label=f"Cluster {cluster_id}",
                   alpha=0.5, s=15)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)",
                  fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)",
                  fontsize=12)
    ax.set_title("K-Means Soil Profile Clusters (PCA Projection)",
                 fontsize=14, fontweight="bold")
    ax.legend(title="Cluster", fontsize=10)
    _save_fig("cluster_pca_scatter.png")

    total_var = sum(pca.explained_variance_ratio_[:2]) * 100
    print(f"  PCA 2D explains {total_var:.1f}% of variance")


# ──────────────────────────────────────────────
#  5. CLUSTER SUMMARY STATISTICS
# ──────────────────────────────────────────────
def cluster_summary(df: pd.DataFrame, features: list):
    """
    Print per-cluster summary: centroid values, size, and dominant crop.
    """
    summary = df.groupby("Soil_Cluster")[features].mean()
    sizes = df["Soil_Cluster"].value_counts().sort_index()
    summary["Cluster_Size"] = sizes

    if "Crop" in df.columns:
        dominant_crop = (
            df.groupby("Soil_Cluster")["Crop"]
            .agg(lambda x: x.value_counts().index[0])
        )
        summary["Dominant_Crop"] = dominant_crop

    print("\n  Cluster Summary:")
    print(summary.to_string())

    os.makedirs(FIGURE_DIR, exist_ok=True)
    summary.to_csv(os.path.join(FIGURE_DIR, "cluster_summary.csv"))
    print(f"  [SAVED] cluster_summary.csv")

    return summary


# ──────────────────────────────────────────────
#  6. CLUSTER SIZE BAR CHART
# ──────────────────────────────────────────────
def cluster_size_chart(df: pd.DataFrame):
    """Bar chart of cluster sizes."""
    sizes = df["Soil_Cluster"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("bright", len(sizes))
    ax.bar(sizes.index.astype(str), sizes.values, color=colors,
           edgecolor="white")
    for i, v in enumerate(sizes.values):
        ax.text(i, v + max(sizes) * 0.01, str(v),
                ha="center", fontweight="bold")

    ax.set_title("Soil Cluster Sizes", fontsize=14, fontweight="bold")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Records")
    _save_fig("cluster_sizes.png")


# ──────────────────────────────────────────────
#  MAIN CLUSTERING RUNNER
# ──────────────────────────────────────────────
def run_clustering(df: pd.DataFrame, k: int = None) -> pd.DataFrame:
    """
    Execute the full K-Means soil profiling pipeline.

    Returns the DataFrame with a new 'Soil_Cluster' column.
    """
    print("=" * 60)
    print(" K-MEANS SOIL PROFILING")
    print("=" * 60)

    print("\n1. Selecting Soil Features")
    soil_df = select_soil_features(df)

    print("\n2. Scaling Features")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(soil_df)

    print("\n3. Finding Optimal K")
    if k is None:
        k = find_optimal_k(X_scaled)
    else:
        print(f"  Using user-specified k = {k}")
        # Still plot the elbow for reference
        find_optimal_k(X_scaled)

    print(f"\n4. Fitting K-Means (k={k})")
    km = fit_kmeans(X_scaled, k)

    print("\n5. Visualising Clusters (PCA)")
    visualize_clusters_pca(X_scaled, km.labels_, k)

    # Assign cluster labels to the DataFrame
    df = df.copy()
    df["Soil_Cluster"] = km.labels_

    print("\n6. Cluster Summary")
    available_features = [c for c in SOIL_FEATURES if c in df.columns]
    cluster_summary(df, available_features)

    print("\n7. Cluster Size Chart")
    cluster_size_chart(df)

    print(f"\n[INFO] Clustering complete. "
          f"Added 'Soil_Cluster' column with {k} clusters.")

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/master_dataset.csv")
    df = run_clustering(df)
