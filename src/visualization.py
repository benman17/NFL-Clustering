"""
Publication-quality data visualizations for NFL player clusters and model diagnostics.
"""

import os
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd


def plot_vor_distribution(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """Plots histogram and KDE of Value Over Replacement."""
    plt.figure(figsize=(10, 4))
    sns.histplot(df['VOR'], bins=50, kde=True, color='#38bdf8')
    plt.title('Value Over Replacement (VOR) Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('VOR')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_elbow(
    k_values: List[int],
    inertias: List[float],
    save_path: Optional[str] = None
) -> None:
    """Plots the Elbow Method curve."""
    plt.figure(figsize=(8, 4))
    plt.plot(k_values, inertias, marker='o', color='#38bdf8', linewidth=2)
    plt.title('Elbow Method (Sum of Squared Errors)', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (SSE)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_silhouette(
    k_values: List[int],
    scores: List[float],
    save_path: Optional[str] = None
) -> None:
    """Plots Silhouette Scores across k."""
    plt.figure(figsize=(8, 4))
    plt.plot(k_values, scores, marker='o', color='#34d399', linewidth=2)
    plt.title('Silhouette Analysis across Cluster Counts', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Silhouette Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_gap_statistic(
    gaps: List[float],
    save_path: Optional[str] = None
) -> None:
    """Plots the Gap Statistic."""
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(gaps) + 1), gaps, marker='o', color='#818cf8', linewidth=2)
    plt.title('Gap Statistic Analysis', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Gap Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_ranked_clusters(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """Plots 1D scatter distribution of players across ranked tier clusters with legend."""
    unique_tiers = sorted(df['Tier'].unique())
    tier_labels_dict = dict(df.groupby('Tier')['TierLabel'].first())

    plt.figure(figsize=(16, 5))
    scatter = plt.scatter(
        df['VOR'],
        [0] * len(df),
        c=df['Tier'],
        cmap='plasma',
        alpha=0.7,
        s=40
    )

    plt.yticks([])
    plt.grid(True, alpha=0.3)
    plt.xlabel('Value Over Replacement (VOR)', fontsize=11, fontweight='bold')
    plt.title('NFL Fantasy Player Tiers (Ranked K-Means Clustering)', fontsize=13, fontweight='bold')

    norm = plt.Normalize(min(unique_tiers), max(unique_tiers))
    legend_handles = [
        mpatches.Patch(color=plt.cm.plasma(norm(t)), label=tier_labels_dict.get(t, f"Tier {t}"))
        for t in unique_tiers
    ]

    plt.legend(
        handles=legend_handles,
        title='Cluster Tiers',
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(unique_tiers),
        frameon=True
    )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_tier_counts(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """Plots bar chart of player counts per tier."""
    counts = df['TierLabel'].value_counts()
    plt.figure(figsize=(9, 4.5))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette='plasma', legend=False)
    plt.title('Player Distribution per Clustered Fantasy Tier', fontsize=12, fontweight='bold')
    plt.xlabel('Fantasy Tier', fontsize=10)
    plt.ylabel('Player Count', fontsize=10)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
