"""
Clustering module: Elbow analysis, Silhouette scoring, Gap Statistic, and K-Means modeling.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from .config import RANDOM_SEED, TIER_LABELS


def compute_elbow(
    X: np.ndarray,
    k_range: range = range(1, 11),
    random_state: int = RANDOM_SEED
) -> Tuple[List[int], List[float]]:
    """
    Calculates K-Means inertia across range of k for the Elbow Method.
    """
    inertias = []
    k_values = list(k_range)
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
    return k_values, inertias


def compute_silhouette(
    X: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = RANDOM_SEED
) -> Tuple[List[int], List[float]]:
    """
    Calculates average Silhouette scores across range of k.
    """
    scores = []
    k_values = list(k_range)
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        score = float(silhouette_score(X, labels))
        scores.append(score)
    return k_values, scores


def compute_gap_statistic(
    X: np.ndarray,
    n_refs: int = 10,
    max_clusters: int = 10,
    random_state: int = RANDOM_SEED
) -> Tuple[List[float], List[float]]:
    """
    Calculates the Gap Statistic by comparing log dispersion against uniform reference null distributions.
    """
    shape = X.shape
    tops = X.max(axis=0)
    bottoms = X.min(axis=0)
    dists = tops - bottoms
    np.random.seed(random_state)
    refs = np.random.rand(n_refs, shape[0], shape[1]) * dists + bottoms

    gaps = []
    deviations = []

    for k in range(1, max_clusters + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X)
        orig_dispersion = np.mean(np.min(pairwise_distances(X, km.cluster_centers_), axis=1))

        ref_disps = np.zeros(n_refs)
        for i in range(n_refs):
            km.fit(refs[i])
            ref_disp = np.mean(np.min(pairwise_distances(refs[i], km.cluster_centers_), axis=1))
            ref_disps[i] = ref_disp

        gap = float(np.log(np.mean(ref_disps)) - np.log(orig_dispersion))
        sdk = float(np.std(np.log(ref_disps)) * np.sqrt(1 + 1 / n_refs))

        gaps.append(gap)
        deviations.append(sdk)

    return gaps, deviations


def fit_kmeans_tiers(
    df: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = RANDOM_SEED,
    tier_labels: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Fits K-Means on VOR, sorts clusters by mean VOR (highest to lowest), and assigns ranked tier labels.
    """
    if tier_labels is None:
        tier_labels = TIER_LABELS

    result_df = df.copy()
    X = result_df[['VOR']].values

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    result_df['Cluster'] = kmeans.fit_predict(X)

    # Rank clusters by descending average VOR so Tier 0 / 1 is highest
    cluster_summary = result_df.groupby('Cluster')['VOR'].mean().sort_values(ascending=False)
    tier_map = {old: new for new, old in enumerate(cluster_summary.index)}
    result_df['Tier'] = result_df['Cluster'].map(tier_map)

    # Map human-readable tier label
    label_count = len(tier_labels)
    result_df['TierLabel'] = result_df['Tier'].apply(
        lambda t: tier_labels[t] if t < label_count else f"Tier {t + 1}"
    )

    return result_df
