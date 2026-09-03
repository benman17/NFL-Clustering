"""
CLI Pipeline Runner for NFL Player Performance & Archetype Clustering.
Executes data loading, preprocessing, VOR calculation, K-Means modeling, and export.
"""

import argparse
import os
import sys

# Configure UTF-8 encoding for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
from src.data_loader import load_data
from src.preprocessing import clean_positions, calculate_fantasy_points, calculate_vor, filter_outliers
from src.clustering import compute_elbow, compute_silhouette, compute_gap_statistic, fit_kmeans_tiers
from src.visualization import (
    plot_vor_distribution,
    plot_elbow,
    plot_silhouette,
    plot_gap_statistic,
    plot_ranked_clusters,
    plot_tier_counts
)


def run_pipeline(
    data_path: str = None,
    api_key: str = None,
    n_clusters: int = 4,
    random_state: int = 418,
    save_plots: bool = True,
    output_dir: str = "data/processed",
    figures_dir: str = "reports/figures"
):
    print("=" * 65)
    print("NFL FANTASY PLAYER PERFORMANCE & TIER CLUSTERING PIPELINE")
    print("=" * 65)

    # 1. Load Data
    print("\n[1/5] Ingesting player statistics...")
    df_raw = load_data(filepath=data_path, api_key=api_key)
    print(f"      Loaded {len(df_raw)} raw player records.")

    # 2. Preprocess & Clean
    print("[2/5] Cleaning positions and computing fantasy scoring...")
    df_clean = clean_positions(df_raw)
    df_scored = calculate_fantasy_points(df_clean)
    print(f"      Filtered non-scoring roles -> {len(df_scored)} eligible fantasy players.")

    # 3. Value Over Replacement (VOR)
    print("[3/5] Calculating Value Over Replacement (VOR)...")
    df_vor = calculate_vor(df_scored)
    df_filtered = filter_outliers(df_vor, min_vor=-50.0)
    print(f"      Active evaluation pool: {len(df_filtered)} players (outliers pruned).")

    # 4. K-Means Tier Clustering
    print(f"[4/5] Executing K-Means clustering (k={n_clusters}, seed={random_state})...")
    df_tiered = fit_kmeans_tiers(
        df_filtered,
        n_clusters=n_clusters,
        random_state=random_state
    )

    # 5. Export Processed Results
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "nfl_player_tiers_2024.csv")
    cols_to_export = [
        'Name', 'Team', 'Position', 'CalculatedFantasyPoints_StdPPPRIDP',
        'replacement_points', 'VOR', 'Cluster', 'Tier', 'TierLabel'
    ]
    export_df = df_tiered[[c for c in cols_to_export if c in df_tiered.columns]]
    export_df.to_csv(out_file, index=False)
    print(f"      Exported tiered dataset to: {out_file}")

    # Summary Statistics per Tier
    print("\n[+] TIER SUMMARY STATISTICS:")
    tier_summary = df_tiered.groupby('TierLabel').agg(
        Players=('Name', 'count'),
        Avg_Points=('CalculatedFantasyPoints_StdPPPRIDP', 'mean'),
        Avg_VOR=('VOR', 'mean'),
        Min_VOR=('VOR', 'min'),
        Max_VOR=('VOR', 'max')
    ).round(2)
    print(tier_summary.to_string())

    # Top Players per Tier Preview
    print("\n[*] SAMPLE ELITE PLAYERS (Tier 1):")
    t1_players = df_tiered[df_tiered['Tier'] == 0].sort_values('VOR', ascending=False).head(8)
    for _, p in t1_players.iterrows():
        print(f"   - {p['Name']} ({p['Position']} - {p['Team']}): {p['CalculatedFantasyPoints_StdPPPRIDP']} pts | VOR: +{p['VOR']}")

    # 6. Generate Diagnostics & Visualizations if requested
    if save_plots:
        print(f"\n[5/5] Generating publication-quality charts in {figures_dir}/...")
        os.makedirs(figures_dir, exist_ok=True)
        plot_vor_distribution(df_filtered, os.path.join(figures_dir, "vor_distribution.png"))
        
        # Diagnostic evaluations
        X = df_filtered[['VOR']].values
        k_vals, inertias = compute_elbow(X)
        plot_elbow(k_vals, inertias, os.path.join(figures_dir, "elbow_method.png"))

        k_vals, silhouettes = compute_silhouette(X)
        plot_silhouette(k_vals, silhouettes, os.path.join(figures_dir, "silhouette_score.png"))

        gaps, _ = compute_gap_statistic(X, n_refs=5, max_clusters=10)
        plot_gap_statistic(gaps, os.path.join(figures_dir, "gap_statistic.png"))

        plot_ranked_clusters(df_tiered, os.path.join(figures_dir, "fantasy_player_tiers_plasma.png"))
        plot_tier_counts(df_tiered, os.path.join(figures_dir, "players_per_tier_bar.png"))
        print("      Diagnostic and tier figures successfully generated.")

    print("\nPipeline execution complete successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="NFL Player Performance & Fantasy Tier Clustering CLI"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to raw NFL player stats CSV or JSON."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional SportsData.io API Key."
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=4,
        help="Number of K-Means clusters (default: 4)."
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=418,
        help="Random seed for reproducibility (default: 418)."
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable generating diagnostic and tier figures."
    )
    args = parser.parse_args()

    run_pipeline(
        data_path=args.data_path,
        api_key=args.api_key,
        n_clusters=args.clusters,
        random_state=args.random_state,
        save_plots=not args.no_plots
    )
