"""
Data preprocessing, custom PPR/IDP fantasy points calculation, and VOR computation.
"""

from typing import Dict, List, Optional
import pandas as pd
from .config import EXCLUDED_POSITIONS, POSITION_MAP, SCORING_WEIGHTS, REPLACEMENT_RANKS


def clean_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out non-fantasy positions and maps granular roles to core fantasy groups.
    """
    clean_df = df.copy()
    # Filter out excluded positions
    clean_df = clean_df[~clean_df['Position'].isin(EXCLUDED_POSITIONS)]
    # Remap sub-positions
    clean_df['Position'] = clean_df['Position'].replace(POSITION_MAP)
    return clean_df.reset_index(drop=True)


def calculate_fantasy_points(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Applies custom fantasy scoring formula (Standard PPR + IDP) to compute total points.
    """
    if weights is None:
        weights = SCORING_WEIGHTS

    scored_df = df.copy()
    points = 0.0
    for col, weight in weights.items():
        if col in scored_df.columns:
            points += scored_df[col].fillna(0.0) * weight

    scored_df['CalculatedFantasyPoints_StdPPPRIDP'] = points.round(2)
    return scored_df


def calculate_vor(
    df: pd.DataFrame,
    replacement_ranks: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    """
    Calculates Value Over Replacement (VOR) for each player relative to position replacement rank.
    """
    if replacement_ranks is None:
        replacement_ranks = REPLACEMENT_RANKS

    df_vor = df.copy()
    replacement_points: Dict[str, float] = {}

    for pos, rank in replacement_ranks.items():
        pos_df = df_vor[df_vor['Position'] == pos].sort_values(
            by='CalculatedFantasyPoints_StdPPPRIDP', ascending=False
        ).reset_index(drop=True)

        if len(pos_df) >= rank:
            # Baseline is the rank-th player (0-indexed rank - 1)
            val = pos_df.loc[rank - 1, 'CalculatedFantasyPoints_StdPPPRIDP']
        elif len(pos_df) > 0:
            val = pos_df['CalculatedFantasyPoints_StdPPPRIDP'].min()
        else:
            val = 0.0

        replacement_points[pos] = val

    df_vor['replacement_points'] = df_vor['Position'].map(replacement_points)
    df_vor['VOR'] = (
        df_vor['CalculatedFantasyPoints_StdPPPRIDP'] - df_vor['replacement_points']
    ).round(2)

    return df_vor


def filter_outliers(
    df: pd.DataFrame,
    min_vor: float = -50.0
) -> pd.DataFrame:
    """
    Filters out extreme negative outliers (players with minimal/no snaps or deep negative production).
    """
    return df[df['VOR'] >= min_vor].copy().reset_index(drop=True)
