"""
Configuration parameters for NFL Fantasy Player Clustering.
Defines scoring multipliers, position mappings, and replacement baselines.
"""

from typing import Dict, List

# Positions to exclude (cannot score fantasy points or non-fantasy offensive line / special teams)
EXCLUDED_POSITIONS: List[str] = ['LS', 'P', 'OL', 'G', 'C', 'OT', 'K']

# Standardization map for sub-positions into core fantasy scoring groups
POSITION_MAP: Dict[str, str] = {
    'FB': 'RB',
    'OLB': 'LB',
    'ILB': 'LB',
    'NT': 'DL',
    'DT': 'DL',
    'DE': 'DL',
    'CB': 'DB',
    'SS': 'DB',
    'FS': 'DB',
    'S': 'DB'
}

# Standard PPR + IDP Fantasy Scoring Multipliers
SCORING_WEIGHTS: Dict[str, float] = {
    'PassingYards': 0.04,
    'PassingTouchdowns': 4.0,
    'PassingInterceptions': -2.0,
    'RushingYards': 0.10,
    'RushingTouchdowns': 6.0,
    'ReceivingYards': 0.10,
    'ReceivingTouchdowns': 6.0,
    'Receptions': 1.0,
    'FumblesLost': -2.0,
    'Tackles': 1.5,
    'Sacks': 4.0,
    'Interceptions': 6.0,
    'ForcedFumbles': 3.0,
    'DefensiveTouchdowns': 6.0
}

# Value Over Replacement (VOR) baseline ranks by position
# Defines the N-th player at each position considered "replacement level" in standard 12-team leagues
REPLACEMENT_RANKS: Dict[str, int] = {
    'QB': 24,
    'RB': 48,
    'WR': 60,
    'TE': 24,
    'DL': 24,
    'LB': 36,
    'DB': 36
}

# Default human-readable labels for 4-tier K-Means clustering (ordered top to bottom)
TIER_LABELS: List[str] = [
    "Tier 1 — Elite",
    "Tier 2 — High-End Starters",
    "Tier 3 — Average Performers",
    "Tier 4 — Negative Performers"
]

# Random seed for reproducible K-Means clustering
RANDOM_SEED: int = 418
