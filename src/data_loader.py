"""
Data ingestion module for NFL Player Season Statistics.
Supports live fetching via SportsData.io API or local CSV/JSON loading.
"""

import os
from typing import Optional
import pandas as pd
import requests


def load_data(
    filepath: Optional[str] = None,
    api_key: Optional[str] = None,
    season: int = 2024
) -> pd.DataFrame:
    """
    Loads NFL player statistics from a local file or the SportsData.io API.

    Parameters:
        filepath (str, optional): Path to local CSV or JSON file.
        api_key (str, optional): SportsData.io API key.
        season (int): Season year to query if using API (default: 2024).

    Returns:
        pd.DataFrame: Raw player performance DataFrame.
    """
    if filepath and os.path.exists(filepath):
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            return pd.read_json(filepath)

    if api_key:
        url = f"https://api.sportsdata.io/v3/nfl/stats/json/PlayerSeasonStats/{season}?key={api_key}"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame({
            'Name': [p.get('Name') for p in data],
            'Team': [p.get('Team') for p in data],
            'Position': [p.get('Position') for p in data],
            'PassingYards': [p.get('PassingYards', 0.0) for p in data],
            'PassingTouchdowns': [p.get('PassingTouchdowns', 0.0) for p in data],
            'PassingInterceptions': [p.get('PassingInterceptions', 0.0) for p in data],
            'RushingYards': [p.get('RushingYards', 0.0) for p in data],
            'RushingTouchdowns': [p.get('RushingTouchdowns', 0.0) for p in data],
            'ReceivingYards': [p.get('ReceivingYards', 0.0) for p in data],
            'ReceivingTouchdowns': [p.get('ReceivingTouchdowns', 0.0) for p in data],
            'Receptions': [p.get('Receptions', 0.0) for p in data],
            'FumblesLost': [p.get('FumblesLost', 0.0) for p in data],
            'Tackles': [p.get('Tackles', 0.0) for p in data],
            'Sacks': [p.get('Sacks', 0.0) for p in data],
            'Interceptions': [p.get('Interceptions', 0.0) for p in data],
            'ForcedFumbles': [p.get('ForcedFumbles', 0.0) for p in data],
            'DefensiveTouchdowns': [p.get('DefensiveTouchdowns', 0.0) for p in data],
        })

    # Default fallback to sample data if exists
    default_sample = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'sample_nfl_player_stats_2024.csv')
    if os.path.exists(default_sample):
        return pd.read_csv(default_sample)

    raise ValueError(
        "No data source found. Please supply a valid filepath, API key, or generate data/raw/sample_nfl_player_stats_2024.csv."
    )
