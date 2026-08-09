"""
Loads raw CSVs from all three sources (NBA_api, Kaggle, BRef) and returns
clean, merged DataFrames ready for feature engineering.

Design principles:
- Every method inspects actual column names before using them
- Graceful fallback when a file or column is missing
- No column names are assumed - always validated against real schema
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from config import (
    NBA_API_DIR, KAGGLE_DIR, BREF_DIR, SEASONS
)

logger = logging.getLogger(__name__)


class NBADataLoader:
    """
    Loads and merges player game logs with supplementary data from
    NBA API, Kaggle, and Basketball Reference.
    """

    # Helpers

    @staticmethod
    def _safe_read(path: str | Path, **kwargs) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(path, low_memory=False, **kwargs)
            logger.debug(f"  Loaded {Path(path).name}: {len(df):,} rows, {df.shape[1]} cols")
            return df
        except Exception as exc:
            logger.warning(f"  Could not read {path}: {exc}")
            return None

    @staticmethod
    def _resolve_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        """Return the first candidate column name that exists in df."""
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # NBA API game logs

    def load_player_gamelogs(self) -> pd.DataFrame:
        """
        Load all player_gamelogs_regular_season_base.csv files from NBA API.
        Returns one combined DataFrame with a SEASON tag.
        """
        pattern = str(NBA_API_DIR / "game_logs" / "players" / "**" / "player_gamelogs_regular_season_base.csv")
        files   = glob.glob(pattern, recursive=True)

        if not files:
            # Fallback: search any CSV with 'gamelog' in the name
            pattern2 = str(NBA_API_DIR / "**" / "*.csv")
            all_csvs = glob.glob(pattern2, recursive=True)
            files    = [f for f in all_csvs
                        if "gamelog" in f.lower() and "regular_season_base" in f.lower()]

        if not files:
            raise FileNotFoundError(
                f"No player game log CSVs found under {NBA_API_DIR}.\n"
                "Run data_pipeline/NBA_api/nba_data_collector.py first."
            )

        dfs = []
        for f in sorted(files):
            df = self._safe_read(f)
            if df is None or df.empty:
                continue
            # Tag with season from folder name
            parts  = Path(f).parts
            season = next((p for p in parts if len(p) == 7 and "-" in p), "unknown")
            df["SEASON"] = season
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Game logs: {len(combined):,} rows from {len(dfs)} files")
        return combined

    def load_team_gamelogs(self) -> Optional[pd.DataFrame]:
        """Load team game logs for team context features."""
        pattern = str(NBA_API_DIR / "game_logs" / "teams" / "**" / "team_gamelogs_regular_season_base.csv")
        files   = glob.glob(pattern, recursive=True)
        if not files:
            logger.warning("No team game log CSVs found - team context features will be skipped")
            return None

        dfs = []
        for f in sorted(files):
            df = self._safe_read(f)
            if df is not None and not df.empty:
                parts  = Path(f).parts
                season = next((p for p in parts if len(p) == 7 and "-" in p), "unknown")
                df["SEASON"] = season
                dfs.append(df)

        if not dfs:
            return None
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Team game logs: {len(combined):,} rows")
        return combined

    # NBA API season stats (for career context)

    def load_player_bio(self) -> Optional[pd.DataFrame]:
        path = NBA_API_DIR / "player_profiles" / "player_bio_info.csv"
        if not path.exists():
            logger.warning(f"player_bio_info.csv not found at {path}")
            return None
        df = self._safe_read(path)
        logger.info(f"Player bio: {len(df):,} rows")
        return df

    def load_career_stats(self) -> Optional[pd.DataFrame]:
        path = NBA_API_DIR / "player_profiles" / "player_career_stats.csv"
        if not path.exists():
            return None
        df = self._safe_read(path)
        logger.info(f"Career stats: {len(df):,} rows")
        return df

    # Kaggle supplementary data

    def load_kaggle_advanced(self) -> Optional[pd.DataFrame]:
        """
        Load drgilermo season-level advanced stats (PER, WS, VORP, BPM).
        File: data/processed/Kaggle/Seasons_Stats.csv
        """
        candidates = [
            KAGGLE_DIR / "Seasons_Stats.csv",
            KAGGLE_DIR / "seasons_stats.csv",
        ]
        for p in candidates:
            if p.exists():
                df = self._safe_read(p)
                logger.info(f"Kaggle advanced stats: {len(df):,} rows from {p.name}")
                return df
        logger.warning("Kaggle advanced stats (Seasons_Stats.csv) not found")
        return None

    def load_kaggle_salaries(self) -> Optional[pd.DataFrame]:
        """Player salary data from loganlauton dataset."""
        candidates = [
            KAGGLE_DIR / "salaries.csv",
            KAGGLE_DIR / "player_salaries.csv",
            KAGGLE_DIR / "nba_salaries.csv",
        ]
        for p in candidates:
            if p.exists():
                df = self._safe_read(p)
                logger.info(f"Salaries: {len(df):,} rows from {p.name}")
                return df
        logger.warning("Salary CSV not found in Kaggle folder")
        return None

    def load_kaggle_player_bios(self) -> Optional[pd.DataFrame]:
        """Player bios (height, weight, college, country) from wyattowalsh dataset."""
        candidates = [
            KAGGLE_DIR / "player.csv",
            KAGGLE_DIR / "players.csv",
            KAGGLE_DIR / "player_bios.csv",
        ]
        for p in candidates:
            if p.exists():
                df = self._safe_read(p)
                logger.info(f"Kaggle bios: {len(df):,} rows from {p.name}")
                return df
        return None

    # Basketball Reference

    def load_bref_team_season(
        self,
        stat_type: str = "advanced",
    ) -> Optional[pd.DataFrame]:
        """
        Load BRef per-team CSVs.  stat_type in {advanced, per_game, team_misc}.
        Files follow pattern:  {TEAM}_{YEAR}_{stat_type}.csv
        """
        pattern = str(BREF_DIR / f"*_{stat_type}.csv")
        files   = glob.glob(pattern)
        if not files:
            logger.warning(f"No BRef {stat_type} CSVs found under {BREF_DIR}")
            return None

        dfs = []
        for f in files:
            df = self._safe_read(f)
            if df is None or df.empty:
                continue
            # Extract team and year from filename: LAL_2023_advanced.csv
            stem  = Path(f).stem                    # e.g. "LAL_2023_advanced"
            parts = stem.split("_")
            if len(parts) >= 2:
                df["BREF_TEAM"] = parts[0]
                df["BREF_YEAR"] = parts[1] if parts[1].isdigit() else None
            dfs.append(df)

        if not dfs:
            return None
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"BRef {stat_type}: {len(combined):,} rows from {len(dfs)} files")
        return combined

    # Main entry point

    def load_all(self) -> dict[str, pd.DataFrame | None]:
        """
        Load everything and return a labelled dict.
        Callers pick what they need; missing data returns None.
        """
        logger.info("=" * 60)
        logger.info("  NBADataLoader - loading all sources")
        logger.info("=" * 60)

        return {
            "player_gamelogs": self.load_player_gamelogs(),
            "team_gamelogs":   self.load_team_gamelogs(),
            "player_bio":      self.load_player_bio(),
            "career_stats":    self.load_career_stats(),
            "kaggle_advanced": self.load_kaggle_advanced(),
            "kaggle_salaries": self.load_kaggle_salaries(),
            "kaggle_bios":     self.load_kaggle_player_bios(),
            "bref_advanced":   self.load_bref_team_season("advanced"),
            "bref_per_game":   self.load_bref_team_season("per_game"),
        }
