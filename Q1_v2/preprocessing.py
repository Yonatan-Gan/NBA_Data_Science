"""
Cleans raw game logs before feature engineering.
All operations are logged and reversible for reproducibility.
"""

from __future__ import annotations
import logging
import re
from typing import Optional
import pandas as pd
import numpy as np
from config import MIN_GAMES, RANDOM_SEED

logger = logging.getLogger(__name__)


class GameLogPreprocessor:
    """Clean raw player game logs into a consistent tabular format."""

    # Expected columns after cleaning (superset - not all must exist)
    REQUIRED_COLS = ["PLAYER_ID", "GAME_DATE", "PTS", "MIN"]

    def __init__(self):
        self.drop_stats: dict[str, int] = {}

    # Individual cleaning steps

    @staticmethod
    def _parse_minutes(s: pd.Series) -> pd.Series:
        """
        Handle both numeric (35.5) and string ("35:30") minute formats.
        Returns float Series of decimal minutes.
        """
        if s.dtype in [float, int]:
            return pd.to_numeric(s, errors="coerce")

        def convert(x):
            if pd.isna(x):
                return np.nan
            x = str(x).strip()
            if ":" in x:
                parts = x.split(":")
                try:
                    return int(parts[0]) + int(parts[1]) / 60
                except ValueError:
                    return np.nan
            try:
                return float(x)
            except ValueError:
                return np.nan

        return s.apply(convert)

    def clean(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Full cleaning pipeline.  Returns a clean DataFrame.
        """
        logger.info(f"Preprocessing: {len(raw):,} raw rows")
        df = raw.copy()
        n0 = len(df)

        # 1. Required columns check
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Raw data missing required columns: {missing}\n"
                             f"Available: {list(df.columns)}")

        # 2. Parse dates
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        bad_dates = df["GAME_DATE"].isna().sum()
        if bad_dates > 0:
            logger.warning(f"  {bad_dates} rows with unparseable GAME_DATE - dropped")
        df = df.dropna(subset=["GAME_DATE"])

        # 3. Parse minutes
        df["MIN"] = self._parse_minutes(df["MIN"])

        # 4. Drop DNP rows (played 0 or NaN minutes)
        n_before_dnp = len(df)
        df = df[df["MIN"].notna() & (df["MIN"] > 0)].copy()
        self.drop_stats["dnp_rows"] = n_before_dnp - len(df)
        logger.info(f"  Dropped {self.drop_stats['dnp_rows']:,} DNP rows (MIN=0 or NaN)")

        # 5. Drop rows missing the prediction target
        df = df.dropna(subset=["PTS"])
        self.drop_stats["missing_pts"] = n0 - len(df) - self.drop_stats["dnp_rows"]

        # 6. Deduplicate on (PLAYER_ID, GAME_ID)
        id_col   = "GAME_ID" if "GAME_ID" in df.columns else None
        key_cols = ["PLAYER_ID", "GAME_ID"] if id_col else ["PLAYER_ID", "GAME_DATE"]
        n_before_dup = len(df)
        df = df.drop_duplicates(subset=key_cols)
        self.drop_stats["duplicates"] = n_before_dup - len(df)
        if self.drop_stats["duplicates"] > 0:
            logger.warning(f"  Dropped {self.drop_stats['duplicates']:,} duplicate rows")

        # 7. Numeric coercion for stat columns
        stat_cols = ["PTS","REB","AST","STL","BLK","TOV",
                     "FGA","FGM","FG_PCT","FG3A","FG3M","FG3_PCT",
                     "FTA","FTM","FT_PCT","PLUS_MINUS"]
        for col in stat_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 8. Sort chronologically
        df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

        # 9. Home/Away from MATCHUP string
        if "MATCHUP" in df.columns:
            df["is_home"] = df["MATCHUP"].str.contains(r"vs\.", regex=True, na=False).astype(int)
        else:
            df["is_home"] = 0
            logger.warning("  MATCHUP column not found - is_home set to 0")

        # 10. Filter players with too few games
        game_counts = df.groupby("PLAYER_ID")["GAME_DATE"].count()
        valid_ids   = game_counts[game_counts >= MIN_GAMES].index
        n_before_filter = len(df)
        df = df[df["PLAYER_ID"].isin(valid_ids)].copy()
        self.drop_stats["too_few_games"] = n_before_filter - len(df)
        logger.info(f"  Kept {df['PLAYER_ID'].nunique()} players with ≥{MIN_GAMES} games "
                    f"({self.drop_stats['too_few_games']:,} rows removed)")

        # 11. eFG%
        if all(c in df.columns for c in ["FGM","FGA","FG3M"]):
            df["eFG_PCT"] = (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"].replace(0, np.nan)

        # 12. Defragment
        df = df.copy()

        logger.info(f"  Clean result: {len(df):,} rows, "
                    f"{df['PLAYER_ID'].nunique()} players, "
                    f"{df['SEASON'].nunique() if 'SEASON' in df.columns else '?'} seasons")
        return df

    def clean_team_logs(self, raw: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Light cleaning of team game logs for context features."""
        if raw is None or raw.empty:
            return None
        df = raw.copy()
        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        df = df.dropna(subset=["GAME_DATE"]) if "GAME_DATE" in df.columns else df
        df = df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True) \
             if all(c in df.columns for c in ["TEAM_ID","GAME_DATE"]) else df
        return df
