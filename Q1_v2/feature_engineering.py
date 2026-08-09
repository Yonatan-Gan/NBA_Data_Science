"""
Builds all feature groups from clean game logs.
Each group is a standalone method so it can be toggled for ablation studies.

Feature groups:
  1. player_form       - rolling stats, EWMA, streaks, consistency
  2. schedule          - rest days, B2B, 3-in-4, home/away streaks
  3. opponent          - rolling opp defensive stats
  4. team_context      - rolling team offensive/defensive context
  5. career            - age, experience, career averages

All rolling windows are SHIFTED by 1 to prevent data leakage.
"""

from __future__ import annotations
import logging
from typing import Optional
import numpy as np
import pandas as pd
from config import WINDOWS, FEATURE_GROUPS

logger = logging.getLogger(__name__)


def _shift_roll(series: pd.Series, window: int, func: str = "mean") -> pd.Series:
    """Shift-then-roll to prevent leakage."""
    shifted = series.shift(1)
    if func == "mean":   return shifted.rolling(window, min_periods=1).mean()
    if func == "std":    return shifted.rolling(window, min_periods=2).std()
    if func == "median": return shifted.rolling(window, min_periods=1).median()
    if func == "max":    return shifted.rolling(window, min_periods=1).max()
    if func == "min":    return shifted.rolling(window, min_periods=1).min()
    raise ValueError(f"Unknown func: {func}")


class FeatureEngineer:
    """
    Transform a clean game-log DataFrame into a feature matrix.
    Call build() to run all groups, or individual methods for ablation.
    """

    def __init__(
        self,
        player_logs: pd.DataFrame,
        team_logs:   Optional[pd.DataFrame] = None,
        player_bio:  Optional[pd.DataFrame] = None,
        career_stats: Optional[pd.DataFrame] = None,
    ):
        self.df          = player_logs.copy()
        self.team_logs   = team_logs
        self.player_bio  = player_bio
        self.career_stats = career_stats
        self._built: set[str] = set()

    #  GROUP 1: PLAYER FORM

    def build_player_form(self) -> "FeatureEngineer":
        """
        Rolling averages, EWMA, std, trend, hot/cold streaks.
        All features use shift(1) so only past games inform the prediction.
        """
        df  = self.df
        pid = "PLAYER_ID"

        for col in ["PTS", "MIN", "REB", "AST", "TOV", "FG_PCT", "FG3_PCT",
                    "FT_PCT", "eFG_PCT"]:
            if col not in df.columns:
                continue
            grp = df.groupby(pid)[col]

            # Rolling mean
            for w in WINDOWS:
                df[f"{col}_roll{w}"] = grp.transform(lambda x, w=w: _shift_roll(x, w, "mean"))

            # Rolling std (consistency proxy)
            df[f"{col}_std5"]  = grp.transform(lambda x: _shift_roll(x, 5, "std"))

            # Exponentially weighted moving average (more weight on recent games)
            df[f"{col}_ewma5"] = grp.transform(
                lambda x: x.shift(1).ewm(span=5, adjust=False).mean()
            )

        # USG%
        # Usage rate is the % of team plays used by the player while on floor.
        # If a 'USG_PCT' column exists use it; otherwise approximate from FGA.
        usg_col = "USG_PCT" if "USG_PCT" in df.columns else None
        if usg_col:
            grp = df.groupby(pid)[usg_col]
            df["USG_roll3"] = grp.transform(lambda x: _shift_roll(x, 3, "mean"))

        # Season-to-date average (expanding window, shifted)
        df["season_avg_pts"] = (
            df.groupby([pid, "SEASON"])["PTS"]
              .transform(lambda x: x.expanding().mean().shift(1))
            if "SEASON" in df.columns else
            df.groupby(pid)["PTS"].transform(lambda x: x.expanding().mean().shift(1))
        )
        df["season_avg_min"] = (
            df.groupby([pid, "SEASON"])["MIN"]
              .transform(lambda x: x.expanding().mean().shift(1))
            if "SEASON" in df.columns else
            df.groupby(pid)["MIN"].transform(lambda x: x.expanding().mean().shift(1))
        )

        # Consistency score: inverse of pts volatility
        df["pts_volatility"]    = df.groupby(pid)["PTS"].transform(lambda x: _shift_roll(x, 10, "std"))
        df["consistency_score"] = 1 / (df["pts_volatility"] + 1)   # avoid div/0

        # Recent trend: slope of last 5 games
        def _trend(x: pd.Series) -> pd.Series:
            """Slope of a linear fit over the last 5 games (shifted)."""
            s = x.shift(1)
            out = pd.Series(np.nan, index=x.index)
            for i in range(4, len(s)):
                window = s.iloc[i-4:i+1].dropna()
                if len(window) >= 3:
                    out.iloc[i] = np.polyfit(range(len(window)), window, 1)[0]
            return out

        df["PTS_trend5"] = df.groupby(pid)["PTS"].transform(_trend)

        # Hot / cold streak
        # Hot if last 3-game avg > (season avg + 3 pts), cold if <(season avg - 3)
        df["pts_vs_season_avg"] = df["PTS_roll3"] - df["season_avg_pts"]
        df["hot_streak"]  = (df["pts_vs_season_avg"] >  3).astype(int)
        df["cold_streak"] = (df["pts_vs_season_avg"] < -3).astype(int)

        self._built.add("player_form")
        logger.info("  player_form features built")
        self.df = df
        return self

    #  GROUP 2: SCHEDULE CONTEXT

    def build_schedule(self) -> "FeatureEngineer":
        """Rest days, back-to-back, 3-in-4, home/away streaks."""
        df  = self.df
        pid = "PLAYER_ID"

        # Rest days
        df["rest_days"] = (
            df.groupby(pid)["GAME_DATE"]
              .diff().dt.days
              .fillna(7)
              .clip(upper=7)
        )

        # Back-to-back
        df["is_back_to_back"] = (df["rest_days"] == 1).astype(int)

        # 3 games in 4 nights
        def _three_in_four(dates: pd.Series) -> pd.Series:
            out = pd.Series(0, index=dates.index)
            dates_arr = dates.values
            for i in range(2, len(dates_arr)):
                span = (dates_arr[i] - dates_arr[i-2]) / np.timedelta64(1, "D")
                if span <= 3:
                    out.iloc[i] = 1
            return out

        df["is_3_in_4"] = df.groupby(pid)["GAME_DATE"].transform(_three_in_four)

        # Games in last 7 days
        def _games_in_7(dates: pd.Series) -> pd.Series:
            out = pd.Series(0, index=dates.index)
            for i in range(1, len(dates)):
                cutoff = dates.iloc[i] - pd.Timedelta(days=7)
                out.iloc[i] = (dates.iloc[:i] >= cutoff).sum()
            return out

        df["games_in_last_7"] = df.groupby(pid)["GAME_DATE"].transform(_games_in_7)

        # Home / away streaks
        if "is_home" in df.columns:
            def _streak(series: pd.Series) -> pd.Series:
                out = pd.Series(0, index=series.index)
                count = 0
                for i in range(len(series)):
                    if series.iloc[i] == 1:
                        count += 1
                        out.iloc[i] = count
                    else:
                        count = 0
                return out

            df["home_streak"] = df.groupby(pid)["is_home"].transform(_streak)
            df["away_streak"] = df.groupby(pid)["is_home"].transform(
                lambda x: _streak(1 - x)
            )

        self._built.add("schedule")
        logger.info("  schedule features built")
        self.df = df
        return self

    #  GROUP 3: OPPONENT CONTEXT

    def build_opponent(self) -> "FeatureEngineer":
        """
        Rolling opponent defensive quality derived from what TeamGameLogs actually contains.

        TeamGameLogs has per-game box scores (PTS, FGM, FGA, FG3M, FG3A, OREB, DREB, etc.)
        but NOT DEF_RATING or PACE - those are season-aggregate endpoints.

        So we compute defensive proxies from what IS available:
          - opp_pts_allowed_roll5:  rolling pts the opponent scored in their own games
                                    -> reflects offensive quality, NOT defensive
          - opp_eFG_roll5:          opponent's own shooting efficiency -> proxy for
                                    how good they are offensively (inverse = how hard
                                    they are to defend against)
          - opp_win_pct_roll5:      rolling win %, overall team quality signal
          - opp_pts_conceded_roll5: rolling avg pts scored against the opponent team,
                                    computed from the MATCHUP column - this IS a real
                                    defensive quality proxy

        The MATCHUP approach (opp_pts_conceded):
          For each team, find all games where another team played against them.
          Average the PTS the opposing team scored in those games.
          Shift by 1 so no leakage.
        """
        df  = self.df
        tl  = self.team_logs

        opp_tid_col = next(
            (c for c in ["OPP_TEAM_ID", "opp_team_id"] if c in df.columns), None
        )

        # Compute opponent features from team game logs
        if opp_tid_col is not None and tl is not None and not tl.empty:
            tl = tl.copy()
            if "GAME_DATE" in tl.columns:
                tl["GAME_DATE"] = pd.to_datetime(tl["GAME_DATE"], errors="coerce")
            tl = tl.dropna(subset=["GAME_DATE"]) if "GAME_DATE" in tl.columns else tl

            tid_col  = next((c for c in ["TEAM_ID","team_id"]   if c in tl.columns), None)
            pts_col  = next((c for c in ["PTS","pts"]           if c in tl.columns), None)
            fgm_col  = next((c for c in ["FGM","fgm"]          if c in tl.columns), None)
            fga_col  = next((c for c in ["FGA","fga"]          if c in tl.columns), None)
            fg3m_col = next((c for c in ["FG3M","fg3m"]        if c in tl.columns), None)
            wl_col   = next((c for c in ["WL","wl"]            if c in tl.columns), None)
            pm_col   = next((c for c in ["PLUS_MINUS","plus_minus"] if c in tl.columns), None)

            if tid_col and pts_col and "GAME_DATE" in tl.columns:
                tl = tl.sort_values([tid_col, "GAME_DATE"]).reset_index(drop=True)

                # Opponent offensive quality (rolling avg pts scored)
                tl["_opp_pts_roll5"] = tl.groupby(tid_col)[pts_col].transform(
                    lambda x: _shift_roll(x, 5, "mean")
                )

                # Opponent shooting efficiency (eFG%)
                if fgm_col and fga_col and fg3m_col:
                    tl["_opp_eFG"] = (
                        (tl[fgm_col] + 0.5 * tl[fg3m_col]) /
                        tl[fga_col].replace(0, np.nan)
                    )
                    tl["_opp_eFG_roll5"] = tl.groupby(tid_col)["_opp_eFG"].transform(
                        lambda x: _shift_roll(x, 5, "mean")
                    )

                # Opponent win % (overall strength)
                if wl_col:
                    tl["_win_flag"] = (tl[wl_col] == "W").astype(float)
                    tl["_opp_win_pct_roll5"] = tl.groupby(tid_col)["_win_flag"].transform(
                        lambda x: _shift_roll(x, 5, "mean")
                    )

                # Opponent net rating proxy (plus/minus rolling avg)
                if pm_col:
                    tl["_opp_net_roll5"] = tl.groupby(tid_col)[pm_col].transform(
                        lambda x: _shift_roll(x, 5, "mean")
                    )

                # Pts CONCEDED by opponent: best defensive proxy available
                # For each game, the opponent team conceded the pts scored by
                # the player's team.  We compute this as:
                #   pts_conceded_by_opp = pts scored by player_team in that game
                # We join player game PTS to team game log by GAME_DATE + OPP_TEAM_ID.
                # We can proxy this as: for each team, compute rolling avg pts
                # scored by opponents against them.  Since PLUS_MINUS = pts_scored -
                # pts_conceded per game, pts_conceded = pts_scored - PLUS_MINUS.
                if pts_col and pm_col:
                    tl["_pts_conceded"] = tl[pts_col] - tl[pm_col]
                    tl["_opp_pts_conceded_roll5"] = tl.groupby(tid_col)["_pts_conceded"].transform(
                        lambda x: _shift_roll(x, 5, "mean")
                    )

                # Build lookup: one row per (OPP_TEAM_ID, GAME_DATE)
                lookup_cols = [tid_col, "GAME_DATE", "_opp_pts_roll5"]
                for c in ["_opp_eFG_roll5","_opp_win_pct_roll5",
                           "_opp_net_roll5","_opp_pts_conceded_roll5"]:
                    if c in tl.columns:
                        lookup_cols.append(c)

                lookup = (
                    tl[lookup_cols]
                    .drop_duplicates(subset=[tid_col, "GAME_DATE"])
                    .rename(columns={tid_col: opp_tid_col})
                )

                df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
                n_before = len(df)
                df = df.merge(lookup, on=[opp_tid_col, "GAME_DATE"], how="left")

                # Rename to final feature names
                rename = {
                    "_opp_pts_roll5":          "opp_pts_scored_roll5",
                    "_opp_eFG_roll5":          "opp_eFG_roll5",
                    "_opp_win_pct_roll5":      "opp_win_pct_roll5",
                    "_opp_net_roll5":          "opp_net_rating_roll5",
                    "_opp_pts_conceded_roll5": "opp_pts_allowed_roll5",  # KEY defensive proxy
                }
                df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

                # opp_def_rating_roll5: use pts_conceded as the primary proxy
                # (higher pts allowed = weaker defence = easier to score against)
                if "opp_pts_allowed_roll5" in df.columns:
                    df["opp_def_rating_roll5"] = df["opp_pts_allowed_roll5"].fillna(112.0)
                    n_filled = df["opp_def_rating_roll5"].notna().sum()
                    logger.info(
                        f"  Opponent defensive proxy (pts_conceded) from team logs: "
                        f"{n_filled:,}/{n_before:,} rows filled ({n_filled/n_before*100:.1f}%)"
                    )
                else:
                    df["opp_def_rating_roll5"] = 112.0
                    logger.warning("  Could not compute opp_pts_conceded - falling back to 112.0")

            else:
                df["opp_def_rating_roll5"] = 112.0
                logger.warning("  Team logs missing TEAM_ID or PTS - using 112.0 fallback")

        else:
            df["opp_def_rating_roll5"] = 112.0
            logger.warning(
                "  OPP_DEF_RATING not available from any source - "
                "using league average 112.0. "
                "Ensure OPP_TEAM_ID exists in player logs and team_gamelogs are loaded."
            )

        # Fill any remaining columns with neutral constants
        defaults = {
            "opp_pace_roll5":       100.0,
            "opp_win_pct_roll5":    0.50,
            "opp_pts_allowed_roll5": 112.0,
            "opp_eFG_allowed_roll5": 0.52,
            "opp_eFG_roll5":         0.52,
            "opp_net_rating_roll5":  0.0,
        }
        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val

        self._built.add("opponent")
        logger.info("  opponent features built")
        self.df = df
        return self

        # Path 1: compute from team game logs (best)
        if opp_tid_col is not None and tl is not None and not tl.empty:
            tl = tl.copy()
            if "GAME_DATE" in tl.columns:
                tl["GAME_DATE"] = pd.to_datetime(tl["GAME_DATE"], errors="coerce")

            tid_col  = next((c for c in ["TEAM_ID","team_id"]       if c in tl.columns), None)
            def_col  = next((c for c in ["DEF_RATING","E_DEF_RATING","def_rating"] if c in tl.columns), None)
            pace_col = next((c for c in ["PACE","E_PACE","pace"]     if c in tl.columns), None)
            pts_col  = next((c for c in ["PTS","pts"]                if c in tl.columns), None)
            wl_col   = next((c for c in ["WL","wl"]                  if c in tl.columns), None)

            if tid_col and def_col and "GAME_DATE" in tl.columns:
                tl = tl.dropna(subset=[tid_col, "GAME_DATE", def_col])
                tl = tl.sort_values([tid_col, "GAME_DATE"]).reset_index(drop=True)

                # Rolling defensive rating for each team (shifted - no leakage)
                tl["_def_roll5"] = tl.groupby(tid_col)[def_col].transform(
                    lambda x: _shift_roll(x, 5, "mean")
                )
                if pace_col:
                    tl["_pace_roll5"] = tl.groupby(tid_col)[pace_col].transform(
                        lambda x: _shift_roll(x, 5, "mean")
                    )

                # Rolling win % (shifted)
                if wl_col:
                    tl["_win_flag"] = (tl[wl_col] == "W").astype(float)
                    tl["_win_pct_roll5"] = tl.groupby(tid_col)["_win_flag"].transform(
                        lambda x: _shift_roll(x, 5, "mean")
                    )

                # Rolling points allowed per game = proxy for eFG allowed
                if pts_col:
                    tl["_pts_roll5"] = tl.groupby(tid_col)[pts_col].transform(
                        lambda x: _shift_roll(x, 5, "mean")
                    )

                # Build lookup table: one row per (TEAM_ID, GAME_DATE)
                lookup_cols = [tid_col, "GAME_DATE", "_def_roll5"]
                if "_pace_roll5"   in tl.columns: lookup_cols.append("_pace_roll5")
                if "_win_pct_roll5" in tl.columns: lookup_cols.append("_win_pct_roll5")
                if "_pts_roll5"    in tl.columns: lookup_cols.append("_pts_roll5")

                lookup = tl[lookup_cols].drop_duplicates(subset=[tid_col, "GAME_DATE"])
                lookup = lookup.rename(columns={tid_col: opp_tid_col})

                # Merge onto player log: match opp team ID + game date
                df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
                df = df.merge(lookup, on=[opp_tid_col, "GAME_DATE"], how="left")

                # Rename to final feature names
                rename = {
                    "_def_roll5":     "opp_def_rating_roll5",
                    "_pace_roll5":    "opp_pace_roll5",
                    "_win_pct_roll5": "opp_win_pct_roll5",
                    "_pts_roll5":     "opp_pts_allowed_roll5",
                }
                df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

                n_filled = df["opp_def_rating_roll5"].notna().sum()
                n_total  = len(df)
                logger.info(
                    f"  Opponent DEF_RATING from team logs: "
                    f"{n_filled:,}/{n_total:,} rows filled ({n_filled/n_total*100:.1f}%)"
                )
                # Fill any gaps (early-season rows with no prior games) with league avg
                df["opp_def_rating_roll5"] = df["opp_def_rating_roll5"].fillna(112.0)

            else:
                logger.warning(
                    "  Team logs lack TEAM_ID or DEF_RATING column - "
                    "falling through to static opponent context"
                )
                df["opp_def_rating_roll5"] = 112.0

        # Path 2: static column already in player log
        elif any(c in df.columns for c in ["OPP_DEF_RATING", "opp_def_rating"]):
            opp_col = next(c for c in ["OPP_DEF_RATING","opp_def_rating"] if c in df.columns)
            df["opp_def_rating_roll5"] = df.groupby("PLAYER_ID")[opp_col].transform(
                lambda x: _shift_roll(x, 5, "mean")
            )
            logger.info("  Opponent DEF_RATING from static player-log column")

        # Path 3: league average fallback
        else:
            df["opp_def_rating_roll5"] = 112.0
            logger.warning(
                "  OPP_DEF_RATING not available from any source - "
                "using league average 112.0. "
                "Ensure team_gamelogs are loaded and contain DEF_RATING."
            )

        # Fill remaining opponent columns with neutral constants if not yet built
        if "opp_pace_roll5"        not in df.columns: df["opp_pace_roll5"]        = 100.0
        if "opp_win_pct_roll5"     not in df.columns: df["opp_win_pct_roll5"]     = 0.50
        if "opp_pts_allowed_roll5" not in df.columns: df["opp_pts_allowed_roll5"] = 112.0

        # Derived: eFG allowed proxy from pts allowed (normalised to [0,1])
        if "opp_eFG_allowed_roll5" not in df.columns:
            df["opp_eFG_allowed_roll5"] = (df["opp_pts_allowed_roll5"] / 220).clip(0.4, 0.65)

        self._built.add("opponent")
        logger.info("  opponent features built")
        self.df = df
        return self

    #  GROUP 4: TEAM CONTEXT

    def build_team_context(self) -> "FeatureEngineer":
        """
        Rolling team offensive / defensive / pace context from team game logs.
        Joins on (TEAM_ID, GAME_DATE) using the team logs provided at init.
        """
        df = self.df

        if self.team_logs is None or self.team_logs.empty:
            # Fill with neutral values
            df["team_off_rating_roll5"] = 112.0
            df["team_def_rating_roll5"] = 112.0
            df["team_net_rating_roll5"] = 0.0
            df["team_pace_roll5"]       = 100.0
            df["team_ast_pct_roll5"]    = 0.60
            logger.warning("  Team game logs not available - using neutral team context")
            self._built.add("team_context")
            self.df = df
            return self

        tl = self.team_logs.copy()
        if "GAME_DATE" in tl.columns:
            tl["GAME_DATE"] = pd.to_datetime(tl["GAME_DATE"], errors="coerce")
        tl = tl.dropna(subset=["GAME_DATE"]) if "GAME_DATE" in tl.columns else tl

        # Identify team stat columns (flexible naming)
        off_col  = next((c for c in ["OFF_RATING","E_OFF_RATING","off_rating"] if c in tl.columns), None)
        def_col  = next((c for c in ["DEF_RATING","E_DEF_RATING","def_rating"] if c in tl.columns), None)
        pace_col = next((c for c in ["PACE","E_PACE","pace"] if c in tl.columns), None)
        tid_col  = next((c for c in ["TEAM_ID","team_id"] if c in tl.columns), None)

        if not tid_col:
            logger.warning("  No TEAM_ID in team logs - skipping team context")
            df["team_off_rating_roll5"] = 112.0
            df["team_def_rating_roll5"] = 112.0
            df["team_net_rating_roll5"] = 0.0
            df["team_pace_roll5"]       = 100.0
            df["team_ast_pct_roll5"]    = 0.60
            self.df = df
            self._built.add("team_context")
            return self

        # Compute rolling team stats
        if off_col:
            tl["team_off_roll5"] = tl.groupby(tid_col)[off_col].transform(
                lambda x: _shift_roll(x, 5, "mean")
            )
        if def_col:
            tl["team_def_roll5"] = tl.groupby(tid_col)[def_col].transform(
                lambda x: _shift_roll(x, 5, "mean")
            )
        if pace_col:
            tl["team_pace_roll5"] = tl.groupby(tid_col)[pace_col].transform(
                lambda x: _shift_roll(x, 5, "mean")
            )

        # Merge on TEAM_ID + GAME_DATE
        tid_player = next((c for c in ["TEAM_ID","team_id"] if c in df.columns), None)
        date_col   = "GAME_DATE"

        if tid_player and date_col in df.columns and date_col in tl.columns:
            merge_cols = [c for c in ["team_off_roll5","team_def_roll5","team_pace_roll5"]
                          if c in tl.columns]
            if merge_cols:
                tl_sub = tl[[tid_col, date_col] + merge_cols].drop_duplicates(subset=[tid_col, date_col])
                df = df.merge(
                    tl_sub.rename(columns={tid_col: tid_player}),
                    on=[tid_player, date_col],
                    how="left",
                )
                rename_map = {
                    "team_off_roll5": "team_off_rating_roll5",
                    "team_def_roll5": "team_def_rating_roll5",
                    "team_pace_roll5": "team_pace_roll5",
                }
                df = df.rename(columns=rename_map)

        # Net rating and fill any remaining NaNs
        if "team_off_rating_roll5" in df.columns and "team_def_rating_roll5" in df.columns:
            df["team_net_rating_roll5"] = (
                df["team_off_rating_roll5"].fillna(112) - df["team_def_rating_roll5"].fillna(112)
            )
        else:
            df["team_net_rating_roll5"] = 0.0

        for col, val in [("team_off_rating_roll5", 112.0), ("team_def_rating_roll5", 112.0),
                         ("team_pace_roll5", 100.0), ("team_ast_pct_roll5", 0.60)]:
            if col not in df.columns:
                df[col] = val

        self._built.add("team_context")
        logger.info("  team_context features built")
        self.df = df
        return self

    #  GROUP 5: CAREER CONTEXT

    def build_career(self) -> "FeatureEngineer":
        """
        Age, experience, career averages, position.
        Sources: player_bio_info.csv and player_career_stats.csv.
        """
        df = self.df
        pid = "PLAYER_ID"

        # Age from BIRTHDATE
        if self.player_bio is not None and not self.player_bio.empty:
            bio = self.player_bio.copy()

            # Find person/player ID column
            id_col = next((c for c in ["PERSON_ID","PLAYER_ID","person_id"] if c in bio.columns), None)
            bd_col = next((c for c in ["BIRTHDATE","birthdate","birth_date"] if c in bio.columns), None)
            pos_col= next((c for c in ["POSITION","position","POS"] if c in bio.columns), None)
            ht_col = next((c for c in ["HEIGHT","height"] if c in bio.columns), None)
            wt_col = next((c for c in ["WEIGHT","weight"] if c in bio.columns), None)

            if id_col and bd_col:
                bio[bd_col] = pd.to_datetime(bio[bd_col], errors="coerce")
                bio_sub = bio[[id_col, bd_col]].copy()
                if pos_col: bio_sub[pos_col] = bio[pos_col]
                if ht_col:  bio_sub[ht_col]  = bio[ht_col]
                if wt_col:  bio_sub[wt_col]  = bio[wt_col]

                bio_sub = bio_sub.rename(columns={id_col: pid})
                df = df.merge(bio_sub, on=pid, how="left")

                df["player_age"] = (
                    (df["GAME_DATE"] - df[bd_col]).dt.days / 365.25
                ).round(1)
                df = df.drop(columns=[bd_col])

                if pos_col in df.columns:
                    df["POSITION"] = df[pos_col]
        else:
            df["player_age"] = np.nan

        if "player_age" not in df.columns:
            df["player_age"] = np.nan

        # Experience (seasons played before this game)
        if "SEASON" in df.columns:
            seasons_sorted = sorted(df["SEASON"].dropna().unique())
            season_rank    = {s: i for i, s in enumerate(seasons_sorted)}
            df["experience_years"] = df["SEASON"].map(season_rank).fillna(0)
        else:
            df["experience_years"] = 0

        # Career averages from career_stats.csv
        if self.career_stats is not None and not self.career_stats.empty:
            cs = self.career_stats.copy()
            cs_pid = next((c for c in ["PLAYER_ID","player_id"] if c in cs.columns), None)
            cs_pts = next((c for c in ["PTS","pts","career_pts"] if c in cs.columns), None)
            cs_min = next((c for c in ["MIN","min"] if c in cs.columns), None)

            if cs_pid and cs_pts:
                career_avg = (
                    cs.groupby(cs_pid)[[cs_pts] + ([cs_min] if cs_min else [])]
                      .mean()
                      .reset_index()
                      .rename(columns={cs_pid: pid, cs_pts: "career_avg_pts"})
                )
                if cs_min:
                    career_avg = career_avg.rename(columns={cs_min: "career_avg_min"})

                df = df.merge(career_avg, on=pid, how="left")

        for col in ["career_avg_pts", "career_avg_min", "career_efg"]:
            if col not in df.columns:
                df[col] = np.nan

        self._built.add("career")
        logger.info("  career features built")
        self.df = df
        return self

    #  MAIN ENTRY POINT

    def build(self, groups: list[str] | None = None) -> pd.DataFrame:
        """
        Build all (or a subset of) feature groups.
        groups=None -> build everything.
        groups=["player_form","schedule"] -> build only those two.
        """
        all_groups = list(FEATURE_GROUPS.keys())
        groups_to_build = groups if groups else all_groups

        for group in groups_to_build:
            if group not in self._built:
                getattr(self, f"build_{group}")()

        n_features = sum(
            1 for c in self.df.columns
            if c not in ["PLAYER_ID","GAME_ID","GAME_DATE","SEASON","PLAYER_NAME",
                         "TEAM_ID","TEAM_ABBREVIATION","MATCHUP","WL","PTS",
                         "REB","AST","STL","BLK","TOV","FGM","FGA","FTM","FTA",
                         "FG3M","FG3A","PLUS_MINUS","MIN"]
        )
        logger.info(f"  Feature engineering complete: ~{n_features} new columns built")
        return self.df

    def get_feature_names(self, groups: list[str] | None = None) -> list[str]:
        """Return ordered list of feature column names for given groups."""
        from config import FEATURE_GROUPS, ALL_FEATURES
        if groups is None:
            return [f for f in ALL_FEATURES if f in self.df.columns]
        selected = [f for g in groups for f in FEATURE_GROUPS.get(g, [])
                    if f in self.df.columns]
        return selected
