"""
NBA Data Collector for Data Science Project
============================================
Collects data from nba_api to support three research questions:
  1. Player next-game performance prediction
  2. Team chemistry & roster composition analysis
  3. Regular season vs. playoff performance comparison

Run this script on your local machine (not in a sandboxed environment).
All data is saved as CSVs inside a structured folder: nba_data/

Usage:
    pip install nba_api pandas
    python nba_data_collector.py

Estimated run time: 2-6 hours (due to rate-limiting delays).
You can run it in sections by toggling the COLLECT_* flags below.
"""

import os
import time
import logging
import pandas as pd
from datetime import datetime

# nba_api imports
from nba_api.stats.static import players as static_players, teams as static_teams
from nba_api.stats.endpoints import (
    # League-wide player stats
    LeagueDashPlayerStats,
    LeagueDashPlayerBioStats,
    LeagueHustleStatsPlayer,
    LeagueDashPlayerClutch,
    PlayerEstimatedMetrics,
    # League-wide team stats
    LeagueDashTeamStats,
    TeamEstimatedMetrics,
    LeagueStandingsV3,
    # Game logs (bulk)
    PlayerGameLogs,
    TeamGameLogs,
    LeagueGameLog,
    # Per-player deep dives
    PlayerCareerStats,
    CommonPlayerInfo,
    PlayerAwards,
    PlayerDashboardByLastNGames,
    # Per-team deep dives
    CommonTeamRoster,
    TeamDetails,
    TeamInfoCommon,
    TeamYearByYearStats,
    # Draft & bio
    DraftHistory,
    DraftCombineStats,
    DraftCombinePlayerAnthro,
    # Box scores (game-level)
    BoxScoreTraditionalV3,
    BoxScoreAdvancedV3,
    BoxScoreMiscV3,
    BoxScoreHustleV2,
    # Playoffs
    CommonPlayoffSeries,
)

#  CONFIG - toggle sections on/off and choose seasons to collect

SEASONS = [
    "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25",
]

# Set to False to skip a section if you want to run in batches
COLLECT_STATIC_DATA         = False   # players list, teams list, draft history (Done)
COLLECT_LEAGUE_SEASON_STATS = False  # season-level aggregated stats per season + season type (Done)
COLLECT_PLAYER_GAME_LOGS    = False   # every player's game-by-game log (bulk) (Done)
COLLECT_TEAM_GAME_LOGS      = False   # every team's game-by-game log (bulk) (Done)
COLLECT_BOX_SCORES          = False   # per-game box scores (slow - most data) (Failed)
COLLECT_PLAYER_PROFILES     = False   # career stats + bio for active players
COLLECT_ROSTER_DATA         = False   # team rosters per season
COLLECT_PLAYOFF_DATA        = False   # playoff-specific stats + series results

DELAY = 2         # seconds between API calls (respect rate limits)
OUTPUT_DIR = "../../../../../../../year 3/Semester B/מחט בערמת דאטה/NBA_Data/nba_data"  # root folder for all saved CSVs

#  SETUP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"nba_collector_{datetime.now().strftime('%Y%m%d_%H%M')}.log"),
    ],
)
log = logging.getLogger(__name__)


def mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save(df: pd.DataFrame, path: str, label: str = "") -> None:
    """Save a DataFrame to CSV, creating parent dirs as needed."""
    mkdir(os.path.dirname(path))
    df.to_csv(path, index=False)
    log.info(f"  Saved {label or os.path.basename(path)}  ({len(df):,} rows)  ->  {path}")


def safe_call(fn, *args, retries: int = 4, **kwargs):
    for attempt in range(retries):
        try:
            time.sleep(DELAY)

            # Call the endpoint with the supplied positional and keyword arguments.
            return fn(*args, **kwargs)

        except TypeError as e:
            log.error(f"Bad arguments for {fn.__name__}: {e}")
            return None  # do NOT retry

        except Exception as exc:
            wait = 10 * (2 ** attempt)
            log.warning(
                f"{fn.__name__} failed (attempt {attempt+1}): {exc}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)

    log.error(f"All retries exhausted for {fn.__name__}")
    return None


#  1.  STATIC / REFERENCE DATA

def collect_static_data():
    log.info("[1/8] Static Reference Data")
    base = mkdir(f"{OUTPUT_DIR}/reference")

    # Draft history (all years)
    log.info("  Fetching draft history...")
    result = safe_call(
        DraftHistory,
        league_id="00"
    )
    if result:
        save(result.get_data_frames()[0], f"{base}/draft_history.csv", "draft history")

    # All players (historical + active)
    all_players = static_players.get_players()
    save(pd.DataFrame(all_players), f"{base}/all_players.csv", "all players")

    # All 30 teams
    all_teams = static_teams.get_teams()
    save(pd.DataFrame(all_teams), f"{base}/all_teams.csv", "all teams")


#  2.  LEAGUE-WIDE SEASON STATS  (per season, regular + playoffs)

def collect_league_season_stats():
    log.info("[2/8] League-Wide Season Stats")

    for season in SEASONS:
        log.info(f"  Season: {season}")
        base = mkdir(f"{OUTPUT_DIR}/season_stats/{season}")

        for season_type in ["Regular Season", "Playoffs"]:
            st_slug = season_type.replace(" ", "_").lower()

            # Traditional player stats (Base, Advanced, Usage, Defense)
            for measure in ["Base", "Advanced", "Usage", "Defense", "Scoring"]:
                result = safe_call(
                    LeagueDashPlayerStats,
                    season=season,
                    season_type_all_star=season_type,
                    per_mode_detailed="PerGame",
                    measure_type_detailed_defense=measure,
                    last_n_games=0, month=0, opponent_team_id=0,
                    pace_adjust="N", plus_minus="N", rank="N", period=0,
                )
                if result:
                    df = result.get_data_frames()[0]
                    save(df, f"{base}/player_stats_{st_slug}_{measure.lower()}.csv",
                         f"player {measure} {season_type}")

            # Bio stats (age, height, weight, country, college)
            result = safe_call(
                LeagueDashPlayerBioStats,
                season=season,
                season_type_all_star=season_type,
                per_mode_simple="PerGame",
                league_id="00",
            )
            if result:
                save(result.get_data_frames()[0],
                     f"{base}/player_bio_{st_slug}.csv", f"player bio {season_type}")

            # Hustle stats (screens, deflections, loose balls, charges)
            result = safe_call(
                LeagueHustleStatsPlayer,
                season=season,
                season_type_all_star=season_type,
                per_mode_time="PerGame",
            )
            if result:
                save(result.get_data_frames()[0],
                     f"{base}/player_hustle_{st_slug}.csv", f"player hustle {season_type}")

            # Clutch stats
            result = safe_call(
                LeagueDashPlayerClutch,
                season=season,
                season_type_all_star=season_type,
                per_mode_detailed="PerGame",
                last_n_games=0, month=0, opponent_team_id=0,
                pace_adjust="N", plus_minus="N", rank="N", period=0,
                ahead_behind="Ahead or Behind",
                clutch_time="Last 5 Minutes",
                point_diff=5,
            )
            if result:
                save(result.get_data_frames()[0],
                     f"{base}/player_clutch_{st_slug}.csv", f"player clutch {season_type}")

            # Player estimated metrics (RAPTOR-style)
            result = safe_call(
                PlayerEstimatedMetrics,
                league_id="00",
                season=season,
                season_type=season_type,
            )
            if result:
                save(result.get_data_frames()[0],
                     f"{base}/player_estimated_metrics_{st_slug}.csv",
                     f"player estimated metrics {season_type}")

            # Team stats (Base, Advanced, Defense, Scoring)
            for measure in ["Base", "Advanced", "Defense", "Scoring"]:
                result = safe_call(
                    LeagueDashTeamStats,
                    season=season,
                    season_type_all_star=season_type,
                    per_mode_detailed="PerGame",
                    measure_type_detailed_defense=measure,
                    last_n_games=0, month=0, opponent_team_id=0,
                    pace_adjust="N", plus_minus="N", rank="N", period=0,
                )
                if result:
                    save(result.get_data_frames()[0],
                         f"{base}/team_stats_{st_slug}_{measure.lower()}.csv",
                         f"team {measure} {season_type}")

            # Team estimated metrics
            result = safe_call(
                TeamEstimatedMetrics,
                league_id="00",
                season=season,
                season_type=season_type,
            )
            if result:
                save(result.get_data_frames()[0],
                     f"{base}/team_estimated_metrics_{st_slug}.csv",
                     f"team estimated metrics {season_type}")

            # Standings
            result = safe_call(
                LeagueStandingsV3,
                league_id="00",
                season=season,
                season_type=season_type,
            )
            if result:
                save(result.get_data_frames()[0],
                     f"{base}/standings_{st_slug}.csv", f"standings {season_type}")


#  3.  PLAYER GAME LOGS  (every game, every player, bulk endpoint)

def collect_player_game_logs():
    log.info("[3/8] Player Game Logs")

    for season in SEASONS:
        log.info(f"  Season: {season}")
        base = mkdir(f"{OUTPUT_DIR}/game_logs/players/{season}")

        for season_type in ["Regular Season", "Playoffs"]:
            st_slug = season_type.replace(" ", "_").lower()

            for measure in ["Base", "Advanced", "Usage"]:
                result = safe_call(
                    PlayerGameLogs,
                    season_nullable=season,
                    season_type_nullable=season_type,
                    per_mode_simple_nullable="PerGame",
                    measure_type_player_game_logs_nullable=measure,
                )
                if result:
                    df = result.get_data_frames()[0]
                    save(df,
                         f"{base}/player_gamelogs_{st_slug}_{measure.lower()}.csv",
                         f"player gamelogs {measure} {season} {season_type}")


#  4.  TEAM GAME LOGS

def collect_team_game_logs():
    log.info("[4/8] Team Game Logs")

    for season in SEASONS:
        log.info(f"  Season: {season}")
        base = mkdir(f"{OUTPUT_DIR}/game_logs/teams/{season}")

        for season_type in ["Regular Season", "Playoffs"]:
            st_slug = season_type.replace(" ", "_").lower()

            for measure in ["Base", "Advanced"]:
                result = safe_call(
                    TeamGameLogs,
                    season_nullable=season,
                    season_type_nullable=season_type,
                    per_mode_simple_nullable="PerGame",
                    measure_type_player_game_logs_nullable=measure,
                )
                if result:
                    df = result.get_data_frames()[0]
                    save(df,
                         f"{base}/team_gamelogs_{st_slug}_{measure.lower()}.csv",
                         f"team gamelogs {measure} {season} {season_type}")


#  5.  BOX SCORES  (Traditional + Advanced, per game)

def _get_all_game_ids(season: str, season_type: str) -> list:
    """Return a sorted, deduplicated list of game IDs for a season."""
    result = safe_call(
        LeagueGameLog,
        season=season,
        season_type_all_star=season_type,
        league_id="00",
        player_or_team_abbreviation="T",
        direction="ASC",
        sorter="DATE",
        counter=0,
    )
    if result is None:
        return []
    df = result.get_data_frames()[0]
    return sorted(df["GAME_ID"].unique().tolist())


def collect_box_scores():
    log.info("[5/8] Box Scores")

    for season in SEASONS:
        for season_type in ["Regular Season", "Playoffs"]:
            st_slug = season_type.replace(" ", "_").lower()
            base = mkdir(f"{OUTPUT_DIR}/box_scores/{season}")

            log.info(f"  Fetching game IDs: {season} {season_type}...")
            game_ids = _get_all_game_ids(season, season_type)
            log.info(f"  Found {len(game_ids)} games")

            trad_rows, adv_rows, misc_rows = [], [], []

            for i, game_id in enumerate(game_ids):
                if i % 50 == 0:
                    log.info(f"  Box scores: {i}/{len(game_ids)} games...")

                # Traditional box score
                res = safe_call(BoxScoreTraditionalV3, game_id=game_id)
                if res:
                    dfs = res.get_data_frames()
                    dfs[0]["SEASON"] = season
                    dfs[0]["SEASON_TYPE"] = season_type
                    trad_rows.append(dfs[0])

                # Advanced box score
                res = safe_call(BoxScoreAdvancedV3, game_id=game_id)
                if res:
                    dfs = res.get_data_frames()
                    dfs[0]["SEASON"] = season
                    dfs[0]["SEASON_TYPE"] = season_type
                    adv_rows.append(dfs[0])

                # Misc box score (pts off turnovers, 2nd chance pts, etc.)
                res = safe_call(BoxScoreMiscV3, game_id=game_id)
                if res:
                    dfs = res.get_data_frames()
                    dfs[0]["SEASON"] = season
                    dfs[0]["SEASON_TYPE"] = season_type
                    misc_rows.append(dfs[0])

            # Merge and save all games for this season/type
            if trad_rows:
                save(pd.concat(trad_rows, ignore_index=True),
                     f"{base}/boxscore_traditional_{st_slug}.csv",
                     f"traditional box scores {season} {season_type}")
            if adv_rows:
                save(pd.concat(adv_rows, ignore_index=True),
                     f"{base}/boxscore_advanced_{st_slug}.csv",
                     f"advanced box scores {season} {season_type}")
            if misc_rows:
                save(pd.concat(misc_rows, ignore_index=True),
                     f"{base}/boxscore_misc_{st_slug}.csv",
                     f"misc box scores {season} {season_type}")


#  6.  PLAYER PROFILES  (career stats, bio, awards - active players only)

def collect_player_profiles():
    log.info("[6/8] Player Profiles (active players)")
    base = mkdir(f"{OUTPUT_DIR}/player_profiles")

    active = [p for p in static_players.get_players() if p["is_active"]]
    log.info(f"  Collecting profiles for {len(active)} active players...")

    bio_rows, career_rows, award_rows = [], [], []

    for i, player in enumerate(active):
        pid = player["id"]
        if i % 50 == 0:
            log.info(f"  Profiles: {i}/{len(active)}...")

        # Bio (height, weight, country, position, draft, college)
        res = safe_call(CommonPlayerInfo, player_id=pid)
        if res:
            dfs = res.get_data_frames()
            bio_rows.append(dfs[0])

        # Full career stats (per season, regular + playoffs)
        res = safe_call(PlayerCareerStats, player_id=pid, per_mode36="PerGame")
        if res:
            dfs = res.get_data_frames()
            for df_idx, label in [(0, "regular"), (2, "playoffs")]:
                if df_idx < len(dfs) and not dfs[df_idx].empty:
                    dfs[df_idx]["PLAYER_ID"] = pid
                    dfs[df_idx]["SEASON_TYPE"] = label
                    career_rows.append(dfs[df_idx])

        # Awards (All-Star, All-NBA, championships, etc.)
        res = safe_call(PlayerAwards, player_id=pid)
        if res:
            df = res.get_data_frames()[0]
            if not df.empty:
                award_rows.append(df)

    if bio_rows:
        save(pd.concat(bio_rows, ignore_index=True),
             f"{base}/player_bio_info.csv", "player bio info")
    if career_rows:
        save(pd.concat(career_rows, ignore_index=True),
             f"{base}/player_career_stats.csv", "player career stats")
    if award_rows:
        save(pd.concat(award_rows, ignore_index=True),
             f"{base}/player_awards.csv", "player awards")


#  7.  ROSTER DATA  (who was on each team each season)

def collect_roster_data():
    log.info("[7/8] Team Rosters")
    base = mkdir(f"{OUTPUT_DIR}/rosters")

    all_teams = static_teams.get_teams()
    all_roster_rows = []

    for season in SEASONS:
        log.info(f"  Rosters: {season}")
        for team in all_teams:
            tid = team["id"]
            res = safe_call(CommonTeamRoster, team_id=tid, season=season)
            if res:
                dfs = res.get_data_frames()
                roster_df = dfs[0]
                roster_df["SEASON"] = season
                roster_df["TEAM_ID"] = tid
                roster_df["TEAM_ABBREVIATION"] = team["abbreviation"]
                all_roster_rows.append(roster_df)

    if all_roster_rows:
        save(pd.concat(all_roster_rows, ignore_index=True),
             f"{base}/all_rosters.csv", "all team rosters")

    # Also save team year-by-year records
    log.info("  Team year-by-year stats...")
    yby_rows = []
    for team in all_teams:
        res = safe_call(TeamYearByYearStats, team_id=team["id"], per_mode_simple="PerGame")
        if res:
            df = res.get_data_frames()[0]
            df["TEAM_ID"] = team["id"]
            df["TEAM_NAME"] = team["full_name"]
            yby_rows.append(df)
    if yby_rows:
        save(pd.concat(yby_rows, ignore_index=True),
             f"{base}/team_year_by_year.csv", "team year-by-year")


#  8.  PLAYOFF-SPECIFIC DATA

def collect_playoff_data():
    log.info("[8/8] Playoff Data")
    base = mkdir(f"{OUTPUT_DIR}/playoffs")

    # Playoff series results (who played who, outcomes)
    series_rows = []
    for season in SEASONS:
        res = safe_call(CommonPlayoffSeries, league_id="00", season=season)
        if res:
            df = res.get_data_frames()[0]
            df["SEASON"] = season
            series_rows.append(df)
    if series_rows:
        save(pd.concat(series_rows, ignore_index=True),
             f"{base}/playoff_series.csv", "playoff series")



#  MAIN

if __name__ == "__main__":
    start = datetime.now()
    log.info("=" * 60)
    log.info("NBA Data Collector - starting")
    log.info(f"Seasons: {SEASONS}")
    log.info(f"Output:  {os.path.abspath(OUTPUT_DIR)}/")
    log.info("=" * 60)

    mkdir(OUTPUT_DIR)

    if COLLECT_STATIC_DATA:
        collect_static_data()

    if COLLECT_LEAGUE_SEASON_STATS:
        collect_league_season_stats()

    if COLLECT_PLAYER_GAME_LOGS:
        collect_player_game_logs()

    if COLLECT_TEAM_GAME_LOGS:
        collect_team_game_logs()

    if COLLECT_BOX_SCORES:
        collect_box_scores()

    if COLLECT_PLAYER_PROFILES:
        collect_player_profiles()

    if COLLECT_ROSTER_DATA:
        collect_roster_data()

    if COLLECT_PLAYOFF_DATA:
        collect_playoff_data()

    elapsed = datetime.now() - start
    log.info("=" * 60)
    log.info(f"Done! Total time: {elapsed}")
    log.info(f"Data saved in: {os.path.abspath(OUTPUT_DIR)}/")
    log.info("=" * 60)