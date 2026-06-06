#!/usr/bin/env python3
"""
NBA Data Collection & Feature Engineering Pipeline
===================================================
Downloads three Kaggle datasets and produces analysis-ready CSVs for:

  Q1  Player next-game performance prediction
      → data/processed/q1_player_game_logs.csv   (game-level rows, rolling features)

  Q2  Team chemistry from roster composition
      → data/processed/q2_team_season.csv         (team-season aggregates)
      → data/processed/q2_player_attributes.csv   (player-level bio + advanced stats)

  Q3  Playoff vs. regular-season performance
      → data/processed/q3_player_split_wide.csv   (one row per player-season, REG vs POF columns)
      → data/processed/q3_game_logs_typed.csv      (raw game logs with GAME_TYPE label)

Datasets used
  nathanlauga/nba-games       – per-game player box scores + team standings
  wyattowalsh/basketball      – SQLite with draft history, player bio, advanced
  drgilermo/nba-players-stats – season-level advanced metrics (PER, WS, VORP…)

Usage
  pip install -r requirements.txt
  # Auth (either one works):
  #   A) ~/.kaggle/access_token   containing your KGAT_… token
  #   B) ~/.kaggle/kaggle.json    containing {"username":…,"key":…}
  #   C) export KAGGLE_API_TOKEN=KGAT_…
  python fetch_nba_data.py
"""

import os
import sys
import subprocess
import json
import sqlite3
import shutil
from pathlib import Path

# ── 0. Bootstrap dependencies ─────────────────────────────────────────────────

REQUIRED = {
    "pandas":    "pandas>=2.0.0",
    "numpy":     "numpy>=1.24.0",
    "tqdm":      "tqdm>=4.65.0",
    "scipy":     "scipy>=1.10.0",
    "kagglehub": "kagglehub",
}

def _ensure_packages():
    missing = []
    for mod, spec in REQUIRED.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(spec)
    if missing:
        print(f"Installing: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
        print("Done. Re-running script…")
        os.execv(sys.executable, [sys.executable] + sys.argv)

_ensure_packages()

import pandas as pd
import numpy as np
from tqdm import tqdm

# ── 1. Directories ────────────────────────────────────────────────────────────

BASE = Path(__file__).parent
RAW  = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"

for d in (RAW, PROC):
    d.mkdir(parents=True, exist_ok=True)

# ── 2. Kaggle auth ────────────────────────────────────────────────────────────
# kagglehub reads KAGGLE_API_TOKEN (the newer KGAT_… format) automatically.
# We load it from ~/.kaggle/access_token if the env var isn't already set.

def _setup_kaggle_auth():
    if os.environ.get("KAGGLE_API_TOKEN"):
        return

    access_token = Path.home() / ".kaggle" / "access_token"
    json_cred    = Path.home() / ".kaggle" / "kaggle.json"

    if access_token.exists():
        os.chmod(access_token, 0o600)
        tok = access_token.read_text().strip()
        if tok:
            os.environ["KAGGLE_API_TOKEN"] = tok
            return

    if json_cred.exists():
        os.chmod(json_cred, 0o600)
        creds = json.loads(json_cred.read_text())
        # kagglehub also reads KAGGLE_USERNAME + KAGGLE_KEY for classic credentials
        os.environ.setdefault("KAGGLE_USERNAME", creds["username"])
        os.environ.setdefault("KAGGLE_KEY",      creds["key"])
        return

    print("""
╔══════════════════════════════════════════════════════════════════╗
║  No Kaggle credentials found.                                    ║
║                                                                  ║
║  Save your KGAT_… token:                                         ║
║    mkdir -p ~/.kaggle                                            ║
║    echo YOUR_KGAT_TOKEN > ~/.kaggle/access_token                 ║
║    chmod 600 ~/.kaggle/access_token                              ║
╚══════════════════════════════════════════════════════════════════╝
""")
    sys.exit(1)

_setup_kaggle_auth()

import kagglehub  # noqa: E402 — must come after env var is set

print(f"✓  Kaggle credentials loaded\n")

# ── 3. Download datasets ──────────────────────────────────────────────────────
# kagglehub caches downloads in ~/.cache/kagglehub/; we copy to data/raw/
# so the project is self-contained.

DATASETS = {
    "nba_games":    "nathanlauga/nba-games",       # game + player box scores
    "basketball":   "wyattowalsh/basketball",       # SQLite: draft, bio, advanced
    "player_stats": "drgilermo/nba-players-stats",  # season stats + bio
}

def _download(name: str, slug: str):
    dest = RAW / name
    if dest.exists() and any(dest.iterdir()):
        print(f"  '{name}'  already present — skip")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading  {slug} …")
    cache_path = kagglehub.dataset_download(slug)
    # Copy from kagglehub cache into data/raw/<name>/
    for item in Path(cache_path).iterdir():
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    print(f"  ✓  {name}  ({len(list(dest.iterdir()))} files)")

print("── Downloading ─────────────────────────────────────────")
for _name, _slug in DATASETS.items():
    _download(_name, _slug)
print()

# ── Helpers ───────────────────────────────────────────────────────────────────

def read_csv(path: Path, **kw) -> pd.DataFrame:
    """Read CSV with automatic encoding fallback."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path}")


def _sqlite_tables(con: sqlite3.Connection) -> list:
    return pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table'", con
    )["name"].tolist()


def _sqlite_columns(con: sqlite3.Connection, table: str) -> list:
    return pd.read_sql(f"PRAGMA table_info({table})", con)["name"].tolist()


def _find_col(cols: list, *candidates):
    """Return the first candidate found in cols (case-insensitive)."""
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Q1 — Player next-game performance prediction
# ═════════════════════════════════════════════════════════════════════════════

print("── Q1: player prediction dataset ──────────────────────")

# ── Load raw sources ──────────────────────────────────────────────────────────

games   = read_csv(RAW / "nba_games" / "games.csv", parse_dates=["GAME_DATE_EST"],
                   low_memory=False)
details = read_csv(RAW / "nba_games" / "games_details.csv", low_memory=False)

# ── Clean details ─────────────────────────────────────────────────────────────

# MIN comes as "32:45" or numeric — convert to minutes played
details["MIN"] = (
    details["MIN"].astype(str)
    .str.split(":").str[0]
    .pipe(pd.to_numeric, errors="coerce")
)

NUM_COLS = [
    "PTS", "REB", "AST", "STL", "BLK",
    "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "TO", "PF", "PLUS_MINUS", "MIN",
]
for col in NUM_COLS:
    if col in details.columns:
        details[col] = pd.to_numeric(details[col], errors="coerce")

details = details.dropna(subset=["PLAYER_ID", "GAME_ID", "PTS"])

# ── Merge game metadata into player logs ─────────────────────────────────────

GAME_META = ["GAME_ID", "GAME_DATE_EST", "SEASON",
             "HOME_TEAM_ID", "VISITOR_TEAM_ID", "HOME_TEAM_WINS"]
game_meta = games[GAME_META].drop_duplicates("GAME_ID")

df = details.merge(game_meta, on="GAME_ID", how="left")

df["IS_HOME"] = (df["TEAM_ID"] == df["HOME_TEAM_ID"]).astype("int8")
df["OPP_TEAM_ID"] = np.where(
    df["IS_HOME"] == 1,
    df["VISITOR_TEAM_ID"],
    df["HOME_TEAM_ID"],
)
df = df.sort_values(["PLAYER_ID", "GAME_DATE_EST"]).reset_index(drop=True)

# ── Rolling features (last 5 / 10 games, using only past games) ───────────────

ROLL_STATS = [c for c in NUM_COLS if c in df.columns]

def _add_rolling(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("GAME_DATE_EST")
    for stat in ROLL_STATS:
        shifted = g[stat].shift(1)          # shift so the current game is excluded
        g[f"L5_{stat}"]  = shifted.rolling(5,  min_periods=1).mean()
        g[f"L10_{stat}"] = shifted.rolling(10, min_periods=1).mean()
    return g

print("  Computing rolling averages …")
df = (
    df.groupby("PLAYER_ID", group_keys=False)
      .apply(_add_rolling)
      .reset_index(drop=True)
)

# ── Rest & fatigue features ───────────────────────────────────────────────────

df["PREV_DATE"]    = df.groupby("PLAYER_ID")["GAME_DATE_EST"].shift(1)
df["DAYS_REST"]    = (df["GAME_DATE_EST"] - df["PREV_DATE"]).dt.days
df["BACK_TO_BACK"] = (df["DAYS_REST"] == 1).astype("int8")
df["DAYS_REST"]    = df["DAYS_REST"].fillna(7).clip(upper=14)  # cap at 14

# ── Opponent defensive strength ───────────────────────────────────────────────
# Proxy: average PTS that the opponent team concedes per player per season.

opp_pts_allowed = (
    details
    .merge(game_meta[["GAME_ID", "SEASON", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]],
           on="GAME_ID", how="left")
    .assign(
        OPP=lambda x: np.where(
            x["TEAM_ID"] == x["HOME_TEAM_ID"],
            x["VISITOR_TEAM_ID"],
            x["HOME_TEAM_ID"],
        )
    )
    .groupby(["SEASON", "OPP"])["PTS"]
    .mean()
    .reset_index()
    .rename(columns={"OPP": "TEAM_ID", "PTS": "OPP_AVG_PTS_ALLOWED"})
)

df = df.merge(
    opp_pts_allowed,
    left_on=["SEASON", "OPP_TEAM_ID"],
    right_on=["SEASON", "TEAM_ID"],
    how="left",
    suffixes=("", "_dup"),
).drop(columns=[c for c in df.columns if c.endswith("_dup")])

# ── Season-to-date averages (expanding, shifted to avoid leakage) ─────────────

def _season_avg(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("GAME_DATE_EST")
    for stat in ROLL_STATS:
        g[f"STD_{stat}"] = g[stat].shift(1).expanding().mean()
    return g

print("  Computing season-to-date averages …")
df = (
    df.groupby(["PLAYER_ID", "SEASON"], group_keys=False)
      .apply(_season_avg)
      .reset_index(drop=True)
)

# ── Assemble Q1 output ────────────────────────────────────────────────────────

ID_COLS  = ["GAME_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID",
            "GAME_DATE_EST", "SEASON"]
TGT_COLS = ["PTS", "REB", "AST", "STL", "BLK", "FG_PCT", "PLUS_MINUS", "MIN"]
FEAT_COLS = (
    ["IS_HOME", "DAYS_REST", "BACK_TO_BACK", "OPP_TEAM_ID", "OPP_AVG_PTS_ALLOWED"]
    + [c for c in df.columns if c.startswith(("L5_", "L10_", "STD_"))]
)

q1 = df[ID_COLS + TGT_COLS + FEAT_COLS].dropna(subset=["PTS"])
q1.to_csv(PROC / "q1_player_game_logs.csv", index=False)
print(f"  ✓  q1_player_game_logs.csv    {len(q1):>8,} rows  ×  {q1.shape[1]} cols")

# ═════════════════════════════════════════════════════════════════════════════
# Q2 — Team chemistry from roster composition
# ═════════════════════════════════════════════════════════════════════════════

print("\n── Q2: team chemistry dataset ──────────────────────────")

# ── Load drgilermo seasonal stats ────────────────────────────────────────────

season_stats = read_csv(RAW / "player_stats" / "Seasons_Stats.csv",
                        index_col=0, low_memory=False)
season_stats.columns = season_stats.columns.str.strip()
season_stats = season_stats.dropna(subset=["Year", "Player", "Tm"])
season_stats["Year"] = pd.to_numeric(season_stats["Year"], errors="coerce").dropna().astype(int)
# Drop mid-season-trade duplicate total rows (Tm == "TOT")
season_stats = season_stats[season_stats["Tm"] != "TOT"].copy()

# ── Load player bio ───────────────────────────────────────────────────────────

bio = read_csv(RAW / "player_stats" / "player_data.csv")
bio.columns = bio.columns.str.lower().str.replace(" ", "_")
bio["birth_year"] = pd.to_datetime(bio.get("birth_date", ""), errors="coerce").dt.year

# ── Try to load draft history & nationality from SQLite ──────────────────────

db_path = next(
    (p for ext in ("*.sqlite", "*.db") for p in (RAW / "basketball").glob(ext)),
    None,
)

draft_df    = pd.DataFrame()
player_info = pd.DataFrame()

if db_path:
    con = sqlite3.connect(str(db_path))
    tables = _sqlite_tables(con)
    print(f"  SQLite tables: {tables}")

    # Draft history
    if "draft_history" in tables:
        draft_df = pd.read_sql("SELECT * FROM draft_history", con)
        draft_df.columns = draft_df.columns.str.lower()

    # Player biographical / nationality info
    for tbl in ("common_player_info", "player_info", "player"):
        if tbl in tables:
            cols = _sqlite_columns(con, tbl)
            nat_col  = _find_col(cols, "country", "birthcountry", "nation", "nationality")
            name_col = _find_col(cols, "display_first_last", "full_name", "player_name",
                                 "display_fi_last")
            if nat_col and name_col:
                player_info = pd.read_sql(
                    f"SELECT {name_col}, {nat_col} FROM {tbl}", con
                ).rename(columns={name_col: "Player", nat_col: "nationality"})
                break

    con.close()
else:
    print("  WARNING: basketball.sqlite not found — nationality & draft from CSV fallback")

# ── Merge bio into season stats ───────────────────────────────────────────────

q2p = season_stats.merge(
    bio[["name", "birth_year", "college", "height", "weight", "position"]].rename(
        columns={"name": "Player"}
    ),
    on="Player",
    how="left",
)
q2p["age_computed"] = q2p["Year"] - q2p["birth_year"]
q2p["experience"]   = q2p.groupby("Player").cumcount() + 1  # seasons in league

# Nationality
if not player_info.empty:
    q2p = q2p.merge(player_info.drop_duplicates("Player"), on="Player", how="left")

# Draft info
if not draft_df.empty:
    name_col  = _find_col(draft_df.columns.tolist(),
                          "display_first_last", "player_name", "full_name")
    round_col = _find_col(draft_df.columns.tolist(), "draft_round", "round_number", "round")
    num_col   = _find_col(draft_df.columns.tolist(), "draft_number", "round_pick", "pick")
    if name_col:
        draft_slim = (
            draft_df[[c for c in [name_col, round_col, num_col] if c]]
            .rename(columns={
                name_col:  "Player",
                **({"draft_round": "draft_round"} if round_col else {}),
                **({"draft_pick":  "draft_pick"}  if num_col   else {}),
            })
            .drop_duplicates("Player")
        )
        if round_col and round_col != "draft_round":
            draft_slim = draft_slim.rename(columns={round_col: "draft_round"})
        if num_col and num_col != "draft_pick":
            draft_slim = draft_slim.rename(columns={num_col: "draft_pick"})
        q2p = q2p.merge(draft_slim, on="Player", how="left")

# Numeric coercion
for col in ("draft_pick", "draft_round", "PER", "WS", "VORP", "BPM",
            "TS%", "WS/48", "PTS", "TRB", "AST"):
    if col in q2p.columns:
        q2p[col] = pd.to_numeric(q2p[col], errors="coerce")

# Star proxy: PER > 20
if "PER" in q2p.columns:
    q2p["is_star"] = (q2p["PER"].fillna(0) > 20).astype("int8")

# ── Save player-level Q2 file ─────────────────────────────────────────────────

PLAYER_COLS = [
    "Player", "Tm", "Year", "Pos", "age_computed", "experience",
    "G", "MP", "PER", "TS%", "WS", "WS/48", "VORP", "BPM",
    "PTS", "TRB", "AST", "STL", "BLK", "TOV",
    "height", "weight", "college",
]
for opt in ("nationality", "draft_round", "draft_pick", "is_star"):
    if opt in q2p.columns:
        PLAYER_COLS.append(opt)

q2_players = q2p[[c for c in PLAYER_COLS if c in q2p.columns]].copy()
q2_players.to_csv(PROC / "q2_player_attributes.csv", index=False)
print(f"  ✓  q2_player_attributes.csv   {len(q2_players):>8,} rows  ×  {q2_players.shape[1]} cols")

# ── Aggregate to team-season ──────────────────────────────────────────────────

agg_spec: dict = {}
for col, fns in [
    ("age_computed",  ["mean", "std", "min", "max"]),
    ("experience",    ["mean", "std", "max"]),
    ("PER",           ["mean", "max", "std"]),
    ("WS",            ["sum",  "mean"]),
    ("WS/48",         ["mean"]),
    ("VORP",          ["sum",  "mean"]),
    ("BPM",           ["mean"]),
    ("PTS",           ["mean"]),
    ("TRB",           ["mean"]),
    ("AST",           ["mean"]),
    ("G",             ["mean"]),        # avg games played per player
]:
    if col in q2p.columns:
        agg_spec[col] = fns

team_season = q2p.groupby(["Tm", "Year"]).agg(agg_spec)
team_season.columns = ["_".join(c) for c in team_season.columns]
team_season = team_season.reset_index()

# Roster size
roster_size = q2p.groupby(["Tm", "Year"]).size().reset_index(name="roster_size")
team_season  = team_season.merge(roster_size, on=["Tm", "Year"], how="left")

# Nationality diversity
if "nationality" in q2p.columns:
    nat_div = (
        q2p.groupby(["Tm", "Year"])["nationality"]
           .nunique()
           .reset_index(name="nationality_diversity")
    )
    team_season = team_season.merge(nat_div, on=["Tm", "Year"], how="left")

# Draft pedigree (lower = higher pick = more valued)
if "draft_pick" in q2p.columns:
    dp = (
        q2p.groupby(["Tm", "Year"])["draft_pick"]
           .mean()
           .reset_index(name="avg_draft_pick")
    )
    team_season = team_season.merge(dp, on=["Tm", "Year"], how="left")

# Star count
if "is_star" in q2p.columns:
    sc = (
        q2p.groupby(["Tm", "Year"])["is_star"]
           .sum()
           .reset_index(name="star_player_count")
    )
    team_season = team_season.merge(sc, on=["Tm", "Year"], how="left")

# ── Merge win % from nathanlauga rankings ────────────────────────────────────

ranking_path = RAW / "nba_games" / "ranking.csv"
if ranking_path.exists():
    ranking = read_csv(ranking_path)
    ranking.columns = ranking.columns.str.upper()
    if "SEASON_ID" in ranking.columns:
        ranking["SEASON_YEAR"] = (
            ranking["SEASON_ID"].astype(str).str[-4:]
        )
        ranking["SEASON_YEAR"] = pd.to_numeric(ranking["SEASON_YEAR"], errors="coerce")
    elif "SEASON" in ranking.columns:
        ranking["SEASON_YEAR"] = pd.to_numeric(ranking["SEASON"].astype(str).str[:4],
                                                errors="coerce")
    # Keep the end-of-season row (max games played per team-season)
    if "G" in ranking.columns:
        team_wins = (
            ranking.sort_values("G", ascending=False)
                   .groupby(["TEAM_ID", "SEASON_YEAR"])
                   .first()
                   .reset_index()[["TEAM_ID", "SEASON_YEAR", "W", "L", "W_PCT",
                                   "HOME_RECORD", "ROAD_RECORD"]]
        )
    else:
        team_wins = pd.DataFrame()

    # Map team abbreviations to TEAM_ID via teams.csv
    teams_df = read_csv(RAW / "nba_games" / "teams.csv")
    teams_df.columns = teams_df.columns.str.upper()
    abbr_to_id = teams_df.set_index("ABBREVIATION")["TEAM_ID"].to_dict()
    team_season["TEAM_ID"] = team_season["Tm"].map(abbr_to_id)

    if not team_wins.empty and "TEAM_ID" in team_season.columns:
        team_season = team_season.merge(
            team_wins,
            left_on=["TEAM_ID", "Year"],
            right_on=["TEAM_ID", "SEASON_YEAR"],
            how="left",
        ).drop(columns=["SEASON_YEAR"], errors="ignore")

team_season.to_csv(PROC / "q2_team_season.csv", index=False)
print(f"  ✓  q2_team_season.csv          {len(team_season):>8,} rows  ×  {team_season.shape[1]} cols")

# ═════════════════════════════════════════════════════════════════════════════
# Q3 — Playoff vs. regular-season performance
# ═════════════════════════════════════════════════════════════════════════════

print("\n── Q3: playoff vs. regular-season dataset ──────────────")

# ── Classify each game as Regular Season or Playoffs ─────────────────────────
#
# Strategy (in order of preference):
#   A. wyattowalsh SQLite: season_id prefix ('2' = regular, '4' = playoffs)
#   B. nathanlauga games.csv: approximate by calendar month
#      (Oct–Mar = regular season, Apr–Jun = playoffs)

game_type_map = None  # GAME_ID → "Regular Season" | "Playoffs"

if db_path:
    con = sqlite3.connect(str(db_path))
    tables = _sqlite_tables(con)
    # Look for a table with both game_id and season_id
    for tbl in ("game", "game_info", "game_summary"):
        if tbl not in tables:
            continue
        cols = _sqlite_columns(con, tbl)
        gid_col = _find_col(cols, "game_id")
        sid_col = _find_col(cols, "season_id", "season_type_id")
        stype_col = _find_col(cols, "season_type", "game_type",
                              "season_type_description")
        if gid_col and (sid_col or stype_col):
            sel_cols = ", ".join(c for c in [gid_col, sid_col, stype_col] if c)
            g_db = pd.read_sql(f"SELECT {sel_cols} FROM {tbl}", con)
            g_db.columns = g_db.columns.str.lower()
            if stype_col:
                col = stype_col.lower()
                g_db["GAME_TYPE"] = g_db[col].astype(str).str.strip()
            elif sid_col:
                col = sid_col.lower()
                sid_str = g_db[col].astype(str).str.strip()
                g_db["GAME_TYPE"] = np.where(
                    sid_str.str.startswith("4"),
                    "Playoffs",
                    np.where(sid_str.str.startswith("2"), "Regular Season", "Other"),
                )
            game_type_map = g_db.set_index(gid_col.lower())["GAME_TYPE"].to_dict()
            print(f"  Game type from SQLite table '{tbl}': "
                  f"{g_db['GAME_TYPE'].value_counts().to_dict()}")
            break
    con.close()

if game_type_map is None:
    # Month-based fallback using nathanlauga games.csv
    games["MONTH"] = games["GAME_DATE_EST"].dt.month
    game_type_map = (
        games
        .assign(GAME_TYPE=lambda x: np.where(
            x["MONTH"].between(4, 6), "Playoffs", "Regular Season"
        ))
        .set_index("GAME_ID")["GAME_TYPE"]
        .to_dict()
    )
    print("  Game type approximated by calendar month (Apr–Jun = Playoffs)")

# ── Build typed game-log file ─────────────────────────────────────────────────

details_q3 = details.merge(
    game_meta[["GAME_ID", "GAME_DATE_EST", "SEASON"]].drop_duplicates("GAME_ID"),
    on="GAME_ID",
    how="left",
)
details_q3["GAME_TYPE"] = (
    details_q3["GAME_ID"]
    .map(game_type_map)
    .fillna("Regular Season")
)

STAT_COLS = [c for c in NUM_COLS if c in details_q3.columns]

# Filter to meaningful games (played some minutes)
details_q3 = details_q3[details_q3["MIN"].fillna(0) >= 1].copy()

# Long file — all game logs with type label
q3_long_cols = (
    ["GAME_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID",
     "GAME_DATE_EST", "SEASON", "GAME_TYPE"]
    + STAT_COLS
)
q3_long = details_q3[[c for c in q3_long_cols if c in details_q3.columns]].dropna(subset=["PTS"])
q3_long.to_csv(PROC / "q3_game_logs_typed.csv", index=False)
print(f"  ✓  q3_game_logs_typed.csv      {len(q3_long):>8,} rows  ×  {q3_long.shape[1]} cols")

# ── Aggregate per player × season × game_type ────────────────────────────────

q3_agg = (
    details_q3
    .groupby(["PLAYER_ID", "PLAYER_NAME", "SEASON", "GAME_TYPE"])[STAT_COLS]
    .agg(["mean", "std", "count"])
)
q3_agg.columns = ["_".join(c) for c in q3_agg.columns]
q3_agg = q3_agg.reset_index()

# Rename *_count columns → games_played (use first stat as proxy)
first_cnt = next((c for c in q3_agg.columns if c.endswith("_count")), None)
if first_cnt:
    q3_agg = q3_agg.rename(columns={first_cnt: "games_played"})
    q3_agg = q3_agg.drop(
        columns=[c for c in q3_agg.columns if c.endswith("_count")]
    )

# ── Pivot to wide format (REG_* vs POF_* columns) ────────────────────────────

reg  = q3_agg[q3_agg["GAME_TYPE"].str.contains("Regular",  na=False)].drop(columns="GAME_TYPE")
poff = q3_agg[q3_agg["GAME_TYPE"].str.contains("Playoff",  na=False)].drop(columns="GAME_TYPE")

MERGE_KEYS = ["PLAYER_ID", "PLAYER_NAME", "SEASON"]
reg.columns  = [c if c in MERGE_KEYS else f"REG_{c}"  for c in reg.columns]
poff.columns = [c if c in MERGE_KEYS else f"POF_{c}" for c in poff.columns]

q3_wide = reg.merge(poff, on=MERGE_KEYS, how="outer")

# Playoff uplift / decline deltas
for stat in ("PTS", "REB", "AST", "FG_PCT", "PLUS_MINUS", "MIN"):
    r = f"REG_{stat}_mean"
    p = f"POF_{stat}_mean"
    if r in q3_wide.columns and p in q3_wide.columns:
        q3_wide[f"DELTA_{stat}"]     = q3_wide[p] - q3_wide[r]
        q3_wide[f"DELTA_PCT_{stat}"] = (
            (q3_wide[p] - q3_wide[r])
            / q3_wide[r].abs().replace(0, np.nan)
            * 100
        )

# Classify each player as "playoff riser", "playoff decliner", or "neutral"
if "DELTA_PTS" in q3_wide.columns:
    q3_wide["PLAYOFF_TENDENCY"] = pd.cut(
        q3_wide["DELTA_PTS"],
        bins=[-np.inf, -2, 2, np.inf],
        labels=["Decliner", "Neutral", "Riser"],
    )

q3_wide.to_csv(PROC / "q3_player_split_wide.csv", index=False)
print(f"  ✓  q3_player_split_wide.csv    {len(q3_wide):>8,} rows  ×  {q3_wide.shape[1]} cols")

# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  Pipeline complete!                                                  ║
║                                                                      ║
║  data/processed/                                                     ║
║  ├── q1_player_game_logs.csv     ← game logs + rolling features      ║
║  ├── q2_team_season.csv          ← team-season composition & wins    ║
║  ├── q2_player_attributes.csv    ← player bio + advanced stats       ║
║  ├── q3_player_split_wide.csv    ← REG vs POF per player-season      ║
║  └── q3_game_logs_typed.csv      ← game logs with GAME_TYPE label    ║
║                                                                      ║
║  Key columns                                                         ║
║  Q1  target: PTS/REB/AST/…  features: L5_*, L10_*, STD_*,          ║
║              IS_HOME, DAYS_REST, BACK_TO_BACK, OPP_AVG_PTS_ALLOWED  ║
║  Q2  target: W_PCT           features: age_*, experience_*,          ║
║              PER_*, WS_*, nationality_diversity, star_player_count   ║
║  Q3  deltas: DELTA_PTS/REB/… DELTA_PCT_*  label: PLAYOFF_TENDENCY   ║
╚══════════════════════════════════════════════════════════════════════╝
""")
