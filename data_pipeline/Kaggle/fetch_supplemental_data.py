#!/usr/bin/env python3
"""
Supplemental NBA data — fills gaps in fetch_nba_data.py.

  Downloads 2 new Kaggle datasets and mines 4 unused SQLite tables.

  Adds to data/processed/:

    Fix for Q3:
      q3_player_split_v2.csv        REAL regular-vs-playoff player stats (1995-2023)
      q3_advanced_split.csv         Advanced metric (PER/WS/VORP) regular vs playoff

    New for Q1:
      player_inactive_log.csv       Which players sat out each game (injury proxy)
      game_context.csv              National-TV flag + game-flow stats per game

    New for Q2:
      player_salaries.csv           Individual salaries 1990-2023
      team_payroll.csv              Team payrolls 1990-2023
      player_combine.csv            Wingspan, vertical leap, sprint, body fat
      player_bio_enhanced.csv       Country, greatest-75 flag, school, draft

Usage
  python fetch_supplemental_data.py
  (assumes fetch_nba_data.py has already been run)
"""

import os
import sys
import json
import sqlite3
import shutil
import subprocess
import re
from pathlib import Path

# ── 0. dependencies ──────────────────────────────────────────────────────────
REQUIRED = {"pandas": "pandas>=2.0.0", "numpy": "numpy>=1.24.0",
            "kagglehub": "kagglehub", "pyarrow": "pyarrow"}

def _ensure_pkgs():
    missing = [s for m, s in REQUIRED.items() if not _import_ok(m)]
    if missing:
        print(f"Installing: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
        os.execv(sys.executable, [sys.executable] + sys.argv)

def _import_ok(name):
    try:
        __import__(name); return True
    except ImportError:
        return False

_ensure_pkgs()

import pandas as pd
import numpy as np

# ── 1. paths ─────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
RAW  = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# ── 2. Kaggle auth (reuses access_token) ─────────────────────────────────────
if not os.environ.get("KAGGLE_API_TOKEN"):
    tok = (Path.home() / ".kaggle" / "access_token")
    if tok.exists():
        os.environ["KAGGLE_API_TOKEN"] = tok.read_text().strip()
import kagglehub  # noqa: E402

# ── 3. Download supplemental Kaggle datasets ─────────────────────────────────

SUPP = {
    "seasonal_history": "bendikfltaas/nba-history-seasonal-data-1995-2023",  # Regular-season and playoff history
    "salaries":         "loganlauton/nba-players-and-team-data",
}

def _download(name: str, slug: str) -> Path:
    dest = RAW / name
    if dest.exists() and any(dest.iterdir()):
        print(f"  '{name}' already present — skip")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {slug} …")
    src = Path(kagglehub.dataset_download(slug))
    for f in src.iterdir():
        if f.is_dir():
            shutil.copytree(f, dest / f.name, dirs_exist_ok=True)
        else:
            shutil.copy2(f, dest / f.name)
    print(f"  ✓ {name} ({len(list(dest.iterdir()))} files)")
    return dest

print("── Downloading supplemental datasets ───────────────────")
for _n, _s in SUPP.items():
    _download(_n, _s)
print()


# ═════════════════════════════════════════════════════════════════════════════
# A. Q3 — season-level regular-vs-playoff player statistics (1995-2023)
# ═════════════════════════════════════════════════════════════════════════════

print("── A. Q3: regular-vs-playoff player stats ──────────────")

HIST_DIR = RAW / "seasonal_history"

reg_avg  = pd.read_parquet(HIST_DIR / "average.parq")
pof_avg  = pd.read_parquet(HIST_DIR / "average_playoffs.parq")
reg_adv  = pd.read_parquet(HIST_DIR / "advanced.parq")
pof_adv  = pd.read_parquet(HIST_DIR / "advanced_playoffs.parq")

# Convert season "1997/1998" -> 1997 (start year)
for d in (reg_avg, pof_avg, reg_adv, pof_adv):
    d["season_start"] = d["season"].str.split("/").str[0].astype(int)

# Standardize column names (snake_case)
def _slim(df, keep):
    return df[keep].copy()

KEY = ["Player", "team_retcon", "season_start"]
PG_STATS = ["G", "MP", "PTS", "TRB", "AST", "STL", "BLK", "TOV",
            "FG%", "3P%", "FT%", "eFG%"]

reg_keep = _slim(reg_avg, KEY + PG_STATS).rename(columns={c: f"REG_{c}" for c in PG_STATS})
pof_keep = _slim(pof_avg, KEY + PG_STATS).rename(columns={c: f"POF_{c}" for c in PG_STATS})

# Drop "TOT" rows (mid-season trades shown as totals) when a real team row also exists
reg_keep = reg_keep[reg_keep["team_retcon"] != "TOT"]
pof_keep = pof_keep[pof_keep["team_retcon"] != "TOT"]

# Player-season is the merge key — collapse multi-team rows by summing G and weighting averages
def _collapse(df, prefix):
    """Weighted average by games when the same player appears for >1 team in a season."""
    if df.empty:
        return df
    g_col = f"{prefix}_G"
    # Calculate games-weighted means for the remaining statistics.
    stat_cols = [c for c in df.columns if c.startswith(prefix + "_") and c != g_col]
    out = (df
           .assign(_w=df[g_col])
           .groupby(["Player", "season_start"], as_index=False)
           .apply(lambda x: pd.Series({
               g_col: x[g_col].sum(),
               **{c: np.average(x[c].fillna(0), weights=x["_w"].clip(lower=1))
                   for c in stat_cols},
           })))
    return out.reset_index(drop=True)

reg_one = _collapse(reg_keep, "REG")
pof_one = _collapse(pof_keep, "POF")

# Outer merge — keep players who played either regular season, playoffs, or both
q3v2 = reg_one.merge(pof_one, on=["Player", "season_start"], how="outer")

# Add delta columns for the most-watched stats
for stat in ["PTS", "TRB", "AST", "FG%", "eFG%"]:
    r = f"REG_{stat}"; p = f"POF_{stat}"
    if r in q3v2.columns and p in q3v2.columns:
        q3v2[f"DELTA_{stat}"] = q3v2[p] - q3v2[r]
        q3v2[f"DELTA_PCT_{stat}"] = (
            (q3v2[p] - q3v2[r]) / q3v2[r].abs().replace(0, np.nan) * 100
        )

# Classify playoff tendency by DELTA_PTS
q3v2["PLAYOFF_TENDENCY"] = pd.cut(
    q3v2["DELTA_PTS"],
    bins=[-np.inf, -2, 2, np.inf],
    labels=["Decliner", "Neutral", "Riser"],
)

q3v2.to_csv(PROC / "q3_player_split_v2.csv", index=False)
print(f"  ✓ q3_player_split_v2.csv         {len(q3v2):>7,} rows × {q3v2.shape[1]} cols")

# ── Advanced metric split (PER, WS, VORP, BPM) ──────────────────────────────

ADV_KEEP = ["PER", "TS%", "USG%", "WS", "WS/48", "VORP", "BPM", "OBPM", "DBPM"]
reg_a = _slim(reg_adv[reg_adv["team_retcon"] != "TOT"],
              KEY + ADV_KEEP).rename(columns={c: f"REG_{c}" for c in ADV_KEEP})
pof_a = _slim(pof_adv[pof_adv["team_retcon"] != "TOT"],
              KEY + ADV_KEEP).rename(columns={c: f"POF_{c}" for c in ADV_KEEP})

# Identify the team for which the player recorded the most games that season.
def _primary_team(df_avg, prefix):
    g_col = f"{prefix}_G" if f"{prefix}_G" in df_avg.columns else "G"
    if g_col not in df_avg.columns:
        return df_avg.drop_duplicates(["Player", "season_start"])
    return (df_avg.sort_values(g_col, ascending=False)
                  .drop_duplicates(["Player", "season_start"]))

reg_a_one = (reg_a.merge(reg_avg[["Player", "team_retcon", "season_start", "G"]],
                          on=["Player", "team_retcon", "season_start"], how="left")
                  .sort_values("G", ascending=False)
                  .drop_duplicates(["Player", "season_start"])
                  .drop(columns="G"))
pof_a_one = (pof_a.merge(pof_avg[["Player", "team_retcon", "season_start", "G"]],
                          on=["Player", "team_retcon", "season_start"], how="left")
                  .sort_values("G", ascending=False)
                  .drop_duplicates(["Player", "season_start"])
                  .drop(columns="G"))

adv_split = reg_a_one.merge(pof_a_one, on=["Player", "season_start"],
                             how="outer", suffixes=("_r", "_p"))
# Reconcile team_retcon columns
adv_split["team"] = adv_split["team_retcon_r"].fillna(adv_split.get("team_retcon_p"))
adv_split = adv_split.drop(columns=[c for c in ["team_retcon_r", "team_retcon_p"] if c in adv_split.columns])
adv_split.to_csv(PROC / "q3_advanced_split.csv", index=False)
print(f"  ✓ q3_advanced_split.csv          {len(adv_split):>7,} rows × {adv_split.shape[1]} cols")


# ═════════════════════════════════════════════════════════════════════════════
# B.  SALARY DATA — player salaries + team payrolls
# ═════════════════════════════════════════════════════════════════════════════

print("\n── B. Salary data (for Q2 chemistry) ───────────────────")

SAL_DIR = RAW / "salaries"

def _money_to_int(s):
    """Convert strings like '$4,250,000' to 4250000."""
    return pd.to_numeric(
        s.astype(str).str.replace(r"[\$,]", "", regex=True),
        errors="coerce",
    )

sal = pd.read_csv(SAL_DIR / "NBA Salaries(1990-2023).csv")
sal = sal.drop(columns=[c for c in sal.columns if c.startswith("Unnamed")])
sal = sal.rename(columns={
    "playerName":          "player",
    "seasonStartYear":     "season_start",
    "salary":              "salary",
    "inflationAdjSalary":  "salary_real_2023",
})
sal["salary"]           = _money_to_int(sal["salary"])
sal["salary_real_2023"] = _money_to_int(sal["salary_real_2023"])
sal.to_csv(PROC / "player_salaries.csv", index=False)
print(f"  ✓ player_salaries.csv            {len(sal):>7,} rows × {sal.shape[1]} cols")

pay = pd.read_csv(SAL_DIR / "NBA Payroll(1990-2023).csv")
pay = pay.drop(columns=[c for c in pay.columns if c.startswith("Unnamed")])
pay = pay.rename(columns={
    "team":                 "team",
    "seasonStartYear":      "season_start",
    "payroll":              "payroll",
    "inflationAdjPayroll":  "payroll_real_2023",
})
pay["payroll"]           = _money_to_int(pay["payroll"])
pay["payroll_real_2023"] = _money_to_int(pay["payroll_real_2023"])
pay.to_csv(PROC / "team_payroll.csv", index=False)
print(f"  ✓ team_payroll.csv               {len(pay):>7,} rows × {pay.shape[1]} cols")


# ═════════════════════════════════════════════════════════════════════════════
# C.  EXTRACT FROM EXISTING SQLITE (no download needed)
# ═════════════════════════════════════════════════════════════════════════════

print("\n── C. Mining unused SQLite tables ──────────────────────")

DB = next((RAW / "basketball").glob("*.sqlite"))
con = sqlite3.connect(str(DB))

# ── C1. player_inactive_log — injury / rest proxy for Q1 ────────────────────
inactive = pd.read_sql("""
    SELECT
      i.game_id,
      i.player_id,
      i.first_name || ' ' || i.last_name AS player_name,
      i.team_abbreviation,
      g.game_date                         AS game_date,
      CASE
        WHEN substr(g.season_id, 1, 1) = '4' THEN 'Playoffs'
        WHEN substr(g.season_id, 1, 1) = '2' THEN 'Regular Season'
        ELSE 'Other'
      END                                 AS game_type,
      CAST(substr(g.season_id, 2) AS INT) AS season_start
    FROM inactive_players i
    LEFT JOIN game g ON i.game_id = g.game_id
""", con)
inactive["game_date"] = pd.to_datetime(inactive["game_date"], errors="coerce")
inactive.to_csv(PROC / "player_inactive_log.csv", index=False)
print(f"  ✓ player_inactive_log.csv        {len(inactive):>7,} rows × {inactive.shape[1]} cols")

# Create a per-player-season summary of inactive games.
miss_summary = (
    inactive.dropna(subset=["season_start"])
            .groupby(["player_id", "player_name", "season_start"])
            .size()
            .reset_index(name="games_inactive")
)
miss_summary.to_csv(PROC / "player_games_missed_by_season.csv", index=False)
print(f"  ✓ player_games_missed_by_season.csv {len(miss_summary):>7,} rows")

# ── C2. game_context — national-TV flag + game-flow stats for Q1 ───────────
gs = pd.read_sql("""
    SELECT
      g.game_id,
      g.game_date_est                     AS game_date,
      g.season,
      g.natl_tv_broadcaster_abbreviation  AS natl_tv,
      o.pts_paint_home,     o.pts_paint_away,
      o.pts_2nd_chance_home, o.pts_2nd_chance_away,
      o.pts_fb_home,        o.pts_fb_away,
      o.largest_lead_home,  o.largest_lead_away,
      o.lead_changes,       o.times_tied
    FROM game_summary g
    LEFT JOIN other_stats o USING (game_id)
""", con)
gs["game_date"] = pd.to_datetime(gs["game_date"], errors="coerce")
gs["is_national_tv"] = gs["natl_tv"].notna().astype("int8")
gs.to_csv(PROC / "game_context.csv", index=False)
print(f"  ✓ game_context.csv               {len(gs):>7,} rows × {gs.shape[1]} cols")

# ── C3. player_combine — physical attributes for Q2 ────────────────────────
combine_cols = [
    "season", "player_id", "player_name", "position",
    "height_wo_shoes", "height_w_shoes", "weight", "wingspan",
    "standing_reach", "body_fat_pct", "hand_length", "hand_width",
    "standing_vertical_leap", "max_vertical_leap",
    "lane_agility_time", "three_quarter_sprint", "bench_press",
]
combine = pd.read_sql(
    f"SELECT {', '.join(combine_cols)} FROM draft_combine_stats", con
)
for c in combine.columns:
    if c not in {"season", "player_id", "player_name", "position"}:
        combine[c] = pd.to_numeric(combine[c], errors="coerce")
combine.to_csv(PROC / "player_combine.csv", index=False)
print(f"  ✓ player_combine.csv             {len(combine):>7,} rows × {combine.shape[1]} cols")

# ── C4. player_bio_enhanced — country, greatest-75 flag, school ────────────
bio = pd.read_sql("""
    SELECT
      person_id           AS player_id,
      display_first_last  AS player_name,
      birthdate,
      country,
      school,
      height, weight,
      season_exp,
      position,
      from_year, to_year,
      draft_year, draft_round, draft_number,
      greatest_75_flag
    FROM common_player_info
""", con)
bio["birthdate"]      = pd.to_datetime(bio["birthdate"], errors="coerce")
bio["birth_year"]     = bio["birthdate"].dt.year
bio["is_greatest_75"] = (bio["greatest_75_flag"] == "Y").astype("int8")
bio["is_usa"]         = (bio["country"].str.upper() == "USA").astype("int8")
for c in ("season_exp", "from_year", "to_year",
          "draft_year", "draft_round", "draft_number"):
    bio[c] = pd.to_numeric(bio[c], errors="coerce")
bio.to_csv(PROC / "player_bio_enhanced.csv", index=False)
print(f"  ✓ player_bio_enhanced.csv        {len(bio):>7,} rows × {bio.shape[1]} cols")

con.close()


# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  Supplemental data pipeline complete!                                ║
║                                                                      ║
║  data/processed/                                                     ║
║  ── for Q3 ──                                                        ║
║  ├── q3_player_split_v2.csv      Reg-vs-playoff statistics 1995-2023 ║
║  ├── q3_advanced_split.csv       PER/WS/VORP reg vs playoff          ║
║                                                                      ║
║  ── for Q1 ──                                                        ║
║  ├── player_inactive_log.csv     Who sat out each game               ║
║  ├── player_games_missed_by_season.csv   season-level injury count   ║
║  ├── game_context.csv            National TV + paint/fastbreak pts   ║
║                                                                      ║
║  ── for Q2 ──                                                        ║
║  ├── player_salaries.csv         Individual salaries 1990-2023       ║
║  ├── team_payroll.csv            Team payrolls 1990-2023             ║
║  ├── player_combine.csv          Wingspan / vertical / sprint        ║
║  └── player_bio_enhanced.csv     Country, greatest-75, school        ║
╚══════════════════════════════════════════════════════════════════════╝
""")
