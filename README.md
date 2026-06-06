# NBA Data Pipeline — File Reference

A data science project exploring three questions about NBA player and team performance.

## Research Questions

1. **Q1 — Predicting a player's next game.** Can we use past performance, rest days, opponent strength, and home/away context to predict how a player will perform tomorrow?
2. **Q2 — Team chemistry.** Can a team's success be explained by the *composition* of its roster (age mix, experience, nationality, draft pedigree, star power, salary distribution)?
3. **Q3 — Playoffs vs. regular season.** Which players raise their game in the playoffs, which ones fall off, and what do the risers have in common?

## How to reproduce

```bash
pip install -r requirements.txt
echo YOUR_KGAT_TOKEN > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token

python fetch_nba_data.py            # base pipeline (downloads + processes 3 datasets)
python fetch_supplemental_data.py   # adds 2 more datasets + mines unused SQLite tables
python sanity_checks.py             # sanity-check visualizations
python playoff_risers_decliners.py  # bonus chart: top risers/decliners dumbbell plot
```

## Source datasets (Kaggle)

| Dataset | Years | What it has |
|---|---|---|
| `nathanlauga/nba-games` | 2003-2022 | Per-game player box scores + team standings |
| `wyattowalsh/basketball` | 1946-2023 | SQLite DB: draft history, player bio, advanced stats, play-by-play |
| `drgilermo/nba-players-stats` | 1950-2017 | Season-level advanced metrics (PER, WS, VORP) |
| `bendikfltaas/nba-history-seasonal-data-1995-2023` | 1995-2023 | **Includes playoff stats** — fixes Q3 |
| `loganlauton/nba-players-and-team-data` | 1990-2022 | Salaries + team payrolls |

---

# Files in `data/processed/`

## For Q1 — Player Prediction

### `q1_player_game_logs.csv` — 558,938 rows × 79 cols
Every player's box score in every regular-season game from 2003 to 2022, with pre-engineered features ready for modeling.

**Key columns:**
- IDs / context: `GAME_ID`, `PLAYER_ID`, `PLAYER_NAME`, `TEAM_ID`, `GAME_DATE_EST`, `SEASON`
- Target stats (what you might predict): `PTS`, `REB`, `AST`, `STL`, `BLK`, `FG_PCT`, `PLUS_MINUS`, `MIN`
- Recent-form features: `L5_PTS`, `L5_REB`, …, `L5_MIN` — average over the last 5 games
- Wider-form features: `L10_PTS`, … — average over the last 10 games
- Season-to-date averages: `STD_PTS`, `STD_REB`, … — expanding mean, shifted by 1 game (no data leakage)
- Context: `IS_HOME` (1/0), `DAYS_REST`, `BACK_TO_BACK`, `OPP_TEAM_ID`, `OPP_AVG_PTS_ALLOWED` (defense proxy)

### `player_inactive_log.csv` — 110,218 rows × 7 cols
Every time a player was inactive for a game (injury, rest, suspension). Source: SQLite `inactive_players`.

**Columns:** `game_id`, `player_id`, `player_name`, `team_abbreviation`, `game_date`, `game_type` (Regular Season / Playoffs), `season_start`

### `player_games_missed_by_season.csv` — 7,260 rows
Season-level summary: how many games each player was inactive for, per season.

### `game_context.csv` — 58,130 rows × 15 cols
Per-game context including national-TV flag and game-flow stats. Source: SQLite `game_summary` + `other_stats`.

**Columns:** `game_id`, `game_date`, `season`, `natl_tv` (broadcaster abbreviation or null), `is_national_tv` (0/1), `pts_paint_home`, `pts_paint_away`, `pts_2nd_chance_*`, `pts_fb_*` (fast break points), `largest_lead_*`, `lead_changes`, `times_tied`

---

## For Q2 — Team Chemistry

### `q2_player_attributes.csv` — 23,054 rows × 27 cols
One row per player per season (1950-2017). Combines drgilermo's seasonal stats with biographical info.

**Key columns:**
- ID: `Player`, `Tm` (team abbrev), `Year`, `Pos`
- Career context: `age_computed`, `experience` (# seasons in league so far)
- Volume: `G` (games), `MP` (minutes played)
- Efficiency: `PER`, `TS%`, `WS`, `WS/48`, `VORP`, `BPM`
- Per-game: `PTS`, `TRB`, `AST`, `STL`, `BLK`, `TOV`
- Physical / origin: `height`, `weight`, `college`, `nationality` (where available)
- Draft: `draft_round`, `draft_pick`
- Derived: `is_star` (1 if PER > 20)

### `q2_team_season.csv` — 1,423 rows × 32 cols
One row per team per season. Aggregates the player attributes above into team-level composition metrics, with the team's actual win % attached.

**Key columns:**
- ID: `Tm`, `Year`, `TEAM_ID`
- Age mix: `age_computed_mean`, `_std`, `_min`, `_max`
- Experience mix: `experience_mean`, `_std`, `_max`
- Quality: `PER_mean`, `PER_max`, `PER_std`, `WS_sum`, `WS_mean`, `VORP_sum`, `BPM_mean`
- Composition: `roster_size`, `nationality_diversity`, `avg_draft_pick`, `star_player_count`
- Outcome: `W`, `L`, `W_PCT`, `HOME_RECORD`, `ROAD_RECORD`

### `player_salaries.csv` — 15,857 rows × 4 cols
Individual player salaries 1990-2022.

**Columns:** `player`, `season_start`, `salary` (nominal $), `salary_real_2023` (inflation-adjusted to 2023 dollars)

### `team_payroll.csv` — 966 rows × 4 cols
Total team payroll per season 1990-2022.

**Columns:** `team`, `season_start`, `payroll`, `payroll_real_2023`

### `player_combine.csv` — 1,633 rows × 17 cols
Pre-draft combine measurements for players who attended the combine. Source: SQLite `draft_combine_stats`.

**Columns:** `season`, `player_id`, `player_name`, `position`, `height_wo_shoes` (inches), `height_w_shoes`, `weight`, `wingspan`, `standing_reach`, `body_fat_pct`, `hand_length`, `hand_width`, `standing_vertical_leap`, `max_vertical_leap`, `lane_agility_time`, `three_quarter_sprint`, `bench_press`

### `player_bio_enhanced.csv` — 3,632 rows × 18 cols
Comprehensive player biographical info. Source: SQLite `common_player_info`.

**Columns:** `player_id`, `player_name`, `birthdate`, `birth_year`, `country` (69 unique), `school`, `height`, `weight`, `season_exp`, `position`, `from_year`, `to_year`, `draft_year`, `draft_round`, `draft_number`, `greatest_75_flag`, `is_greatest_75` (1/0 — there are 59 players on the NBA's 75th anniversary team), `is_usa` (1/0)

---

## For Q3 — Playoffs vs. Regular Season

### `q3_player_split_v2.csv` — 11,508 rows × 37 cols  ✅ Use this one
One row per player per season (1995-2023) with both regular-season and playoff stats side by side.

**Key columns:**
- IDs: `Player`, `season_start`
- Regular season: `REG_G`, `REG_MP`, `REG_PTS`, `REG_TRB`, `REG_AST`, `REG_STL`, `REG_BLK`, `REG_FG%`, `REG_3P%`, `REG_FT%`, `REG_eFG%`
- Playoffs: `POF_G`, `POF_MP`, `POF_PTS`, … (same stats, playoffs only)
- Deltas: `DELTA_PTS` = `POF_PTS − REG_PTS`, `DELTA_PCT_PTS` = percentage change, same for REB/AST/FG%/eFG%
- Label: `PLAYOFF_TENDENCY` ∈ {`Riser`, `Neutral`, `Decliner`} based on DELTA_PTS

**Coverage:** 11,499 player-seasons have regular-season data; 4,999 have playoff data; 4,990 have both (these are the ones useful for comparison).

### `q3_advanced_split.csv` — 11,508 rows × 21 cols
Same idea but with advanced metrics: `REG_PER` vs `POF_PER`, `REG_WS` vs `POF_WS`, `REG_VORP` vs `POF_VORP`, `REG_BPM` vs `POF_BPM`, plus `OBPM`, `DBPM`, `TS%`, `USG%`, `WS/48`.

### `q3_player_split_wide.csv` ❌ Broken — do not use
Created by `fetch_nba_data.py` but has no real playoff data because `nathanlauga/nba-games` only includes regular-season games. Replaced by `q3_player_split_v2.csv`.

### `q3_game_logs_typed.csv` — 553,135 rows × 27 cols
Game-level logs with a `GAME_TYPE` label. **Caveat:** despite the label column existing, all current rows happen to be `"Regular Season"` for the same reason as above. Useful as raw material if you find another source of playoff player box scores.

---

## Visualizations in `figures/`

| File | What it shows |
|---|---|
| `fig1_age_sanity.png` | Player age distribution — confirms the bulk is realistic, flags 315 bad-data rows |
| `fig2_points_sanity.png` | Points-per-game distribution per season (violin plots) — confirms no impossible values |
| `playoff_risers_decliners.png` | Dumbbell plot: top 10 playoff risers (LeBron, Steve Nash, Kawhi-era Spurs) vs top 10 decliners (Jordan Poole, Joel Embiid's 2019 knee year, etc.) |

---

## Script reference

| Script | What it does |
|---|---|
| `fetch_nba_data.py` | Downloads 3 base Kaggle datasets and builds Q1/Q2/Q3 files. Run first. |
| `fetch_supplemental_data.py` | Adds 2 more Kaggle datasets, mines unused SQLite tables, fixes Q3. Run second. |
| `sanity_checks.py` | Builds the two sanity-check figures. |
| `playoff_risers_decliners.py` | Builds the risers/decliners dumbbell chart. |
