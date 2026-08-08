"""Alternative Q1: next-game scoring forecasts tested on a future NBA season."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    BLUE,
    FIGURES,
    GRID,
    INK,
    MUTED,
    NBA,
    ORANGE,
    PALE_BLUE,
    RANDOM_SEED,
    RED,
    SEASONS,
    TABLES,
    TEAL,
    add_source,
    cluster_bootstrap_difference,
    finish_figure,
    seed_everything,
    set_style,
)


TEST_SEASON = "2024-25"


def _rolling_prior(grouped: pd.core.groupby.SeriesGroupBy, window: int, min_periods: int = 1):
    return grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())


def load_player_games() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        path = NBA / "game_logs" / "players" / season / "player_gamelogs_regular_season_base.csv"
        frame = pd.read_csv(path, low_memory=False)
        frame["SEASON"] = season
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    for col in ["MIN", "PTS", "FGA", "FG_PCT", "REB", "AST", "TOV"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["PLAYER_ID", "GAME_DATE", "MIN", "PTS"])
    df = df[df["MIN"] > 0].drop_duplicates(["PLAYER_ID", "GAME_ID"])
    return df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)


def load_opponent_defense() -> pd.DataFrame:
    """Pregame five-game points-allowed lookup for every opponent and date."""
    frames = []
    for season in SEASONS:
        path = NBA / "game_logs" / "teams" / season / "team_gamelogs_regular_season_base.csv"
        frame = pd.read_csv(path, low_memory=False)
        frame["SEASON"] = season
        frames.append(frame)
    teams = pd.concat(frames, ignore_index=True)
    teams["GAME_DATE"] = pd.to_datetime(teams["GAME_DATE"], errors="coerce")
    teams["PTS"] = pd.to_numeric(teams["PTS"], errors="coerce")
    teams["PLUS_MINUS"] = pd.to_numeric(teams["PLUS_MINUS"], errors="coerce")
    teams["PTS_ALLOWED"] = teams["PTS"] - teams["PLUS_MINUS"]
    teams = teams.sort_values(["TEAM_ABBREVIATION", "SEASON", "GAME_DATE"])
    teams["opp_pts_allowed_roll5"] = (
        teams.groupby(["TEAM_ABBREVIATION", "SEASON"])["PTS_ALLOWED"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
    )
    return teams[["SEASON", "GAME_DATE", "TEAM_ABBREVIATION", "opp_pts_allowed_roll5"]].drop_duplicates()


def build_features() -> pd.DataFrame:
    df = load_player_games()
    key = ["PLAYER_ID", "SEASON"]
    df["games_played_prior"] = df.groupby(key).cumcount()
    grouped = df.groupby(key, sort=False)

    df["prev_pts"] = grouped["PTS"].shift(1)
    for window in [3, 5, 10]:
        df[f"pts_roll{window}"] = _rolling_prior(grouped["PTS"], window)
    df["pts_season"] = grouped["PTS"].transform(lambda s: s.shift(1).expanding().mean())
    df["pts_std10"] = grouped["PTS"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=5).std()
    )
    for col in ["MIN", "FGA", "FG_PCT", "REB", "AST", "TOV"]:
        df[f"{col.lower()}_roll5"] = _rolling_prior(grouped[col], 5, min_periods=2)

    df["rest_days"] = df.groupby("PLAYER_ID")["GAME_DATE"].diff().dt.days.clip(0, 7).fillna(7)
    df["is_home"] = df["MATCHUP"].str.contains(r"vs\.", regex=True, na=False).astype(int)
    df["is_b2b"] = (df["rest_days"] == 1).astype(int)
    df["OPPONENT_ABBREVIATION"] = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s+([A-Z]{3})", expand=False)

    defense = load_opponent_defense().rename(columns={"TEAM_ABBREVIATION": "OPPONENT_ABBREVIATION"})
    df = df.merge(defense, on=["SEASON", "GAME_DATE", "OPPONENT_ABBREVIATION"], how="left")

    # Ten prior games makes the player-specific summaries stable and ensures
    # every prediction is a real next-game forecast rather than a cold start.
    return df[df["games_played_prior"] >= 10].reset_index(drop=True)


FEATURES = [
    "prev_pts",
    "pts_roll3",
    "pts_roll5",
    "pts_roll10",
    "pts_season",
    "pts_std10",
    "min_roll5",
    "fga_roll5",
    "fg_pct_roll5",
    "reb_roll5",
    "ast_roll5",
    "tov_roll5",
    "rest_days",
    "is_home",
    "is_b2b",
    "opp_pts_allowed_roll5",
    "games_played_prior",
]


def _metric_row(name: str, actual: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "model": name,
        "mae": mean_absolute_error(actual, pred),
        "rmse": mean_squared_error(actual, pred) ** 0.5,
        "r2": r2_score(actual, pred),
    }


def fit_and_evaluate(df: pd.DataFrame):
    train = df[df["SEASON"] != TEST_SEASON].copy()
    test = df[df["SEASON"] == TEST_SEASON].copy()
    X_train, y_train = train[FEATURES], train["PTS"].to_numpy()
    X_test, y_test = test[FEATURES], test["PTS"].to_numpy()

    models = {
        "Ridge regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=20.0)
        ),
        "Gradient boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                learning_rate=0.055,
                max_iter=260,
                max_leaf_nodes=31,
                min_samples_leaf=50,
                l2_regularization=8.0,
                random_state=RANDOM_SEED,
            ),
        ),
    }
    predictions = {
        "Previous game": test["prev_pts"].to_numpy(),
        "Five-game average": test["pts_roll5"].to_numpy(),
        "Season-to-date average": test["pts_season"].to_numpy(),
    }
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)
        fitted[name] = model

    results = pd.DataFrame([_metric_row(name, y_test, pred) for name, pred in predictions.items()])
    results = results.sort_values("mae").reset_index(drop=True)
    best_name = results.iloc[0]["model"]
    test["prediction"] = predictions[best_name]
    test["baseline_prediction"] = predictions["Season-to-date average"]
    test["abs_error"] = np.abs(test["PTS"] - test["prediction"])
    test["baseline_abs_error"] = np.abs(test["PTS"] - test["baseline_prediction"])

    diff, lo, hi = cluster_bootstrap_difference(
        test["abs_error"].to_numpy(),
        test["baseline_abs_error"].to_numpy(),
        test["PLAYER_ID"].to_numpy(),
    )
    uncertainty = {
        "best_model": best_name,
        "mae_difference_vs_season_average": diff,
        "ci95_low": lo,
        "ci95_high": hi,
        "n_test_games": int(len(test)),
        "n_test_players": int(test["PLAYER_ID"].nunique()),
        "train_seasons": [s for s in SEASONS if s != TEST_SEASON],
        "test_season": TEST_SEASON,
    }

    return train, test, fitted, predictions, results, uncertainty


def figure_model_ladder(results: pd.DataFrame, uncertainty: dict) -> None:
    ordered = results.sort_values("mae", ascending=True)
    colors = [TEAL if name == uncertainty["best_model"] else BLUE if "average" not in name.lower() and "previous" not in name.lower() else MUTED for name in ordered["model"]]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    y = np.arange(len(ordered))
    bars = ax.barh(y, ordered["mae"], color=colors, height=0.62)
    ax.set_yticks(y, ordered["model"])
    ax.set_xlim(0, max(ordered["mae"]) * 1.18)
    ax.set_xlabel("Mean absolute error on 2024-25 games (points; lower is better)")
    delta = -uncertainty["mae_difference_vs_season_average"]
    ci_low = -uncertainty["ci95_high"]
    ci_high = -uncertainty["ci95_low"]
    fig.suptitle(
        "The models only slightly beat a player's season average",
        x=0.01,
        y=0.98,
        ha="left",
        color=INK,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.91,
        f"Tested on 2024-25; the best model improves MAE by {delta:.2f} points (95% CI: {ci_low:.2f} to {ci_high:.2f})",
        color=MUTED,
        fontsize=10,
        va="bottom",
    )
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    for bar, (_, row) in zip(bars, ordered.iterrows()):
        ax.text(row["mae"] + 0.06, bar.get_y() + bar.get_height() / 2, f"{row['mae']:.3f}", va="center", fontweight="bold", color=INK)

    add_source(fig, f"Source: NBA Stats API. Test: {uncertainty['n_test_games']:,} games from {uncertainty['n_test_players']:,} players in 2024-25.")
    fig.tight_layout(rect=[0, 0.03, 1, 0.86])
    finish_figure(fig, "q1_fig1_forecast_ladder")


def figure_error_by_role(test: pd.DataFrame, model_name: str) -> None:
    labels = ["Role player\n<8 PPG", "Contributor\n8-14", "Starter\n14-20", "Star\n20+"]
    test = test.copy()
    test["role"] = pd.cut(test["pts_season"], [-np.inf, 8, 14, 20, np.inf], labels=labels)
    summary = test.groupby("role", observed=False).agg(
        n=("PTS", "size"),
        mean_points=("PTS", "mean"),
        model_mae=("abs_error", "mean"),
        baseline_mae=("baseline_abs_error", "mean"),
    ).reset_index()
    summary["model_relative"] = 100 * summary["model_mae"] / summary["mean_points"]
    summary["baseline_relative"] = 100 * summary["baseline_mae"] / summary["mean_points"]
    summary.to_csv(TABLES / "q1_error_by_role.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), gridspec_kw={"wspace": 0.23})
    x = np.arange(len(summary))
    width = 0.34
    for ax, model_col, base_col, ylabel, title in [
        (axes[0], "model_mae", "baseline_mae", "Mean absolute error (points)", "Error measured in points"),
        (axes[1], "model_relative", "baseline_relative", "Error as % of average points", "Error relative to average scoring"),
    ]:
        ax.bar(x - width / 2, summary[base_col], width, label="Season average", color="#B8C1CC")
        ax.bar(x + width / 2, summary[model_col], width, label=model_name, color=TEAL)
        ax.set_xticks(x, summary["role"])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, loc="left", color=INK)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        for i, n in enumerate(summary["n"]):
            ax.text(i, -0.13, f"n={n:,}", transform=ax.get_xaxis_transform(), ha="center", va="top", color=MUTED, fontsize=8.5)
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Prediction error changes by scoring role", x=0.01, ha="left", color=INK, fontsize=15, fontweight="bold")
    add_source(fig, "Source: NBA Stats API, held-out 2024-25 season. Roles are defined from pre-game season-to-date scoring.")
    fig.tight_layout(rect=[0, 0.03, 1, 0.91])
    finish_figure(fig, "q1_fig2_error_by_role")


def figure_feature_importance(model, train: pd.DataFrame, test: pd.DataFrame) -> None:
    sample = test.sample(min(6000, len(test)), random_state=RANDOM_SEED)
    importance = permutation_importance(
        model,
        sample[FEATURES],
        sample["PTS"],
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    readable = {
        "pts_season": "Season-to-date points",
        "min_roll5": "Minutes, last 5",
        "fga_roll5": "Shot attempts, last 5",
        "pts_roll10": "Points, last 10",
        "pts_roll5": "Points, last 5",
        "pts_roll3": "Points, last 3",
        "prev_pts": "Previous-game points",
        "pts_std10": "Scoring volatility",
        "fg_pct_roll5": "FG%, last 5",
        "ast_roll5": "Assists, last 5",
        "reb_roll5": "Rebounds, last 5",
        "tov_roll5": "Turnovers, last 5",
        "opp_pts_allowed_roll5": "Opponent points allowed",
        "rest_days": "Rest days",
        "is_home": "Home court",
        "is_b2b": "Back-to-back",
        "games_played_prior": "Season games played",
    }
    imp = pd.DataFrame({"feature": FEATURES, "mae_increase": importance.importances_mean, "std": importance.importances_std})
    imp["label"] = imp["feature"].map(readable)
    imp = imp.sort_values("mae_increase", ascending=False)
    imp.to_csv(TABLES / "q1_permutation_importance.csv", index=False)
    shown = imp.head(10).sort_values("mae_increase")

    fig, ax = plt.subplots(figsize=(9.1, 5.5))
    colors = [TEAL if "Season-to-date" in label or "Minutes" in label else BLUE for label in shown["label"]]
    ax.barh(shown["label"], shown["mae_increase"], xerr=shown["std"], color=colors, ecolor=MUTED, capsize=2)
    ax.axvline(0, color=GRID, linewidth=1)
    ax.set_xlabel("Increase in test MAE after shuffling feature (points)")
    fig.suptitle(
        "A player's usual scoring level is the strongest predictor",
        x=0.01,
        y=0.98,
        ha="left",
        color=INK,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(0.01, 0.91, "Permutation importance on held-out 2024-25 games", color=MUTED, fontsize=10, va="bottom")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    add_source(fig, "Source: NBA Stats API. Error bars show variation across five independent shuffles.")
    fig.tight_layout(rect=[0, 0.03, 1, 0.86])
    finish_figure(fig, "q1_fig3_feature_importance")


def main() -> None:
    seed_everything()
    set_style()
    df = build_features()
    train, test, fitted, predictions, results, uncertainty = fit_and_evaluate(df)
    results.to_csv(TABLES / "q1_model_results.csv", index=False)
    test[["PLAYER_ID", "PLAYER_NAME", "GAME_DATE", "SEASON", "PTS", "prediction", "baseline_prediction", "abs_error"]].to_csv(
        TABLES / "q1_test_predictions.csv", index=False
    )
    (TABLES / "q1_summary.json").write_text(json.dumps(uncertainty, indent=2) + "\n", encoding="utf-8")

    figure_model_ladder(results, uncertainty)
    figure_error_by_role(test, uncertainty["best_model"])
    if uncertainty["best_model"] in fitted:
        figure_feature_importance(fitted[uncertainty["best_model"]], train, test)

    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(json.dumps(uncertainty, indent=2))


if __name__ == "__main__":
    main()
