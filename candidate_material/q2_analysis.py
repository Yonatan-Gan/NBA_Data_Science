"""Alternative Q2: does roster continuity add value beyond prior talent?"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    BLUE,
    GRID,
    INK,
    MUTED,
    NBA,
    ORANGE,
    PALE_BLUE,
    PALE_ORANGE,
    RANDOM_SEED,
    RED,
    SEASONS,
    SEASON_ORDER,
    TABLES,
    TEAL,
    add_source,
    cluster_bootstrap_difference,
    finish_figure,
    seed_everything,
    set_style,
)


ANALYSIS_SEASONS = SEASONS[1:]  # 2020-21 onward; the first season supplies lagged data.


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def _weighted_std(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    x = values[mask].to_numpy(float)
    w = weights[mask].to_numpy(float)
    mu = np.average(x, weights=w)
    return float(np.sqrt(np.average((x - mu) ** 2, weights=w)))


def load_player_team_minutes() -> pd.DataFrame:
    """Rebuild exact player-team membership from game logs, including trades."""
    frames = []
    for season in SEASONS:
        path = NBA / "game_logs" / "players" / season / "player_gamelogs_regular_season_base.csv"
        frame = pd.read_csv(
            path,
            usecols=["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "MIN"],
            low_memory=False,
        )
        frame["SEASON"] = season
        frame["MIN"] = pd.to_numeric(frame["MIN"], errors="coerce")
        frame = frame[frame["MIN"] > 0].drop_duplicates(["PLAYER_ID", "TEAM_ID", "GAME_ID"])
        frames.append(frame)
    logs = pd.concat(frames, ignore_index=True)
    return (
        logs.groupby(["SEASON", "TEAM_ID", "TEAM_ABBREVIATION", "PLAYER_ID", "PLAYER_NAME"], as_index=False)
        .agg(MINUTES=("MIN", "sum"), GAMES=("GAME_ID", "nunique"))
    )


def load_player_attributes() -> pd.DataFrame:
    """Season-level age/usage and strictly lagged player talent."""
    frames = []
    for season in SEASONS:
        usage_path = NBA / "season_stats" / season / "player_stats_regular_season_usage.csv"
        adv_path = NBA / "season_stats" / season / "player_stats_regular_season_advanced.csv"
        usage = pd.read_csv(usage_path, usecols=["PLAYER_ID", "AGE", "USG_PCT"], low_memory=False)
        advanced = pd.read_csv(adv_path, usecols=["PLAYER_ID", "PIE", "TS_PCT"], low_memory=False)
        attrs = usage.merge(advanced, on="PLAYER_ID", how="outer").drop_duplicates("PLAYER_ID")
        attrs["SEASON"] = season
        frames.append(attrs)
    current = pd.concat(frames, ignore_index=True)

    # Merge each roster with the player's previous-season performance. This is
    # available before the current season and therefore avoids outcome leakage.
    previous = current[["PLAYER_ID", "SEASON", "PIE", "TS_PCT"]].copy()
    previous["SEASON"] = previous["SEASON"].map(
        lambda season: SEASONS[SEASON_ORDER[season] + 1] if SEASON_ORDER[season] + 1 < len(SEASONS) else None
    )
    previous = previous.rename(columns={"PIE": "PREV_PIE", "TS_PCT": "PREV_TS"}).dropna(subset=["SEASON"])
    return current.merge(previous, on=["PLAYER_ID", "SEASON"], how="left")


def load_team_outcomes() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        path = NBA / "season_stats" / season / "team_stats_regular_season_advanced.csv"
        frame = pd.read_csv(
            path,
            usecols=["TEAM_ID", "TEAM_NAME", "W_PCT", "NET_RATING", "AST_PCT"],
            low_memory=False,
        )
        frame["SEASON"] = season
        frames.append(frame)
    outcomes = pd.concat(frames, ignore_index=True)

    previous = outcomes[["TEAM_ID", "SEASON", "NET_RATING"]].copy()
    previous["SEASON"] = previous["SEASON"].map(
        lambda season: SEASONS[SEASON_ORDER[season] + 1] if SEASON_ORDER[season] + 1 < len(SEASONS) else None
    )
    previous = previous.rename(columns={"NET_RATING": "PREV_TEAM_NET_RATING"}).dropna(subset=["SEASON"])
    return outcomes.merge(previous, on=["TEAM_ID", "SEASON"], how="left")


def build_team_panel() -> pd.DataFrame:
    roster = load_player_team_minutes()
    attrs = load_player_attributes()
    roster = roster.merge(attrs, on=["PLAYER_ID", "SEASON"], how="left")

    membership = set(zip(roster["SEASON"], roster["TEAM_ID"], roster["PLAYER_ID"]))
    previous_season = {SEASONS[i]: SEASONS[i - 1] for i in range(1, len(SEASONS))}
    roster["RETURNER"] = [
        float((previous_season.get(season), team, player) in membership)
        for season, team, player in zip(roster["SEASON"], roster["TEAM_ID"], roster["PLAYER_ID"])
    ]

    # Keep meaningful contributors; ultra-short stints otherwise make roster
    # size and diversity metrics unstable. Minutes still weight every feature.
    rotation = roster[roster["MINUTES"] >= 150].copy()
    season_median_pie = rotation.groupby("SEASON")["PREV_PIE"].transform("median")
    rotation["PREV_PIE_FILLED"] = rotation["PREV_PIE"].fillna(season_median_pie)
    rotation["PREV_PIE_MISSING"] = rotation["PREV_PIE"].isna().astype(float)

    rows = []
    for (season, team_id, abbr), group in rotation.groupby(["SEASON", "TEAM_ID", "TEAM_ABBREVIATION"]):
        weights = group["MINUTES"]
        shares = weights / weights.sum()
        prior_sorted = group.sort_values("PREV_PIE_FILLED", ascending=False)
        rows.append(
            {
                "SEASON": season,
                "TEAM_ID": team_id,
                "TEAM_ABBREVIATION": abbr,
                "ROTATION_PLAYERS": len(group),
                "CONTINUITY": float(np.average(group["RETURNER"], weights=weights)),
                "AGE_MEAN": _weighted_mean(group["AGE"], weights),
                "AGE_SD": _weighted_std(group["AGE"], weights),
                "ROLE_HHI": float(np.square(shares).sum()),
                "TOP2_MIN_SHARE": float(shares.nlargest(2).sum()),
                "USAGE_MEAN": _weighted_mean(group["USG_PCT"], weights),
                "USAGE_SD": _weighted_std(group["USG_PCT"], weights),
                "PRIOR_PIE_MEAN": _weighted_mean(group["PREV_PIE_FILLED"], weights),
                "PRIOR_STAR_PIE": float(prior_sorted["PREV_PIE_FILLED"].head(2).mean()),
                "PRIOR_PIE_MISSING_SHARE": _weighted_mean(group["PREV_PIE_MISSING"], weights),
            }
        )
    panel = pd.DataFrame(rows)
    panel = panel.merge(load_team_outcomes(), on=["SEASON", "TEAM_ID"], how="inner")
    panel = panel[panel["SEASON"].isin(ANALYSIS_SEASONS)].copy()
    panel["TALENT_X_CONTINUITY"] = panel["PRIOR_PIE_MEAN"] * panel["CONTINUITY"]
    return panel.sort_values(["SEASON", "TEAM_ABBREVIATION"]).reset_index(drop=True)


FEATURE_SETS = {
    "Prior team only": ["PREV_TEAM_NET_RATING"],
    "+ roster talent": [
        "PREV_TEAM_NET_RATING",
        "PRIOR_PIE_MEAN",
        "PRIOR_STAR_PIE",
        "PRIOR_PIE_MISSING_SHARE",
    ],
    "+ roster composition": [
        "PREV_TEAM_NET_RATING",
        "PRIOR_PIE_MEAN",
        "PRIOR_STAR_PIE",
        "PRIOR_PIE_MISSING_SHARE",
        "AGE_MEAN",
        "AGE_SD",
        "ROLE_HHI",
        "TOP2_MIN_SHARE",
        "USAGE_SD",
    ],
    "+ continuity": [
        "PREV_TEAM_NET_RATING",
        "PRIOR_PIE_MEAN",
        "PRIOR_STAR_PIE",
        "PRIOR_PIE_MISSING_SHARE",
        "AGE_MEAN",
        "AGE_SD",
        "ROLE_HHI",
        "TOP2_MIN_SHARE",
        "USAGE_SD",
        "CONTINUITY",
        "TALENT_X_CONTINUITY",
    ],
}


def _ridge() -> object:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=12.0))


def leave_one_season_out(panel: pd.DataFrame):
    logo = LeaveOneGroupOut()
    y = panel["NET_RATING"].to_numpy()
    groups = panel["SEASON"].to_numpy()
    all_predictions = {}
    rows = []
    for name, features in FEATURE_SETS.items():
        pred = np.full(len(panel), np.nan)
        for train_idx, test_idx in logo.split(panel, y, groups):
            model = _ridge()
            model.fit(panel.iloc[train_idx][features], y[train_idx])
            pred[test_idx] = model.predict(panel.iloc[test_idx][features])
        all_predictions[name] = pred
        rows.append(
            {
                "model": name,
                "n_features": len(features),
                "mae": mean_absolute_error(y, pred),
                "r2": r2_score(y, pred),
            }
        )
    return pd.DataFrame(rows), all_predictions


def figure_model_ladder(results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), gridspec_kw={"wspace": 0.3})
    labels = results["model"].tolist()
    x = np.arange(len(labels))
    colors = [MUTED, BLUE, ORANGE, TEAL]
    axes[0].bar(x, results["mae"], color=colors, width=0.66)
    axes[1].bar(x, results["r2"], color=colors, width=0.66)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("MAE in net-rating points")
    axes[1].set_ylabel("Out-of-season R-squared")
    axes[1].axhline(0, color=GRID, linewidth=1)
    for ax, metric, fmt in [(axes[0], "mae", ".3f"), (axes[1], "r2", ".3f")]:
        for i, value in enumerate(results[metric]):
            va = "bottom" if value >= 0 else "top"
            offset = 0.05 if value >= 0 else -0.05
            ax.text(i, value + offset, format(value, fmt), ha="center", va=va, fontweight="bold", color=INK)
    fig.suptitle("Most of the improvement comes from prior roster talent", x=0.01, y=0.98, ha="left", color=INK, fontsize=15, fontweight="bold")
    fig.text(0.01, 0.91, "Full-season roster summaries, tested by leaving out one season at a time", color=MUTED, fontsize=10)
    add_source(fig, "Source: NBA Stats API, 150 team-seasons from 2020-21 through 2024-25.")
    fig.tight_layout(rect=[0, 0.03, 1, 0.84])
    finish_figure(fig, "q2_fig1_incremental_prediction")


def _clustered_line(x: np.ndarray, y: np.ndarray, clusters: np.ndarray) -> dict:
    """Fit y = intercept + slope*x with standard errors clustered by team."""
    design = np.column_stack([np.ones(len(x)), x])
    bread = np.linalg.inv(design.T @ design)
    beta = bread @ design.T @ y
    residual = y - design @ beta
    meat = np.zeros((2, 2))
    unique = np.unique(clusters)
    for cluster in unique:
        mask = clusters == cluster
        score = design[mask].T @ residual[mask]
        meat += np.outer(score, score)
    n, k, g = len(x), design.shape[1], len(unique)
    correction = (g / (g - 1)) * ((n - 1) / (n - k))
    covariance = correction * bread @ meat @ bread
    slope_se = float(np.sqrt(covariance[1, 1]))
    t_value = float(beta[1] / slope_se)
    p_value = float(2 * stats.t.sf(abs(t_value), df=g - 1))
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "covariance": covariance,
        "pvalue": p_value,
        "df": g - 1,
    }


def residualized_continuity(panel: pd.DataFrame):
    controls = ["PREV_TEAM_NET_RATING", "PRIOR_PIE_MEAN", "PRIOR_STAR_PIE", "PRIOR_PIE_MISSING_SHARE"]
    clean = panel.dropna(subset=controls + ["CONTINUITY", "NET_RATING"]).copy()
    season_dummies = pd.get_dummies(clean["SEASON"], drop_first=True, dtype=float)
    X = np.column_stack([clean[controls].to_numpy(), season_dummies.to_numpy()])
    y_resid = clean["NET_RATING"].to_numpy() - LinearRegression().fit(X, clean["NET_RATING"]).predict(X)
    x_resid = clean["CONTINUITY"].to_numpy() - LinearRegression().fit(X, clean["CONTINUITY"]).predict(X)
    fit = _clustered_line(x_resid, y_resid, clean["TEAM_ID"].to_numpy())
    clean["continuity_residual"] = x_resid
    clean["net_rating_residual"] = y_resid
    return clean, fit


def figure_continuity(panel: pd.DataFrame):
    clean, fit = residualized_continuity(panel)
    x = clean["continuity_residual"].to_numpy()
    y = clean["net_rating_residual"].to_numpy()
    order = np.argsort(x)
    xs = np.linspace(x.min(), x.max(), 200)
    yhat = fit["intercept"] + fit["slope"] * xs
    prediction_design = np.column_stack([np.ones(len(xs)), xs])
    se_mean = np.sqrt(np.einsum("ij,jk,ik->i", prediction_design, fit["covariance"], prediction_design))
    critical = stats.t.ppf(0.975, fit["df"])

    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    ax.scatter(100 * x, y, s=38, alpha=0.62, color=BLUE, edgecolor="white", linewidth=0.45)
    ax.plot(100 * xs, yhat, color=TEAL, linewidth=2.6)
    ax.fill_between(100 * xs, yhat - critical * se_mean, yhat + critical * se_mean, color=TEAL, alpha=0.16)
    ax.axhline(0, color=GRID, linewidth=1)
    ax.axvline(0, color=GRID, linewidth=1)
    ax.set_xlabel("Adjusted continuity (percentage points)")
    ax.set_ylabel("Adjusted net rating")
    ax.set_title("Continuity has a small positive association after controlling for talent", loc="left", color=INK)
    ax.text(
        0.02,
        0.97,
        f"+10 percentage points continuity is linked to {fit['slope'] * 0.10:+.2f} net-rating points\np = {fit['pvalue']:.3f}; team-clustered 95% confidence band",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color=INK,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": PALE_BLUE, "edgecolor": "none"},
    )
    add_source(fig, "Controls: season, prior team net rating, prior roster PIE, star PIE, and missing prior data. Association, not causation.")
    fig.tight_layout()
    finish_figure(fig, "q2_fig2_continuity_signal")
    return fit


def figure_talent_continuity_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy()
    data["talent_group"] = pd.qcut(data["PRIOR_PIE_MEAN"], 2, labels=["Lower prior talent", "Higher prior talent"])
    data["continuity_group"] = pd.qcut(data["CONTINUITY"], 2, labels=["Lower continuity", "Higher continuity"])
    matrix = data.pivot_table(index="talent_group", columns="continuity_group", values="NET_RATING", aggfunc="mean", observed=False)
    counts = data.pivot_table(index="talent_group", columns="continuity_group", values="NET_RATING", aggfunc="size", observed=False)

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    vmax = max(abs(matrix.to_numpy().min()), abs(matrix.to_numpy().max()))
    image = ax.imshow(matrix.to_numpy(), cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.suptitle("Teams do best when they have both talent and continuity", x=0.01, y=0.98, ha="left", color=INK, fontsize=15, fontweight="bold")
    fig.text(0.01, 0.91, "Average current-season net rating; median splits", color=MUTED, fontsize=10, va="bottom")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            color = "white" if abs(value) > 0.55 * vmax else INK
            ax.text(j, i - 0.04, f"{value:+.1f}", ha="center", va="center", fontsize=19, fontweight="bold", color=color)
            ax.text(j, i + 0.27, f"n={int(counts.iloc[i, j])}", ha="center", va="center", fontsize=9, color=color)
    cbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.04)
    cbar.set_label("Net rating")
    add_source(fig, "Source: NBA Stats API, 2020-21 through 2024-25. Talent uses current-roster players' prior-season PIE.")
    fig.tight_layout(rect=[0, 0.03, 1, 0.85])
    finish_figure(fig, "q2_fig3_talent_continuity_matrix")

    out = matrix.stack().rename("mean_net_rating").reset_index()
    out["n"] = counts.stack().to_numpy()
    return out


def main() -> None:
    seed_everything()
    set_style()
    panel = build_team_panel()
    results, predictions = leave_one_season_out(panel)
    panel.to_csv(TABLES / "q2_team_season_panel.csv", index=False)
    results.to_csv(TABLES / "q2_model_results.csv", index=False)
    for name, pred in predictions.items():
        panel[f"PRED_{name}"] = pred
    panel.to_csv(TABLES / "q2_oof_predictions.csv", index=False)

    y = panel["NET_RATING"].to_numpy()
    with_continuity = np.abs(y - predictions["+ continuity"])
    without_continuity = np.abs(y - predictions["+ roster composition"])
    diff, diff_lo, diff_hi = cluster_bootstrap_difference(
        with_continuity,
        without_continuity,
        panel["TEAM_ID"].to_numpy(),
        n_boot=5000,
    )

    figure_model_ladder(results)
    fit = figure_continuity(panel)
    matrix = figure_talent_continuity_matrix(panel)
    matrix.to_csv(TABLES / "q2_talent_continuity_matrix.csv", index=False)

    summary = {
        "n_team_seasons": int(len(panel)),
        "seasons": ANALYSIS_SEASONS,
        "continuity_slope_per_10pp": float(fit["slope"] * 0.10),
        "continuity_p_value": float(fit["pvalue"]),
        "continuity_mae_gain": float(-diff),
        "continuity_gain_ci95_low": float(-diff_hi),
        "continuity_gain_ci95_high": float(-diff_lo),
        "best_model": results.sort_values("mae").iloc[0]["model"],
        "best_mae": float(results["mae"].min()),
        "best_r2": float(results.loc[results["mae"].idxmin(), "r2"]),
    }
    (TABLES / "q2_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
