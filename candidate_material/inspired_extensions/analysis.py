"""Build three research-inspired extensions for the NBA project."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
ROOT = CANDIDATE.parent
sys.path.insert(0, str(CANDIDATE))

import q1_analysis as q1  # noqa: E402
import q2_analysis as q2  # noqa: E402


OUTPUT = HERE / "output"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"
PDF_FIGURES = OUTPUT / "figures"
for directory in [FIGURES, TABLES, OUTPUT / "pdf"]:
    directory.mkdir(parents=True, exist_ok=True)

INK = "#182230"
MUTED = "#64748B"
BLUE = "#2563A6"
TEAL = "#138A7E"
ORANGE = "#D97706"
GRID = "#DDE3EA"
PALE_BLUE = "#E8F1F8"

SEED = 42
ROLE_LABELS = ["Role player", "Contributor", "Starter", "Star"]
ROLE_BINS = [-np.inf, 8, 14, 20, np.inf]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_FIGURES / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def higher_quantile(values: pd.Series | np.ndarray, level: float) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, level, method="higher"))


def build_q1_intervals() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Train, calibrate, and test time-ordered empirical scoring ranges."""
    games = q1.build_features()
    train_seasons = ["2019-20", "2020-21", "2021-22", "2022-23"]
    train = games[games["SEASON"].isin(train_seasons)].copy()
    calibration = games[games["SEASON"] == "2023-24"].copy()
    test = games[games["SEASON"] == "2024-25"].copy()

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=20.0),
    )
    model.fit(train[q1.FEATURES], train["PTS"])
    calibration["prediction"] = model.predict(calibration[q1.FEATURES])
    test["prediction"] = model.predict(test[q1.FEATURES])
    calibration["abs_error"] = (calibration["PTS"] - calibration["prediction"]).abs()
    test["abs_error"] = (test["PTS"] - test["prediction"]).abs()

    coverage_rows = []
    for target in [0.50, 0.80, 0.90]:
        half_width = higher_quantile(calibration["abs_error"], target)
        observed = float((test["abs_error"] <= half_width).mean())
        coverage_rows.append(
            {
                "target_coverage": target,
                "half_width_points": half_width,
                "observed_coverage": observed,
                "n_calibration_games": len(calibration),
                "n_test_games": len(test),
            }
        )
    coverage = pd.DataFrame(coverage_rows)

    for frame in [calibration, test]:
        frame["scoring_role"] = pd.cut(
            frame["pts_season"], bins=ROLE_BINS, labels=ROLE_LABELS
        )

    role_rows = []
    for role in ROLE_LABELS:
        calibration_role = calibration[calibration["scoring_role"] == role]
        test_role = test[test["scoring_role"] == role]
        half_width = higher_quantile(calibration_role["abs_error"], 0.80)
        role_rows.append(
            {
                "scoring_role": role,
                "half_width_points": half_width,
                "observed_coverage": float((test_role["abs_error"] <= half_width).mean()),
                "n_calibration_games": len(calibration_role),
                "n_test_games": len(test_role),
            }
        )
    roles = pd.DataFrame(role_rows)

    coverage.to_csv(TABLES / "q1_interval_coverage.csv", index=False)
    roles.to_csv(TABLES / "q1_interval_by_role.csv", index=False)

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(11.4, 5.4), gridspec_kw={"width_ratios": [1.08, 0.92], "wspace": 0.34}
    )
    fig.suptitle(
        "A useful scoring forecast is a range, not one number",
        x=0.01,
        y=0.985,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.01,
        0.92,
        "The tested 80% range is almost twice as wide for stars as it is for role players.",
        color=MUTED,
        fontsize=10.5,
    )

    role_plot = roles.iloc[::-1].reset_index(drop=True)
    colors = [TEAL if role == "Star" else BLUE for role in role_plot["scoring_role"]]
    bars = ax_left.barh(
        role_plot["scoring_role"], role_plot["half_width_points"], color=colors, height=0.58
    )
    ax_left.set_title("Half-width of the 80% scoring range", loc="left", fontweight="bold")
    ax_left.set_xlabel("Points above and below the model prediction")
    ax_left.set_xlim(0, 11.2)
    ax_left.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax_left.set_axisbelow(True)
    for bar, value in zip(bars, role_plot["half_width_points"]):
        ax_left.text(
            value - 0.16,
            bar.get_y() + bar.get_height() / 2,
            f"+/- {value:.1f} pts",
            va="center",
            ha="right",
            fontweight="bold",
            color="white",
        )

    target = 100 * coverage["target_coverage"].to_numpy()
    observed = 100 * coverage["observed_coverage"].to_numpy()
    ax_right.plot(
        [45, 95], [45, 95], color=MUTED, linewidth=1.4, linestyle="--", label="Perfect calibration"
    )
    ax_right.plot(target, observed, color=TEAL, linewidth=2.5, marker="o", markersize=7)
    ax_right.set_title("The ranges stay calibrated one season later", loc="left", fontweight="bold")
    ax_right.set_xlabel("Target coverage")
    ax_right.set_ylabel("Observed 2024-25 coverage")
    ax_right.set_xlim(45, 95)
    ax_right.set_ylim(45, 95)
    ax_right.set_xticks(target, [f"{value:.0f}%" for value in target])
    ax_right.set_yticks([50, 60, 70, 80, 90], ["50%", "60%", "70%", "80%", "90%"])
    ax_right.grid(True, color=GRID, linewidth=0.8)
    ax_right.set_axisbelow(True)
    for x, y in zip(target, observed):
        ax_right.annotate(
            f"{y:.1f}%",
            (x, y),
            xytext=(7, -12 if x == 90 else 7),
            textcoords="offset points",
            fontweight="bold",
            color=INK,
        )
    ax_right.text(
        0.03,
        0.96,
        "Dashed line: perfect calibration",
        transform=ax_right.transAxes,
        va="top",
        color=MUTED,
        fontsize=8.5,
    )

    fig.text(
        0.01,
        0.015,
        "Method: Ridge model trained through 2022-23; range sizes learned on 2023-24; final check on 2024-25. Roles use pregame scoring averages.",
        color=MUTED,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.14, top=0.84, wspace=0.34)
    save_figure(fig, "q1_forecast_ranges")

    summary = {
        "train_seasons": train_seasons,
        "calibration_season": "2023-24",
        "test_season": "2024-25",
        "n_calibration_games": int(len(calibration)),
        "n_test_games": int(len(test)),
        "test_mae": float(test["abs_error"].mean()),
        "global_80_half_width": float(
            coverage.loc[coverage["target_coverage"] == 0.80, "half_width_points"].iloc[0]
        ),
        "global_80_coverage": float(
            coverage.loc[coverage["target_coverage"] == 0.80, "observed_coverage"].iloc[0]
        ),
    }
    return coverage, roles, summary


def clustered_ols(
    data: pd.DataFrame, controls: list[str]
) -> dict[str, float | int]:
    """OLS with season indicators and standard errors clustered by team."""
    required = ["NET_RATING", "CONTINUITY", "TEAM_ID", "SEASON"] + controls
    clean = data.dropna(subset=required).copy()
    season_dummies = pd.get_dummies(clean["SEASON"], drop_first=True, dtype=float)
    columns = [np.ones(len(clean)), clean["CONTINUITY"].to_numpy(float)]
    columns.extend(clean[column].to_numpy(float) for column in controls)
    columns.extend(season_dummies[column].to_numpy(float) for column in season_dummies)
    design = np.column_stack(columns)
    outcome = clean["NET_RATING"].to_numpy(float)

    bread = np.linalg.inv(design.T @ design)
    beta = bread @ design.T @ outcome
    residual = outcome - design @ beta
    clusters = clean["TEAM_ID"].to_numpy()
    unique_clusters = np.unique(clusters)
    meat = np.zeros((design.shape[1], design.shape[1]))
    for cluster in unique_clusters:
        mask = clusters == cluster
        score = design[mask].T @ residual[mask]
        meat += np.outer(score, score)

    n = len(clean)
    k = design.shape[1]
    g = len(unique_clusters)
    correction = (g / (g - 1)) * ((n - 1) / (n - k))
    covariance = correction * bread @ meat @ bread
    slope = float(beta[1])
    slope_se = float(np.sqrt(covariance[1, 1]))
    critical = float(stats.t.ppf(0.975, g - 1))
    t_value = slope / slope_se

    return {
        "effect_per_10pp": 0.10 * slope,
        "ci95_low": 0.10 * (slope - critical * slope_se),
        "ci95_high": 0.10 * (slope + critical * slope_se),
        "p_value": float(2 * stats.t.sf(abs(t_value), df=g - 1)),
        "n_team_seasons": n,
        "n_teams": g,
    }


def build_q2_adjustment() -> tuple[pd.DataFrame, dict]:
    """Show how the continuity coefficient changes as confounders are added."""
    panel = q2.build_team_panel()
    specifications = [
        ("Continuity only", []),
        ("Add prior team strength", ["PREV_TEAM_NET_RATING"]),
        (
            "Add prior roster talent",
            [
                "PREV_TEAM_NET_RATING",
                "PRIOR_PIE_MEAN",
                "PRIOR_STAR_PIE",
                "PRIOR_PIE_MISSING_SHARE",
            ],
        ),
        (
            "Add roster composition",
            [
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
        ),
    ]

    rows = []
    for order, (label, controls) in enumerate(specifications):
        result = clustered_ols(panel, controls)
        rows.append({"order": order, "model": label, "n_controls": len(controls), **result})
    results = pd.DataFrame(rows)
    results.to_csv(TABLES / "q2_continuity_adjustment.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    fig.suptitle(
        "The continuity effect shrinks after accounting for roster talent",
        x=0.01,
        y=0.985,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.01,
        0.92,
        "Good teams often keep good players. The raw continuity link includes both effects.",
        color=MUTED,
        fontsize=10.5,
    )

    display = results.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(display))
    colors = [TEAL if name == "Add roster composition" else BLUE for name in display["model"]]
    x = display["effect_per_10pp"].to_numpy()
    xerr = np.vstack(
        [x - display["ci95_low"].to_numpy(), display["ci95_high"].to_numpy() - x]
    )
    for index, color in enumerate(colors):
        ax.errorbar(
            x[index],
            y[index],
            xerr=xerr[:, index].reshape(2, 1),
            fmt="none",
            ecolor=color,
            elinewidth=3,
            capsize=6,
            capthick=2,
            zorder=2,
        )
    ax.scatter(x, y, s=110, c=colors, edgecolor="white", linewidth=1.1, zorder=3)
    ax.axvline(0, color=INK, linewidth=1.1)
    ax.set_yticks(y, display["model"])
    ax.set_xlabel("Estimated net-rating change for 10 percentage points more continuity")
    ax.set_xlim(-0.55, 2.15)
    ax.set_ylim(-0.65, len(display) - 0.35)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for yi, row in display.iterrows():
        ax.text(
            min(row["ci95_high"] + 0.08, 2.03),
            yi,
            f"{row['effect_per_10pp']:+.2f}  [{row['ci95_low']:+.2f}, {row['ci95_high']:+.2f}]",
            va="center",
            fontsize=9,
            fontweight="bold" if row["model"] == "Add roster composition" else "normal",
            color=INK,
        )
    shrink = 1 - results.iloc[-1]["effect_per_10pp"] / results.iloc[0]["effect_per_10pp"]
    ax.text(
        0.01,
        0.04,
        f"After all controls, the estimate is {100 * shrink:.0f}% smaller and its interval includes zero.",
        transform=ax.transAxes,
        fontsize=10,
        color=INK,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": PALE_BLUE, "edgecolor": "none"},
    )
    fig.text(
        0.01,
        0.015,
        "All models include season indicators. Lines are 95% confidence intervals with standard errors clustered by team. Association, not causation.",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.88])
    save_figure(fig, "q2_continuity_confounding")

    summary = {
        "n_team_seasons": int(len(panel)),
        "n_teams": int(panel["TEAM_ID"].nunique()),
        "raw_effect_per_10pp": float(results.iloc[0]["effect_per_10pp"]),
        "full_effect_per_10pp": float(results.iloc[-1]["effect_per_10pp"]),
        "full_ci95_low": float(results.iloc[-1]["ci95_low"]),
        "full_ci95_high": float(results.iloc[-1]["ci95_high"]),
        "full_p_value": float(results.iloc[-1]["p_value"]),
        "percent_shrink": float(100 * shrink),
    }
    return results, summary


def cluster_bootstrap_mean(
    data: pd.DataFrame, value: str, cluster: str = "Player", n_boot: int = 4000
) -> tuple[float, float, float]:
    """Bootstrap a row mean while keeping repeated seasons for a player together."""
    grouped = data.groupby(cluster)[value].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    samples = np.empty(n_boot)
    for index in range(n_boot):
        draw = rng.integers(0, len(grouped), len(grouped))
        samples[index] = sums[draw].sum() / counts[draw].sum()
    return (
        float(data[value].mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def build_q3_decomposition() -> tuple[pd.DataFrame, dict]:
    """Split playoff PPG change into an exact minutes part and scoring-rate part."""
    path = ROOT / "data" / "processed" / "Kaggle" / "q3_player_split_v2.csv"
    data = pd.read_csv(path, low_memory=False)
    data = data.dropna(subset=["REG_PTS", "POF_PTS", "REG_MP", "POF_MP"]).copy()
    data = data[
        (data["REG_G"] >= 20)
        & (data["POF_G"] >= 5)
        & (data["REG_MP"] > 0)
        & (data["POF_MP"] > 0)
    ].copy()

    data["minutes_effect"] = (data["POF_MP"] - data["REG_MP"]) * (
        data["REG_PTS"] / data["REG_MP"]
    )
    data["scoring_rate_effect"] = data["POF_MP"] * (
        data["POF_PTS"] / data["POF_MP"] - data["REG_PTS"] / data["REG_MP"]
    )
    data["net_change"] = data["minutes_effect"] + data["scoring_rate_effect"]
    if not np.allclose(data["net_change"], data["POF_PTS"] - data["REG_PTS"]):
        raise AssertionError("Scoring decomposition does not reproduce the PPG change")

    data["scoring_role"] = pd.cut(
        data["REG_PTS"], bins=ROLE_BINS, labels=ROLE_LABELS
    )
    rows = []
    for role in ROLE_LABELS:
        group = data[data["scoring_role"] == role]
        net_mean, net_low, net_high = cluster_bootstrap_mean(group, "net_change")
        rows.append(
            {
                "scoring_role": role,
                "n_player_seasons": len(group),
                "n_players": group["Player"].nunique(),
                "regular_ppg": group["REG_PTS"].mean(),
                "playoff_ppg": group["POF_PTS"].mean(),
                "regular_minutes": group["REG_MP"].mean(),
                "playoff_minutes": group["POF_MP"].mean(),
                "minutes_effect": group["minutes_effect"].mean(),
                "scoring_rate_effect": group["scoring_rate_effect"].mean(),
                "net_change": net_mean,
                "net_ci95_low": net_low,
                "net_ci95_high": net_high,
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(TABLES / "q3_scoring_decomposition.csv", index=False)

    overall_net, overall_low, overall_high = cluster_bootstrap_mean(data, "net_change")
    overall = {
        "n_player_seasons": int(len(data)),
        "n_players": int(data["Player"].nunique()),
        "mean_net_change": overall_net,
        "net_ci95_low": overall_low,
        "net_ci95_high": overall_high,
        "mean_minutes_effect": float(data["minutes_effect"].mean()),
        "mean_scoring_rate_effect": float(data["scoring_rate_effect"].mean()),
    }

    fig, axes = plt.subplots(1, 4, figsize=(11.8, 5.35), sharey=True)
    fig.suptitle(
        "Stars maintain their scoring because extra minutes offset a lower scoring rate",
        x=0.01,
        y=0.985,
        ha="left",
        fontsize=15.5,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.01,
        0.92,
        "Each waterfall exactly splits the playoff change in points per game into two parts.",
        color=MUTED,
        fontsize=10.5,
    )

    for ax, (_, row) in zip(axes, results.iterrows()):
        minutes = float(row["minutes_effect"])
        rate = float(row["scoring_rate_effect"])
        net = float(row["net_change"])
        ax.axhline(0, color=INK, linewidth=1)
        ax.bar(0, minutes, width=0.62, color=BLUE)
        ax.plot([0.31, 0.69], [minutes, minutes], color=MUTED, linewidth=1)
        ax.bar(1, rate, bottom=minutes, width=0.62, color=ORANGE)
        lower_error = net - float(row["net_ci95_low"])
        upper_error = float(row["net_ci95_high"]) - net
        ax.errorbar(
            2,
            net,
            yerr=np.array([[lower_error], [upper_error]]),
            fmt="D",
            color=TEAL,
            markerfacecolor=TEAL,
            markeredgecolor="white",
            markersize=8,
            capsize=4,
            linewidth=1.5,
            zorder=4,
        )
        ax.set_title(
            f"{row['scoring_role']}\nn = {int(row['n_player_seasons']):,}",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.set_xticks([0, 1, 2], ["Minutes", "Scoring\nrate", "Net"])
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(-3.1, 3.1)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

        def label_value(x_position: float, start: float, change: float, color: str) -> None:
            end = start + change
            offset = 0.14 if change >= 0 else -0.14
            ax.text(
                x_position,
                end + offset,
                f"{change:+.2f}",
                ha="center",
                va="bottom" if change >= 0 else "top",
                fontsize=9,
                fontweight="bold",
                color=color,
            )

        label_value(0, 0, minutes, BLUE)
        label_value(1, minutes, rate, ORANGE)
        net_label_y = (
            float(row["net_ci95_high"]) + 0.14
            if net >= 0
            else float(row["net_ci95_low"]) - 0.14
        )
        ax.text(
            2,
            net_label_y,
            f"{net:+.2f}",
            ha="center",
            va="bottom" if net >= 0 else "top",
            fontsize=9,
            fontweight="bold",
            color=TEAL,
        )
    axes[0].set_ylabel("Change in playoff points per game")
    fig.text(0.18, 0.075, "Blue: minutes effect", color=BLUE, fontweight="bold", fontsize=9)
    fig.text(0.43, 0.075, "Orange: points-per-minute effect", color=ORANGE, fontweight="bold", fontsize=9)
    fig.text(0.76, 0.075, "Diamond: net change", color=TEAL, fontweight="bold", fontsize=9)
    fig.text(
        0.01,
        0.015,
        "Data: 1995-96 through 2022-23. Minimum 20 regular-season games and 5 playoff games. Net intervals resample players, keeping repeated seasons together.",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.11, 1, 0.88], w_pad=1.0)
    save_figure(fig, "q3_playoff_scoring_decomposition")
    return results, overall


def main() -> None:
    np.random.seed(SEED)
    set_style()
    _, _, q1_summary = build_q1_intervals()
    _, q2_summary = build_q2_adjustment()
    _, q3_summary = build_q3_decomposition()
    summary = {"q1": q1_summary, "q2": q2_summary, "q3": q3_summary}
    (TABLES / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
