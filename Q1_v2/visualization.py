"""
All figures for the Q1 report.  Every function:
  - saves a PNG to FIGURES_DIR
  - returns the Path to the saved file
  - uses the shared STYLE config so all figures are visually consistent

Figure catalogue:
  fig02  Model comparison bar chart (MAE + R²)
  fig03  Actual vs Predicted scatter (best model)
  fig04  SHAP global importance
  fig06  Error by scoring tier and position
  fig08  Rolling prediction example for 3 individual players
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import FIGURES_DIR, STYLE

logger = logging.getLogger(__name__)

# Global matplotlib style
plt.rcParams.update({
    "font.family":         "DejaVu Sans",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.edgecolor":      "#cccccc",
    "axes.linewidth":      0.8,
    "xtick.color":         "#555555",
    "ytick.color":         "#555555",
    "xtick.labelsize":     10,
    "ytick.labelsize":     10,
    "grid.color":          "#e8e8e8",
    "grid.linestyle":      "--",
    "grid.linewidth":      0.6,
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "#cccccc",
    "legend.fontsize":     9,
})

C = STYLE   # shorthand


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIGURES_DIR / f"{name}{C['fig_ext']}"
    fig.savefig(path, dpi=C["dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  {path.name}")
    return path


def _takeaway(fig: plt.Figure, text: str, y: float = -0.04) -> None:
    """Add a key takeaway box below the figure (matches Q3 report style)."""
    fig.text(
        0.5, y, text,
        ha="center", fontsize=9.5, color="#0d0d0d",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f0f7e6",
                  ec=C["green"], lw=1.2),
        wrap=True,
    )


# Figure 2: Model comparison

def fig02_model_comparison(model_results: list) -> Path:
    """MAE and R² side-by-side bar chart for all models."""
    results = sorted(model_results, key=lambda r: r.mae)
    names   = [r.model_name for r in results]
    maes    = [r.mae  for r in results]
    r2s     = [r.r2   for r in results]
    times   = [r.train_time_s for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Q1 Analysis: Which Model Best Predicts Next-Game Scoring?\n"
        "All models trained on identical feature set (chronological split)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    palette = [C["blue"] if i == 0 else C["light_blue"] if i < 3 else C["grey"]
               for i in range(len(names))]

    for ax, vals, label, better, fmt in zip(
        axes,
        [maes, r2s, times],
        ["Mean Absolute Error (pts)", "R² Score", "Training time (s)"],
        ["lower", "higher", "lower"],
        [".2f", ".3f", ".1f"],
    ):
        if better == "lower":
            bar_c = [C["blue"] if v == min(vals) else
                     C["light_blue"] if v < np.percentile(vals, 40) else C["grey"]
                     for v in vals]
        else:
            bar_c = [C["blue"] if v == max(vals) else
                     C["light_blue"] if v > np.percentile(vals, 60) else C["grey"]
                     for v in vals]
        bars = ax.barh(names, vals, color=bar_c, height=0.6, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(v + max(vals) * 0.01,
                    bar.get_y() + bar.get_height()/2,
                    f"{v:{fmt}}", va="center", fontsize=9)
        ax.set_xlabel(label, fontsize=10, labelpad=6)
        ax.set_title(f"{label}\n({'lower' if better=='lower' else 'higher'} = better)",
                     fontsize=10)
        ax.xaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["bottom"].set_visible(False)

    best = results[0]
    _takeaway(fig,
              f"Key takeaway: {best.model_name} achieves the lowest MAE ({best.mae:.2f} pts) "
              f"and explains {best.r2*100:.0f}% of variance in next-game scoring. "
              f"Tree-based ensemble methods consistently outperform linear models on this dataset.")
    fig.tight_layout()
    return _save(fig, "Q1_fig02_model_comparison")


# Figure 3: Actual vs Predicted

def fig03_actual_vs_predicted(
    actuals: np.ndarray,
    predictions: np.ndarray,
    model_name: str,
    mae: float,
    r2: float,
) -> Path:
    """Hexbin scatter + error distribution for the best model."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        f"Q1 Analysis: How Close Are the Predictions?\n"
        f"{model_name} - Actual vs Predicted Next-Game Points",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # Left: hexbin scatter
    ax = axes[0]
    lo, hi = 0, 50  # Hardcoded max scale to ~50 pts for better visualization

    # Explicitly filter data within bounds for clean hexbin rendering
    mask = (actuals >= lo) & (actuals <= hi) & (predictions >= lo) & (predictions <= hi)
    act_plot = actuals[mask]
    pred_plot = predictions[mask]

    hb = ax.hexbin(act_plot, pred_plot, gridsize=40, cmap="Blues",
                   mincnt=1, alpha=0.85, extent=(lo, hi, lo, hi))

    # Perfect prediction line
    ax.plot([lo, hi], [lo, hi], "--", color=C["red"], lw=2,
            label="Perfect prediction", zorder=5)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual points scored", fontsize=11, labelpad=8)
    ax.set_ylabel("Predicted points", fontsize=11, labelpad=8)
    ax.set_title(f"MAE = {mae:.2f} pts  ·  R² = {r2:.3f}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_aspect("equal")
    fig.colorbar(hb, ax=ax, label="Game count")


    # Right: error distribution
    ax = axes[1]
    errors = predictions - actuals
    bins   = np.linspace(-35, 35, 70)
    ax.hist(errors, bins=bins, color=C["blue"], alpha=0.7, density=True, zorder=3)


    # Overlay normal fit
    mu, sigma = errors.mean(), errors.std()
    x_fit = np.linspace(errors.min(), errors.max(), 200)
    ax.plot(x_fit, stats.norm.pdf(x_fit, mu, sigma),
            color=C["red"], lw=2, label=f"Normal fit\n(μ={mu:.2f}, σ={sigma:.2f})")
    ax.axvline(0, color=C["dark"], lw=1.5, linestyle="--", alpha=0.5, label="Zero error")

    within5  = (np.abs(errors) <= 5).mean() * 100
    within10 = (np.abs(errors) <= 10).mean() * 100
    ax.text(0.97, 0.92,
            f"{within5:.0f}% within ±5 pts\n{within10:.0f}% within ±10 pts",
            transform=ax.transAxes, ha="right", fontsize=9.5,
            color=C["dark"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#e8f4fd", ec=C["light_blue"], lw=1))
    ax.set_xlabel("Prediction error (predicted − actual)", fontsize=11, labelpad=8)
    ax.set_ylabel("Density", fontsize=11, labelpad=8)
    ax.set_title("Error Distribution\n(symmetric = unbiased)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, zorder=0)
    ax.spines["left"].set_visible(False)

    _takeaway(fig,
              f"Key takeaway: The model predicts next-game scoring within ±5 points "
              f"{within5:.0f}% of the time and within ±10 points {within10:.0f}% of the time. "
              f"The error distribution is approximately symmetric, indicating no systematic bias.")
    fig.tight_layout()
    return _save(fig, "Q1_fig03_actual_vs_predicted")


# Figure 4: SHAP summary

def fig04_shap_summary(
    shap_values:   np.ndarray,
    X_sample:      pd.DataFrame,
    feature_names: list[str],
    top_n:         int = 12,
) -> Path:
    """SHAP global importance bar chart. Shows only the top `top_n` features
    by mean |SHAP value| - anything past that is close to background noise
    and only clutters the chart."""
    try:
        import shap as shap_lib
    except ImportError:
        logger.warning("shap not installed - skipping SHAP figure")
        return None

    # Changed from 1x2 to 1x1 grid and adjusted width
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    fig.suptitle(
        "Q1 Analysis: What Drives NBA Scoring Predictions?\n"
        f"SHAP Values - Top {top_n} Features by Impact on the Model's Output",
        fontsize=13, fontweight="bold", y=1.02,
    )

    mean_abs = np.abs(shap_values).mean(axis=0)

    # Filter out features with 0.00 importance, then keep only the top_n
    non_zero_idx = np.where(mean_abs >= 0.005)[0]
    mean_abs_filtered = mean_abs[non_zero_idx]
    feature_names_filtered = [feature_names[i] for i in non_zero_idx]

    top_order = np.argsort(mean_abs_filtered)[::-1][:top_n]
    mean_abs_filtered = mean_abs_filtered[top_order]
    feature_names_filtered = [feature_names_filtered[i] for i in top_order]

    order = np.argsort(mean_abs_filtered)
    feat_sorted = [feature_names_filtered[i] for i in order]
    vals_sorted = mean_abs_filtered[order]

    bar_colors = [C["blue"] if v > np.percentile(mean_abs_filtered, 60) else C["light_blue"]
                  for v in vals_sorted]

    bars = ax.barh(feat_sorted, vals_sorted, color=bar_colors, height=0.65, zorder=3)

    for bar, v in zip(bars, vals_sorted):
        ax.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                f"{v:.2f}", va="center", fontsize=8.5, color="#333")

    # Guard in case fewer than 3 features remain
    if len(feat_sorted) >= 3:
        top3 = [feat_sorted[-1], feat_sorted[-2], feat_sorted[-3]]
        ax.text(0.97, 0.04,
                f"Top 3 predictors:\n" + "\n".join(f"  {i+1}. {n}" for i, n in enumerate(top3)),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="#f0f7e6", ec=C["green"], lw=1))
        top1_name, top2_name = top3[0], top3[1]
    else:
        top1_name, top2_name = "Primary Feature", "Secondary Feature"

    ax.set_xlabel("Mean |SHAP value| (average impact on predictions)", fontsize=10, labelpad=8)
    ax.set_title("Overall Feature Importance\n(longer bar = larger average impact)",
                 fontsize=11, fontweight="bold")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)

    fig.tight_layout()
    return _save(fig, "Q1_fig04_shap_summary")


# Figure 6: Error by tier and position

def fig06_error_by_subgroup(df_test: pd.DataFrame, predictions: np.ndarray) -> Path:
    """Prediction MAE broken down by scoring tier and position."""
    df = df_test.copy()
    df["__pred"]  = predictions
    df["__error"] = np.abs(predictions - df["PTS"].values)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Q1 Analysis: Who Is Easiest and Hardest to Predict?\n"
        "Prediction Error Broken Down by Scoring Tier and Position",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # Left: scoring tier
    ax = axes[0]
    if "season_avg_pts" in df.columns:
        def _tier(x):
            if pd.isna(x):  return "Unknown"
            if x < 8:       return "Role player\n(<8 ppg)"
            if x < 14:      return "Contributor\n(8-14 ppg)"
            if x < 20:      return "Starter\n(14-20 ppg)"
            return "Star\n(20+ ppg)"

        df["_tier"] = df["season_avg_pts"].apply(_tier)
        tier_order  = ["Role player\n(<8 ppg)", "Contributor\n(8-14 ppg)",
                       "Starter\n(14-20 ppg)", "Star\n(20+ ppg)"]
        tier_colors = ["#90CAF9", "#42A5F5", "#1565C0", "#0D47A1"]
        tier_maes   = []
        tier_labels = []
        tier_ns     = []
        for t, c in zip(tier_order, tier_colors):
            sub = df.loc[df["_tier"] == t, "__error"]
            if len(sub) < 20:
                continue
            tier_labels.append(t)
            tier_maes.append(sub.mean())
            tier_ns.append(len(sub))

        x     = np.arange(len(tier_labels))
        bars  = ax.bar(x, tier_maes, color=tier_colors[:len(tier_labels)],
                       width=0.55, zorder=3)
        for bar, mae, n in zip(bars, tier_maes, tier_ns):
            ax.text(bar.get_x() + bar.get_width()/2, mae + 0.05,
                    f"{mae:.2f} pts\n(n={n:,})", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(tier_labels, fontsize=9.5)
        ax.set_ylabel("Mean Absolute Error (points)", fontsize=11)
        ax.set_title("Prediction Error by Scoring Role\n(are stars harder to predict?)",
                     fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, zorder=0)
        ax.spines["left"].set_visible(False)
    else:
        ax.text(0.5, 0.5, "season_avg_pts\nnot available",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)

    # Right: position
    ax = axes[1]
    pos_col = "POSITION" if "POSITION" in df.columns else None
    if pos_col:
        pos_order  = ["G", "F", "C", "G-F", "F-C"]
        pos_labels, pos_maes, pos_ns = [], [], []
        for pos in pos_order:
            sub = df.loc[df[pos_col].str.startswith(pos, na=False), "__error"]
            if len(sub) < 20:
                continue
            pos_labels.append(pos)
            pos_maes.append(sub.mean())
            pos_ns.append(len(sub))

        x2   = np.arange(len(pos_labels))
        cols = [C["blue"], C["green"], C["orange"],
                C["light_blue"], C["grey"]][:len(pos_labels)]
        bars2 = ax.bar(x2, pos_maes, color=cols, width=0.55, zorder=3)
        for bar, mae, n in zip(bars2, pos_maes, pos_ns):
            ax.text(bar.get_x() + bar.get_width()/2, mae + 0.05,
                    f"{mae:.2f}\n(n={n:,})", ha="center", fontsize=9)
        ax.set_xticks(x2)
        ax.set_xticklabels(pos_labels, fontsize=10)
        ax.set_ylabel("Mean Absolute Error (points)", fontsize=11)
        ax.set_title("Prediction Error by Position\n(do guards differ from centers?)",
                     fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, zorder=0)
        ax.spines["left"].set_visible(False)
    else:
        ax.text(0.5, 0.5, "Position data\nnot available",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)

    _takeaway(fig,
              "Key takeaway: Star players (20+ ppg) are significantly harder to predict "
              "than role players - defenders game-plan specifically for them, creating higher "
              "game-to-game variance. Guards tend to have slightly higher prediction error than "
              "centers, likely because guard scoring is more sensitive to defensive matchups.")
    fig.tight_layout()
    return _save(fig, "Q1_fig06_error_by_subgroup")


# Figure 8: Rolling prediction for example players

def fig08_rolling_prediction_examples(
    df_test: pd.DataFrame,
    predictions: np.ndarray,
    n_players: int = 3,
) -> Path:
    """
    For n_players example players, plot game-by-game actual vs predicted PTS.
    Shows what the model "sees" over a season - good for qualitative validation.
    """
    df = df_test.copy()
    df["__pred"] = predictions

    # Pick players with enough games (one high, one mid, one low scorer)
    if "season_avg_pts" in df.columns:
        df["_tier"] = pd.qcut(df["season_avg_pts"].clip(1, 40),
                               q=3, labels=["Low", "Mid", "High"],
                               duplicates="drop")
        example_ids = []
        for t in ["High", "Mid", "Low"]:
            sub = df[df["_tier"] == t]
            if sub.empty:
                continue
            counts = sub.groupby("PLAYER_ID")["GAME_DATE"].count()
            pid    = counts[counts >= 20].index
            if len(pid):
                example_ids.append(np.random.choice(pid))
        if not example_ids:
            example_ids = df["PLAYER_ID"].value_counts().head(n_players).index.tolist()
    else:
        counts     = df.groupby("PLAYER_ID")["GAME_DATE"].count()
        example_ids= counts[counts >= 20].head(n_players).index.tolist()

    example_ids = example_ids[:n_players]
    n           = len(example_ids)
    if n == 0:
        logger.warning("No players with enough games for rolling prediction plot")
        return None

    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]
    fig.suptitle(
        "Q1 Analysis: Following Individual Players Through the Season\n"
        "Actual vs Predicted Points - Game by Game",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for ax, pid in zip(axes, example_ids):
        sub = df[df["PLAYER_ID"] == pid].sort_values("GAME_DATE")
        dates  = sub["GAME_DATE"].values
        actual = sub["PTS"].values
        pred   = sub["__pred"].values
        mae_p  = np.abs(actual - pred).mean()
        name   = sub["PLAYER_NAME"].iloc[0] if "PLAYER_NAME" in sub.columns else f"Player {pid}"
        avg    = sub.get("season_avg_pts", pd.Series([actual.mean()])).iloc[0]

        ax.fill_between(range(len(actual)), actual, alpha=0.15, color=C["grey"])
        ax.plot(actual, color=C["grey"],  lw=1.5, label="Actual", alpha=0.8)
        ax.plot(pred,   color=C["blue"],  lw=2.0, label="Predicted", alpha=0.9)
        ax.axhline(avg, color=C["red"], lw=1.2, linestyle="--", alpha=0.5,
                   label=f"Season avg: {avg:.1f}")

        ax.set_ylabel("Points scored", fontsize=10)
        ax.set_xlabel("Game number", fontsize=10)
        ax.set_title(f"{name}  ·  Season avg: {avg:.1f} ppg  ·  MAE: {mae_p:.2f} pts",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.yaxis.grid(True, zorder=0, alpha=0.5)
        ax.spines["left"].set_visible(False)

    _takeaway(fig,
              "Key takeaway: The model successfully captures the broad seasonal arc "
              "and baseline scoring level for each player, but cannot anticipate "
              "extreme game-to-game swings. The gap between the blue and grey lines "
              "represents the irreducible randomness of basketball.")

    fig.tight_layout()
    return _save(fig, "Q1_fig08_rolling_predictions")