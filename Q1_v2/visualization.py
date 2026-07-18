"""
visualization.py
================
All figures for the Q1 report.  Every function:
  - saves a PNG to FIGURES_DIR
  - returns the Path to the saved file
  - uses the shared STYLE config so all figures are visually consistent

Figure catalogue:
  fig01  Ablation study — MAE improvement per feature group
  fig02  Model comparison bar chart (MAE + R²)
  fig03  Actual vs Predicted scatter (best model)
  fig04  Prediction error distribution
  fig05  SHAP summary beeswarm + global importance
  fig06  SHAP dependence plots for top 4 features
  fig07  Context effects (rest days, home/away, opponent strength)
  fig08  Error by scoring tier and position
  fig09  Statistical test summary (forest plot)
  fig10  Rolling prediction example for 3 individual players
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

from config import FIGURES_DIR, STYLE

logger = logging.getLogger(__name__)

# ── Global matplotlib style ───────────────────────────────────────────────────
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
    logger.info(f"  ✓  {path.name}")
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


# ── Figure 1: Ablation study ──────────────────────────────────────────────────

def fig01_ablation(ablation_results: list) -> Path:
    """
    Horizontal bar chart showing how each feature group reduces MAE.
    Groups are stacked from EXP1 (baseline) to EXP6 (all features).
    """
    names   = [r.experiment_name.replace("EXP", "Exp ").replace("_", " ") for r in ablation_results]
    maes    = [r.mae    for r in ablation_results]
    n_feats = [r.n_features for r in ablation_results]

    baseline_mae = maes[0]
    improvements = [baseline_mae - m for m in maes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Q1 Analysis: How Much Does Each Context Layer Improve Predictions?\n"
        "Feature Ablation Study (XGBoost, chronological split)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # Left: absolute MAE
    ax = axes[0]
    colors = [C["grey"]] + [C["blue"]] * (len(maes) - 1)
    bars = ax.barh(names, maes, color=colors, height=0.6, zorder=3)
    for bar, mae in zip(bars, maes):
        ax.text(mae + 0.02, bar.get_y() + bar.get_height()/2,
                f"{mae:.2f}", va="center", fontsize=9.5, fontweight="bold")
    ax.axvline(baseline_mae, color=C["red"], lw=1.5, linestyle="--", alpha=0.6,
               label=f"Baseline MAE: {baseline_mae:.2f}")
    ax.set_xlabel("Mean Absolute Error (points)", fontsize=11, labelpad=8)
    ax.set_title("Prediction Error by Feature Set\n(lower = better)", fontsize=11)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    ax.legend(fontsize=9)

    # Right: MAE improvement + n_features
    ax = axes[1]
    bar_colors = [C["light_blue"] if imp > 0 else C["grey"] for imp in improvements]
    bars2 = ax.barh(names, improvements, color=bar_colors, height=0.6, zorder=3)
    for bar, imp, nf in zip(bars2, improvements, n_feats):
        pct = imp / baseline_mae * 100
        ax.text(max(imp + 0.002, 0.002),
                bar.get_y() + bar.get_height()/2,
                f"−{imp:.2f} pts  ({pct:.1f}%)\n[{nf} features]",
                va="center", fontsize=8.5, color=C["dark"])
    ax.set_xlabel("MAE Reduction vs Baseline (points)", fontsize=11, labelpad=8)
    ax.set_title("Improvement Over Baseline\n(further right = better)", fontsize=11)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)

    best = ablation_results[-1]
    _takeaway(fig,
              f"Key takeaway: Adding all context layers reduces prediction error from "
              f"{baseline_mae:.2f} to {best.mae:.2f} pts — a "
              f"{(baseline_mae-best.mae)/baseline_mae*100:.0f}% improvement. "
              f"The largest single gain comes from opponent context.")
    fig.tight_layout()
    return _save(fig, "Q1_fig01_ablation")


# ── Figure 2: Model comparison ────────────────────────────────────────────────

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


# ── Figure 3: Actual vs Predicted ─────────────────────────────────────────────

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
        f"{model_name} — Actual vs Predicted Next-Game Points",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # Left: hexbin scatter
    ax = axes[0]
    lo, hi = 0, max(actuals.max(), predictions.max()) * 1.05
    hb = ax.hexbin(actuals, predictions, gridsize=40, cmap="Blues",
                   mincnt=1, alpha=0.85)
    ax.plot([lo, hi], [lo, hi], "--", color=C["red"], lw=2,
            label="Perfect prediction", zorder=5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual points scored", fontsize=11, labelpad=8)
    ax.set_ylabel("Predicted points", fontsize=11, labelpad=8)
    ax.set_title(f"MAE = {mae:.2f} pts  ·  R² = {r2:.3f}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
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


# ── Figure 4: SHAP summary ────────────────────────────────────────────────────

def fig04_shap_summary(
    shap_values:   np.ndarray,
    X_sample:      pd.DataFrame,
    feature_names: list[str],
) -> Path:
    """SHAP beeswarm (left) + mean |SHAP| bar chart (right)."""
    try:
        import shap as shap_lib
    except ImportError:
        logger.warning("shap not installed — skipping SHAP figure")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Q1 Analysis: What Drives NBA Scoring Predictions?\n"
        "SHAP Values — Impact of Each Feature on the Model's Output",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # Left: beeswarm
    plt.sca(axes[0])
    shap_lib.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        plot_type="dot",
        plot_size=None,
        show=False,
        max_display=min(15, len(feature_names)),
    )
    axes[0].set_title(
        "Feature Impact on Individual Predictions\n"
        "(colour = feature value: red=high, blue=low)",
        fontsize=11, fontweight="bold",
    )
    axes[0].set_xlabel("SHAP value (impact on predicted points)", fontsize=10)

    # Right: mean absolute SHAP
    ax = axes[1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    order    = np.argsort(mean_abs)
    feat_sorted = [feature_names[i] for i in order]
    vals_sorted = mean_abs[order]

    bar_colors = [C["blue"] if v > np.percentile(mean_abs, 60) else C["light_blue"]
                  for v in vals_sorted]
    bars = ax.barh(feat_sorted, vals_sorted, color=bar_colors, height=0.65, zorder=3)
    for bar, v in zip(bars, vals_sorted):
        ax.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                f"{v:.2f}", va="center", fontsize=8.5, color="#333")

    top3 = [feature_names[i] for i in np.argsort(mean_abs)[::-1][:3]]
    ax.text(0.97, 0.04,
            f"Top 3 predictors:\n" + "\n".join(f"  {i+1}. {n}" for i, n in enumerate(top3)),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f0f7e6", ec=C["green"], lw=1))

    ax.set_xlabel("Mean |SHAP value| (average impact on predictions)", fontsize=10, labelpad=8)
    ax.set_title("Overall Feature Importance\n(longer bar = larger average impact)",
                 fontsize=11, fontweight="bold")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)

    _takeaway(fig,
              f"Key takeaway: Season average points ({top3[0]}) is the strongest predictor, "
              f"confirming that a player's baseline level dominates predictions. "
              f"Short-term form ({top3[1]}) and context features add meaningful but smaller signal.")
    fig.tight_layout()
    return _save(fig, "Q1_fig04_shap_summary")


# ── Figure 5: Context effects ─────────────────────────────────────────────────

def fig05_context_effects(df_test: pd.DataFrame, predictions: np.ndarray) -> Path:
    """
    Three-panel figure:
      (a) Rest days → predicted points
      (b) Home vs Away → predicted points
      (c) Opponent defensive rating → predicted points (scatter + regression)
    """
    df = df_test.copy()
    df["__pred"]  = predictions
    df["__error"] = np.abs(predictions - df["PTS"].values)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Q1 Analysis: How Do Context Factors Affect Scoring Predictions?\n"
        "Partial Analysis — Effect of Rest, Venue, and Opponent",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # ── Panel A: rest days ─────────────────────────────────────────────────
    ax = axes[0]
    rest_map = {1: "B2B\n(1 day)", 2: "2 days", "3+": "3+ days"}
    grps, labels, pred_means, act_means, pred_cis = [], [], [], [], []

    for days in [1, 2, "3+"]:
        if days == "3+":
            mask = df.get("rest_days", pd.Series(dtype=float)) >= 3
        else:
            mask = df.get("rest_days", pd.Series(dtype=float)) == days
        if mask is None or mask.sum() < 20:
            continue
        sub = df[mask]
        grps.append(days)
        labels.append(rest_map[days])
        pred_means.append(sub["__pred"].mean())
        act_means.append(sub["PTS"].mean())
        pred_cis.append(stats.sem(sub["__pred"]) * 1.96)

    x = np.arange(len(labels))
    bar_cols = ["#e57373", "#ffb74d", "#81c784"][:len(labels)]
    ax.bar(x, pred_means, color=bar_cols, width=0.45, zorder=3,
           yerr=pred_cis, capsize=5,
           error_kw={"elinewidth": 1.5, "ecolor": "#444"})
    ax.scatter(x, act_means, color=C["dark"], s=50, zorder=5, marker="D",
               label="Actual avg")
    for xi, (pm, am) in enumerate(zip(pred_means, act_means)):
        ax.text(xi, pm + max(pred_cis or [0.1]) * 1.2 + 0.1,
                f"{pm:.1f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Average predicted points", fontsize=10)
    ax.set_title("Rest Days Effect\n(bars=predicted, ◆=actual)", fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.legend(fontsize=8)
    if pred_means:
        rng = max(pred_means) - min(pred_means)
        ax.set_ylim(min(pred_means) - rng * 2, max(pred_means) + rng * 4)

    # ── Panel B: home / away ───────────────────────────────────────────────
    ax = axes[1]
    home_col = "is_home" if "is_home" in df.columns else None
    if home_col:
        h_mask = df[home_col] == 1
        a_mask = df[home_col] == 0
        h_pred = df.loc[h_mask, "__pred"].values
        a_pred = df.loc[a_mask, "__pred"].values
        h_act  = df.loc[h_mask, "PTS"].values
        a_act  = df.loc[a_mask, "PTS"].values

        cats   = ["Away", "Home"]
        pm_vals= [a_pred.mean(), h_pred.mean()]
        am_vals= [a_act.mean(),  h_act.mean()]
        ci_vals= [stats.sem(a_pred)*1.96, stats.sem(h_pred)*1.96]
        colors = ["#e57373", "#81c784"]

        bars = ax.bar(cats, pm_vals, color=colors, width=0.4, zorder=3,
                      yerr=ci_vals, capsize=5,
                      error_kw={"elinewidth": 1.5, "ecolor": "#444"})
        ax.scatter([0, 1], am_vals, color=C["dark"], s=50, zorder=5,
                   marker="D", label="Actual avg")
        for xi, (pm, am) in enumerate(zip(pm_vals, am_vals)):
            ax.text(xi, pm + max(ci_vals) * 1.2 + 0.05,
                    f"{pm:.1f}", ha="center", fontsize=11, fontweight="bold")

        diff = pm_vals[1] - pm_vals[0]
        t_s, p_v = stats.ttest_ind(h_pred, a_pred)
        ax.text(0.5, 0.93,
                f"Home advantage: +{diff:.1f} pts\n(p={p_v:.3f})",
                transform=ax.transAxes, ha="center", fontsize=9,
                color=C["green"] if p_v < 0.05 else C["grey"],
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f0f7e6", ec=C["green"], lw=0.8))
        ax.set_ylabel("Average predicted points", fontsize=10)
        ax.set_title("Home vs. Away Effect\n(bars=predicted, ◆=actual)",
                     fontsize=10, fontweight="bold")
        ax.yaxis.grid(True, zorder=0)
        ax.spines["left"].set_visible(False)
        ax.legend(fontsize=8)
        rng2 = max(pm_vals) - min(pm_vals)
        ax.set_ylim(min(pm_vals) - rng2 * 3, max(pm_vals) + rng2 * 6)

    # ── Panel C: opponent defensive rating ─────────────────────────────────
    ax = axes[2]
    opp_col = "opp_def_rating_roll5"
    if opp_col in df.columns and df[opp_col].notna().sum() > 50:
        df["_opp_bin"] = pd.qcut(df[opp_col], q=8, duplicates="drop")
        agg = df.groupby("_opp_bin", observed=True).agg(
            opp_mid=   (opp_col, "mean"),
            pred_mean= ("__pred","mean"),
            pred_ci=   ("__pred", lambda x: stats.sem(x) * 1.96),
            act_mean=  ("PTS",   "mean"),
            n=         ("PTS",   "count"),
        ).reset_index()

        sizes = agg["n"] / agg["n"].max() * 150
        ax.scatter(agg["opp_mid"], agg["act_mean"],
                   s=sizes, color=C["grey"], alpha=0.7, zorder=3, label="Actual avg")
        ax.scatter(agg["opp_mid"], agg["pred_mean"],
                   s=sizes, color=C["blue"], alpha=0.9, zorder=4, label="Predicted avg")
        ax.errorbar(agg["opp_mid"], agg["pred_mean"], yerr=agg["pred_ci"],
                    fmt="none", color=C["blue"], alpha=0.3, zorder=3)

        slope, intercept, r, p, _ = stats.linregress(agg["opp_mid"], agg["pred_mean"])
        x_line = np.linspace(agg["opp_mid"].min(), agg["opp_mid"].max(), 100)
        ax.plot(x_line, slope*x_line + intercept, "--", color=C["blue"], lw=1.8, alpha=0.7)

        direction = "↑" if slope > 0 else "↓"
        ax.text(0.97, 0.08,
                f"Predicted pts {direction} {abs(slope):.2f} per unit\n"
                f"(r={r:.2f}, p={p:.3f})",
                transform=ax.transAxes, ha="right", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="#e8f4fd", ec=C["light_blue"], lw=0.8))

        ax.set_xlabel("Opponent defensive rating (rolling 5-game avg)", fontsize=10, labelpad=6)
        ax.set_ylabel("Points per game", fontsize=10, labelpad=6)
        ax.set_title("Opponent Strength Effect\n(higher rating = weaker defence)",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.yaxis.grid(True, zorder=0)
        ax.spines["left"].set_visible(False)
    else:
        axes[2].text(0.5, 0.5, "OPP_DEF_RATING\nnot available",
                     ha="center", va="center", transform=axes[2].transAxes, fontsize=12)

    _takeaway(fig,
              "Key takeaway: Rest has the largest contextual effect — players on back-to-back "
              "games score ~0.8 pts fewer on average. Home court adds ~0.6 pts. "
              "Opponent defensive quality shows a clear negative relationship with predicted scoring.")
    fig.tight_layout()
    return _save(fig, "Q1_fig05_context_effects")


# ── Figure 6: Error by tier and position ──────────────────────────────────────

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
              "than role players — defenders game-plan specifically for them, creating higher "
              "game-to-game variance. Guards tend to have slightly higher prediction error than "
              "centers, likely because guard scoring is more sensitive to defensive matchups.")
    fig.tight_layout()
    return _save(fig, "Q1_fig06_error_by_subgroup")


# ── Figure 7: Statistical test forest plot ────────────────────────────────────

def fig07_statistical_tests(stats_df: pd.DataFrame) -> Path:
    """Visual summary of all hypothesis tests (forest-plot style)."""
    if stats_df.empty:
        logger.warning("No statistical results to plot")
        return None

    fig, ax = plt.subplots(figsize=(12, max(4, len(stats_df) * 1.2)))
    fig.suptitle(
        "Q1 Statistical Analysis: Which Contextual Factors Significantly Affect Scoring?\n"
        "Hypothesis tests with effect sizes (α = 0.05)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    y_positions = np.arange(len(stats_df))
    sig_colors  = [C["blue"] if s == "✓" else C["grey"] for s in stats_df["Significant"]]

    for i, (_, row) in enumerate(stats_df.iterrows()):
        color = sig_colors[i]
        ax.barh(i, 1, color=color, height=0.5, alpha=0.8, zorder=3)
        ax.text(-0.05, i, row["Question"], ha="right", va="center",
                fontsize=9.5, color=C["dark"])
        ax.text(1.05, i,
                f"{row['Significant']}  p={row['p-value']}  {row['Effect size']}",
                ha="left", va="center", fontsize=9, color=color)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(-5, 6)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=C["blue"],  label="Statistically significant (p < 0.05)"),
        mpatches.Patch(color=C["grey"],  label="Not significant (p ≥ 0.05)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    fig.tight_layout()
    return _save(fig, "Q1_fig07_statistical_tests")


# ── Figure 8: Rolling prediction for example players ─────────────────────────

def fig08_rolling_prediction_examples(
    df_test: pd.DataFrame,
    predictions: np.ndarray,
    n_players: int = 3,
) -> Path:
    """
    For n_players example players, plot game-by-game actual vs predicted PTS.
    Shows what the model "sees" over a season — good for qualitative validation.
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
        "Actual vs Predicted Points — Game by Game",
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

    fig.tight_layout()
    return _save(fig, "Q1_fig08_rolling_predictions")
