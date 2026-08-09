"""
Statistical tests that answer the research sub-questions:

  Q: Does fatigue (back-to-back) significantly reduce scoring?
  Q: Does opponent defense significantly predict error magnitude?
  Q: Are stars (high-scoring players) harder or easier to predict?
  Q: Does prediction error differ by position?
  Q: Is the improvement from each feature group statistically significant?

All tests report: test statistic, p-value, effect size (Cohen's d), and
95% bootstrap confidence interval for the effect.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

np.random.seed(42)


@dataclass
class TestResult:
    question:    str
    stat_name:   str
    stat_value:  float
    p_value:     float
    effect_size: float
    effect_label: str        # "Cohen's d", "eta²", etc.
    ci_low:      float
    ci_high:     float
    significant: bool
    interpretation: str


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0


def _bootstrap_ci(
    a: np.ndarray, b: np.ndarray,
    n_boot: int = 2000, alpha: float = 0.05
) -> tuple[float, float]:
    """Bootstrap 95% CI for the difference in means (a - b)."""
    diffs = []
    for _ in range(n_boot):
        ai = np.random.choice(a, size=len(a), replace=True)
        bi = np.random.choice(b, size=len(b), replace=True)
        diffs.append(ai.mean() - bi.mean())
    lo = np.percentile(diffs, alpha/2*100)
    hi = np.percentile(diffs, (1-alpha/2)*100)
    return float(lo), float(hi)


class StatisticalAnalyzer:
    """
    Run hypothesis tests on the test set predictions.
    Pass the test-set DataFrame with predictions attached.
    """

    def __init__(self, df_test: pd.DataFrame, predictions: np.ndarray):
        self.df   = df_test.copy()
        self.df["__pred"]  = predictions
        self.df["__error"] = np.abs(predictions - df_test["PTS"].values)
        self.results: list[TestResult] = []

    # Test 1: Does fatigue affect scoring?

    def test_fatigue_effect(self) -> Optional[TestResult]:
        """
        H0: Mean points on back-to-back == mean points on rested games,
            controlling for playing time (MIN > 20 filter).

        WHY the MIN filter matters:
        Without it, the naive result is often that B2B players score MORE
        than rested players - a selection bias artefact.  Stars play more
        total games (hence more B2Bs) AND score more, so the raw B2B group
        is inflated by high-usage players.  Restricting to MIN > 20 ensures
        we compare meaningful starters-level performances, not garbage-time
        DNP-adjacent minutes.

        We also run a WITHIN-player version using paired differences: for
        each player, compare their avg PTS on B2B nights vs rested nights.
        This eliminates the between-player confound entirely.
        """
        if "is_back_to_back" not in self.df.columns:
            logger.warning("  is_back_to_back not in data - skipping fatigue test")
            return None

        # Filter to meaningful playing time only
        min_col = "MIN"
        if min_col in self.df.columns:
            df_starters = self.df[self.df[min_col] > 20].copy()
        else:
            df_starters = self.df.copy()
            logger.warning("  MIN column not found - fatigue test uses all rows")

        b2b    = df_starters.loc[df_starters["is_back_to_back"] == 1, "PTS"].dropna().values
        rested = df_starters.loc[df_starters["is_back_to_back"] == 0, "PTS"].dropna().values

        if len(b2b) < 30 or len(rested) < 30:
            logger.warning(f"  Too few B2B rows after MIN>20 filter ({len(b2b)}) - skipping")
            return None

        # Within-player paired test
        # For each player compute mean B2B PTS − mean rested PTS
        pid_col = "PLAYER_ID"
        paired_diffs = []
        for pid, grp in df_starters.groupby(pid_col):
            b2b_g    = grp.loc[grp["is_back_to_back"] == 1, "PTS"].mean()
            rest_g   = grp.loc[grp["is_back_to_back"] == 0, "PTS"].mean()
            n_b2b    = (grp["is_back_to_back"] == 1).sum()
            n_rest   = (grp["is_back_to_back"] == 0).sum()
            if n_b2b >= 3 and n_rest >= 3 and not np.isnan(b2b_g) and not np.isnan(rest_g):
                paired_diffs.append(b2b_g - rest_g)   # negative = B2B hurts scoring

        if len(paired_diffs) >= 10:
            # One-sample t-test: is the mean within-player difference ≠ 0?
            t_stat, p_val = stats.ttest_1samp(paired_diffs, popmean=0)
            mean_diff     = float(np.mean(paired_diffs))
            d             = mean_diff / float(np.std(paired_diffs)) if np.std(paired_diffs) > 0 else 0.0
            ci_lo         = float(np.percentile(paired_diffs, 2.5))
            ci_hi         = float(np.percentile(paired_diffs, 97.5))
            test_type     = "paired t (within-player)"
            n_players_used = len(paired_diffs)
        else:
            # Fall back to independent samples if too few players have both conditions
            t_stat, p_val = stats.ttest_ind(b2b, rested)
            mean_diff     = float(b2b.mean() - rested.mean())
            d             = _cohens_d(b2b, rested)
            ci_lo, ci_hi  = _bootstrap_ci(b2b, rested)
            test_type     = "independent t"
            n_players_used = df_starters[pid_col].nunique()

        sig    = p_val < 0.05
        direction = "reduces" if mean_diff < 0 else "does not reduce (or slightly increases)"
        interp = (
            f"Within-player analysis ({n_players_used} players, MIN>20): "
            f"B2B games {direction} scoring by {abs(mean_diff):.2f} pts on average "
            f"(mean diff={mean_diff:+.2f}, CI=[{ci_lo:.2f},{ci_hi:.2f}], "
            f"{'significant' if sig else 'not significant'} at α=0.05). "
            f"Test: {test_type}."
        )
        r = TestResult("Does back-to-back fatigue reduce scoring? (MIN>20, within-player)",
                       test_type, float(t_stat), float(p_val), d, "Cohen's d",
                       ci_lo, ci_hi, sig, interp)
        self.results.append(r)
        logger.info(f"  Fatigue test: t={t_stat:.2f}, p={p_val:.4f}, d={d:.3f} - {interp}")
        return r

    # Test 2: Does opponent defense affect prediction error?

    def test_opponent_defense_effect(self) -> Optional[TestResult]:
        """
        Do players have higher absolute prediction error against tough defenses?
        Split by median opp def rating -> weak vs strong defense.
        """
        opp_col = "opp_def_rating_roll5"
        if opp_col not in self.df.columns:
            return None

        median_opp = self.df[opp_col].median()
        tough      = self.df.loc[self.df[opp_col] <= median_opp, "__error"].dropna().values
        weak       = self.df.loc[self.df[opp_col] >  median_opp, "__error"].dropna().values

        if len(tough) < 30 or len(weak) < 30:
            return None

        t_stat, p_val = stats.ttest_ind(tough, weak)
        d             = _cohens_d(tough, weak)
        ci_lo, ci_hi  = _bootstrap_ci(tough, weak)
        sig           = p_val < 0.05

        interp = (
            f"Against tough defences (opp def ≤{median_opp:.1f}): "
            f"MAE={tough.mean():.2f} vs weak defences MAE={weak.mean():.2f} "
            f"({'significant' if sig else 'not significant'})"
        )
        r = TestResult("Does opponent defence affect prediction accuracy?",
                       "t", float(t_stat), float(p_val), d, "Cohen's d",
                       ci_lo, ci_hi, sig, interp)
        self.results.append(r)
        logger.info(f"  Opponent defense test: {interp}")
        return r

    # Test 3: Are high-volume scorers harder to predict?

    def test_scoring_tier_predictability(self) -> Optional[TestResult]:
        """
        ANOVA: does absolute prediction error differ across scoring tiers?
        Tiers: role player (<8 ppg), contributor (8-14), starter (14-20), star (20+).
        """
        if "season_avg_pts" not in self.df.columns:
            return None

        def tier(x):
            if x < 8:   return "Role player"
            if x < 14:  return "Contributor"
            if x < 20:  return "Starter"
            return "Star"

        self.df["_tier"] = self.df["season_avg_pts"].apply(tier)
        groups = [
            self.df.loc[self.df["_tier"] == t, "__error"].dropna().values
            for t in ["Role player", "Contributor", "Starter", "Star"]
            if len(self.df.loc[self.df["_tier"] == t]) >= 30
        ]
        if len(groups) < 2:
            return None

        f_stat, p_val = stats.f_oneway(*groups)
        # eta² ≈ SS_between / SS_total
        grand_mean = self.df["__error"].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
        ss_total   = sum(((self.df["__error"] - grand_mean)**2).sum() for _ in [1])
        eta2 = ss_between / max(ss_total, 1e-9)
        sig  = p_val < 0.05

        interp = (
            f"ANOVA F={f_stat:.2f}, p={p_val:.4f}, η²={eta2:.3f}. "
            f"{'Error differs significantly across scoring tiers' if sig else 'No significant tier effect'}."
        )
        r = TestResult("Are stars harder to predict than role players?",
                       "F (ANOVA)", float(f_stat), float(p_val), eta2, "η²",
                       float(np.nan), float(np.nan), sig, interp)
        self.results.append(r)
        logger.info(f"  Scoring tier test: {interp}")
        return r

    # Test 4: Home vs Away error

    def test_home_away_error(self) -> Optional[TestResult]:
        """Are home games easier to predict than away games?"""
        if "is_home" not in self.df.columns:
            return None

        home = self.df.loc[self.df["is_home"] == 1, "__error"].dropna().values
        away = self.df.loc[self.df["is_home"] == 0, "__error"].dropna().values

        if len(home) < 30 or len(away) < 30:
            return None

        t_stat, p_val = stats.ttest_ind(home, away)
        d             = _cohens_d(home, away)
        ci_lo, ci_hi  = _bootstrap_ci(home, away)
        sig           = p_val < 0.05

        interp = (
            f"Home game MAE={home.mean():.2f} vs Away MAE={away.mean():.2f} "
            f"({'significant' if sig else 'not significant'})"
        )
        r = TestResult("Are home games easier to predict?", "t",
                       float(t_stat), float(p_val), d, "Cohen's d",
                       ci_lo, ci_hi, sig, interp)
        self.results.append(r)
        logger.info(f"  Home/away test: {interp}")
        return r

    # Run all tests

    def run_all(self) -> pd.DataFrame:
        self.test_fatigue_effect()
        self.test_opponent_defense_effect()
        self.test_scoring_tier_predictability()
        self.test_home_away_error()

        rows = [{
            "Question":     r.question,
            "Test":         r.stat_name,
            "Statistic":    round(r.stat_value, 3),
            "p-value":      f"{r.p_value:.4f}",
            "Effect size":  f"{r.effect_size:.3f} ({r.effect_label})",
            "Significant":  "Yes" if r.significant else "No",
            "Interpretation": r.interpretation,
        } for r in self.results]

        return pd.DataFrame(rows)
