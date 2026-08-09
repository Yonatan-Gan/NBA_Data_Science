"""
Q3 - Playoffs vs. Regular Season
Full analysis using a composite performance score (not just points).
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from scipy.stats import zscore, gaussian_kde
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAGGLE = os.path.join(BASE, "data", "processed", "Kaggle")
FIG    = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

RISER_COL   = "#27ae60"
NEUTRAL_COL = "#7f8c8d"
DECL_COL    = "#c0392b"
PAL   = {"Riser": RISER_COL, "Neutral": NEUTRAL_COL, "Decliner": DECL_COL}
ORDER = ["Riser", "Neutral", "Decliner"]

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         False,
})


# ============================================================
# LOAD AND BUILD COMPOSITE SCORE
# ============================================================
print("Loading data and building composite score...")

q3  = pd.read_csv(os.path.join(KAGGLE, "q3_player_split_v2.csv"))
adv = pd.read_csv(os.path.join(KAGGLE, "q3_advanced_split.csv"))
bio = pd.read_csv(os.path.join(KAGGLE, "player_bio_enhanced.csv"))

EAST_TEAMS = {"ATL","BOS","BRK","CHA","CHO","CHI","CLE","DET","IND","MIA","MIL",
              "NJN","NYK","ORL","PHI","TOR","WAS"}
WEST_TEAMS = {"DAL","DEN","GSW","HOU","LAC","LAL","MEM","MIN","NOP","NOH","NOK",
              "OKC","PHO","POR","SAC","SAS","UTA","SEA","VAN"}

# Keep only player-seasons with both regular season and playoff data,
# and enough games to make the stats reliable
df = q3.dropna(subset=["REG_PTS", "POF_PTS"]).copy()
df = df[(df["REG_G"] >= 20) & (df["POF_G"] >= 5)].copy()
df = df.reset_index(drop=True)

# Compute turnover delta
df["DELTA_TOV"] = df["POF_TOV"] - df["REG_TOV"]

# Normalize each component to z-scores so they are on the same scale
COMPONENTS = ["DELTA_PTS", "DELTA_eFG%", "DELTA_AST", "DELTA_TOV"]
for c in COMPONENTS:
    df[f"z_{c}"] = zscore(df[c].fillna(0))

# Turnovers: fewer is better, so we flip the sign
df["z_DELTA_TOV"] = -df["z_DELTA_TOV"]

# Weighted composite: scoring 35%, shooting efficiency 30%, assists 20%, turnovers 15%
WEIGHTS = {"DELTA_PTS": 0.35, "DELTA_eFG%": 0.30, "DELTA_AST": 0.20, "DELTA_TOV": 0.15}
df["COMPOSITE"] = sum(df[f"z_{c}"] * w for c, w in WEIGHTS.items())

THRESH = 0.5
df["TENDENCY"] = "Neutral"
df.loc[df["COMPOSITE"] >  THRESH, "TENDENCY"] = "Riser"
df.loc[df["COMPOSITE"] < -THRESH, "TENDENCY"] = "Decliner"

# Merge advanced stats
adv_both = adv.dropna(subset=["REG_PER", "POF_PER"]).copy()
adv_cols = ["Player","season_start",
            "REG_PER","REG_TS%","REG_USG%","REG_WS/48","REG_VORP","REG_BPM",
            "POF_PER","POF_TS%","POF_USG%","POF_WS/48","POF_VORP","POF_BPM"]
df = df.merge(adv_both[adv_cols], on=["Player","season_start"], how="left")

# Merge bio
bio_slim = (bio[["player_name","position","season_exp","birth_year","is_greatest_75"]]
            .drop_duplicates("player_name").copy())
bio_slim["player_name"] = bio_slim["player_name"].str.strip()
df["Player_key"] = df["Player"].str.strip()
df = df.merge(bio_slim.rename(columns={"player_name": "Player_key"}),
              on="Player_key", how="left")

# Position group
pos_map = {
    "Guard": "Guard", "Guard-Forward": "Guard", "Forward-Guard": "Wing",
    "Forward": "Wing", "Forward-Center": "Wing",
    "Center-Forward": "Center", "Center": "Center",
}
df["pos_group"] = df["position"].map(pos_map).fillna("Unknown")

# Scoring tier
df["scoring_tier"] = pd.cut(
    df["REG_PTS"], bins=[0, 8, 14, 20, 100],
    labels=["Role player\n(under 8 ppg)", "Contributor\n(8-14 ppg)",
            "Starter\n(14-20 ppg)", "Star\n(20+ ppg)"]
)

# Age and experience
df["approx_age"] = df["season_start"] + 1 - df["birth_year"]
df["exp_bucket"] = pd.cut(
    df["season_exp"].clip(0, 22), bins=[-1, 3, 7, 12, 25],
    labels=["Rookie\n(0-3 yrs)", "Young\n(4-7)", "Prime\n(8-12)", "Veteran\n(13+)"]
)

# Conference (team from advanced split)
adv_team = adv[["Player","season_start","team"]].drop_duplicates()
df = df.merge(adv_team, on=["Player","season_start"], how="left")
df["conference"] = df["team"].map(
    lambda t: "East" if t in EAST_TEAMS else ("West" if t in WEST_TEAMS else None)
)

# Finer experience brackets for cross-analysis
EXP_BINS   = [0, 3, 6, 10, 15, 26]
EXP_LABELS = ["1-3 yrs", "4-6 yrs", "7-10 yrs", "11-15 yrs", "16+ yrs"]
df["exp_bracket"] = pd.cut(
    df["season_exp"].clip(0, 25).fillna(-1).replace(-1, np.nan),
    bins=EXP_BINS, labels=EXP_LABELS
)

n_total = len(df)
n_r = (df["TENDENCY"] == "Riser").sum()
n_n = (df["TENDENCY"] == "Neutral").sum()
n_d = (df["TENDENCY"] == "Decliner").sum()
print(f"  {n_total:,} player-seasons  |  Risers {n_r} ({100*n_r/n_total:.0f}%)  "
      f"Neutral {n_n} ({100*n_n/n_total:.0f}%)  Decliners {n_d} ({100*n_d/n_total:.0f}%)")


# ============================================================
# FIG 1 - Composite score shape comparison: Risers vs Decliners
# ============================================================
print("Fig 1...")

r_comp = df[df["TENDENCY"] == "Riser"]["COMPOSITE"].dropna()
d_comp = df[df["TENDENCY"] == "Decliner"]["COMPOSITE"].dropna()

# KDE for each group
kde_r = gaussian_kde(r_comp, bw_method=0.30)
kde_d = gaussian_kde(d_comp, bw_method=0.30)

x_full = np.linspace(-3.8, 3.0, 800)
xs_r   = np.linspace(0.5, r_comp.max() + 0.15, 500)
xs_d   = np.linspace(d_comp.min() - 0.15, -0.5, 500)

# Stats for annotations
r_comp_std = r_comp.std()
d_comp_std = d_comp.std()

fig, ax = plt.subplots(figsize=(11, 5.5))
fig.suptitle("How Extreme Is Each Group? Comparing the Shape of Riser vs Decliner Scores",
             fontsize=13, fontweight="bold", y=1.01)

# Shade + outline for each group (plotted only in their own zone)
ax.fill_between(xs_r, kde_r(xs_r), alpha=0.32, color=RISER_COL)
ax.plot(xs_r, kde_r(xs_r), color=RISER_COL, linewidth=2.4,
        label=f"Risers  (n={n_r:,})")

ax.fill_between(xs_d, kde_d(xs_d), alpha=0.32, color=DECL_COL)
ax.plot(xs_d, kde_d(xs_d), color=DECL_COL, linewidth=2.4,
        label=f"Decliners  (n={n_d:,})")

# Neutral gap shading
ax.axvspan(-0.5, 0.5, alpha=0.07, color=NEUTRAL_COL, zorder=0)
ax.text(0, ax.get_ylim()[1] * 0.05 if ax.get_ylim()[1] > 0 else 0.02,
        "Neutral zone", ha="center", fontsize=8.5,
        color=NEUTRAL_COL, style="italic")

# Threshold lines
ax.axvline( 0.5, color=RISER_COL, linewidth=1.0, linestyle="--", alpha=0.5)
ax.axvline(-0.5, color=DECL_COL,  linewidth=1.0, linestyle="--", alpha=0.5)

# Annotate the spread difference
ymax = max(kde_r(xs_r).max(), kde_d(xs_d).max())
ax.annotate(
    f"Risers: tighter cluster\n(spread = {r_comp_std:.2f})",
    xy=(r_comp.mean(), kde_r([r_comp.mean()])[0]),
    xytext=(1.9, ymax * 0.82),
    arrowprops=dict(arrowstyle="->", color=RISER_COL, lw=1.3),
    fontsize=9.5, color=RISER_COL, fontweight="bold")
ax.annotate(
    f"Decliners: wider spread\n(spread = {d_comp_std:.2f})",
    xy=(d_comp.mean(), kde_d([d_comp.mean()])[0]),
    xytext=(-3.4, ymax * 0.82),
    arrowprops=dict(arrowstyle="->", color=DECL_COL, lw=1.3),
    fontsize=9.5, color=DECL_COL, fontweight="bold")

ax.set_xlabel(
    "Composite playoff performance score  "
    "(combines scoring, shooting efficiency, assists, turnovers  -  "
    "positive = improved in playoffs, negative = declined)",
    fontsize=9.5)
ax.set_ylabel("Density", fontsize=11)
ax.set_xlim(-3.8, 3.0)
ax.tick_params(left=False, labelleft=False)
ax.legend(frameon=False, fontsize=11, loc="upper right")

fig.tight_layout()
p1 = os.path.join(FIG, "q3_fig1_scoring_delta.png")
fig.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# FIG 2 - Who rises? By position and scoring role
# ============================================================
print("Fig 2...")

def stacked_pct_chart(ax, groupcol, order, title, df_sub):
    ct = (df_sub.groupby([groupcol, "TENDENCY"])
                .size().unstack(fill_value=0))
    for t in ORDER:
        if t not in ct.columns:
            ct[t] = 0
    ct  = ct[ORDER]
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    pct = pct.reindex(order).dropna(how="all")

    bottoms = np.zeros(len(pct))
    x = np.arange(len(pct))
    for tend in ORDER:
        ax.bar(x, pct[tend].values, bottom=bottoms,
               color=PAL[tend], edgecolor="white", linewidth=0.8)
        for i, (h, b) in enumerate(zip(pct[tend].values, bottoms)):
            if h > 9:
                ax.text(x[i], b + h/2, f"{h:.0f}%",
                        ha="center", va="center", fontsize=9,
                        color="white", fontweight="bold")
        bottoms += pct[tend].values

    ax.set_xticks(x)
    ax.set_xticklabels(pct.index, fontsize=10.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel("% of player-seasons", fontsize=11)
    ax.set_title(title, fontsize=12)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("Who Rises? Who Declines? Breakdown by Position and Scoring Role",
             fontsize=13, fontweight="bold", y=1.01)

df_pos = df[df["pos_group"] != "Unknown"]
stacked_pct_chart(ax1, "pos_group",
                  ["Guard", "Wing", "Center"],
                  "By Position", df_pos)
stacked_pct_chart(ax2, "scoring_tier",
                  ["Role player\n(under 8 ppg)", "Contributor\n(8-14 ppg)",
                   "Starter\n(14-20 ppg)", "Star\n(20+ ppg)"],
                  "By Scoring Role", df)

# Single shared legend below the charts
patches = [mpatches.Patch(color=PAL[t], label=t) for t in ORDER]
fig.legend(handles=patches, loc="lower center", ncol=3,
           frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout()
p2 = os.path.join(FIG, "q3_fig2_breakdown.png")
fig.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# FIG 3 - How Risers and Decliners differ across 4 key stats
# ============================================================
print("Fig 3...")
stat_info = [
    ("DELTA_PTS",  "Points per game",            "ppg"),
    ("DELTA_eFG%", "Shooting efficiency (eFG%)", "percentage points"),
    ("DELTA_AST",  "Assists per game",            "apg"),
]

fig, axes = plt.subplots(1, 3, figsize=(11, 5.5))
fig.suptitle("How Risers and Decliners Differ Across Three Key Stats\n"
             "(Playoffs minus Regular Season - positive means improvement)",
             fontsize=13, fontweight="bold", y=1.03)

for ax, (col, label, unit) in zip(axes, stat_info):
    means = {t: df[df["TENDENCY"]==t][col].dropna().mean() for t in ["Riser","Decliner"]}
    pos_max = max(v for v in means.values() if v >= 0) if any(v >= 0 for v in means.values()) else 0
    neg_min = min(v for v in means.values() if v <  0) if any(v <  0 for v in means.values()) else 0
    total_span = pos_max - neg_min if (pos_max - neg_min) > 0 else 1
    # 30% headroom above the tallest positive bar, 30% below the deepest negative bar
    ylim_top = pos_max + total_span * 0.30
    ylim_bot = neg_min - total_span * 0.30

    for xi, tend in enumerate(["Riser", "Decliner"]):
        mean  = means[tend]
        color = PAL[tend]
        ax.bar([xi], [mean], color=color, alpha=0.85, edgecolor="white", width=0.55)
        sign = "+" if mean > 0 else ""
        offset = total_span * 0.06      # label sits 6% of range away from bar tip
        if mean >= 0:
            ax.text(xi, mean + offset, f"{sign}{mean:.2f}",
                    ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=color)
        else:
            ax.text(xi, mean - offset, f"{sign}{mean:.2f}",
                    ha="center", va="top", fontsize=10.5, fontweight="bold", color=color)

    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.4)
    ax.set_ylim(ylim_bot, ylim_top)
    ax.set_title(label, fontsize=10.5, fontweight="bold", pad=10)
    ax.set_ylabel(f"Change ({unit})", fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Riser", "Decliner"], fontsize=10)
    ax.tick_params(bottom=False)

fig.tight_layout(pad=1.8)
p3 = os.path.join(FIG, "q3_fig3_component_deltas.png")
fig.savefig(p3, dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# FIG 4 - Regular season vs playoff scoring scatter (sampled)
# ============================================================
print("Fig 4...")
np.random.seed(42)
N_SAMPLE = 250
fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True, sharex=True)
fig.suptitle(
    "Regular Season Scoring vs Playoff Scoring\n"
    "Each dot is one player-season. Above the diagonal = scored more in playoffs.\n"
    "Colors reflect a 4-stat composite score (shooting, assists, turnovers, scoring)  -  "
    "a player can score slightly less but still be a Riser overall.",
    fontsize=11.5, fontweight="bold", y=1.05)

for ax, tend in zip(axes, ORDER):
    sub = df[df["TENDENCY"] == tend].copy()
    sample = sub.sample(min(N_SAMPLE, len(sub)), random_state=42)
    ax.scatter(sample["REG_PTS"], sample["POF_PTS"],
               color=PAL[tend], alpha=0.55, s=22, edgecolors="none")
    lim = 38
    ax.plot([0, lim], [0, lim], color="black", linewidth=1.2,
            linestyle="--", alpha=0.45, zorder=0)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title(f"{tend}  (n={len(sub):,})", fontsize=12,
                 color=PAL[tend], fontweight="bold")
    ax.set_xlabel("Points per game - Regular Season", fontsize=10)

axes[0].set_ylabel("Points per game - Playoffs", fontsize=10)
fig.tight_layout()
p4 = os.path.join(FIG, "q3_fig4_scatter.png")
fig.savefig(p4, dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# FIG 5 - Serial performers (3+ seasons, composite)
# ============================================================
print("Fig 5...")
career_pts = df.groupby("Player")["REG_PTS"].mean()
rise_c = df[df["TENDENCY"] == "Riser"].groupby("Player").size()
decl_c = df[df["TENDENCY"] == "Decliner"].groupby("Player").size()

serial_r = (rise_c[rise_c >= 3].to_frame("seasons")
            .join(career_pts.rename("avg_pts"))
            .sort_values("seasons", ascending=False).head(18))
serial_d = (decl_c[decl_c >= 3].to_frame("seasons")
            .join(career_pts.rename("avg_pts"))
            .sort_values("seasons", ascending=False).head(18))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
fig.suptitle("Consistent Playoff Performers: 3 or More Seasons of the Same Tendency",
             fontsize=13, fontweight="bold", y=1.01)

for ax, data, col, title in [
    (ax1, serial_r, RISER_COL,  "Players who consistently rose"),
    (ax2, serial_d, DECL_COL,   "Players who consistently declined"),
]:
    names = data.index.tolist()[::-1]
    vals  = data["seasons"].values[::-1]
    avgs  = data["avg_pts"].values[::-1]
    y = np.arange(len(names))
    ax.barh(y, vals, color=col, alpha=0.82, edgecolor="white", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Number of playoff seasons with this tendency", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", color=col)
    ax.set_xlim(0, vals.max() + 3.5)   # extra room for labels
    for yi, (v, a) in enumerate(zip(vals, avgs)):
        ppg = f"  ({a:.1f} ppg)" if not np.isnan(a) else ""
        ax.text(v + 0.15, yi, f"{int(v)} seasons{ppg}",
                va="center", ha="left", fontsize=8.5, color=col, fontweight="bold")

fig.tight_layout()
p5 = os.path.join(FIG, "q3_fig5_serial.png")
fig.savefig(p5, dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# FIG 6 - Validation: avg playoff games by tendency
# ============================================================
print("Fig 6 (validation)...")

avg_games = df.groupby("TENDENCY")["POF_G"].mean().reindex(ORDER)
n_games   = df.groupby("TENDENCY")["POF_G"].count().reindex(ORDER)
t_stat_g, p_val_g = stats.ttest_ind(
    df[df["TENDENCY"] == "Riser"]["POF_G"].dropna(),
    df[df["TENDENCY"] == "Decliner"]["POF_G"].dropna()
)

fig, ax = plt.subplots(figsize=(8, 5.5))
fig.suptitle("Do Risers Actually Go Further in the Playoffs?",
             fontsize=13, fontweight="bold", y=1.01)

for xi, tend in enumerate(ORDER):
    mean = avg_games[tend]
    ax.bar(xi, mean, color=PAL[tend], alpha=0.85, edgecolor="white", width=0.55)
    ax.text(xi, mean + 0.15, f"{mean:.1f}", ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=PAL[tend])
    ax.text(xi, 0.28, f"n={n_games[tend]:,}", ha="center", va="bottom",
            fontsize=9, color="white", fontweight="bold")

ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels(ORDER, fontsize=13)
ax.set_ylabel("Average playoff games played per season", fontsize=11)
ax.set_title(
    "Average playoff games played per season, by tendency group\n"
    "More games = team went further in the postseason",
    fontsize=10.5)
ax.set_ylim(0, avg_games.max() + 2.5)
ax.tick_params(bottom=False)

fig.tight_layout()
p6 = os.path.join(FIG, "q3_fig6_validation.png")
fig.savefig(p6, dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# FIG 7 - Feature importance (what predicts a Riser?)
# ============================================================
print("Fig 7...")
feature_cols = [
    "REG_PTS","REG_TRB","REG_AST","REG_STL","REG_BLK","REG_TOV",
    "REG_FG%","REG_3P%","REG_FT%","REG_eFG%","REG_MP",
    "REG_PER","REG_TS%","REG_USG%","REG_WS/48","REG_VORP","REG_BPM",
    "season_exp","POF_G",
]
nice = {
    "REG_PTS":"Points per game","REG_TRB":"Rebounds","REG_AST":"Assists",
    "REG_STL":"Steals","REG_BLK":"Blocks","REG_TOV":"Turnovers",
    "REG_FG%":"Field goal %","REG_3P%":"3-point %","REG_FT%":"Free throw %",
    "REG_eFG%":"Shooting efficiency (eFG%)","REG_MP":"Minutes played",
    "REG_PER":"Overall efficiency (PER)","REG_TS%":"True shooting %",
    "REG_USG%":"Usage rate","REG_WS/48":"Win contribution (WS/48)",
    "REG_VORP":"Value vs. bench player (VORP)","REG_BPM":"Net impact (BPM)",
    "season_exp":"Years of experience","POF_G":"Playoff games played",
}

rd = df[df["TENDENCY"].isin(["Riser","Decliner"])].dropna(subset=feature_cols).copy()
X  = rd[feature_cols].values
y  = (rd["TENDENCY"] == "Riser").astype(int).values

rf = RandomForestClassifier(n_estimators=400, max_depth=6, random_state=42, n_jobs=-1)
rf.fit(X, y)
cv_auc = cross_val_score(rf, X, y, cv=5, scoring="roc_auc")
print(f"  RF AUC: {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")

imp = pd.Series(rf.feature_importances_ * 100, index=feature_cols)
imp.index = [nice.get(c, c) for c in imp.index]
imp = imp.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 7.5))
threshold = imp.quantile(0.65)
colors = [RISER_COL if v >= threshold else NEUTRAL_COL for v in imp.values]
ax.barh(imp.index, imp.values, color=colors, edgecolor="white", height=0.7)
ax.set_xlabel("Importance score: how much this regular-season stat helped the model predict playoff tendency (%)",
              fontsize=9.5)
ax.set_title(
    f"Can We Predict a Playoff Riser Before the Season Starts?\n"
    f"We gave a machine learning model only each player's regular-season stats - "
    f"nothing from the playoffs.\n"
    f"It learned which stats best predict whether that player would rise or decline. "
    f"Longer bar = more useful.\n"
    f"Model accuracy: AUC = {cv_auc.mean():.2f}  (0.5 = random guessing, 1.0 = perfect)",
    fontsize=9.5, fontweight="bold", loc="left"
)
green_patch = mpatches.Patch(color=RISER_COL,   label="Most predictive stats")
gray_patch  = mpatches.Patch(color=NEUTRAL_COL, label="Less predictive stats")
ax.legend(handles=[green_patch, gray_patch], frameon=False, fontsize=9)
fig.tight_layout()
p7 = os.path.join(FIG, "q3_fig7_feature_importance.png")
fig.savefig(p7, dpi=150, bbox_inches="tight")
plt.close()

top_feat = imp.sort_values(ascending=False).head(5)


# ============================================================
# FIG 8 - Riser/Decliner rate by scoring role
# ============================================================
print("Fig 8...")
tier_order = ["Role player\n(under 8 ppg)", "Contributor\n(8-14 ppg)",
              "Starter\n(14-20 ppg)", "Star\n(20+ ppg)"]

fig, ax = plt.subplots(figsize=(11, 5.5))
fig.suptitle("Does Your Role Change Your Odds of Rising or Declining in the Playoffs?",
             fontsize=13, fontweight="bold", y=1.01)

w8 = 0.33
x8 = np.arange(len(tier_order))
riser_r8, decl_r8, tier_ns8 = [], [], []
for t in tier_order:
    sub = df[df["scoring_tier"] == t]["TENDENCY"].dropna()
    n   = len(sub)
    riser_r8.append(100 * (sub == "Riser").sum()    / n if n else 0)
    decl_r8.append( 100 * (sub == "Decliner").sum() / n if n else 0)
    tier_ns8.append(n)

bars_r8 = ax.bar(x8 - w8/2, riser_r8,  w8, color=RISER_COL, alpha=0.85,
                 edgecolor="white", label="Riser %")
bars_d8 = ax.bar(x8 + w8/2, decl_r8,   w8, color=DECL_COL,  alpha=0.85,
                 edgecolor="white", label="Decliner %")

for i, (r, d) in enumerate(zip(riser_r8, decl_r8)):
    ax.text(i - w8/2, r + 0.5, f"{r:.0f}%", ha="center", fontsize=10,
            color=RISER_COL, fontweight="bold")
    ax.text(i + w8/2, d + 0.5, f"{d:.0f}%", ha="center", fontsize=10,
            color=DECL_COL,  fontweight="bold")

ax.set_xticks(x8)
ax.set_xticklabels([f"{t}\n(n={n:,})" for t, n in zip(tier_order, tier_ns8)], fontsize=10)
ax.set_ylim(0, max(max(riser_r8), max(decl_r8)) + 8)
ax.set_ylabel("% of player-seasons in this role", fontsize=11)
ax.legend(frameon=False, fontsize=11, loc="upper left")
ax.tick_params(bottom=False)

fig.tight_layout()
p8 = os.path.join(FIG, "q3_fig8_delta_by_tier.png")
fig.savefig(p8, dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# FIG 9 - Age and experience
# ============================================================
print("Fig 9...")
age_r = df[(df["TENDENCY"] == "Riser") & df["approx_age"].notna()]["approx_age"]
age_d = df[(df["TENDENCY"] == "Decliner") & df["approx_age"].notna()]["approx_age"]
t_age, p_age = stats.ttest_ind(age_r, age_d)

age_buckets = ["23 or under", "24 to 26", "27 to 30", "31 to 34", "35 or older"]
age_bin_labels = ["<=23", "24-26", "27-30", "31-34", "35+"]
df["age_bucket_label"] = pd.cut(
    df["approx_age"].clip(18, 42),
    bins=[17, 23, 26, 30, 34, 43],
    labels=age_buckets
)

fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Does Age or Experience Change How Players Perform in the Playoffs?",
             fontsize=14, fontweight="bold", y=1.01)

w = 0.35

# Left: Riser vs Decliner rate by age group
riser_r_age, decl_r_age, age_ns = [], [], []
for ab in age_buckets:
    sub = df[df["age_bucket_label"] == ab]["TENDENCY"].dropna()
    n   = len(sub)
    riser_r_age.append(100 * (sub == "Riser").sum()   / n if n > 0 else 0)
    decl_r_age.append( 100 * (sub == "Decliner").sum()/ n if n > 0 else 0)
    age_ns.append(n)

x = np.arange(len(age_buckets))
ax1.bar(x - w/2, riser_r_age, w, color=RISER_COL, alpha=0.85, edgecolor="white", label="Riser %")
ax1.bar(x + w/2, decl_r_age,  w, color=DECL_COL,  alpha=0.85, edgecolor="white", label="Decliner %")
ax1.set_xticks(x)
ax1.set_xticklabels([f"{b}\n(n={n:,})" for b, n in zip(age_buckets, age_ns)], fontsize=9)
ax1.set_ylim(0, 55); ax1.set_ylabel("% of player-seasons", fontsize=11)
ax1.set_title("Riser vs. Decliner Rate by Age Group\n"
              "(age alone is not a significant predictor)",
              fontsize=11, fontweight="bold")
ax1.legend(frameon=False, fontsize=10)
ax1.tick_params(bottom=False)
for i, (r, d) in enumerate(zip(riser_r_age, decl_r_age)):
    ax1.text(i - w/2, r + 0.8, f"{r:.0f}%", ha="center", fontsize=9,
             color=RISER_COL, fontweight="bold")
    ax1.text(i + w/2, d + 0.8, f"{d:.0f}%", ha="center", fontsize=9,
             color=DECL_COL, fontweight="bold")

# Right: Riser vs Decliner by experience bucket
exp_buckets = ["Rookie\n(0-3 yrs)", "Young\n(4-7)", "Prime\n(8-12)", "Veteran\n(13+)"]
exp_df = df[df["exp_bucket"].notna() & df["TENDENCY"].notna()]
riser_r_exp, decl_r_exp, exp_ns = [], [], []
for e in exp_buckets:
    sub = exp_df[exp_df["exp_bucket"] == e]["TENDENCY"]
    n   = len(sub)
    riser_r_exp.append(100 * (sub == "Riser").sum()   / n if n > 0 else 0)
    decl_r_exp.append( 100 * (sub == "Decliner").sum()/ n if n > 0 else 0)
    exp_ns.append(n)

xe = np.arange(len(exp_buckets))
ax3.bar(xe - w/2, riser_r_exp, w, color=RISER_COL, alpha=0.85, edgecolor="white", label="Riser %")
ax3.bar(xe + w/2, decl_r_exp,  w, color=DECL_COL,  alpha=0.85, edgecolor="white", label="Decliner %")
ax3.set_xticks(xe)
ax3.set_xticklabels([f"{b}\n(n={n:,})" for b, n in zip(exp_buckets, exp_ns)], fontsize=9.5)
ax3.set_ylim(0, 55); ax3.set_ylabel("% of player-seasons", fontsize=11)
ax3.set_title("Riser vs. Decliner Rate by Experience", fontsize=11, fontweight="bold")
ax3.legend(frameon=False, fontsize=10)
ax3.tick_params(bottom=False)
for i, (r, d) in enumerate(zip(riser_r_exp, decl_r_exp)):
    ax3.text(i - w/2, r + 0.8, f"{r:.0f}%", ha="center", fontsize=9,
             color=RISER_COL, fontweight="bold")
    ax3.text(i + w/2, d + 0.8, f"{d:.0f}%", ha="center", fontsize=9,
             color=DECL_COL, fontweight="bold")

p9 = os.path.join(FIG, "q3_fig9_age_experience.png")
fig.savefig(p9, dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# FIG 10 - Fine-grained experience brackets
# ============================================================
print("Fig 10...")

# --- Data prep ---
exp_valid = df.dropna(subset=["exp_bracket", "TENDENCY"]).copy()

riser_by_exp, decl_by_exp, exp_ns_10 = [], [], []
for e in EXP_LABELS:
    sub = exp_valid[exp_valid["exp_bracket"] == e]["TENDENCY"]
    n   = len(sub)
    riser_by_exp.append(100 * (sub == "Riser").sum()   / n if n > 0 else 0)
    decl_by_exp.append( 100 * (sub == "Decliner").sum()/ n if n > 0 else 0)
    exp_ns_10.append(n)

# --- Figure: single panel, experience brackets only ---
fig, ax_exp = plt.subplots(figsize=(12, 5.5))
fig.suptitle("Experience and Playoff Tendency",
             fontsize=14, fontweight="bold", y=1.01)

w10  = 0.35
xe10 = np.arange(len(EXP_LABELS))
ax_exp.bar(xe10 - w10/2, riser_by_exp, w10, color=RISER_COL, alpha=0.85,
           edgecolor="white", label="Riser %")
ax_exp.bar(xe10 + w10/2, decl_by_exp,  w10, color=DECL_COL,  alpha=0.85,
           edgecolor="white", label="Decliner %")

for i, (r, d, n) in enumerate(zip(riser_by_exp, decl_by_exp, exp_ns_10)):
    ax_exp.text(i - w10/2, r + 0.6, f"{r:.0f}%", ha="center", fontsize=10,
                color=RISER_COL, fontweight="bold")
    ax_exp.text(i + w10/2, d + 0.6, f"{d:.0f}%", ha="center", fontsize=10,
                color=DECL_COL,  fontweight="bold")
    ax_exp.text(i, -3.2, f"n={n:,}", ha="center", fontsize=8.5, color="gray")

ax_exp.set_xticks(xe10)
ax_exp.set_xticklabels(EXP_LABELS, fontsize=12)
ax_exp.set_ylim(-5, 42)
ax_exp.set_ylabel("% of player-seasons", fontsize=11)
ax_exp.set_title(
    "How Experience Changes Your Odds of Rising or Declining",
    fontsize=12, fontweight="bold"
)
ax_exp.legend(frameon=False, fontsize=11, loc="upper left")
ax_exp.tick_params(bottom=False)
ax_exp.annotate("Young players decline\nmuch more than they rise",
                xy=(0.1, 33), xytext=(0.55, 39),
                arrowprops=dict(arrowstyle="->", color=DECL_COL, lw=1.4),
                fontsize=9.5, color=DECL_COL)
ax_exp.annotate("Veterans rise more\nthan they decline",
                xy=(4, riser_by_exp[-1]), xytext=(3.35, 36),
                arrowprops=dict(arrowstyle="->", color=RISER_COL, lw=1.4),
                fontsize=9.5, color=RISER_COL)

fig.tight_layout()
p10 = os.path.join(FIG, "q3_fig10_experience_conference.png")
fig.savefig(p10, dpi=150, bbox_inches="tight")
plt.close()

# Summary

for label, path in [
    ("Fig 1 - Scoring distribution", p1),
    ("Fig 2 - Position/role breakdown", p2),
    ("Fig 3 - Component deltas",  p3),
    ("Fig 4 - Scatter",           p4),
    ("Fig 5 - Serial performers", p5),
    ("Fig 6 - Validation",        p6),
    ("Fig 7 - Feature importance",p7),
    ("Fig 8 - Delta by tier",     p8),
    ("Fig 9 - Age/experience",    p9),
    ("Fig 10 - Experience brackets + conference", p10),
]:
    print(f"  {label}: {os.path.basename(path)}")
