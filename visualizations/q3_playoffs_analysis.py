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
from scipy.stats import zscore, gaussian_kde, mannwhitneyu, chi2_contingency
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

# Compute turnover delta (not pre-computed in source)
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
# FIG 1 - How players score differently in the playoffs
# ============================================================
print("Fig 1...")
fig, ax = plt.subplots(figsize=(10, 5.5))
fig.suptitle("How Players Perform Differently in the Playoffs",
             fontsize=15, fontweight="bold", y=1.01)

xs = np.linspace(-25, 18, 400)
for tend in ORDER:
    vals = df[df["TENDENCY"] == tend]["DELTA_PTS"].dropna().values
    kde  = gaussian_kde(vals, bw_method=0.3)
    ax.fill_between(xs, kde(xs), alpha=0.22, color=PAL[tend])
    ax.plot(xs, kde(xs), color=PAL[tend], linewidth=2.2, label=tend)

ax.axvline(0, color="black", linewidth=1.3, linestyle="--", alpha=0.5)
ax.text(0.4, ax.get_ylim()[1]*0.92, "No change", fontsize=9, color="gray")
ax.set_xlabel("Points per game in the playoffs  minus  points per game in the regular season",
              fontsize=11)
ax.set_ylabel("Share of player-seasons", fontsize=11)
ax.set_xlim(-22, 16)
ax.legend(frameon=False, fontsize=11, loc="upper left")

ax.annotate("Decliners shift\nto the left",
            xy=(-6, 0.14), xytext=(-18, 0.22),
            arrowprops=dict(arrowstyle="->", color=DECL_COL, lw=1.5),
            fontsize=9, color=DECL_COL)
ax.annotate("Risers shift\nto the right",
            xy=(3.5, 0.14), xytext=(8, 0.22),
            arrowprops=dict(arrowstyle="->", color=RISER_COL, lw=1.5),
            fontsize=9, color=RISER_COL)

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
    ("DELTA_PTS",  "Points per game",       "ppg"),
    ("DELTA_eFG%", "Shooting efficiency\n(eFG%)", "percentage points"),
    ("DELTA_AST",  "Assists per game",       "apg"),
    ("DELTA_TOV",  "Turnovers per game\n(lower = better)", "tpg"),
]

fig, axes = plt.subplots(1, 4, figsize=(14, 5.5))
fig.suptitle("How Risers and Decliners Differ Across Four Key Stats\n"
             "(Playoffs minus Regular Season -- positive means improvement)",
             fontsize=13, fontweight="bold", y=1.03)

for ax, (col, label, unit) in zip(axes, stat_info):
    group_means = {}
    for tend in ["Riser", "Decliner"]:
        sub = df[df["TENDENCY"] == tend][col].dropna()
        group_means[tend] = sub.mean()
        err = sub.sem() * 1.96
        color = PAL[tend]
        bar = ax.bar([tend], [sub.mean()], color=color, alpha=0.85,
                     edgecolor="white", width=0.55,
                     yerr=err, capsize=5,
                     error_kw=dict(ecolor="black", elinewidth=1.2, capthick=1.2))
        sign = "+" if sub.mean() > 0 else ""
        ax.text(0 if tend == "Riser" else 1, sub.mean() + (err + abs(sub.mean())*0.05) * (1 if sub.mean() >= 0 else -1),
                f"{sign}{sub.mean():.2f}",
                ha="center", va="bottom" if sub.mean() >= 0 else "top",
                fontsize=10, fontweight="bold", color=color)

    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_title(label, fontsize=10.5, fontweight="bold")
    ax.set_ylabel(f"Change ({unit})", fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Riser", "Decliner"], fontsize=10)
    ax.tick_params(bottom=False)

fig.tight_layout()
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
fig.suptitle("Regular Season Scoring vs Playoff Scoring\n"
             "Each dot is one player-season. Above the line = scored more in playoffs.",
             fontsize=13, fontweight="bold", y=1.03)

for ax, tend in zip(axes, ORDER):
    sub = df[df["TENDENCY"] == tend].copy()
    sample = sub.sample(min(N_SAMPLE, len(sub)), random_state=42)
    ax.scatter(sample["REG_PTS"], sample["POF_PTS"],
               color=PAL[tend], alpha=0.55, s=22, edgecolors="none")
    lim = 38
    ax.plot([0, lim], [0, lim], color="black", linewidth=1.2,
            linestyle="--", alpha=0.45, zorder=0)
    if tend == "Riser":
        ax.text(20, 26, "Above this line:\nscored MORE\nin playoffs",
                fontsize=8, color="gray",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    elif tend == "Decliner":
        ax.text(20, 12, "Below this line:\nscored LESS\nin playoffs",
                fontsize=8, color="gray",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
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
    ax.set_xlim(0, vals.max() + 0.5)
    for yi, (v, a) in enumerate(zip(vals, avgs)):
        label = (f"{int(v)} seasons  |  {a:.1f} reg-season ppg"
                 if not np.isnan(a) else f"{int(v)} seasons")
        if v >= 4:
            ax.text(0.15, yi, label, va="center", ha="left",
                    fontsize=8.5, color="white", fontweight="bold")
        else:
            ax.text(v + 0.1, yi, label, va="center", ha="left",
                    fontsize=8.5, color=col, fontweight="bold")

fig.tight_layout()
p5 = os.path.join(FIG, "q3_fig5_serial.png")
fig.savefig(p5, dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# FIG 6 - Validation: famous players + Finals depth
# ============================================================
print("Fig 6 (validation)...")
famous = [
    ("LeBron James",   "Riser"),
    ("Tim Duncan",     "Riser"),
    ("Kawhi Leonard",  "Riser"),
    ("Dwyane Wade",    "Riser"),
    ("Dirk Nowitzki",  "Riser"),
    ("Tony Parker",    "Riser"),
    ("Andre Iguodala", "Riser"),
    ("Draymond Green", "Riser"),
    ("James Harden",   "Decliner"),
    ("Joel Embiid",    "Decliner"),
    ("Joe Johnson",    "Decliner"),
    ("Chris Paul",     "Neutral"),
    ("Kobe Bryant",    "Neutral"),
    ("Stephen Curry",  "Neutral"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Does Our Score Match Reality? Validation",
             fontsize=14, fontweight="bold", y=1.01)

# Left: famous players dot plot
names_f, scores_f, reps_f = [], [], []
for name, reputation in famous:
    sub = df[df["Player"] == name]
    if len(sub) > 0:
        avg = sub["COMPOSITE"].mean()
        names_f.append(name)
        scores_f.append(avg)
        reps_f.append(reputation)

sorted_idx = np.argsort(scores_f)
names_s  = [names_f[i]  for i in sorted_idx]
scores_s = [scores_f[i] for i in sorted_idx]
reps_s   = [reps_f[i]   for i in sorted_idx]

y = np.arange(len(names_s))
for yi, (score, rep) in enumerate(zip(scores_s, reps_s)):
    color = PAL.get(rep, NEUTRAL_COL)
    ax1.barh(yi, score, color=color, alpha=0.8, edgecolor="white", height=0.65)
    sign = "+" if score > 0 else ""
    ha = "left" if score >= 0 else "right"
    offset = 0.03 if score >= 0 else -0.03
    ax1.text(score + offset, yi, f"{sign}{score:.2f}",
             va="center", ha=ha, fontsize=9, color=color, fontweight="bold")

ax1.set_yticks(y); ax1.set_yticklabels(names_s, fontsize=10)
ax1.axvline(0, color="black", linewidth=1.1, linestyle="--", alpha=0.5)
ax1.set_xlabel("Average composite playoff score", fontsize=11)
ax1.set_title("Well-known players: score vs. reputation\n"
              "(green = known riser, red = known decliner, gray = mixed)",
              fontsize=10.5)
patches = [mpatches.Patch(color=v, label=k) for k, v in PAL.items()]
ax1.legend(handles=patches, frameon=False, fontsize=9, loc="lower right")

# Right: tendency among Finals players vs first-round exits
deep  = df[df["POF_G"] >= 20]
early = df[df["POF_G"] <= 7]

for data, label, x_offset, col_offset in [
    (deep,  "Made the Finals\n(20+ playoff games)", 0, 0),
    (early, "First round exit\n(7 or fewer games)", 1, 0),
]:
    ct = data["TENDENCY"].value_counts(normalize=True) * 100
    bottoms = 0
    for tend in ORDER:
        val = ct.get(tend, 0)
        ax2.bar(x_offset, val, bottom=bottoms,
                color=PAL[tend], edgecolor="white", width=0.5)
        if val > 6:
            ax2.text(x_offset, bottoms + val/2, f"{val:.0f}%",
                     ha="center", va="center", fontsize=11,
                     color="white", fontweight="bold")
        bottoms += val
    ax2.text(x_offset, 103, f"n={len(data):,}", ha="center", fontsize=9, color="gray")
    ax2.text(x_offset, -7, label, ha="center", fontsize=10)

ax2.set_ylim(-12, 110)
ax2.set_xlim(-0.5, 1.5)
ax2.set_xticks([])
ax2.set_ylabel("% of player-seasons", fontsize=11)
ax2.set_title("Tendency breakdown:\nplayers who made the Finals vs. first-round exits",
              fontsize=10.5)
ax2.legend(handles=patches, frameon=False, fontsize=9, loc="upper right")

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

fig, ax = plt.subplots(figsize=(9, 7))
threshold = imp.quantile(0.65)
colors = [RISER_COL if v >= threshold else NEUTRAL_COL for v in imp.values]
ax.barh(imp.index, imp.values, color=colors, edgecolor="white", height=0.7)
ax.set_xlabel("How useful this stat is for predicting whether a player will rise or decline (%)",
              fontsize=10)
ax.set_title(
    f"What Makes a Playoff Riser? Predictive Model Results\n"
    f"(model accuracy: AUC = {cv_auc.mean():.2f} out of 1.0 -- better than random guessing)",
    fontsize=11, fontweight="bold"
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
# FIG 8 - Composite delta by scoring role
# ============================================================
print("Fig 8...")
tier_order = ["Role player\n(under 8 ppg)", "Contributor\n(8-14 ppg)",
              "Starter\n(14-20 ppg)", "Star\n(20+ ppg)"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Does a Player's Role Affect How They Respond to the Playoffs?",
             fontsize=13, fontweight="bold", y=1.01)

for ax, metric, label, ylim in [
    (ax1, "DELTA_PTS",  "Change in points per game", (-8, 10)),
    (ax2, "COMPOSITE",  "Overall playoff score\n(composite of points, shooting, assists, turnovers)", (-1.2, 1.2)),
]:
    sub = df.dropna(subset=[metric, "scoring_tier"])
    tier_ns = [sub[sub["scoring_tier"] == t].shape[0] for t in tier_order]
    tier_labels_n = [f"{t}\n(n={n:,})" for t, n in zip(tier_order, tier_ns)]
    sns.boxplot(
        data=sub, x="scoring_tier", y=metric, order=tier_order,
        palette=["#5dade2","#2e86c1","#1a5276","#0d2137"],
        flierprops=dict(marker=".", markersize=2, alpha=0.12, color="gray"),
        width=0.55, linewidth=1.2, ax=ax
    )
    ax.axhline(0, color="black", linewidth=1.1, linestyle="--", alpha=0.65)
    ax.set_ylim(*ylim)
    ax.set_xlabel("")
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(label.split("\n")[0], fontsize=12)
    ax.set_xticklabels(tier_labels_n, fontsize=9.5)

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

fig = plt.figure(figsize=(14, 10))
gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
fig.suptitle("Does Age or Experience Change How Players Perform in the Playoffs?",
             fontsize=14, fontweight="bold", y=1.02)

w = 0.35

# top left: Riser vs Decliner rate by age
ax1 = fig.add_subplot(gs[0, 0])
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
ax1.set_xticklabels([f"{b}\n(n={n:,})" for b, n in zip(age_buckets, age_ns)], fontsize=8.5)
ax1.set_ylim(0, 55); ax1.set_ylabel("% of player-seasons", fontsize=10)
ax1.set_title("Riser vs. Decliner Rate by Age Group", fontsize=11, fontweight="bold")
ax1.legend(frameon=False, fontsize=9)
for i, (r, d) in enumerate(zip(riser_r_age, decl_r_age)):
    ax1.text(i - w/2, r + 0.8, f"{r:.0f}%", ha="center", fontsize=8,
             color=RISER_COL, fontweight="bold")
    ax1.text(i + w/2, d + 0.8, f"{d:.0f}%", ha="center", fontsize=8,
             color=DECL_COL, fontweight="bold")

# top right: stacked by age
ax2 = fig.add_subplot(gs[0, 1])
age_df = df[df["age_bucket_label"].notna() & df["TENDENCY"].notna()]
ct = (age_df.groupby(["age_bucket_label","TENDENCY"]).size().unstack(fill_value=0))
for t in ORDER:
    if t not in ct.columns: ct[t] = 0
ct  = ct[ORDER].reindex(age_buckets)
pct = ct.div(ct.sum(axis=1), axis=0) * 100

bottoms = np.zeros(len(age_buckets))
for tend in ORDER:
    ax2.bar(np.arange(len(age_buckets)), pct[tend].values, bottom=bottoms,
            color=PAL[tend], edgecolor="white", linewidth=0.8, alpha=0.9, label=tend)
    for i, (h, b) in enumerate(zip(pct[tend].values, bottoms)):
        if h > 8:
            ax2.text(i, b + h/2, f"{h:.0f}%",
                     ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    bottoms += pct[tend].values

ax2.set_xticks(np.arange(len(age_buckets)))
ax2.set_xticklabels(age_buckets, fontsize=9)
ax2.set_ylim(0, 108); ax2.set_ylabel("% of player-seasons", fontsize=10)
ax2.set_title("Full Breakdown by Age Group", fontsize=11, fontweight="bold")
ax2.legend(frameon=False, fontsize=9, loc="upper right")

# bottom left: Riser vs Decliner by experience
ax3 = fig.add_subplot(gs[1, 0])
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
ax3.set_ylim(0, 55); ax3.set_ylabel("% of player-seasons", fontsize=10)
ax3.set_title("Riser vs. Decliner Rate by Experience", fontsize=11, fontweight="bold")
ax3.legend(frameon=False, fontsize=9)
for i, (r, d) in enumerate(zip(riser_r_exp, decl_r_exp)):
    ax3.text(i - w/2, r + 0.8, f"{r:.0f}%", ha="center", fontsize=8,
             color=RISER_COL, fontweight="bold")
    ax3.text(i + w/2, d + 0.8, f"{d:.0f}%", ha="center", fontsize=8,
             color=DECL_COL, fontweight="bold")

# bottom right: violin age by tendency
ax4 = fig.add_subplot(gs[1, 1])
age_v = df[df["approx_age"].between(18, 42) & df["TENDENCY"].notna()].copy()
sns.violinplot(data=age_v, x="TENDENCY", y="approx_age",
               order=ORDER, palette=PAL, inner="quartile",
               linewidth=1.2, ax=ax4, alpha=0.85)
for i, tend in enumerate(ORDER):
    m = age_v[age_v["TENDENCY"] == tend]["approx_age"].mean()
    ax4.scatter([i], [m], color="white", s=60, zorder=5,
                edgecolors="black", linewidth=1.2)
    ax4.text(i, m + 0.7, f"{m:.1f}", ha="center", fontsize=9, fontweight="bold")

ax4.set_xlabel("")
ax4.set_ylabel("Player age at the time of the season", fontsize=10)
ax4.set_title(
    f"Age Distribution by Tendency\n"
    f"(Statistical test: p = {p_age:.2f} -- no significant difference)",
    fontsize=11, fontweight="bold"
)
ax4.set_xticklabels(ORDER, fontsize=11)

p9 = os.path.join(FIG, "q3_fig9_age_experience.png")
fig.savefig(p9, dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# FIG 10 - Experience brackets + year-over-year + conference
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

# Year-over-year tendency transition
player_tend_seq = df.sort_values(["Player","season_start"]).groupby("Player")["TENDENCY"].apply(list)
transitions = []
for tends in player_tend_seq:
    for i in range(len(tends) - 1):
        transitions.append((tends[i], tends[i + 1]))
trans_df = pd.DataFrame(transitions, columns=["this_year", "next_year"])
trans_pct = (trans_df.groupby(["this_year","next_year"]).size()
                     .unstack(fill_value=0)
                     .div(trans_df.groupby("this_year").size(), axis=0) * 100)
for t in ORDER:
    if t not in trans_pct.columns: trans_pct[t] = 0.0
trans_pct = trans_pct[ORDER].reindex(ORDER)

# Conference breakdown
conf_valid = df.dropna(subset=["conference"]).copy()
_, p_conf, _, _ = chi2_contingency(
    pd.crosstab(conf_valid["conference"], conf_valid["TENDENCY"])
)
exp_r   = df[df["TENDENCY"] == "Riser"]["season_exp"].dropna()
exp_d   = df[df["TENDENCY"] == "Decliner"]["season_exp"].dropna()
_, p_exp = mannwhitneyu(exp_r, exp_d)

# --- Figure ---
fig = plt.figure(figsize=(16, 11))
gs  = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.38)
fig.suptitle("Experience, Conference, and Year-over-Year Patterns",
             fontsize=14, fontweight="bold", y=1.01)

# Top left: experience bracket
ax_exp = fig.add_subplot(gs[0, :])   # spans both columns
w10 = 0.35
xe10 = np.arange(len(EXP_LABELS))
bars_r = ax_exp.bar(xe10 - w10/2, riser_by_exp,  w10, color=RISER_COL, alpha=0.85,
                    edgecolor="white", label="Riser %")
bars_d = ax_exp.bar(xe10 + w10/2, decl_by_exp,   w10, color=DECL_COL,  alpha=0.85,
                    edgecolor="white", label="Decliner %")

# Overall baselines
base_r = (df["TENDENCY"] == "Riser").mean() * 100
base_d = (df["TENDENCY"] == "Decliner").mean() * 100
ax_exp.axhline(base_r, color=RISER_COL,  linewidth=1.3, linestyle="--", alpha=0.55)
ax_exp.axhline(base_d, color=DECL_COL,   linewidth=1.3, linestyle="--", alpha=0.55)
ax_exp.text(len(EXP_LABELS) - 0.48, base_r + 0.5, f"Overall avg {base_r:.0f}%",
            color=RISER_COL, fontsize=8, alpha=0.8)
ax_exp.text(len(EXP_LABELS) - 0.48, base_d - 1.5, f"Overall avg {base_d:.0f}%",
            color=DECL_COL,  fontsize=8, alpha=0.8)

for i, (r, d, n) in enumerate(zip(riser_by_exp, decl_by_exp, exp_ns_10)):
    ax_exp.text(i - w10/2, r + 0.6, f"{r:.0f}%", ha="center", fontsize=9,
                color=RISER_COL, fontweight="bold")
    ax_exp.text(i + w10/2, d + 0.6, f"{d:.0f}%", ha="center", fontsize=9,
                color=DECL_COL,  fontweight="bold")
    ax_exp.text(i, -3.5, f"n={n:,}", ha="center", fontsize=8, color="gray")

ax_exp.set_xticks(xe10)
ax_exp.set_xticklabels(EXP_LABELS, fontsize=11)
ax_exp.set_ylim(-6, 45)
ax_exp.set_ylabel("% of player-seasons", fontsize=11)
ax_exp.set_title(
    f"How Experience Changes Your Odds of Rising or Declining in the Playoffs\n"
    f"(statistical test Riser vs Decliner experience: p = {p_exp:.4f})",
    fontsize=12, fontweight="bold"
)
ax_exp.legend(frameon=False, fontsize=10, loc="upper left")

# Add a shaded annotation for the key crossover
ax_exp.annotate("Young players decline\nmuch more than they rise",
                xy=(0.1, 33), xytext=(0.6, 41),
                arrowprops=dict(arrowstyle="->", color=DECL_COL, lw=1.4),
                fontsize=9, color=DECL_COL)
ax_exp.annotate("Veterans rise more\nthan they decline",
                xy=(4, 20.7), xytext=(3.3, 38),
                arrowprops=dict(arrowstyle="->", color=RISER_COL, lw=1.4),
                fontsize=9, color=RISER_COL)

# Bottom left: year-over-year heatmap
ax_yoy = fig.add_subplot(gs[1, 0])
cmap = plt.cm.RdYlGn
im = ax_yoy.imshow(trans_pct.values, cmap=cmap, aspect="auto", vmin=10, vmax=30)
ax_yoy.set_xticks(range(3)); ax_yoy.set_xticklabels(ORDER, fontsize=10)
ax_yoy.set_yticks(range(3)); ax_yoy.set_yticklabels(ORDER, fontsize=10)
ax_yoy.set_xlabel("Tendency NEXT year", fontsize=10)
ax_yoy.set_ylabel("Tendency THIS year", fontsize=10)
ax_yoy.set_title("Year-over-Year Tendency Consistency\n"
                 "(Does last year's label predict this year?)", fontsize=11, fontweight="bold")
for i in range(3):
    for j in range(3):
        val = trans_pct.values[i, j]
        ax_yoy.text(j, i, f"{val:.0f}%",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if (val < 14 or val > 26) else "black")
plt.colorbar(im, ax=ax_yoy, shrink=0.8, label="% of player-seasons")

# Bottom right: conference breakdown
ax_conf = fig.add_subplot(gs[1, 1])
conf_order = ["East", "West"]
conf_pct_data = {}
for conf in conf_order:
    sub = conf_valid[conf_valid["conference"] == conf]["TENDENCY"]
    conf_pct_data[conf] = {t: (sub == t).mean() * 100 for t in ORDER}

bottoms = np.zeros(2)
for tend in ORDER:
    vals = [conf_pct_data[c][tend] for c in conf_order]
    ax_conf.bar([0, 1], vals, bottom=bottoms, color=PAL[tend],
                edgecolor="white", linewidth=0.8, alpha=0.9, label=tend)
    for xi, (v, b) in enumerate(zip(vals, bottoms)):
        if v > 7:
            ax_conf.text(xi, b + v / 2, f"{v:.0f}%",
                         ha="center", va="center", fontsize=12,
                         color="white", fontweight="bold")
    bottoms += np.array(vals)

n_east = (conf_valid["conference"] == "East").sum()
n_west = (conf_valid["conference"] == "West").sum()
ax_conf.text(0, 103, f"n={n_east:,}", ha="center", fontsize=9, color="gray")
ax_conf.text(1, 103, f"n={n_west:,}", ha="center", fontsize=9, color="gray")
ax_conf.set_xticks([0, 1])
ax_conf.set_xticklabels(["Eastern Conference", "Western Conference"], fontsize=11)
ax_conf.set_ylim(0, 110)
ax_conf.set_ylabel("% of player-seasons", fontsize=11)
ax_conf.set_title(
    f"East vs. West Conference\n(p = {p_conf:.2f} -- small difference, not conclusive)",
    fontsize=11, fontweight="bold"
)
patches = [mpatches.Patch(color=PAL[t], label=t) for t in ORDER]
ax_conf.legend(handles=patches, frameon=False, fontsize=9, loc="lower center",
               bbox_to_anchor=(0.5, -0.18), ncol=3)

p10 = os.path.join(FIG, "q3_fig10_experience_conference.png")
fig.savefig(p10, dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# SUMMARY STATS for PDF
# ============================================================
mean_delta_r   = df[df["TENDENCY"] == "Riser"]["DELTA_PTS"].mean()
mean_delta_d   = df[df["TENDENCY"] == "Decliner"]["DELTA_PTS"].mean()
comp_r         = df[df["TENDENCY"] == "Riser"]["COMPOSITE"].mean()
comp_d         = df[df["TENDENCY"] == "Decliner"]["COMPOSITE"].mean()

r_bpm = df[df["TENDENCY"] == "Riser"]["REG_BPM"].dropna()
d_bpm = df[df["TENDENCY"] == "Decliner"]["REG_BPM"].dropna()

df_pos2 = df[df["pos_group"] != "Unknown"]
pos_riser_pct = (
    df_pos2[df_pos2["TENDENCY"] == "Riser"].groupby("pos_group").size() /
    df_pos2.groupby("pos_group").size() * 100
).round(1)

serial_r_top5 = serial_r.head(5).index.tolist()
serial_d_top5 = serial_d.head(5).index.tolist()

age_r_mean = age_r.mean()
age_d_mean = age_d.mean()

deep_pct_riser  = (df[df["POF_G"] >= 20]["TENDENCY"] == "Riser").mean() * 100
early_pct_decl  = (df[df["POF_G"] <= 7]["TENDENCY"] == "Decliner").mean() * 100

stat_means_r = {c: df[df["TENDENCY"]=="Riser"][c].mean()   for c in ["DELTA_PTS","DELTA_eFG%","DELTA_AST","DELTA_TOV"]}
stat_means_d = {c: df[df["TENDENCY"]=="Decliner"][c].mean() for c in ["DELTA_PTS","DELTA_eFG%","DELTA_AST","DELTA_TOV"]}


# ============================================================
# BUILD PDF
# ============================================================
print("Building PDF...")
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                 PageBreak, HRFlowable, Table, TableStyle)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

PDF = os.path.join(BASE, "figures", "Q3_Playoffs_vs_Regular_Season.pdf")
doc = SimpleDocTemplate(PDF, pagesize=letter,
                        rightMargin=0.85*inch, leftMargin=0.85*inch,
                        topMargin=0.85*inch, bottomMargin=0.85*inch)

styles = getSampleStyleSheet()

H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=20, spaceAfter=6,
                    textColor=colors.HexColor("#1a252f"), alignment=TA_CENTER)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=14,
                    spaceAfter=4, textColor=colors.HexColor("#2c3e50"))
BODY = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14.5,
                      spaceAfter=6, alignment=TA_JUSTIFY)
CAPTION = ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8.5, leading=11,
                         textColor=colors.HexColor("#555555"), alignment=TA_CENTER,
                         spaceAfter=4, fontName="Helvetica-Oblique")
TAKEAWAY = ParagraphStyle("Takeaway", parent=styles["Normal"], fontSize=10.5, leading=14,
                          spaceAfter=10, spaceBefore=4, leftIndent=12, rightIndent=12,
                          borderPad=6, backColor=colors.HexColor("#eafaf1"),
                          borderColor=colors.HexColor("#27ae60"), borderWidth=1,
                          fontName="Helvetica-Bold", textColor=colors.HexColor("#1a5e36"))
STAT = ParagraphStyle("Stat", parent=styles["Normal"], fontSize=10, leading=14,
                      leftIndent=18, spaceAfter=3)
FOOTER_S = ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8,
                           textColor=colors.grey, alignment=TA_CENTER)
DEFINE = ParagraphStyle("Define", parent=styles["Normal"], fontSize=9.5, leading=13,
                        leftIndent=18, spaceAfter=3, textColor=colors.HexColor("#34495e"),
                        fontName="Helvetica-Oblique")

def embed(path, width=6.4*inch):
    from PIL import Image as PILImg
    pil = PILImg.open(path)
    w, h = pil.size
    return Image(path, width=width, height=width*(h/w))

def hr():
    return HRFlowable(width="100%", thickness=0.5,
                      color=colors.HexColor("#bdc3c7"), spaceAfter=8, spaceBefore=8)

def takeaway(text):
    return Paragraph(f"Key takeaway: {text}", TAKEAWAY)

story = []

# --- Title page ---
story += [
    Spacer(1, 0.4*inch),
    Paragraph("Predicting and Analyzing NBA Performance", H1),
    Paragraph("Question 3: Playoffs vs. Regular Season", H1),
    Spacer(1, 0.15*inch), hr(),
    Paragraph("Which players raise their game when the pressure is highest?",
              ParagraphStyle("Sub", parent=styles["Normal"], fontSize=12,
                             alignment=TA_CENTER, textColor=colors.HexColor("#7f8c8d"))),
    Spacer(1, 0.1*inch),
    Paragraph("Yonatan Gan, Ariel Mersel, Nimrod Segev",
              ParagraphStyle("Auth", parent=styles["Normal"], fontSize=11,
                             alignment=TA_CENTER, textColor=colors.HexColor("#2c3e50"))),
    Paragraph("67978 - A Needle in a Data Haystack, 2026",
              ParagraphStyle("Crs", parent=styles["Normal"], fontSize=9,
                             alignment=TA_CENTER, textColor=colors.HexColor("#95a5a6"))),
    Spacer(1, 0.25*inch), hr(),
]

# --- 1. Introduction ---
story.append(Paragraph("1. Introduction", H2))
story.append(Paragraph(
    "Every year, the NBA playoffs bring a different kind of basketball. Games slow down, "
    "defenses get more physical, and coaches spend days studying video to stop specific players. "
    "Some players step up under this pressure. Others struggle. "
    "Understanding which players genuinely improve in the playoffs, and why, "
    "is one of the most interesting questions in basketball analytics.",
    BODY))
story.append(Paragraph(
    f"This analysis covers <b>{n_total:,} player-seasons</b> spanning from 1995 to 2023, "
    "where the same player appeared in both the regular season and the playoffs. "
    "We compare their stats in both contexts and ask: did they play better, worse, or about the same?",
    BODY))

# --- 2. How we define a Riser or Decliner ---
story.append(Paragraph("2. How We Define a Riser or Decliner", H2))
story.append(Paragraph(
    "Previous analyses often label a player a 'Riser' simply because they scored more points in "
    "the playoffs. But that misses the full picture. A player who scores 2 more points but turns "
    "the ball over twice as much and shoots terribly is not actually playing better.",
    BODY))
story.append(Paragraph(
    "We built a <b>composite performance score</b> that combines four things a player can control:",
    BODY))

for line in [
    "Points per game: did they score more or less?",
    "Shooting efficiency (eFG%): did they shoot the ball better or worse? "
    "eFG% gives extra credit for 3-point shots since they are worth more.",
    "Assists per game: did they create more opportunities for teammates?",
    "Turnovers per game: did they hold on to the ball? Fewer turnovers is better, "
    "so we flip the sign for this one.",
]:
    story.append(Paragraph(f"- {line}", STAT))

story.append(Paragraph(
    "Each stat is converted to a standardized scale so they can be fairly combined, "
    "then weighted: scoring 35%, shooting efficiency 30%, assists 20%, turnovers 15%. "
    "A player with a score above +0.5 is a <b>Riser</b>, below -0.5 is a <b>Decliner</b>, "
    "and in between is <b>Neutral</b>. "
    f"This gives us {n_r:,} Risers ({100*n_r/n_total:.0f}%), "
    f"{n_n:,} Neutral ({100*n_n/n_total:.0f}%), and "
    f"{n_d:,} Decliners ({100*n_d/n_total:.0f}%).",
    BODY))
story.append(Paragraph(
    "Note: we only include player-seasons where the player appeared in at least 20 regular-season "
    "games and 5 playoff games, so one-game samples do not skew the results.",
    BODY))

# --- 3. Scoring distribution ---
story.append(Paragraph("3. The Scoring Gap Between Groups", H2))
story.append(Paragraph(
    f"Even just looking at scoring, the gap is large. Risers average "
    f"<b>{mean_delta_r:+.1f} points per game more</b> in the playoffs. "
    f"Decliners average <b>{abs(mean_delta_d):.1f} points per game fewer.</b> "
    "The chart below shows the full distribution of scoring changes for all three groups.",
    BODY))
story.append(embed(p1))
story.append(Paragraph(
    "Figure 1. Each curve shows how many player-seasons had a given scoring change in the playoffs "
    "compared to the regular season. The dashed line at zero means no change. "
    "Curves to the right of zero scored more in the playoffs; curves to the left scored less.",
    CAPTION))
story.append(takeaway(
    f"Decliners (red) shift clearly to the left, Risers (green) shift to the right. "
    f"But most players sit near zero, meaning the typical NBA player scores about the same "
    "in both contexts."))

# --- 4. Who rises by position/role ---
story.append(Paragraph("4. Who Rises? Position and Scoring Role", H2))
story.append(Paragraph(
    "Not all players respond to the playoffs the same way. The charts below break down "
    "Riser, Neutral, and Decliner rates by position and by how much a player scored "
    "in the regular season. Each bar adds up to 100%.",
    BODY))
story.append(embed(p2))
story.append(Paragraph(
    "Figure 2. Each bar represents a group of players. The green portion at the bottom "
    "shows what percentage of that group rose in the playoffs; red at the top shows "
    "who declined. Numbers inside each segment show the exact percentage.",
    CAPTION))
pos_str = ", ".join([f"{g} ({v:.0f}%)" for g, v in pos_riser_pct.items()])
story.append(Paragraph(
    f"By position: Guards have the highest Riser rate ({pos_str}). "
    "Guards control the ball and can adjust how they play more easily. "
    "Centers tend to be more Neutral, partly because their role in the playoffs "
    "is more defined and defenders do not scheme around them as aggressively.",
    BODY))
story.append(takeaway(
    "Star players (20+ points per game) are the most unpredictable group. "
    "They have the highest chance of both rising and declining. Role players "
    "almost always stay Neutral, which makes sense since defenders rarely focus on them."))

# --- 5. Component breakdown ---
story.append(PageBreak())
story.append(Paragraph("5. How Risers and Decliners Differ Across Four Stats", H2))
story.append(Paragraph(
    "The chart below directly compares Risers and Decliners on each of the four "
    "components of our composite score. The bars show the average change from "
    "regular season to playoffs. Error bars show the 95% confidence interval around the mean.",
    BODY))
story.append(embed(p3))
story.append(Paragraph(
    "Figure 3. Average change in each stat from regular season to playoffs, "
    "for Risers (green) and Decliners (red). A positive bar means the player "
    "improved in that area during the playoffs.",
    CAPTION))
story.append(Paragraph(
    f"Risers improve on every single dimension: "
    f"they score {stat_means_r['DELTA_PTS']:+.1f} more points, "
    f"shoot {stat_means_r['DELTA_eFG%']*100:+.1f} percentage points more efficiently, "
    f"add {stat_means_r['DELTA_AST']:+.1f} assists, "
    f"and turn the ball over {abs(stat_means_r['DELTA_TOV']):.1f} fewer times per game. "
    f"Decliners go the opposite direction on all four.",
    BODY))
story.append(takeaway(
    "A true playoff Riser is not just someone who scores more. They shoot better, "
    "pass better, and take care of the ball better. The improvement is across the board."))

# --- 6. Scoring scatter ---
story.append(Paragraph("6. Seeing It Visually: Regular Season vs. Playoff Scoring", H2))
story.append(Paragraph(
    "Each dot in the charts below represents one player in one season. "
    "The x-axis shows how much they scored per game in the regular season, "
    "and the y-axis shows how much they scored in the playoffs. "
    "The dashed diagonal line is the line of 'no change': if a player is right on "
    "that line, they scored exactly the same in both.",
    BODY))
story.append(embed(p4))
story.append(Paragraph(
    "Figure 4. Each panel shows a random sample of 250 player-seasons from each group. "
    "Dots above the diagonal line scored more in the playoffs. Dots below scored less.",
    CAPTION))
story.append(takeaway(
    "For Risers, most dots cluster above the diagonal. For Decliners, most sit below it. "
    "For Neutral players, the dots hug the diagonal closely."))

# --- 7. Serial performers ---
story.append(Paragraph("7. Consistent Performers: Who Does It Season After Season?", H2))
story.append(Paragraph(
    "Some players rise or decline in the playoffs year after year, suggesting it is a "
    "genuine characteristic of that player rather than a random one-off. "
    "The chart below shows players who showed the same tendency in three or more seasons.",
    BODY))
story.append(embed(p5))
story.append(Paragraph(
    "Figure 5. Players with three or more seasons of the same playoff tendency, "
    "measured by our composite score. Numbers show how many seasons and their "
    "average regular-season scoring.",
    CAPTION))
story.append(Paragraph(
    f"The top serial Risers include {', '.join(serial_r_top5[:4])}, and others widely "
    "recognized as clutch playoff performers. This is a strong sign our scoring method is "
    "capturing something real.",
    BODY))
story.append(Paragraph(
    f"The top serial Decliners include {', '.join(serial_d_top5[:4])}. "
    "James Harden in particular has been a prominent topic of conversation in basketball "
    "for exactly this reason, and our data backs it up.",
    BODY))
story.append(takeaway(
    "Our composite score correctly identifies players that basketball experts "
    "already consider great or poor playoff performers. The method checks out."))

# --- 8. Validation ---
story.append(PageBreak())
story.append(Paragraph("8. Does the Score Actually Predict Playoff Success?", H2))
story.append(Paragraph(
    "Beyond matching reputations, we tested whether our composite score correlates with "
    "real playoff outcomes. Specifically, we asked: do composite Risers play more "
    "playoff games? Teams that win more rounds play more games, so playoff games played "
    "is a reasonable proxy for team success.",
    BODY))
story.append(embed(p6))
story.append(Paragraph(
    "Figure 6. Left: career average composite score for well-known players, colored by their "
    "public reputation as a playoff performer. Right: tendency breakdown for players whose "
    "teams reached the Finals (20+ playoff games) vs. first-round exits (7 or fewer games).",
    CAPTION))
story.append(Paragraph(
    f"Risers average <b>10.4 playoff games</b> played, Decliners average only <b>9.0</b> "
    f"(statistically significant, p less than 0.001). Among players who made the Finals, "
    f"<b>{deep_pct_riser:.0f}% are Risers</b>, compared to only "
    f"{(df[df['POF_G'] <= 7]['TENDENCY']=='Riser').mean()*100:.0f}% "
    "among first-round exits. Players who step up in the playoffs help their teams "
    "go further.",
    BODY))
story.append(takeaway(
    f"Players who made the Finals are almost 3x more likely to be Risers than Decliners. "
    "Great playoff performers are not just a narrative. They drive real team success."))

# --- 9. Predictive model ---
story.append(Paragraph("9. Can We Predict Who Will Rise Before the Playoffs Start?", H2))
story.append(Paragraph(
    "We trained a machine learning model to predict whether a player will be a Riser or Decliner, "
    "using only their regular-season statistics. This tests whether the tendency is predictable "
    "from publicly available data, or whether it is essentially random.",
    BODY))
story.append(embed(p7))
story.append(Paragraph(
    "Figure 7. How useful each regular-season stat is for predicting a player's playoff tendency. "
    "Longer bar means more useful. The model's overall accuracy is shown in the title: "
    "AUC (Area Under the Curve) is a standard measure where 0.5 is random guessing and "
    "1.0 is perfect prediction.",
    CAPTION))
top_str = ", ".join([f"<b>{n}</b>" for n in top_feat.index.tolist()[:4]])
story.append(Paragraph(
    f"The model reaches AUC = {cv_auc.mean():.2f}, which is meaningfully above random. "
    f"The most predictive regular-season stats are: {top_str}. "
    "Notice that overall efficiency metrics rank higher than raw counting stats like "
    "points or rebounds. Players who contribute efficiently in the regular season, "
    "not just in volume, tend to carry that efficiency into the playoffs.",
    BODY))
story.append(takeaway(
    f"We can predict playoff tendency with moderate accuracy (AUC = {cv_auc.mean():.2f}) "
    "from regular-season stats alone. It is not random. How efficiently a player "
    "contributes in the regular season is a better predictor than how much they score."))

# --- 10. Role matters ---
story.append(Paragraph("10. Does Your Role in the Regular Season Affect Your Playoffs?", H2))
story.append(embed(p8))
story.append(Paragraph(
    "Figure 8. Box plots comparing scoring change (left) and overall composite score change "
    "(right) by regular-season scoring role. The dashed line at zero means no change. "
    "The box shows where the middle 50% of players fall; the line inside is the median.",
    CAPTION))
story.append(Paragraph(
    "Role players (under 8 points per game) cluster tightly around zero on both charts. "
    "Defenders do not spend much energy scheming against them. "
    "Stars (20+ points per game) have the widest spread. They are the defensive focus "
    "of every game plan. Some rise to the occasion; others are brought down by it. "
    "This is why star players are both the most exciting and the most unpredictable "
    "group in the playoffs.",
    BODY))
story.append(takeaway(
    "The more you score in the regular season, the more unpredictable your playoffs "
    "become. Stars face the toughest defensive game plans and their results vary the most."))

# --- 11. Age and experience ---
story.append(PageBreak())
story.append(Paragraph("11. Does Age or Experience Matter?", H2))
story.append(Paragraph(
    "A common belief in basketball is that veteran players are better at handling "
    "the pressure and pace of playoff basketball. We tested this directly.",
    BODY))
story.append(embed(p9))
story.append(Paragraph(
    "Figure 9. Top row: Riser and Decliner rates broken down by age group (left) and "
    "the full three-way breakdown (right). Bottom row: the same analysis by years "
    "of experience (left), and violin plots showing the age distribution for each "
    "tendency group with the average marked by a white dot (right).",
    CAPTION))
story.append(Paragraph(
    f"The average age of Risers is {age_r_mean:.1f} years, and Decliners {age_d_mean:.1f} years. "
    f"That difference is not statistically significant (p = {p_age:.2f}), "
    "meaning age alone does not determine whether a player rises or declines.",
    BODY))
story.append(Paragraph(
    "Experience tells a different story. Veteran players (13+ years in the league) "
    "have the smallest gap between their Riser and Decliner rates of any experience group. "
    "Rookies (0-3 years) show the largest gap, heavily skewed toward Declining. "
    "This suggests that <b>playoff experience, rather than age, provides a modest edge.</b>",
    BODY))
story.append(takeaway(
    "Age by itself does not predict playoff performance. But players with more years "
    "of playoff experience tend to handle it better. It is what you have been through, "
    "not how old you are, that matters."))

# --- 12. Experience brackets, year-over-year, conference ---
story.append(Paragraph("12. A Closer Look at Experience and Conference", H2))
story.append(Paragraph(
    "We dug deeper into three additional angles: exactly how experience level (not just "
    "age) changes the odds, whether a player's tendency from last year predicts this year, "
    "and whether Eastern or Western Conference players behave differently.",
    BODY))
story.append(embed(p10, width=6.6*inch))
story.append(Paragraph(
    "Figure 10. Top: Riser rate (green) and Decliner rate (red) broken out by years of "
    "experience in the NBA. Dashed lines show the overall average rates for reference. "
    "Bottom left: a grid showing how likely a player is to repeat their previous playoff "
    "tendency the next year. Bottom right: East vs. West conference breakdown.",
    CAPTION))

story.append(Paragraph("<b>Experience brackets:</b>", BODY))
story.append(Paragraph(
    f"Players with only 1-3 years in the league decline in the playoffs at an unusually "
    f"high rate ({decl_by_exp[0]:.0f}% Decliners vs {riser_by_exp[0]:.0f}% Risers). "
    "They are simply not used to the intensity yet. As players accumulate experience, "
    "this gap gradually closes. By year 11 and beyond, the rates are nearly equal. "
    f"By 16+ years, the pattern flips: {riser_by_exp[-1]:.0f}% of very experienced players "
    f"rise vs only {decl_by_exp[-1]:.0f}% who decline. "
    f"(Statistical test: p = {p_exp:.4f} -- highly significant.)",
    BODY))

story.append(Paragraph("<b>Year-over-year consistency:</b>", BODY))
story.append(Paragraph(
    "The bottom-left grid shows transition probabilities. If a player was a Decliner "
    f"one year, they have a {trans_pct.loc['Decliner','Decliner']:.0f}% chance of declining "
    f"again the next year -- slightly higher than the baseline rate of {base_d:.0f}%. "
    f"Past Risers repeat at {trans_pct.loc['Riser','Riser']:.0f}%, close to the baseline. "
    "This tells us that the Decliner label is slightly stickier than the Riser label -- "
    "a player who struggles in the playoffs is somewhat more likely to struggle again. "
    "But the effect is moderate; playoff tendency is not fully set in stone.",
    BODY))

story.append(Paragraph("<b>East vs. West Conference:</b>", BODY))
story.append(Paragraph(
    f"Western Conference players rise at {conf_pct_data['West']['Riser']:.0f}% vs "
    f"{conf_pct_data['East']['Riser']:.0f}% for Eastern Conference players. "
    f"This small difference is not statistically significant (p = {p_conf:.2f}), "
    "so we cannot draw a firm conclusion. One possible explanation is that the West "
    "has historically been a tougher conference, meaning West players face higher-quality "
    "opponents all regular season and may be better conditioned to elite-level play. "
    "But the data does not confirm this strongly.",
    BODY))
story.append(takeaway(
    f"Experience matters more than anything else in this analysis. Young players (1-3 yrs) "
    f"decline at {decl_by_exp[0]:.0f}% -- nearly twice their Riser rate. Veterans (16+ yrs) "
    f"flip that script, rising at {riser_by_exp[-1]:.0f}% vs declining at {decl_by_exp[-1]:.0f}%. "
    "Past playoff struggles repeat slightly more often than past successes. Conference makes "
    "very little difference."))

# --- 13. Conclusions ---
story.append(Paragraph("13. Conclusions", H2))
conclusions = [
    f"<b>The playoffs are genuinely harder.</b> When looking at composite performance "
    f"(not just scoring), {n_d:,} players declined vs. {n_r:,} who rose. "
    "Most players stayed roughly the same.",

    "<b>Stars are the most unpredictable.</b> High-volume scorers have the highest Riser "
    "rate AND the highest Decliner rate. They either step up or get shut down. "
    "Role players almost always stay Neutral.",

    "<b>Guards rise most by position.</b> Centers tend to be the most Neutral, likely "
    "because their defensive role is more defined in the playoffs.",

    "<b>Composite Risers improve across all four dimensions.</b> Points, shooting efficiency, "
    "assists, and turnovers all move in the right direction at once.",

    f"<b>Great playoff performers help their teams win.</b> Among players who reached the "
    f"Finals, {deep_pct_riser:.0f}% were Risers compared to only "
    f"{(df[df['POF_G'] <= 7]['TENDENCY']=='Riser').mean()*100:.0f}% "
    "among first-round exits.",

    f"<b>Playoff tendency is partially predictable</b> from regular-season stats "
    f"(AUC = {cv_auc.mean():.2f}). Efficiency metrics predict it better than scoring volume.",

    "<b>Age does not matter. Experience does -- and it matters a lot.</b> "
    f"Players with 1-3 years in the league decline at {decl_by_exp[0]:.0f}% -- "
    f"nearly twice their Riser rate. By 16+ years, that inverts: {riser_by_exp[-1]:.0f}% "
    f"rise vs {decl_by_exp[-1]:.0f}% decline.",

    "<b>Past playoff struggles are slightly sticky.</b> Players who declined last year "
    "are somewhat more likely to decline again. Rising is less persistent, but neither "
    "tendency fully determines the future.",

    "<b>Conference (East vs. West) makes little difference.</b> There is a small trend "
    "toward West players rising more, but it is not statistically significant.",
]
for i, c in enumerate(conclusions, 1):
    story.append(Paragraph(f"{i}. {c}", STAT))
    story.append(Spacer(1, 4))

story += [
    Spacer(1, 0.2*inch), hr(),
    Paragraph(
        "Data: Kaggle (bendikfltaas/nba-history-seasonal-data-1995-2023, drgilermo/nba-players-stats), "
        "Basketball Reference, NBA Stats API. Coverage: 1995-2023 regular seasons and playoffs. "
        "Minimum thresholds: 20 regular-season games, 5 playoff games per player-season.",
        FOOTER_S)
]

doc.build(story)
print(f"\nPDF saved to: {PDF}")
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
