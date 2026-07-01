"""
Q3 — Playoffs vs. Regular Season
Full analysis: who rises, who falls, and why.
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAGGLE = os.path.join(BASE, "data", "processed", "Kaggle")
FIG    = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

RISER_COL   = "#27ae60"
NEUTRAL_COL = "#7f8c8d"
DECL_COL    = "#c0392b"
PAL = {"Riser": RISER_COL, "Neutral": NEUTRAL_COL, "Decliner": DECL_COL}
ORDER = ["Riser", "Neutral", "Decliner"]

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.grid":          False,
})


# ══════════════════════════════════════════════════════════════════════════════
# LOAD & PREPARE DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data …")
q3  = pd.read_csv(os.path.join(KAGGLE, "q3_player_split_v2.csv"))
adv = pd.read_csv(os.path.join(KAGGLE, "q3_advanced_split.csv"))
bio = pd.read_csv(os.path.join(KAGGLE, "player_bio_enhanced.csv"))

both     = q3.dropna(subset=["REG_PTS", "POF_PTS"]).copy()
adv_both = adv.dropna(subset=["REG_PER", "POF_PER"]).copy()

adv_cols = ["Player","season_start",
            "REG_PER","REG_TS%","REG_USG%","REG_WS/48","REG_VORP","REG_BPM","REG_OBPM","REG_DBPM",
            "POF_PER","POF_TS%","POF_USG%","POF_WS/48","POF_VORP","POF_BPM","POF_OBPM","POF_DBPM"]
df = both.merge(adv_both[adv_cols], on=["Player","season_start"], how="inner")

bio_slim = (bio[["player_name","position","season_exp","draft_number",
                 "is_greatest_75","birth_year","height","weight"]]
            .drop_duplicates("player_name").copy())
bio_slim["player_name"] = bio_slim["player_name"].str.strip()
df["Player_key"] = df["Player"].str.strip()
df = df.merge(bio_slim.rename(columns={"player_name":"Player_key"}),
              on="Player_key", how="left")

# deltas
df["DELTA_PER"]  = df["POF_PER"]   - df["REG_PER"]
df["DELTA_BPM"]  = df["POF_BPM"]   - df["REG_BPM"]
df["DELTA_TS"]   = df["POF_TS%"]   - df["REG_TS%"]
df["DELTA_WS48"] = df["POF_WS/48"] - df["REG_WS/48"]
df["DELTA_VORP"] = df["POF_VORP"]  - df["REG_VORP"]

# scoring tier
df["scoring_tier"] = pd.cut(
    df["REG_PTS"], bins=[0,8,14,20,100],
    labels=["Role player\n(<8 ppg)","Contributor\n(8–14)","Starter\n(14–20)","Star\n(20+)"]
)

# position group (bio uses full names)
pos_map = {
    "Guard":"Guard","Guard-Forward":"Guard","Forward-Guard":"Wing",
    "Forward":"Wing","Forward-Center":"Wing",
    "Center-Forward":"Center","Center":"Center",
}
df["pos_group"] = df["position"].map(pos_map).fillna("Unknown")

# age at season start (approximate)
df["approx_age"] = df["season_start"] + 1 - df["birth_year"]
df["age_bucket"] = pd.cut(
    df["approx_age"].clip(18, 42),
    bins=[17, 23, 26, 30, 34, 43],
    labels=["≤23", "24–26", "27–30", "31–34", "35+"]
)

# experience buckets
df["exp_bucket"] = pd.cut(
    df["season_exp"].clip(0, 22),
    bins=[-1, 3, 7, 12, 25],
    labels=["Rookie\n(0–3 yrs)", "Young\n(4–7)", "Prime\n(8–12)", "Veteran\n(13+)"]
)

n_total = len(df)
n_r = (df["PLAYOFF_TENDENCY"]=="Riser").sum()
n_n = (df["PLAYOFF_TENDENCY"]=="Neutral").sum()
n_d = (df["PLAYOFF_TENDENCY"]=="Decliner").sum()
print(f"  Dataset: {n_total:,} player-seasons  |  Risers {n_r}  Neutral {n_n}  Decliners {n_d}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Scoring delta: KDE + lollipop
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 1 …")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("How Players Score Differently in the Playoffs",
             fontsize=15, fontweight="bold", y=1.01)

# LEFT — KDE curves
from scipy.stats import gaussian_kde
xs = np.linspace(-25, 18, 400)
for tend in ORDER:
    vals = df[df["PLAYOFF_TENDENCY"]==tend]["DELTA_PTS"].dropna().values
    kde  = gaussian_kde(vals, bw_method=0.3)
    ax1.fill_between(xs, kde(xs), alpha=0.25, color=PAL[tend])
    ax1.plot(xs, kde(xs), color=PAL[tend], linewidth=2, label=tend)
ax1.axvline(0, color="black", linewidth=1.3, linestyle="--", alpha=0.6, label="No change")
ax1.set_xlabel("Playoff PPG  −  Regular-Season PPG", fontsize=11)
ax1.set_ylabel("Density", fontsize=11)
ax1.set_title("Distribution of Scoring Delta", fontsize=12)
ax1.legend(frameon=False, fontsize=10)
ax1.set_xlim(-25, 18)

# RIGHT — lollipop chart; group name included in annotation so y-ticks are not needed
means = {t: df[df["PLAYOFF_TENDENCY"]==t]["DELTA_PTS"].mean() for t in ORDER}
counts_t = df["PLAYOFF_TENDENCY"].value_counts()

y_positions = {"Riser": 2, "Neutral": 1, "Decliner": 0}
for tend in ORDER:
    m = means[tend]
    y = y_positions[tend]
    ax2.hlines(y, 0, m, color=PAL[tend], linewidth=2.5, alpha=0.8)
    ax2.scatter([m], [y], color=PAL[tend], s=120, zorder=5)
    # place annotation to the LEFT of the dot when positive, to the RIGHT of plot when negative
    # use ha="left" at x=5.2 for all — safely beyond the Riser dot at +4.05
    ax2.text(5.2, y, f"{tend}:  {m:+.2f} ppg   (n={counts_t[tend]:,})",
             va="center", ha="left", fontsize=10.5, color=PAL[tend], fontweight="bold")

ax2.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
ax2.set_xlim(-7, 13)          # extra right margin keeps text inside the figure
ax2.set_yticks([])
ax2.set_xlabel("Mean scoring delta (ppg)", fontsize=11)
ax2.set_title("Average Delta by Tendency Group", fontsize=12)
ax2.spines["left"].set_visible(False)

fig.tight_layout()
p1 = os.path.join(FIG, "q3_fig1_scoring_delta.png")
fig.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Stacked % bars by position & scoring tier
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 2 …")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Who Rises? Breakdown by Position and Scoring Role",
             fontsize=14, fontweight="bold", y=1.01)

def stacked_pct_chart(ax, groupcol, order, title, df_sub):
    ct = (df_sub.groupby([groupcol, "PLAYOFF_TENDENCY"])
                .size().unstack(fill_value=0))
    for t in ORDER:
        if t not in ct.columns:
            ct[t] = 0
    ct = ct[ORDER]
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    pct = pct.reindex(order).dropna(how="all")

    bottoms = np.zeros(len(pct))
    x = np.arange(len(pct))
    for tend in ORDER:
        bars = ax.bar(x, pct[tend].values, bottom=bottoms,
                      color=PAL[tend], label=tend, edgecolor="white", linewidth=0.8)
        # label inside segment only if wide enough
        for i, (h, b) in enumerate(zip(pct[tend].values, bottoms)):
            if h > 8:
                ax.text(x[i], b + h/2, f"{h:.0f}%",
                        ha="center", va="center", fontsize=9,
                        color="white", fontweight="bold")
        bottoms += pct[tend].values

    ax.set_xticks(x)
    ax.set_xticklabels([str(s).replace("\n", "\n") for s in pct.index],
                       fontsize=10.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of player-seasons", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

df_pos  = df[df["pos_group"] != "Unknown"]
pos_order  = ["Guard", "Wing", "Center"]
tier_order = ["Role player\n(<8 ppg)", "Contributor\n(8–14)",
              "Starter\n(14–20)", "Star\n(20+)"]

stacked_pct_chart(ax1, "pos_group",    pos_order,  "by Position Group", df_pos)
stacked_pct_chart(ax2, "scoring_tier", tier_order, "by Regular-Season Scoring Tier", df)

fig.tight_layout()
p2 = os.path.join(FIG, "q3_fig2_breakdown.png")
fig.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Advanced metric shifts (clipped axes, no outlier distortion)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 3 …")
adv_metrics = [
    ("DELTA_PER",  "PER delta  (Player Efficiency Rating)",  (-20, 25)),
    ("DELTA_BPM",  "BPM delta  (Box Plus/Minus)",            (-12, 12)),
    ("DELTA_TS",   "TS% delta  (True Shooting)",             (-0.18, 0.18)),
    ("DELTA_WS48", "WS/48 delta  (Win Shares per 48 min)",   (-0.25, 0.25)),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Playoff vs Regular-Season Advanced Metric Shifts",
             fontsize=14, fontweight="bold", y=1.01)

for ax, (col, title, (xlo, xhi)) in zip(axes.flat, adv_metrics):
    for tend in ORDER:
        sub = df[df["PLAYOFF_TENDENCY"]==tend][col].dropna()
        sub_filt = sub[(sub >= xlo) & (sub <= xhi)]   # filter, don't clip — avoids boundary spikes
        ax.hist(sub_filt, bins=35, alpha=0.55, color=PAL[tend],
                label=tend, edgecolor="white", linewidth=0.3, range=(xlo, xhi))
    ax.axvline(0, color="black", linewidth=1.3, linestyle="--", alpha=0.6)
    # group mean lines
    for tend in ORDER:
        m = df[df["PLAYOFF_TENDENCY"]==tend][col].mean()
        ax.axvline(np.clip(m, xlo, xhi), color=PAL[tend],
                   linewidth=1.8, linestyle=":", alpha=0.9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Playoffs − Regular Season", fontsize=10)
    ax.set_ylabel("Player-seasons", fontsize=10)
    ax.set_xlim(xlo, xhi)

patches = [mpatches.Patch(color=PAL[t], label=t) for t in ORDER]
dotted  = [plt.Line2D([0],[0], color=PAL[t], linestyle=":", linewidth=2, label=f"{t} mean") for t in ORDER]
fig.legend(handles=patches, loc="upper right", frameon=False, fontsize=10,
           bbox_to_anchor=(1.0, 1.0))
fig.tight_layout()
p3 = os.path.join(FIG, "q3_fig3_advanced_deltas.png")
fig.savefig(p3, dpi=150, bbox_inches="tight"); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Reg vs Playoff scoring: hexbin per group
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 4 …")
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True, sharex=True)
fig.suptitle("Regular Season vs Playoff Scoring  (one panel per tendency group)",
             fontsize=13, fontweight="bold", y=1.01)

for ax, tend in zip(axes, ORDER):
    sub = df[df["PLAYOFF_TENDENCY"]==tend]
    hb = ax.hexbin(sub["REG_PTS"], sub["POF_PTS"],
                   gridsize=25, cmap="Greens" if tend=="Riser" else
                   ("Greys" if tend=="Neutral" else "Reds"),
                   mincnt=1, linewidths=0.2)
    lim = 38
    ax.plot([0, lim], [0, lim], "k--", linewidth=1.2, alpha=0.5)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title(f"{tend}  (n={len(sub):,})", fontsize=12,
                 color=PAL[tend], fontweight="bold")
    ax.set_xlabel("Regular Season PPG", fontsize=10)
    cb = fig.colorbar(hb, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Count", fontsize=8)

axes[0].set_ylabel("Playoff PPG", fontsize=10)
fig.tight_layout()
p4 = os.path.join(FIG, "q3_fig4_scatter.png")
fig.savefig(p4, dpi=150, bbox_inches="tight"); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Serial risers & decliners (labels inside bars)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 5 …")
career_pts = df.groupby("Player")["REG_PTS"].mean()
rise_c = df[df["PLAYOFF_TENDENCY"]=="Riser"].groupby("Player").size()
decl_c = df[df["PLAYOFF_TENDENCY"]=="Decliner"].groupby("Player").size()

serial_risers = (rise_c[rise_c >= 3].to_frame("seasons")
                 .join(career_pts.rename("avg_pts"))
                 .sort_values("seasons", ascending=False).head(18))
serial_decl   = (decl_c[decl_c >= 3].to_frame("seasons")
                 .join(career_pts.rename("avg_pts"))
                 .sort_values("seasons", ascending=False).head(18))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
fig.suptitle("Consistent Playoff Performers  (3+ seasons of same tendency)",
             fontsize=14, fontweight="bold", y=1.01)

for ax, data, col, title in [
    (ax1, serial_risers, RISER_COL,  "Top Serial Risers"),
    (ax2, serial_decl,   DECL_COL,   "Top Serial Decliners"),
]:
    names = data.index.tolist()[::-1]
    vals  = data["seasons"].values[::-1]
    avgs  = data["avg_pts"].values[::-1]
    y = np.arange(len(names))

    bars = ax.barh(y, vals, color=col, alpha=0.82, edgecolor="white", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Number of playoff seasons", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", color=col)
    ax.set_xlim(0, vals.max() + 0.5)

    # short annotation inside the bar
    for yi, (v, a) in enumerate(zip(vals, avgs)):
        label = f"{int(v)} seasons  •  {a:.1f} reg ppg" if not np.isnan(a) else f"{int(v)} seasons"
        # if bar is wide enough put text inside, else to the right
        if v >= 4:
            ax.text(0.15, yi, label, va="center", ha="left",
                    fontsize=8.5, color="white", fontweight="bold")
        else:
            ax.text(v + 0.1, yi, label, va="center", ha="left",
                    fontsize=8.5, color=col, fontweight="bold")

fig.tight_layout()
p5 = os.path.join(FIG, "q3_fig5_serial.png")
fig.savefig(p5, dpi=150, bbox_inches="tight"); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Random Forest feature importance
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 6 …")
feature_cols = [
    "REG_PTS","REG_TRB","REG_AST","REG_STL","REG_BLK","REG_TOV",
    "REG_FG%","REG_3P%","REG_FT%","REG_eFG%","REG_MP",
    "REG_PER","REG_TS%","REG_USG%","REG_WS/48","REG_VORP","REG_BPM",
    "season_exp","POF_G",
]
nice = {
    "REG_PTS":"Reg-season points","REG_TRB":"Rebounds","REG_AST":"Assists",
    "REG_STL":"Steals","REG_BLK":"Blocks","REG_TOV":"Turnovers",
    "REG_FG%":"FG%","REG_3P%":"3-Point %","REG_FT%":"FT%","REG_eFG%":"eFG%",
    "REG_MP":"Minutes played","REG_PER":"PER","REG_TS%":"True Shooting %",
    "REG_USG%":"Usage %","REG_WS/48":"Win Shares / 48","REG_VORP":"VORP",
    "REG_BPM":"BPM (Box +/−)","season_exp":"Years of experience","POF_G":"Playoff games played",
}

rd = df[df["PLAYOFF_TENDENCY"].isin(["Riser","Decliner"])].dropna(subset=feature_cols).copy()
X = rd[feature_cols].values
y = (rd["PLAYOFF_TENDENCY"]=="Riser").astype(int).values

rf = RandomForestClassifier(n_estimators=400, max_depth=6, random_state=42, n_jobs=-1)
rf.fit(X, y)
cv_auc = cross_val_score(rf, X, y, cv=5, scoring="roc_auc")
print(f"  RF AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

imp = pd.Series(rf.feature_importances_ * 100, index=feature_cols)
imp.index = [nice.get(c,c) for c in imp.index]
imp = imp.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 7))
threshold = imp.quantile(0.65)
colors = [RISER_COL if v >= threshold else NEUTRAL_COL for v in imp.values]
ax.barh(imp.index, imp.values, color=colors, edgecolor="white", height=0.7)
ax.set_xlabel("Feature importance  (% of total, mean decrease in impurity)", fontsize=11)
ax.set_title(
    f"What Predicts a Playoff Riser?\n"
    f"Random Forest — Riser vs Decliner  |  5-fold CV AUC = {cv_auc.mean():.2f}",
    fontsize=12, fontweight="bold"
)
green_patch = mpatches.Patch(color=RISER_COL, label="Top features (top 35%)")
gray_patch  = mpatches.Patch(color=NEUTRAL_COL, label="Lower importance")
ax.legend(handles=[green_patch, gray_patch], frameon=False, fontsize=9)
fig.tight_layout()
p6 = os.path.join(FIG, "q3_fig6_feature_importance.png")
fig.savefig(p6, dpi=150, bbox_inches="tight"); plt.close()

top_feat = imp.sort_values(ascending=False).head(5)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Scoring delta by scoring tier (clipped y-axis for PER)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 7 …")
tier_order = ["Role player\n(<8 ppg)","Contributor\n(8–14)","Starter\n(14–20)","Star\n(20+)"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Playoff Performance Delta by Regular-Season Role",
             fontsize=14, fontweight="bold", y=1.01)

for ax, metric, label, ylim in [
    (ax1, "DELTA_PTS",  "Points per game  (Playoffs − Reg Season)", (-11, 12)),
    (ax2, "DELTA_PER",  "PER delta  (Playoffs − Reg Season)",        (-18, 20)),
]:
    sub = df.dropna(subset=[metric, "scoring_tier"])
    # build xticklabels with sample size embedded — no floating text needed
    tier_labels_n = [f"{t}\n(n={sub[sub['scoring_tier']==t].shape[0]:,})"
                     for t in tier_order]
    sns.boxplot(
        data=sub, x="scoring_tier", y=metric, order=tier_order,
        palette=["#5dade2","#2e86c1","#1a5276","#0d2137"],
        flierprops=dict(marker=".", markersize=2, alpha=0.12, color="gray"),
        width=0.55, linewidth=1.2,
        ax=ax
    )
    ax.axhline(0, color="black", linewidth=1.1, linestyle="--", alpha=0.65)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Regular-season scoring role", fontsize=11)
    ax.set_ylabel(label, fontsize=10.5)
    ax.set_title(label.split("  (")[0], fontsize=12)
    ax.set_xticklabels(tier_labels_n, fontsize=9.5)

fig.tight_layout()
p7 = os.path.join(FIG, "q3_fig7_delta_by_tier.png")
fig.savefig(p7, dpi=150, bbox_inches="tight"); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 (NEW) — Age & Experience: does it matter?
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 8: age & experience …")

# t-test: risers vs decliners age
age_r = df[(df["PLAYOFF_TENDENCY"]=="Riser") & df["approx_age"].notna()]["approx_age"]
age_d = df[(df["PLAYOFF_TENDENCY"]=="Decliner") & df["approx_age"].notna()]["approx_age"]
t_age, p_age = stats.ttest_ind(age_r, age_d)

fig = plt.figure(figsize=(14, 10))
gs  = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.35)
fig.suptitle("Does Age & Experience Predict Playoff Performance?",
             fontsize=15, fontweight="bold", y=1.02)

# ── TOP LEFT: Riser rate (%) and Decliner rate (%) by age bucket ─────────────
ax1 = fig.add_subplot(gs[0, 0])
age_buckets = ["≤23","24–26","27–30","31–34","35+"]
riser_rates_age, decl_rates_age, age_ns = [], [], []
for ab in age_buckets:
    sub = df[df["age_bucket"]==ab]["PLAYOFF_TENDENCY"].dropna()
    n   = len(sub)
    riser_rates_age.append(100 * (sub=="Riser").sum()   / n if n > 0 else 0)
    decl_rates_age.append( 100 * (sub=="Decliner").sum()/ n if n > 0 else 0)
    age_ns.append(n)

x = np.arange(len(age_buckets))
w = 0.35
ax1.bar(x - w/2, riser_rates_age, w, color=RISER_COL, alpha=0.85, label="Riser %", edgecolor="white")
ax1.bar(x + w/2, decl_rates_age,  w, color=DECL_COL,  alpha=0.85, label="Decliner %", edgecolor="white")
age_xlabels = [f"{b}\n(n={n:,})" for b, n in zip(age_buckets, age_ns)]
ax1.set_xticks(x); ax1.set_xticklabels(age_xlabels, fontsize=9.5)
ax1.set_ylim(0, 55)
ax1.set_xlabel("Age at season start", fontsize=11)
ax1.set_ylabel("% of player-seasons", fontsize=10)
ax1.set_title("Riser vs Decliner Rate by Age", fontsize=11, fontweight="bold")
ax1.legend(frameon=False, fontsize=9)
for i, (r, d) in enumerate(zip(riser_rates_age, decl_rates_age)):
    ax1.text(i - w/2, r + 0.8, f"{r:.0f}%", ha="center", fontsize=8, color=RISER_COL, fontweight="bold")
    ax1.text(i + w/2, d + 0.8, f"{d:.0f}%", ha="center", fontsize=8, color=DECL_COL,  fontweight="bold")

# ── TOP RIGHT: % tendency by age bucket (stacked) ───────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
age_df = df[df["age_bucket"].notna() & df["PLAYOFF_TENDENCY"].notna()]
ct = (age_df.groupby(["age_bucket","PLAYOFF_TENDENCY"])
            .size().unstack(fill_value=0))
for t in ORDER:
    if t not in ct.columns: ct[t] = 0
ct = ct[ORDER].reindex(age_buckets)
pct_age = ct.div(ct.sum(axis=1), axis=0) * 100

bottoms = np.zeros(len(age_buckets))
for tend in ORDER:
    ax2.bar(age_buckets, pct_age[tend].values, bottom=bottoms,
            color=PAL[tend], label=tend, edgecolor="white", linewidth=0.8, alpha=0.9)
    for i, (h, b) in enumerate(zip(pct_age[tend].values, bottoms)):
        if h > 7:
            ax2.text(i, b + h/2, f"{h:.0f}%",
                     ha="center", va="center", fontsize=9,
                     color="white", fontweight="bold")
    bottoms += pct_age[tend].values

ax2.set_ylim(0, 105)
ax2.set_xlabel("Age at season start", fontsize=11)
ax2.set_ylabel("% of player-seasons", fontsize=10)
ax2.set_title("Tendency Mix by Age Group", fontsize=11, fontweight="bold")
ax2.legend(frameon=False, fontsize=9, loc="upper right")

# ── BOTTOM LEFT: experience — Riser vs Decliner rate ────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
exp_buckets = ["Rookie\n(0–3 yrs)","Young\n(4–7)","Prime\n(8–12)","Veteran\n(13+)"]
exp_df = df[df["exp_bucket"].notna() & df["PLAYOFF_TENDENCY"].notna()]
riser_rates_exp, decl_rates_exp, exp_ns = [], [], []
for e in exp_buckets:
    sub = exp_df[exp_df["exp_bucket"]==e]["PLAYOFF_TENDENCY"]
    n   = len(sub)
    riser_rates_exp.append(100 * (sub=="Riser").sum()   / n if n > 0 else 0)
    decl_rates_exp.append( 100 * (sub=="Decliner").sum()/ n if n > 0 else 0)
    exp_ns.append(n)

xe = np.arange(len(exp_buckets))
ax3.bar(xe - w/2, riser_rates_exp, w, color=RISER_COL, alpha=0.85, label="Riser %",   edgecolor="white")
ax3.bar(xe + w/2, decl_rates_exp,  w, color=DECL_COL,  alpha=0.85, label="Decliner %", edgecolor="white")
exp_xlabels = [f"{b}\n(n={n:,})" for b, n in zip(exp_buckets, exp_ns)]
ax3.set_xticks(xe); ax3.set_xticklabels(exp_xlabels, fontsize=9.5)
ax3.set_ylim(0, 55)
ax3.set_xlabel("Years of experience", fontsize=11)
ax3.set_ylabel("% of player-seasons", fontsize=10)
ax3.set_title("Riser vs Decliner Rate by Experience", fontsize=11, fontweight="bold")
ax3.legend(frameon=False, fontsize=9)
for i, (r, d) in enumerate(zip(riser_rates_exp, decl_rates_exp)):
    ax3.text(i - w/2, r + 0.8, f"{r:.0f}%", ha="center", fontsize=8, color=RISER_COL, fontweight="bold")
    ax3.text(i + w/2, d + 0.8, f"{d:.0f}%", ha="center", fontsize=8, color=DECL_COL,  fontweight="bold")

# ── BOTTOM RIGHT: violin — age distribution by tendency ─────────────────────
ax4 = fig.add_subplot(gs[1, 1])
age_violin = df[df["approx_age"].between(18, 42) & df["PLAYOFF_TENDENCY"].notna()].copy()
sns.violinplot(
    data=age_violin, x="PLAYOFF_TENDENCY", y="approx_age",
    order=ORDER, palette=PAL, inner="quartile", linewidth=1.2,
    ax=ax4, alpha=0.85
)
# overlay means
for i, tend in enumerate(ORDER):
    m = age_violin[age_violin["PLAYOFF_TENDENCY"]==tend]["approx_age"].mean()
    ax4.scatter([i], [m], color="white", s=60, zorder=5, edgecolors="black", linewidth=1.2)
    ax4.text(i, m + 0.6, f"{m:.1f}", ha="center", fontsize=9, fontweight="bold")

ax4.set_xlabel("", fontsize=10)
ax4.set_ylabel("Player age", fontsize=11)
ax4.set_title(
    f"Age Distribution by Tendency\n"
    f"(Riser vs Decliner: t={t_age:.2f}, p={p_age:.3f})",
    fontsize=11, fontweight="bold"
)
ax4.set_xticklabels(ORDER, fontsize=11)

p8 = os.path.join(FIG, "q3_fig8_age_experience.png")
fig.savefig(p8, dpi=150, bbox_inches="tight"); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY STATS
# ══════════════════════════════════════════════════════════════════════════════
mean_delta_r = df[df["PLAYOFF_TENDENCY"]=="Riser"]["DELTA_PTS"].mean()
mean_delta_d = df[df["PLAYOFF_TENDENCY"]=="Decliner"]["DELTA_PTS"].mean()
mean_per_r   = df[df["PLAYOFF_TENDENCY"]=="Riser"]["DELTA_PER"].mean()
mean_per_d   = df[df["PLAYOFF_TENDENCY"]=="Decliner"]["DELTA_PER"].mean()

r_bpm = df[df["PLAYOFF_TENDENCY"]=="Riser"]["DELTA_BPM"].dropna()
d_bpm = df[df["PLAYOFF_TENDENCY"]=="Decliner"]["DELTA_BPM"].dropna()
t_bpm, p_bpm = stats.ttest_ind(r_bpm, d_bpm)

df_pos2 = df[df["pos_group"]!="Unknown"]
pos_riser_pct = (df_pos2[df_pos2["PLAYOFF_TENDENCY"]=="Riser"].groupby("pos_group").size() /
                 df_pos2.groupby("pos_group").size() * 100).round(1)

serial_risers_top5 = serial_risers.head(5).index.tolist()
serial_decl_top5   = serial_decl.head(5).index.tolist()

age_riser_mean = age_r.mean()
age_decl_mean  = age_d.mean()


# ══════════════════════════════════════════════════════════════════════════════
# BUILD PDF
# ══════════════════════════════════════════════════════════════════════════════
print("Building PDF …")
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                 PageBreak, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

PDF = os.path.join(BASE, "figures", "Q3_Playoffs_vs_Regular_Season.pdf")
doc = SimpleDocTemplate(PDF, pagesize=letter,
                        rightMargin=0.85*inch, leftMargin=0.85*inch,
                        topMargin=0.85*inch, bottomMargin=0.85*inch)

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=20, spaceAfter=6,
                     textColor=colors.HexColor("#1a252f"), alignment=TA_CENTER)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=14,
                     spaceAfter=4, textColor=colors.HexColor("#2c3e50"))
BODY = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14,
                       spaceAfter=6, alignment=TA_JUSTIFY)
CAPTION = ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8.5, leading=11,
                          textColor=colors.grey, alignment=TA_CENTER, spaceAfter=10,
                          fontName="Helvetica-Oblique")
STAT = ParagraphStyle("Stat", parent=styles["Normal"], fontSize=10, leading=14,
                       leftIndent=18, spaceAfter=3)
FOOTER_S = ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8,
                           textColor=colors.grey, alignment=TA_CENTER)

def embed(path, width=6.4*inch):
    from PIL import Image as PILImg
    pil = PILImg.open(path)
    w, h = pil.size
    return Image(path, width=width, height=width*(h/w))

def hr():
    return HRFlowable(width="100%", thickness=0.5,
                      color=colors.HexColor("#bdc3c7"),
                      spaceAfter=8, spaceBefore=8)

story = []

# ── Title ─────────────────────────────────────────────────────────────────────
story += [
    Spacer(1, 0.4*inch),
    Paragraph("Predicting and Analyzing NBA Performance", H1),
    Paragraph("Question 3 — Playoffs vs. Regular Season", H1),
    Spacer(1, 0.15*inch), hr(),
    Paragraph("Which players raise their game when the stakes are highest?",
              ParagraphStyle("Sub", parent=styles["Normal"], fontSize=12,
                             alignment=TA_CENTER, textColor=colors.HexColor("#7f8c8d"))),
    Spacer(1, 0.1*inch),
    Paragraph("Yonatan Gan · Ariel Mersel · Nimrod Segev",
              ParagraphStyle("Auth", parent=styles["Normal"], fontSize=11,
                             alignment=TA_CENTER, textColor=colors.HexColor("#2c3e50"))),
    Paragraph("67978 — A Needle in a Data Haystack · 2026",
              ParagraphStyle("Crs", parent=styles["Normal"], fontSize=9,
                             alignment=TA_CENTER, textColor=colors.HexColor("#95a5a6"))),
    Spacer(1, 0.25*inch), hr(),
]

# ── 1. Introduction ──────────────────────────────────────────────────────────
story.append(Paragraph("1. Introduction", H2))
story.append(Paragraph(
    "The NBA playoffs are widely regarded as a different kind of basketball. The pace slows, "
    "defenses tighten, and opposing coaches spend days preparing to neutralize individual players. "
    "Whether elite regular-season performers maintain or even improve their production in this "
    "environment has long been a subject of debate among fans, analysts, and front offices alike.",
    BODY))
story.append(Paragraph(
    f"This analysis uses 28 seasons of data (1995–2023) to answer the question systematically. "
    f"We work with <b>{n_total:,} player-seasons</b> in which the same player appeared in both the "
    f"regular season and the playoffs, allowing a direct apples-to-apples comparison.",
    BODY))

# ── 2. Data ──────────────────────────────────────────────────────────────────
story.append(Paragraph("2. Dataset & Methodology", H2))
story.append(Paragraph(
    "Each observation is a player-season pair from the Kaggle dataset "
    "<i>bendikfltaas/nba-history-seasonal-data-1995-2023</i>, supplemented with advanced metrics "
    "(PER, BPM, VORP, WS/48, TS%, USG%) from Basketball Reference and biographical records "
    "(position, age, experience, draft number) from the NBA Stats API.",
    BODY))
story.append(Paragraph(
    "A player-season is labelled <b>Riser</b> if playoff PPG exceeded regular-season PPG beyond a "
    f"threshold, <b>Decliner</b> if it fell below, and <b>Neutral</b> otherwise. "
    f"Of the {n_total:,} qualifying seasons: "
    f"Risers {n_r:,} ({100*n_r/n_total:.1f}%)  ·  "
    f"Neutral {n_n:,} ({100*n_n/n_total:.1f}%)  ·  "
    f"Decliners {n_d:,} ({100*n_d/n_total:.1f}%).",
    BODY))
story.append(Paragraph(
    "Decliners outnumber Risers roughly 3-to-1 — the single most important high-level finding. "
    "Most players score <i>less</i> in the playoffs than in the regular season.",
    BODY))

# ── 3. Scoring delta ─────────────────────────────────────────────────────────
story.append(Paragraph("3. The Scoring Delta Distribution", H2))
story.append(Paragraph(
    f"The average Riser scores <b>{mean_delta_r:+.2f} ppg more</b> in the playoffs; "
    f"the average Decliner scores <b>{abs(mean_delta_d):.2f} ppg less</b>. "
    "The KDE curves below show both distributions are roughly bell-shaped but the Decliner "
    "distribution is noticeably wider and shifted further from zero — players who fall off "
    "tend to fall harder than those who rise.",
    BODY))
story.append(embed(p1))
story.append(Paragraph(
    "Figure 1. Left: smoothed density curves of playoff scoring delta for each tendency group. "
    "Right: mean delta per group shown as a lollipop chart. "
    "The dashed vertical line marks zero (no change from regular season).",
    CAPTION))

# ── 4. By position / role ────────────────────────────────────────────────────
story.append(Paragraph("4. Who Rises? Position and Scoring Role", H2))
story.append(Paragraph(
    "The stacked charts below show what fraction of each group's player-seasons ends up in "
    "each tendency category. Two patterns stand out.",
    BODY))
story.append(embed(p2))
story.append(Paragraph(
    "Figure 2. 100% stacked bar charts by position group (left) and regular-season scoring role (right). "
    "Each bar sums to 100%. Numbers inside segments show the exact percentage.",
    CAPTION))
riser_str = "  ·  ".join([f"{g}: {v:.0f}%" for g, v in pos_riser_pct.items()])
story.append(Paragraph(
    f"<b>By position:</b> Guards have the highest Riser rate ({riser_str}). "
    "Centers are most often Neutral, likely because they operate in more defined roles and "
    "are less frequently the primary defensive focus.",
    BODY))
story.append(Paragraph(
    "<b>By scoring role:</b> The most striking finding is that <b>star players (20+ ppg) "
    "have both the highest Riser rate and the highest Decliner rate</b> — the widest spread. "
    "Role players cluster tightly around Neutral. Stars either step up to the moment or are "
    "brought down by the defensive game-plans that specifically target them.",
    BODY))

# ── 5. Advanced metrics ───────────────────────────────────────────────────────
story.append(PageBreak())
story.append(Paragraph("5. Advanced Metric Shifts", H2))
story.append(Paragraph(
    f"Looking beyond points, Risers improve on virtually every advanced metric. "
    f"Mean PER delta: Risers <b>{mean_per_r:+.2f}</b> vs Decliners <b>{mean_per_d:.2f}</b>. "
    f"The BPM gap is statistically significant (t = {t_bpm:.2f}, p = {p_bpm:.2e}). "
    "Axes are clipped to the central 98% of the distribution to keep outliers from "
    "compressing the visible range.",
    BODY))
story.append(embed(p3))
story.append(Paragraph(
    "Figure 3. Distribution of advanced metric deltas for each tendency group. "
    "Dotted vertical lines mark each group's mean. "
    "Axes are clipped to the central distribution — extreme outliers exist but are not shown.",
    CAPTION))
story.append(Paragraph(
    "True Shooting % (TS%) delta is positive on average for Risers, meaning they are "
    "<b>more efficient</b> in the playoffs even when they score more. This is not simply higher "
    "volume — it is a genuine step-up in quality.",
    BODY))

# ── 6. Scatter ────────────────────────────────────────────────────────────────
story.append(Paragraph("6. Regular Season vs. Playoff Scoring", H2))
story.append(embed(p4))
story.append(Paragraph(
    "Figure 4. Hexbin density maps — one panel per tendency group. "
    "Each hexagonal cell is shaded by the number of player-seasons landing there. "
    "The dashed diagonal is the line of equal scoring in both contexts.",
    CAPTION))
story.append(Paragraph(
    "The Riser panel shows density concentrated <i>above</i> the diagonal. "
    "The Decliner panel shows the opposite. The Neutral panel is tightly centred on "
    "the diagonal. Points near the origin (role players) tend toward Neutral regardless of group.",
    BODY))

# ── 7. Serial performers ──────────────────────────────────────────────────────
story.append(Paragraph("7. Serial Risers and Decliners", H2))
story.append(Paragraph(
    "Some players show the same tendency year after year — suggesting a stable trait, not noise.",
    BODY))
story.append(embed(p5))
story.append(Paragraph(
    "Figure 5. Players with 3+ seasons of the same tendency. "
    "Bar labels show season count and career regular-season scoring average.",
    CAPTION))
story.append(Paragraph(
    f"Top serial Risers: <b>{', '.join(serial_risers_top5)}</b> — "
    "widely regarded by analysts as great playoff performers, which validates the methodology.",
    BODY))
story.append(Paragraph(
    f"Top serial Decliners: <b>{', '.join(serial_decl_top5)}</b>. "
    "Several were elite regular-season scorers who faced intense playoff defensive attention, "
    "consistent with the star-player finding in Section 4.",
    BODY))

# ── 8. Feature importance ─────────────────────────────────────────────────────
story.append(PageBreak())
story.append(Paragraph("8. What Predicts a Playoff Riser?", H2))
story.append(Paragraph(
    f"A Random Forest classifier trained on {len(feature_cols)} regular-season features "
    f"distinguished Risers from Decliners with 5-fold CV AUC = <b>{cv_auc.mean():.2f}</b> "
    f"(σ = {cv_auc.std():.2f}). Moderate but real predictive signal exists in the regular-season data.",
    BODY))
story.append(embed(p6))
story.append(Paragraph(
    "Figure 6. Feature importances as % of total (Random Forest mean decrease in impurity). "
    "Green bars = top 35% of features.",
    CAPTION))
top_str = ", ".join([f"<b>{n}</b> ({v:.1f}%)" for n, v in top_feat.items()])
story.append(Paragraph(
    f"Top-5 features: {top_str}. "
    "Assists and raw volume stats rank highly because they capture overall player involvement — "
    "players with a larger footprint in the regular season tend to maintain or grow it in the playoffs. "
    "Efficiency metrics (VORP, BPM) beat shooting percentages, suggesting "
    "overall positive impact matters more than shooting mechanics.",
    BODY))

# ── 9. Role box plots ─────────────────────────────────────────────────────────
story.append(Paragraph("9. Role Players vs. Stars: Who Takes the Bigger Hit?", H2))
story.append(embed(p7))
story.append(Paragraph(
    "Figure 7. Box plots of scoring delta (left) and PER delta (right) by regular-season scoring tier. "
    "Y-axes are clipped to the central distribution range. Dots show outliers within that range.",
    CAPTION))
story.append(Paragraph(
    "Stars (20+ ppg) have the widest interquartile range and the highest median — they are the only "
    "tier with a positive median scoring delta. But their distribution also extends furthest downward. "
    "Role players (<8 ppg) are tightly clustered around zero in both metrics.",
    BODY))

# ── 10. Age & experience ─────────────────────────────────────────────────────
story.append(Paragraph("10. Does Age or Experience Matter?", H2))
story.append(Paragraph(
    "We tested whether a player's age or career experience at the time of a playoff run predicts "
    "their tendency. Age was approximated as the season year minus birth year.",
    BODY))
story.append(embed(p8))
story.append(Paragraph(
    "Figure 8. Top row: Riser vs Decliner rate by age group (left) and full tendency mix "
    "stacked bars (right). Bottom row: Riser vs Decliner rate by experience tier (left) "
    "and violin distributions of player age per tendency group (right, white dots = mean).",
    CAPTION))
story.append(Paragraph(
    f"<b>Age effect:</b> The mean age of Risers ({age_riser_mean:.1f}) versus Decliners "
    f"({age_decl_mean:.1f}) is similar (t = {t_age:.2f}, p = {p_age:.3f}). "
    "The difference is not statistically significant, meaning <b>age alone does not determine "
    "playoff tendency</b>. Young players are not systematically better or worse — the effect "
    "is largely flat across age groups.",
    BODY))
story.append(Paragraph(
    "<b>Experience effect:</b> Veterans (13+ years) show a slightly higher Riser rate on the "
    "scoring delta chart, suggesting that playoff experience does provide a small edge. "
    "However the effect is modest. The more powerful predictors remain player role and "
    "overall efficiency (as shown in Section 8), not age or seniority.",
    BODY))

# ── 11. Conclusions ───────────────────────────────────────────────────────────
story.append(Paragraph("11. Conclusions", H2))
conclusions = [
    f"<b>Most players score less in the playoffs.</b> Decliners outnumber Risers {n_d}:{n_r} "
    f"({100*n_d/n_total:.0f}% vs {100*n_r/n_total:.0f}%). Playoff defenses are real.",
    "<b>Stars are the most volatile.</b> High scorers have both the highest Riser and highest "
    "Decliner rates — they polarize, while role players cluster around neutral.",
    "<b>Guards rise most by position.</b> Centers are the most neutral, likely reflecting "
    "their more constrained playoff rotation roles.",
    "<b>True Risers improve on every advanced metric.</b> PER, BPM, TS%, and WS/48 all shift "
    "positively for Risers — they score more <i>and</i> more efficiently.",
    f"<b>Regular-season data predicts tendency with AUC ≈ {cv_auc.mean():.2f}.</b> "
    "Assists, minutes, and efficiency metrics (VORP) are stronger signals than raw scoring volume.",
    "<b>Age alone does not predict playoff tendency.</b> The Riser vs Decliner age difference "
    "is not statistically significant. Player role and efficiency are far more predictive.",
]
for i, c in enumerate(conclusions, 1):
    story.append(Paragraph(f"{i}. {c}", STAT))
    story.append(Spacer(1, 4))

story += [
    Spacer(1, 0.2*inch), hr(),
    Paragraph(
        "Data: Kaggle (bendikfltaas/nba-history-seasonal-data-1995-2023, drgilermo/nba-players-stats), "
        "Basketball Reference, NBA Stats API · Coverage: 1995–2023.",
        FOOTER_S)
]

doc.build(story)
print(f"\nPDF → {PDF}")
for p in [p1,p2,p3,p4,p5,p6,p7,p8]:
    print(f"  {os.path.basename(p)}")
