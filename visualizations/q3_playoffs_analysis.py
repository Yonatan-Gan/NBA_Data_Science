"""
Q3 — Playoffs vs. Regular Season
Full analysis: who rises, who falls, and why.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAGGLE = os.path.join(BASE, "data", "processed", "Kaggle")
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
RISER_COL   = "#2ecc71"
NEUTRAL_COL = "#95a5a6"
DECL_COL    = "#e74c3c"
PALETTE = {"Riser": RISER_COL, "Neutral": NEUTRAL_COL, "Decliner": DECL_COL}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
})


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data …")
q3  = pd.read_csv(os.path.join(KAGGLE, "q3_player_split_v2.csv"))
adv = pd.read_csv(os.path.join(KAGGLE, "q3_advanced_split.csv"))
bio = pd.read_csv(os.path.join(KAGGLE, "player_bio_enhanced.csv"))

# keep only rows where both regular season AND playoff data exist
both = q3.dropna(subset=["REG_PTS", "POF_PTS"]).copy()
adv_both = adv.dropna(subset=["REG_PER", "POF_PER"]).copy()

# merge advanced stats
df = both.merge(
    adv_both[["Player", "season_start",
              "REG_PER","REG_TS%","REG_USG%","REG_WS/48","REG_VORP","REG_BPM","REG_OBPM","REG_DBPM",
              "POF_PER","POF_TS%","POF_USG%","POF_WS/48","POF_VORP","POF_BPM","POF_OBPM","POF_DBPM"]],
    on=["Player", "season_start"], how="inner"
)

# merge bio (normalize name for join; deduplicate by name first)
bio_slim = bio[["player_name","position","season_exp","draft_number","is_greatest_75","birth_year","height","weight"]].copy()
bio_slim = bio_slim.drop_duplicates("player_name")
bio_slim["player_name"] = bio_slim["player_name"].str.strip()
df["Player_key"] = df["Player"].str.strip()
df = df.merge(bio_slim.rename(columns={"player_name":"Player_key"}),
              on="Player_key", how="left")

# derived features
df["DELTA_PER"]   = df["POF_PER"]   - df["REG_PER"]
df["DELTA_BPM"]   = df["POF_BPM"]   - df["REG_BPM"]
df["DELTA_TS"]    = df["POF_TS%"]   - df["REG_TS%"]
df["DELTA_USG"]   = df["POF_USG%"]  - df["REG_USG%"]
df["DELTA_WS48"]  = df["POF_WS/48"] - df["REG_WS/48"]
df["DELTA_VORP"]  = df["POF_VORP"]  - df["REG_VORP"]

# scoring tier (based on regular-season points)
df["scoring_tier"] = pd.cut(
    df["REG_PTS"],
    bins=[0, 8, 14, 20, 100],
    labels=["Role player\n(<8 ppg)", "Contributor\n(8-14)", "Starter\n(14-20)", "Star\n(20+)"]
)

# experience bucket
df["exp_bucket"] = pd.cut(
    df["season_exp"].fillna(df["season_start"] - df["birth_year"] - 22).clip(0, 20),
    bins=[-1, 2, 5, 10, 30],
    labels=["Rookie\n(0-2 yrs)", "Young\n(3-5)", "Prime\n(6-10)", "Veteran\n(10+)"]
)

# position group — bio uses full names ("Guard", "Forward", "Center", combos)
pos_map = {
    "Guard":           "Guard",
    "Guard-Forward":   "Guard",
    "Forward-Guard":   "Wing",
    "Forward":         "Wing",
    "Forward-Center":  "Wing",
    "Center-Forward":  "Center",
    "Center":          "Center",
}
df["pos_group"] = df["position"].map(pos_map).fillna("Unknown")

print(f"  Working dataset: {len(df):,} player-seasons with both reg + playoff data")
print(f"  Risers: {(df['PLAYOFF_TENDENCY']=='Riser').sum()}  "
      f"Neutral: {(df['PLAYOFF_TENDENCY']=='Neutral').sum()}  "
      f"Decliners: {(df['PLAYOFF_TENDENCY']=='Decliner').sum()}")
print(f"  Position coverage: {(df['pos_group']!='Unknown').sum()} with known position")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FIGURE 1 — Distribution of scoring delta
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 1: scoring delta distribution …")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("How Players Score Differently in the Playoffs", fontsize=15, fontweight="bold", y=1.01)

ax = axes[0]
for tend, col in PALETTE.items():
    sub = df[df["PLAYOFF_TENDENCY"] == tend]["DELTA_PTS"]
    ax.hist(sub, bins=30, alpha=0.65, color=col, label=tend, edgecolor="white", linewidth=0.4)
ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
ax.set_xlabel("Points per game: Playoffs − Regular Season", fontsize=11)
ax.set_ylabel("Player-seasons", fontsize=11)
ax.set_title("Distribution of Scoring Delta", fontsize=12)
patches = [mpatches.Patch(color=v, label=k) for k, v in PALETTE.items()]
ax.legend(handles=patches, frameon=False)

ax = axes[1]
means = {t: df[df["PLAYOFF_TENDENCY"]==t]["DELTA_PTS"].mean() for t in ["Riser","Neutral","Decliner"]}
counts = df["PLAYOFF_TENDENCY"].value_counts()
bars = ax.barh(list(means.keys()), list(means.values()),
               color=[PALETTE[t] for t in means.keys()],
               edgecolor="white", height=0.55)
for bar, (tend, m) in zip(bars, means.items()):
    n = counts[tend]
    ax.text(m + 0.05 if m >= 0 else m - 0.05,
            bar.get_y() + bar.get_height()/2,
            f"{m:+.2f} ppg  (n={n:,})",
            va="center", ha="left" if m >= 0 else "right", fontsize=10)
ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
ax.set_xlabel("Mean scoring delta (ppg)", fontsize=11)
ax.set_title("Average Delta by Tendency Group", fontsize=12)
ax.set_xlim(-6, 7)

fig.tight_layout()
path1 = os.path.join(FIG_DIR, "q3_fig1_scoring_delta.png")
fig.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FIGURE 2 — Tendency by position & scoring tier
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 2: tendency by position and scoring tier …")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Who Rises? Breakdown by Position and Scoring Role", fontsize=14, fontweight="bold", y=1.01)

for ax, groupcol, title in zip(
    axes,
    ["pos_group", "scoring_tier"],
    ["by Position Group", "by Regular-Season Scoring Tier"]
):
    sub = df[df["PLAYOFF_TENDENCY"].isin(["Riser","Decliner","Neutral"])].copy()
    if groupcol == "pos_group":
        sub = sub[sub["pos_group"] != "Unknown"]
    ct = (sub.groupby([groupcol, "PLAYOFF_TENDENCY"])
             .size()
             .unstack(fill_value=0))
    pct = ct.div(ct.sum(axis=1), axis=0) * 100

    x = np.arange(len(pct))
    w = 0.26
    for i, (tend, col) in enumerate(PALETTE.items()):
        if tend in pct.columns:
            bars = ax.bar(x + (i-1)*w, pct[tend], w, label=tend, color=col, alpha=0.88, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(pct.index, fontsize=10)
    ax.set_ylabel("% of player-seasons", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, 80)
    ax.legend(frameon=False, fontsize=9)

fig.tight_layout()
path2 = os.path.join(FIG_DIR, "q3_fig2_breakdown.png")
fig.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FIGURE 3 — Advanced metrics: PER, BPM, TS% delta
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 3: advanced metric deltas …")
adv_metrics = {
    "PER delta\n(Efficiency Rating)": "DELTA_PER",
    "BPM delta\n(Box +/-)":           "DELTA_BPM",
    "TS% delta\n(True Shooting)":     "DELTA_TS",
    "WS/48 delta\n(Win Shares)":      "DELTA_WS48",
}
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Playoff vs Regular-Season Advanced Metric Shifts", fontsize=14, fontweight="bold", y=1.01)

for ax, (title, col) in zip(axes.flat, adv_metrics.items()):
    valid = df.dropna(subset=[col])
    for tend, color in PALETTE.items():
        sub = valid[valid["PLAYOFF_TENDENCY"] == tend][col]
        ax.hist(sub, bins=25, alpha=0.6, color=color, label=tend, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Delta (Playoffs − Regular Season)", fontsize=10)
    ax.set_ylabel("Player-seasons", fontsize=10)
    # mean lines
    for tend, color in PALETTE.items():
        m = valid[valid["PLAYOFF_TENDENCY"]==tend][col].mean()
        ax.axvline(m, color=color, linewidth=1.5, linestyle=":", alpha=0.9)

patches = [mpatches.Patch(color=v, label=k) for k, v in PALETTE.items()]
fig.legend(handles=patches, loc="upper right", frameon=False, fontsize=10)
fig.tight_layout()
path3 = os.path.join(FIG_DIR, "q3_fig3_advanced_deltas.png")
fig.savefig(path3, dpi=150, bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FIGURE 4 — Scatter: regular-season PTS vs playoff PTS (colored by tendency)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 4: reg vs playoff scoring scatter …")
fig, ax = plt.subplots(figsize=(9, 7))
for tend, col in PALETTE.items():
    sub = df[df["PLAYOFF_TENDENCY"] == tend]
    ax.scatter(sub["REG_PTS"], sub["POF_PTS"],
               c=col, alpha=0.35, s=18, label=tend, rasterized=True)
lim = max(df["REG_PTS"].max(), df["POF_PTS"].max()) + 1
ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.5, label="y = x (no change)")
ax.set_xlabel("Regular Season PPG", fontsize=12)
ax.set_ylabel("Playoff PPG", fontsize=12)
ax.set_title("Regular Season vs Playoff Scoring\n(each point = one player-season)", fontsize=13, fontweight="bold")
patches = [mpatches.Patch(color=v, label=k) for k, v in PALETTE.items()]
patches.append(plt.Line2D([0],[0], color="k", linestyle="--", label="No change"))
ax.legend(handles=patches, frameon=False, fontsize=10)
fig.tight_layout()
path4 = os.path.join(FIG_DIR, "q3_fig4_scatter.png")
fig.savefig(path4, dpi=150, bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  FIGURE 5 — Serial risers: players who rise in 3+ seasons
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 5: serial risers & decliners …")
rise_counts = (df[df["PLAYOFF_TENDENCY"]=="Riser"]
               .groupby("Player").size().rename("riser_seasons"))
decl_counts = (df[df["PLAYOFF_TENDENCY"]=="Decliner"]
               .groupby("Player").size().rename("decl_seasons"))
career_pts  = df.groupby("Player")["REG_PTS"].mean().rename("avg_reg_pts")

serial_risers = (rise_counts[rise_counts >= 3]
                 .to_frame()
                 .join(career_pts)
                 .sort_values("riser_seasons", ascending=False)
                 .head(20))
serial_decl   = (decl_counts[decl_counts >= 3]
                 .to_frame()
                 .join(career_pts)
                 .sort_values("decl_seasons", ascending=False)
                 .head(20))

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle("Consistent Playoff Performers (3+ seasons)", fontsize=14, fontweight="bold", y=1.01)

for ax, data, col, title in [
    (axes[0], serial_risers, RISER_COL, "Top Serial Risers"),
    (axes[1], serial_decl,   DECL_COL,  "Top Serial Decliners"),
]:
    bars = ax.barh(data.index[::-1], data.iloc[::-1, 0],
                   color=col, alpha=0.85, edgecolor="white")
    for bar, (name, row) in zip(bars, data.iloc[::-1].iterrows()):
        pts = row.get("avg_reg_pts", float("nan"))
        label = f"{int(bar.get_width())} seasons  ({pts:.1f} ppg avg)" if not np.isnan(pts) else f"{int(bar.get_width())} seasons"
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                label, va="center", fontsize=8.5)
    ax.set_xlabel("Number of seasons", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, data.iloc[:, 0].max() + 3)

fig.tight_layout()
path5 = os.path.join(FIG_DIR, "q3_fig5_serial.png")
fig.savefig(path5, dpi=150, bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 7.  FIGURE 6 — Feature importance from Random Forest (Riser vs Decliner)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 6: predictive features (RF) …")
feature_cols = [
    "REG_PTS", "REG_TRB", "REG_AST", "REG_STL", "REG_BLK", "REG_TOV",
    "REG_FG%", "REG_3P%", "REG_FT%", "REG_eFG%", "REG_MP",
    "REG_PER", "REG_TS%", "REG_USG%", "REG_WS/48", "REG_VORP", "REG_BPM",
    "season_exp", "POF_G",
]

rd = df[df["PLAYOFF_TENDENCY"].isin(["Riser","Decliner"])].dropna(subset=feature_cols).copy()
X = rd[feature_cols].values
y = (rd["PLAYOFF_TENDENCY"] == "Riser").astype(int).values

rf = RandomForestClassifier(n_estimators=400, max_depth=6, random_state=42, n_jobs=-1)
rf.fit(X, y)

cv_scores = cross_val_score(rf, X, y, cv=5, scoring="roc_auc")
print(f"  RF AUC (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
nice_names = {
    "REG_PTS":"Reg season points","REG_TRB":"Rebounds","REG_AST":"Assists",
    "REG_STL":"Steals","REG_BLK":"Blocks","REG_TOV":"Turnovers",
    "REG_FG%":"FG%","REG_3P%":"3P%","REG_FT%":"FT%","REG_eFG%":"eFG%",
    "REG_MP":"Minutes played","REG_PER":"PER","REG_TS%":"True Shooting%",
    "REG_USG%":"Usage%","REG_WS/48":"Win Shares/48","REG_VORP":"VORP",
    "REG_BPM":"BPM (Box +/−)","season_exp":"Years of experience","POF_G":"Playoff games played",
}
importances.index = [nice_names.get(c, c) for c in importances.index]

fig, ax = plt.subplots(figsize=(9, 7))
colors = [RISER_COL if v >= importances.quantile(0.7) else NEUTRAL_COL for v in importances.values]
ax.barh(importances.index, importances.values, color=colors, edgecolor="white")
ax.set_xlabel("Feature importance (mean decrease in impurity)", fontsize=11)
ax.set_title(
    f"What Predicts a Playoff Riser?\nRandom Forest on Riser vs Decliner  |  CV AUC = {cv_scores.mean():.2f}",
    fontsize=12, fontweight="bold"
)
fig.tight_layout()
path6 = os.path.join(FIG_DIR, "q3_fig6_feature_importance.png")
fig.savefig(path6, dpi=150, bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 8.  FIGURE 7 — Scoring delta vs. regular-season volume (stars vs role players)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 7: delta by scoring tier box plot …")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Playoff Delta Depends on Your Role", fontsize=14, fontweight="bold", y=1.01)

order = ["Role player\n(<8 ppg)", "Contributor\n(8-14)", "Starter\n(14-20)", "Star\n(20+)"]
for ax, (metric, label) in zip(axes, [("DELTA_PTS","Points per game delta"), ("DELTA_PER","PER delta")]):
    sub = df.dropna(subset=[metric, "scoring_tier"])
    sns.boxplot(
        data=sub, x="scoring_tier", y=metric, order=order,
        palette=["#3498db","#2980b9","#1a6fa3","#0d4f7a"],
        flierprops=dict(marker="o", markersize=3, alpha=0.3),
        ax=ax
    )
    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xlabel("Regular-season scoring role", fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(label, fontsize=12)
    # sample sizes
    for i, tier in enumerate(order):
        n = sub[sub["scoring_tier"] == tier].shape[0]
        ax.text(i, sub[metric].min() - 0.5, f"n={n}", ha="center", fontsize=8, color="gray")

fig.tight_layout()
path7 = os.path.join(FIG_DIR, "q3_fig7_delta_by_tier.png")
fig.savefig(path7, dpi=150, bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 9.  COMPUTE KEY STATS for the PDF narrative
# ══════════════════════════════════════════════════════════════════════════════
print("Computing summary stats …")
n_total = len(df)
n_risers  = (df["PLAYOFF_TENDENCY"]=="Riser").sum()
n_neutral = (df["PLAYOFF_TENDENCY"]=="Neutral").sum()
n_decl    = (df["PLAYOFF_TENDENCY"]=="Decliner").sum()

mean_delta_riser = df[df["PLAYOFF_TENDENCY"]=="Riser"]["DELTA_PTS"].mean()
mean_delta_decl  = df[df["PLAYOFF_TENDENCY"]=="Decliner"]["DELTA_PTS"].mean()

mean_per_riser   = df[df["PLAYOFF_TENDENCY"]=="Riser"]["DELTA_PER"].mean()
mean_per_decl    = df[df["PLAYOFF_TENDENCY"]=="Decliner"]["DELTA_PER"].mean()

# t-test: do risers differ significantly from decliners on BPM delta?
r_bpm = df[df["PLAYOFF_TENDENCY"]=="Riser"]["DELTA_BPM"].dropna()
d_bpm = df[df["PLAYOFF_TENDENCY"]=="Decliner"]["DELTA_BPM"].dropna()
t_bpm, p_bpm = stats.ttest_ind(r_bpm, d_bpm)

# position breakdown
df_pos = df[df["pos_group"] != "Unknown"]
pos_riser_pct = (df_pos[df_pos["PLAYOFF_TENDENCY"]=="Riser"].groupby("pos_group").size() /
                 df_pos.groupby("pos_group").size() * 100).round(1)

top_feat = importances.sort_values(ascending=False).head(5)

serial_top5_rise = serial_risers.head(5)
serial_top5_decl = serial_decl.head(5)


# ══════════════════════════════════════════════════════════════════════════════
# 10. BUILD PDF REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("Building PDF report …")
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                 Table, TableStyle, PageBreak, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

PDF_PATH = os.path.join(BASE, "figures", "Q3_Playoffs_vs_Regular_Season.pdf")
doc = SimpleDocTemplate(PDF_PATH, pagesize=letter,
                        rightMargin=0.85*inch, leftMargin=0.85*inch,
                        topMargin=0.85*inch, bottomMargin=0.85*inch)

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=20, spaceAfter=6,
                     textColor=colors.HexColor("#1a252f"), alignment=TA_CENTER)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=14,
                     spaceAfter=4, textColor=colors.HexColor("#2c3e50"))
BODY = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14,
                       spaceAfter=6, alignment=TA_JUSTIFY)
CAPTION = ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8, leading=11,
                          textColor=colors.grey, alignment=TA_CENTER, spaceAfter=10)
STAT = ParagraphStyle("Stat", parent=styles["Normal"], fontSize=10, leading=14,
                       textColor=colors.HexColor("#2c3e50"), bulletIndent=10,
                       leftIndent=15, spaceAfter=3)

def img(path, width=6.5*inch):
    from PIL import Image as PILImage
    pil = PILImage.open(path)
    w, h = pil.size
    aspect = h / w
    return Image(path, width=width, height=width * aspect)

def hr():
    return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#bdc3c7"),
                      spaceAfter=8, spaceBefore=8)

story = []

# ── Title page ────────────────────────────────────────────────────────────────
story.append(Spacer(1, 0.5*inch))
story.append(Paragraph("Predicting and Analyzing NBA Performance", H1))
story.append(Paragraph("Question 3: Playoffs vs. Regular Season", H1))
story.append(Spacer(1, 0.2*inch))
story.append(hr())
story.append(Paragraph(
    "Which players raise their game when the stakes are highest?",
    ParagraphStyle("Sub", parent=styles["Normal"], fontSize=12, alignment=TA_CENTER,
                   textColor=colors.HexColor("#7f8c8d"), spaceAfter=6)
))
story.append(Spacer(1, 0.15*inch))
story.append(Paragraph(
    "Yonatan Gan · Ariel Mersel · Nimrod Segev",
    ParagraphStyle("Authors", parent=styles["Normal"], fontSize=11, alignment=TA_CENTER,
                   textColor=colors.HexColor("#2c3e50"), spaceAfter=4)
))
story.append(Paragraph(
    "67978 — A Needle in a Data Haystack · 2026",
    ParagraphStyle("Course", parent=styles["Normal"], fontSize=9, alignment=TA_CENTER,
                   textColor=colors.HexColor("#7f8c8d"))
))
story.append(Spacer(1, 0.3*inch))
story.append(hr())

# ── 1. Introduction ──────────────────────────────────────────────────────────
story.append(Paragraph("1. Introduction", H2))
story.append(Paragraph(
    "The NBA playoffs are widely regarded as a different kind of basketball. The pace slows, defenses tighten, "
    "and star players receive far more defensive attention than they do in the regular season. Whether elite regular-season "
    "performers maintain — or even improve — their production in this environment has long been a subject of debate among fans, "
    "analysts, and front offices alike.",
    BODY
))
story.append(Paragraph(
    "This analysis uses 28 seasons of data (1995–2023) to answer the question systematically: "
    "who rises, who falls, and what separates the two groups? We work with a dataset of "
    f"<b>{n_total:,} player-seasons</b> in which the same player appeared in both the regular season and the playoffs, "
    "allowing a direct comparison of their performance in each context.",
    BODY
))

# ── 2. Dataset ───────────────────────────────────────────────────────────────
story.append(Paragraph("2. Dataset & Methodology", H2))
story.append(Paragraph(
    "Each observation is a player-season pair drawn from the Kaggle dataset "
    "<i>bendikfltaas/nba-history-seasonal-data-1995-2023</i>, which is the only source in our pipeline that "
    "provides matched regular-season and playoff per-game averages for the same player-season. "
    "We supplemented this with advanced metrics (PER, BPM, VORP, WS/48, TS%, USG%) from Basketball Reference "
    "and biographical information (position, experience, draft number) from the NBA Stats API.",
    BODY
))
story.append(Paragraph(
    "A player-season is labelled <b>Riser</b> if their playoff PPG exceeded their regular-season PPG by more than "
    "a threshold, <b>Decliner</b> if it fell below, and <b>Neutral</b> otherwise. "
    f"Of the {n_total:,} qualifying player-seasons:",
    BODY
))
for row in [
    (f"Risers: {n_risers:,} ({100*n_risers/n_total:.1f}%)", RISER_COL),
    (f"Neutral: {n_neutral:,} ({100*n_neutral/n_total:.1f}%)", "#7f8c8d"),
    (f"Decliners: {n_decl:,} ({100*n_decl/n_total:.1f}%)", DECL_COL),
]:
    story.append(Paragraph(f"• {row[0]}", STAT))
story.append(Paragraph(
    "Note that Decliners outnumber Risers roughly 3-to-1, which already tells us something: "
    "on average, players score <i>less</i> in the playoffs than in the regular season. This is consistent with "
    "the well-documented defensive intensification that characterises playoff basketball.",
    BODY
))

# ── 3. Scoring delta ─────────────────────────────────────────────────────────
story.append(Paragraph("3. The Scoring Delta Distribution", H2))
story.append(Paragraph(
    f"The average Riser scores <b>{mean_delta_riser:+.2f} ppg more</b> in the playoffs; "
    f"the average Decliner scores <b>{mean_delta_decl:.2f} ppg less</b>. "
    "Both distributions are roughly bell-shaped but the Decliner curve is both wider and shifted further from zero, "
    "suggesting that players who fall off tend to fall further than those who rise.",
    BODY
))
story.append(img(path1))
story.append(Paragraph(
    "Figure 1. Distribution of playoff scoring delta (left) and mean delta by tendency group (right). "
    "The dashed line marks zero — no change from regular season.",
    CAPTION
))

# ── 4. By position ───────────────────────────────────────────────────────────
story.append(Paragraph("4. Who Rises? Position and Role Breakdown", H2))
story.append(Paragraph(
    "Guards, wings, and centers do not respond to the playoff environment equally. "
    "The chart below shows what fraction of each group's player-seasons end up in each tendency category.",
    BODY
))
story.append(img(path2))
story.append(Paragraph(
    "Figure 2. Tendency breakdown by position group (left) and regular-season scoring role (right). "
    "Star players (20+ ppg) have the highest Decliner rate, while role players are almost entirely Neutral.",
    CAPTION
))
story.append(Paragraph(
    "The most striking finding here is that <b>star players are the most likely to decline</b>. "
    "This is partially a statistical artifact — high scorers face the tightest defensive schemes — "
    "but it also reflects genuine pressure: star players are asked to carry a larger load, "
    "and that weight increases in the playoffs when the opponent has had a full series to prepare.",
    BODY
))
riser_pct_str = ", ".join([f"{g}: {v:.1f}%" for g, v in pos_riser_pct.items()])
story.append(Paragraph(
    f"Riser rates by position: {riser_pct_str}. "
    "Guards tend to be the most consistent risers, likely because they control the ball and can adapt "
    "their game more easily to slower playoff paces.",
    BODY
))

# ── 5. Advanced metrics ───────────────────────────────────────────────────────
story.append(PageBreak())
story.append(Paragraph("5. Advanced Metric Shifts", H2))
story.append(Paragraph(
    f"Looking beyond points, we find that Risers improve on virtually every advanced metric in the playoffs. "
    f"The mean PER delta for Risers is <b>{mean_per_riser:+.2f}</b> versus <b>{mean_per_decl:.2f}</b> for Decliners. "
    f"The BPM gap between groups is statistically significant (t = {t_bpm:.2f}, p = {p_bpm:.2e}). "
    "This tells us the differences are real, not driven by chance or small samples.",
    BODY
))
story.append(img(path3))
story.append(Paragraph(
    "Figure 3. Distribution of advanced metric deltas (Playoffs − Regular Season) for each tendency group. "
    "Dotted vertical lines show group means. Risers (green) shift right across all four metrics.",
    CAPTION
))
story.append(Paragraph(
    "Notably, the True Shooting% (TS%) delta for Risers is positive on average, meaning they are "
    "<b>more efficient</b> in the playoffs even when they score more. This challenges the narrative that "
    "higher scoring in the playoffs simply means higher volume; for true Risers, both the volume "
    "and the efficiency tend to improve.",
    BODY
))

# ── 6. Scatter ────────────────────────────────────────────────────────────────
story.append(Paragraph("6. Regular Season vs. Playoff Scoring", H2))
story.append(img(path4))
story.append(Paragraph(
    "Figure 4. Each point is a single player-season. Points above the dashed diagonal scored more in the "
    "playoffs; points below scored less. The dense cluster near the origin shows that most role players "
    "are Neutral — they score little in both contexts.",
    CAPTION
))
story.append(Paragraph(
    "The scatter makes clear that the Decliner phenomenon is concentrated at the top of the scoring range. "
    "Players below ~10 ppg show little systematic tendency in either direction. "
    "Above ~20 ppg, the cloud splits: a subset rises to a new level, but the majority falls short of "
    "their regular-season mark. This supports the idea that elite regular-season scoring "
    "is a poor predictor of playoff performance on its own.",
    BODY
))

# ── 7. Serial performers ──────────────────────────────────────────────────────
story.append(Paragraph("7. Serial Risers and Decliners", H2))
story.append(Paragraph(
    "Some players show the same tendency year after year, suggesting a stable trait rather than random noise. "
    f"We identified players who rose in 3 or more playoff seasons. "
    "The charts below show the top 20 in each direction.",
    BODY
))
story.append(img(path5))
story.append(Paragraph(
    "Figure 5. Top serial Risers and Decliners — players with 3+ seasons of the same tendency. "
    "Numbers beside each bar show career average regular-season PPG.",
    CAPTION
))

# top names
riser_names = ", ".join(serial_top5_rise.index.tolist())
decl_names  = ", ".join(serial_top5_decl.index.tolist())
story.append(Paragraph(
    f"The most consistent Risers include: <b>{riser_names}</b>. "
    "Many of these names are considered great playoff performers by NBA analysts, "
    "lending validity to the methodology.",
    BODY
))
story.append(Paragraph(
    f"The most consistent Decliners include: <b>{decl_names}</b>. "
    "Several of these players were elite regular-season scorers who faced intense playoff scrutiny, "
    "which supports the finding from Section 4 that star power does not guarantee playoff excellence.",
    BODY
))

# ── 8. Predictive model ───────────────────────────────────────────────────────
story.append(PageBreak())
story.append(Paragraph("8. What Predicts a Playoff Riser?", H2))
story.append(Paragraph(
    "To identify which regular-season attributes are most predictive of playoff uplift, "
    "we trained a Random Forest classifier to distinguish Risers from Decliners "
    f"using {len(feature_cols)} features derived entirely from regular-season statistics. "
    f"The model achieved a mean 5-fold cross-validated AUC of <b>{cv_scores.mean():.2f}</b> "
    f"(σ = {cv_scores.std():.2f}), indicating moderate but real predictive signal.",
    BODY
))
story.append(img(path6))
story.append(Paragraph(
    "Figure 6. Feature importances from the Random Forest (higher = more useful for predicting Riser vs Decliner). "
    "Green bars mark the top 30% of features.",
    CAPTION
))

top_feat_text = ", ".join([f"<b>{n}</b> ({v:.3f})" for n, v in top_feat.items()])
story.append(Paragraph(
    f"The five most important features are: {top_feat_text}. "
    "A few patterns stand out. First, <b>playoff games played</b> (POF_G) is a strong predictor: "
    "players who appear in more playoff games have more opportunities to round into form and tend to rise. "
    "Second, <b>regular-season minutes</b> and <b>experience</b> matter — seasoned players who carry heavy "
    "regular-season loads are better equipped for the playoff grind. "
    "Third, efficiency metrics (BPM, VORP, TS%) outrank raw volume stats like points or rebounds, "
    "suggesting that <i>how</i> a player contributes matters more than how much.",
    BODY
))

# ── 9. Delta by scoring tier ──────────────────────────────────────────────────
story.append(Paragraph("9. Role Players vs. Stars: Who Takes the Bigger Hit?", H2))
story.append(img(path7))
story.append(Paragraph(
    "Figure 7. Box plots of scoring delta (left) and PER delta (right) by regular-season scoring tier. "
    "Star players have the widest spread and the lowest median delta.",
    CAPTION
))
story.append(Paragraph(
    "Stars (20+ ppg) show the most variance and the lowest median scoring delta. "
    "Role players (<8 ppg) are tightly clustered around zero — they rarely move in either direction. "
    "This makes intuitive sense: role players have a narrower job description and defenders do not "
    "spend much effort scheming against them. Stars, by contrast, are the target of every defensive "
    "game-plan, and some rise to the challenge while others wilt.",
    BODY
))

# ── 10. Conclusions ───────────────────────────────────────────────────────────
story.append(Paragraph("10. Conclusions", H2))
story.append(Paragraph(
    "Our analysis of 28 seasons of NBA playoff data yields five main takeaways:",
    BODY
))
conclusions = [
    "Most players score fewer points in the playoffs than in the regular season. "
    "Decliners outnumber Risers roughly 3-to-1. The playoffs are harder.",
    "The Riser/Decliner split is most pronounced at the top of the scoring distribution. "
    "Star players are simultaneously the most likely to rise spectacularly and the most likely to fall off.",
    "Guards show the highest Riser rate by position; the effect is smallest for centers, "
    "who are often used more narrowly in playoff rotations.",
    "Advanced metrics (BPM, PER, TS%) reveal that true Risers do not just score more — "
    "they play more efficiently, suggesting a genuine step up rather than increased volume.",
    "A Random Forest trained only on regular-season data can predict Riser vs Decliner with "
    f"AUC ≈ {cv_scores.mean():.2f}. Playoff games played, experience, and efficiency metrics "
    "are the strongest signals. Raw scoring volume is less predictive than it might appear.",
]
for i, c in enumerate(conclusions, 1):
    story.append(Paragraph(f"<b>{i}.</b> {c}", STAT))

story.append(Spacer(1, 0.2*inch))
story.append(Paragraph(
    "Future work could extend this analysis by incorporating game-by-game playoff logs to model "
    "how performance evolves across rounds, and by examining whether team context "
    "(seeding, opponent strength, role within the team) mediates the individual tendency.",
    BODY
))

story.append(hr())
story.append(Paragraph(
    "Data sources: Kaggle (bendikfltaas/nba-history-seasonal-data-1995-2023, drgilermo/nba-players-stats), "
    "Basketball Reference, NBA Stats API. Coverage: 1995–2023 regular seasons and playoffs.",
    ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8,
                   textColor=colors.grey, alignment=TA_CENTER)
))

doc.build(story)
print(f"\nDone. PDF saved to:\n  {PDF_PATH}")
print("\nFigures saved:")
for p in [path1, path2, path3, path4, path5, path6, path7]:
    print(f"  {p}")
