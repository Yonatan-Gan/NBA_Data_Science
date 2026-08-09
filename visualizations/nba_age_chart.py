"""
NBA Age Distribution Chart
"""

import io, os, sys, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, HRFlowable
)

# CONFIG
DATA_DIR    = "../../../../../../year 3/Semester B/מחט בערמת דאטה/NBA_Data/nba_data"
OUTPUT_FILE = "../../../../../../year 3/Semester B/מחט בערמת דאטה/NBA_Data/nba_age_distribution.pdf"

C_DARK  = HexColor("#0d0d0d")
C_MUTED = HexColor("#555555")
C_RULE  = HexColor("#cccccc")
C_BLUE  = HexColor("#1565C0")

PAGE_W, PAGE_H = landscape(A4)
MARGIN = 2.0 * cm
CONTENT_W = PAGE_W - 2 * MARGIN

# LOAD DATA
print("="*60)
print("  NBA Age Distribution Chart Generator")
print(f"  Data dir : {os.path.abspath(DATA_DIR)}/")
print("="*60)

bio_path = f"{DATA_DIR}/player_profiles/player_bio_info.csv"
if not os.path.exists(bio_path):
    print(f"\nNot found: {bio_path}")
    print("  Run nba_data_collector.py first.\n")
    sys.exit(1)

bio = pd.read_csv(bio_path, low_memory=False)
print(f"  Loaded player_bio_info.csv  ({len(bio):,} rows)")

# DERIVE AGE
# Use the midpoint of the 2024-25 season as reference date
REF_DATE = pd.Timestamp("2025-01-15")

age_col = None
name_col = next((c for c in ["DISPLAY_FIRST_LAST","PLAYER_NAME","display_first_last"]
                 if c in bio.columns), None)

if "BIRTHDATE" in bio.columns:
    bio["BIRTHDATE"] = pd.to_datetime(bio["BIRTHDATE"], errors="coerce")
    bio["_age"] = ((REF_DATE - bio["BIRTHDATE"]).dt.days / 365.25).fillna(-1).astype(int)
    age_col = "_age"
    print("  -> Age derived from BIRTHDATE")
elif "AGE" in bio.columns:
    bio["_age"] = pd.to_numeric(bio["AGE"], errors="coerce")
    age_col = "_age"
    print("  -> Age from AGE column")
else:
    # Fallback: season bio stats
    fb = f"{DATA_DIR}/season_stats/2024-25/player_bio_regular_season.csv"
    if os.path.exists(fb):
        sb = pd.read_csv(fb, low_memory=False)
        col = next((c for c in ["PLAYER_AGE","AGE"] if c in sb.columns), None)
        if col:
            bio = sb.copy()
            bio["_age"] = pd.to_numeric(bio[col], errors="coerce")
            age_col = "_age"
            name_col = next((c for c in ["PLAYER_NAME","DISPLAY_FIRST_LAST"] if c in bio.columns), None)
            print(f"  -> Age from season bio stats ({col})")
    if age_col is None:
        print("Cannot find age data. Checked BIRTHDATE, AGE, PLAYER_AGE.")
        sys.exit(1)

bio = bio.dropna(subset=[age_col])
bio["_age"] = bio["_age"].astype(int)
bio = bio[(bio["_age"] >= 18) & (bio["_age"] <= 50)]

age_counts  = bio["_age"].value_counts().sort_index()
age_series  = bio["_age"]
avg_age     = float(age_series.mean())
peak_age    = int(age_counts.idxmax())
n_players   = len(age_series)
ages        = list(age_counts.index)
counts      = list(age_counts.values)
peak_count  = max(counts)

# FIND OUTLIER NAME
# Find the oldest player(s) and get their name
outlier_age   = int(age_counts[age_counts > 0].index.max())
outlier_name  = "Veteran"   # fallback

if name_col and outlier_age in bio["_age"].values:
    oldest_rows = bio[bio["_age"] == outlier_age]
    if not oldest_rows.empty:
        outlier_name = oldest_rows[name_col].iloc[0]
        # If multiple players at that age, list them
        all_oldest = oldest_rows[name_col].tolist()
        if len(all_oldest) > 1:
            outlier_name = " & ".join(all_oldest[:2])

print(f"  -> {n_players} players | avg age {avg_age: .2f} | peak {peak_age} | oldest: {outlier_name} ({outlier_age})")

# BUILD CHART
print("\n  Rendering chart...")

fig, ax = plt.subplots(figsize=(15, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Bar colours - blue gradient based on height, matching reference image style
bar_colors = []
for c in counts:
    t = c / peak_count
    if   t > 0.85: bar_colors.append("#1565C0")
    elif t > 0.55: bar_colors.append("#1976D2")
    elif t > 0.30: bar_colors.append("#42A5F5")
    else:          bar_colors.append("#90CAF9")

ax.bar(ages, counts, color=bar_colors, width=0.72, zorder=3, linewidth=0)

# Value label on every bar
for age, count in zip(ages, counts):
    if count > 0:
        ax.text(age, count + 0.5, str(count),
                ha="center", va="bottom",
                fontsize=9.5, color="#1565C0", fontweight="bold")

# AVERAGE AGE LINE (red, like reference's red annotation)
ax.axvline(avg_age, color="#E53935", linewidth=2.2,
           linestyle="-", zorder=5, alpha=0.9)

# Position label: left of line if it'd collide with peak, otherwise right
label_x_offset = 0.35 if avg_age < peak_age else -0.35
ha_align = "left" if avg_age < peak_age else "right"
ax.text(avg_age + label_x_offset,
        peak_count * 0.70,
        f"Avg age\n{avg_age:.2f}",
        color="#E53935", fontsize=12, fontweight="bold",
        va="center", ha=ha_align, zorder=6,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E53935",
                  lw=1, alpha=0.85))

# OUTLIER ANNOTATION (green, like reference's event annotations)
if outlier_age in age_counts.index:
    outlier_count = int(age_counts[outlier_age])
    # Decide arrow direction based on whether outlier is left or right of center
    center_age = np.mean(ages)
    x_text_offset = -4 if outlier_age > center_age else 4

    ax.annotate(
        f"{outlier_name}\n(age {outlier_age})",
        xy=(outlier_age, outlier_count),
        xytext=(outlier_age + x_text_offset, outlier_count + peak_count * 0.18),
        color="#2E7D32", fontsize=11, fontweight="bold", ha="center",
        arrowprops=dict(
            arrowstyle="-",
            color="#2E7D32",
            lw=1.8,
        ),
        zorder=6,
    )

# TITLES
ax.set_title(
    "NBA Players Age Distribution",
    fontsize=22, fontweight="bold", color="#0d0d0d",
    pad=30, loc="center"
)
ax.text(
    0.5, 1.012,
    f"Age distribution of active NBA players  ·  2024-25 season  ·  {n_players} players",
    transform=ax.transAxes, ha="center", fontsize=12,
    color="#555555", style="italic"
)

# AXES & GRID
ax.set_xlabel("Age", fontsize=20,fontweight="bold", color="#444444", labelpad=8)
ax.set_ylabel("Number of players", fontsize=20,fontweight="bold", color="#444444", labelpad=8)

ax.set_xticks(ages)
ax.set_xticklabels([str(a) for a in ages], fontsize=10, color="#555555")
ax.tick_params(axis="y", labelsize=10, colors="#555555", length=0)
ax.tick_params(axis="x", length=3, colors="#888888")

ax.yaxis.grid(True, color="#e8e8e8", linewidth=0.9, zorder=0)
ax.set_axisbelow(True)

# Remove all spines except bottom
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#cccccc")
ax.spines["bottom"].set_linewidth(0.8)

ax.set_ylim(0, peak_count * 1.22)

fig.tight_layout(pad=1.5)

# RENDER TO BUFFER
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
buf.seek(0)
plt.close(fig)
print("  Chart rendered")

# ASSEMBLE PDF
print("  Building PDF...")

story = []

# chart image - fill the content width
chart_h_cm = CONTENT_W / cm * (7 / 15)   # preserve aspect ratio (15:7 figsize)
story.append(RLImage(buf, width=CONTENT_W, height=chart_h_cm * cm))

story.append(Spacer(1, 0.35 * cm))
story.append(HRFlowable(width="100%", thickness=0.5,
                         color=HexColor("#cccccc"), spaceAfter=4))

doc = SimpleDocTemplate(
    OUTPUT_FILE,
    pagesize=landscape(A4),
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=MARGIN,  bottomMargin=MARGIN,
    author="NBA Data Science Project",
)
doc.build(story)

print(f"\n{'='*60}")
print(f"  PDF saved -> {os.path.abspath(OUTPUT_FILE)}")
print(f"{'='*60}\n")
