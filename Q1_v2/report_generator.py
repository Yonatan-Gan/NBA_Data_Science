"""
report_generator.py
====================
Generates a publication-quality PDF report for Q1.

Reads:
  - results/ablation_results.csv
  - results/model_comparison.csv
  - results/statistical_tests.csv
  - figures/Q1/*.png

Outputs:
  - Q1_report.pdf  (in the repo root)
"""

from __future__ import annotations

import io
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import config as cfg
import pandas as pd

# ── ReportLab ────────────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.colors import HexColor, white, black, Color
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, PageBreak, KeepTogether,
    CondPageBreak,
)
from reportlab.platypus.flowables import Flowable

# ── Colours ───────────────────────────────────────────────────────────────────
C_BLUE      = HexColor("#1565C0")
C_LBLUE     = HexColor("#42A5F5")
C_RED       = HexColor("#C62828")
C_GREEN     = HexColor("#2E7D32")
C_ORANGE    = HexColor("#E65100")
C_DARK      = HexColor("#0d0d0d")
C_MID       = HexColor("#3a3a3a")
C_MUTED     = HexColor("#6b6b6b")
C_RULE      = HexColor("#d4d0c8")
C_PAPER2    = HexColor("#f2f0eb")
C_PAPER3    = HexColor("#e8f4fd")

PAGE_W, PAGE_H = A4
MARGIN_L = MARGIN_R = 2.2 * cm
MARGIN_T = MARGIN_B = 2.0 * cm
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R

OUTPUT_PATH = REPO_ROOT / "Q1_report.pdf"


# ══════════════════════════════════════════════════════════════════════════════
#  STYLES
# ══════════════════════════════════════════════════════════════════════════════

def _styles() -> dict:
    base = getSampleStyleSheet()
    s = {}

    s["kicker"] = ParagraphStyle("kicker",
        fontSize=8, textColor=C_BLUE, spaceAfter=2,
        fontName="Helvetica-Bold", leading=10)

    s["title"] = ParagraphStyle("title",
        fontSize=24, textColor=C_DARK, spaceAfter=6,
        fontName="Helvetica-Bold", leading=28)

    s["subtitle"] = ParagraphStyle("subtitle",
        fontSize=11, textColor=C_MUTED, spaceAfter=4,
        fontName="Helvetica", leading=15)

    s["authors"] = ParagraphStyle("authors",
        fontSize=10, textColor=C_MID, spaceAfter=2,
        fontName="Helvetica-Bold", leading=13)

    s["affil"] = ParagraphStyle("affil",
        fontSize=9, textColor=C_MUTED, spaceAfter=0,
        fontName="Helvetica", leading=12)

    s["h1"] = ParagraphStyle("h1",
        fontSize=14, textColor=C_DARK, spaceBefore=14, spaceAfter=5,
        fontName="Helvetica-Bold", leading=18)

    s["h2"] = ParagraphStyle("h2",
        fontSize=11.5, textColor=C_BLUE, spaceBefore=10, spaceAfter=4,
        fontName="Helvetica-Bold", leading=15)

    s["body"] = ParagraphStyle("body",
        fontSize=9.5, textColor=C_MID, spaceAfter=6,
        fontName="Helvetica", leading=14, alignment=TA_JUSTIFY)

    s["body_sm"] = ParagraphStyle("body_sm",
        fontSize=8.5, textColor=C_MID, spaceAfter=4,
        fontName="Helvetica", leading=12, alignment=TA_JUSTIFY)

    s["caption"] = ParagraphStyle("caption",
        fontSize=8, textColor=C_MUTED, spaceAfter=8, spaceBefore=3,
        fontName="Helvetica-Oblique", leading=11, alignment=TA_CENTER)

    s["takeaway"] = ParagraphStyle("takeaway",
        fontSize=9.5, textColor=C_DARK, spaceAfter=0,
        fontName="Helvetica-Bold", leading=13, alignment=TA_JUSTIFY)

    s["mono"] = ParagraphStyle("mono",
        fontSize=8.5, textColor=C_MID, spaceAfter=0,
        fontName="Courier", leading=12)

    s["tbl_hdr"] = ParagraphStyle("tbl_hdr",
        fontSize=8, textColor=C_MUTED,
        fontName="Helvetica-Bold", leading=10)

    s["tbl_cell"] = ParagraphStyle("tbl_cell",
        fontSize=8.5, textColor=C_MID,
        fontName="Helvetica", leading=11)

    s["tbl_cell_mono"] = ParagraphStyle("tbl_cell_mono",
        fontSize=8, textColor=C_MID,
        fontName="Courier", leading=11)

    s["footer"] = ParagraphStyle("footer",
        fontSize=7.5, textColor=C_MUTED,
        fontName="Helvetica", alignment=TA_CENTER, leading=10)

    s["abstract"] = ParagraphStyle("abstract",
        fontSize=9, textColor=C_MID, spaceAfter=0,
        fontName="Helvetica", leading=13, alignment=TA_JUSTIFY,
        leftIndent=10, rightIndent=10)

    return s


# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM FLOWABLES
# ══════════════════════════════════════════════════════════════════════════════

class TakeawayBox(Flowable):
    """Green-bordered key takeaway box (matches Q3 style)."""
    def __init__(self, text: str, S: dict, width: float):
        super().__init__()
        self._para  = Paragraph(text, S["takeaway"])
        self._width = width
        self._pad   = 8

    def wrap(self, aW, aH):
        inner = self._width - self._pad * 2
        _, h  = self._para.wrap(inner, aH)
        self.height = h + self._pad * 2
        return self._width, self.height

    def draw(self):
        c = self.canv
        c.setFillColor(HexColor("#f0f7e6"))
        c.setStrokeColor(C_GREEN)
        c.setLineWidth(1.2)
        c.roundRect(0, 0, self._width, self.height, 4, fill=1, stroke=1)
        inner = self._width - self._pad * 2
        _, h  = self._para.wrap(inner, 9999)
        self._para.drawOn(c, self._pad, (self.height - h) / 2)


class SectionRule(Flowable):
    """Thick left-accent bar for section headings."""
    def __init__(self, width: float, color=C_BLUE, height: float = 3):
        super().__init__()
        self.width  = width
        self._color = color
        self.height = height

    def draw(self):
        self.canv.setFillColor(self._color)
        self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=0)


def _rule(width=None, color=C_RULE, thickness=0.5) -> HRFlowable:
    return HRFlowable(
        width=width or "100%", thickness=thickness,
        color=color, spaceAfter=4, spaceBefore=4,
    )


def _fig(name: str, width_cm: float, height_cm: float,
         caption: str, S: dict) -> list:
    """Load a figure PNG and return [Image, Caption] flowables."""
    path = cfg.FIGURES_DIR / f"{name}.png"
    if not path.exists():
        return [Paragraph(f"[Figure {name} not found]", S["caption"])]
    img = RLImage(str(path), width=width_cm * cm, height=height_cm * cm)
    cap = Paragraph(caption, S["caption"])
    return [img, cap]


def _table(headers: list, rows: list, col_widths: list,
           S: dict) -> Table:
    data = [[Paragraph(str(h), S["tbl_hdr"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), S["tbl_cell"]) for c in row])

    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("LINEBELOW",     (0,0), (-1,0),  1.2, C_DARK),
        ("LINEBELOW",     (0,1), (-1,-1), 0.4, C_RULE),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [white, C_PAPER2]),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 5),
        ("RIGHTPADDING",  (0,0), (-1,-1), 5),
    ]))
    return tbl


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE TEMPLATE (header / footer)
# ══════════════════════════════════════════════════════════════════════════════

def _on_page(canvas, doc):
    canvas.saveState()
    # Header rule
    canvas.setStrokeColor(C_RULE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN_L, PAGE_H - 1.3*cm, PAGE_W - MARGIN_R, PAGE_H - 1.3*cm)
    # Header text
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(C_MUTED)
    canvas.drawString(MARGIN_L, PAGE_H - 1.1*cm,
                      "Q1: Predicting NBA Player Next-Game Performance")
    canvas.drawRightString(PAGE_W - MARGIN_R, PAGE_H - 1.1*cm,
                           "NBA Data Science Project · 2026")
    # Footer
    canvas.line(MARGIN_L, 1.3*cm, PAGE_W - MARGIN_R, 1.3*cm)
    canvas.drawCentredString(PAGE_W / 2, 0.85*cm, f"{doc.page}")
    canvas.restoreState()

def _on_first_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(C_MUTED)
    canvas.line(MARGIN_L, 1.3*cm, PAGE_W - MARGIN_R, 1.3*cm)
    canvas.drawCentredString(PAGE_W / 2, 0.85*cm, "1")
    canvas.restoreState()


# ══════════════════════════════════════════════════════════════════════════════
#  REPORT ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def build_report(
    ablation_df: Optional[pd.DataFrame] = None,
    model_df:    Optional[pd.DataFrame] = None,
    stats_df:    Optional[pd.DataFrame] = None,
) -> Path:

    S = _styles()
    story = []

    # helper shortcuts
    P  = lambda text, style="body": Paragraph(text, S[style])
    SP = lambda h=6: Spacer(1, h)
    KB = lambda items: KeepTogether(items)

    FW = CONTENT_W   # full content width

    # ══════════════════════════════════════════════════════════════════════════
    #  TITLE PAGE
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        SP(20),
        SectionRule(FW, C_BLUE, 4),
        SP(10),
        P("NBA DATA SCIENCE PROJECT", "kicker"),
        SP(4),
        P("Understanding and Predicting NBA Player<br/>Performance Using Context-Aware Machine Learning",
          "title"),
        SP(6),
        P("Research Question 1: Can we predict how a player will perform in their next game\n"
          "using past performance, rest days, opponent strength, and team context?", "subtitle"),
        SP(16),
        _rule(color=C_RULE),
        SP(6),
        P("Yonatan Gan · Ariel Mersel · Nimrod Segev", "authors"),
        P("67978 — A Needle in a Data Haystack, 2026", "affil"),
        SP(4),
        P(f"Report generated: {datetime.now().strftime('%B %d, %Y')}", "affil"),
        SP(16),
        _rule(color=C_RULE),
        SP(10),
    ]

    # Abstract box
    abstract_text = (
        "We investigate whether machine learning can predict an NBA player's next-game "
        "point total using five categories of contextual features: player form (rolling "
        "statistics, shooting efficiency, streak indicators), schedule context (rest days, "
        "back-to-back games), opponent context (defensive quality proxies from team logs), "
        "team context (rolling offensive and defensive ratings), and career context (age, "
        "experience, career averages). Across 149,316 player-game observations from six "
        "seasons (2019-20 to 2024-25) and ten models, we find that all models converge "
        "to a mean absolute error of approximately 4.54-4.62 points — a 35% improvement "
        "over a naive baseline. Critically, model architecture matters far less than "
        "feature quality: Ridge regression matches XGBoost within 0.001 MAE points. "
        "Statistical analysis reveals that star players (20+ ppg) are significantly "
        "harder to predict (ANOVA F=609, p&lt;0.001), while back-to-back fatigue has no "
        "measurable within-player effect on scoring (p=0.928). These findings suggest "
        "that the dominant source of prediction difficulty is intrinsic game-to-game "
        "variance in player performance, not model limitation."
    )
    abstract_table = Table(
        [[Paragraph("<b>Abstract.</b> " + abstract_text, S["abstract"])]],
        colWidths=[FW],
    )
    abstract_table.setStyle(TableStyle([
        ("BOX",            (0,0), (-1,-1), 0.5, C_RULE),
        ("BACKGROUND",     (0,0), (-1,-1), C_PAPER2),
        ("TOPPADDING",     (0,0), (-1,-1), 10),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 10),
        ("LEFTPADDING",    (0,0), (-1,-1), 12),
        ("RIGHTPADDING",   (0,0), (-1,-1), 12),
    ]))
    story += [abstract_table, SP(8)]
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    #  1. INTRODUCTION
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        SectionRule(FW), SP(4),
        P("1. Introduction", "h1"),
        P("Every NBA broadcast makes it sound simple: you know what a player averages, you know the opponent, you've seen the last ten games. How hard can tomorrow's points be to predict? As it turns out — very. And the reason why tells us something fundamental about the nature of basketball itself."),
        P("The difficulty arises from a fundamental tension: NBA scoring is simultaneously "
          "structured and noisy. It is structured because a player's baseline ability is "
          "stable over time — a 20-point-per-game scorer rarely posts single-digit totals "
          "repeatedly. But it is noisy because each individual game is shaped by dozens of "
          "unobservable factors: defensive schemes, foul trouble, fatigue, team chemistry, "
          "and random variation in shooting. No model can observe all of these, which means "
          "prediction error has a hard floor that more data or better models cannot remove."),
        P("This paper investigates two questions simultaneously. First, how accurately "
          "can we predict next-game scoring, and which contextual factors contribute most "
          "to that accuracy? Second, and more importantly, what do the residual errors tell "
          "us about the nature of NBA performance — which players and situations are "
          "inherently harder to predict, and why?"),
        SP(4),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    #  2. RESEARCH QUESTION
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        SectionRule(FW), SP(4),
        P("2. Research Question", "h1"),
        P("<b>Primary question:</b> Can we use a player's recent form, schedule context, "
          "opponent quality, and team environment to predict their next-game point total?"),
        P("<b>Secondary questions:</b>"),
        P("&nbsp;&nbsp;• Which contextual feature group contributes most to prediction accuracy?<br/>"
          "&nbsp;&nbsp;• Do all model architectures converge to the same accuracy floor?<br/>"
          "&nbsp;&nbsp;• Does fatigue (back-to-back games) measurably reduce scoring?<br/>"
          "&nbsp;&nbsp;• Are star players harder to predict than role players?<br/>"
          "&nbsp;&nbsp;• Is home court advantage predictable from the features we have?"),
        SP(4),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    #  3. DATASET
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        SectionRule(FW), SP(4),
        P("3. Dataset Description", "h1"),
        P("All data was collected from the official NBA Stats API (stats.nba.com) via the "
          "<font face='Courier' size=9>nba_api</font> Python package, supplemented by "
          "Basketball Reference and Kaggle datasets. The core dataset consists of player "
          "game logs — one row per player per game — spanning six complete NBA seasons "
          "from 2019-20 through 2024-25."),
        SP(4),
    ]

    ds_rows = [
        ["Player game logs",   "NBA API", "149,316 rows", "910 players, 6 seasons"],
        ["Team game logs",     "NBA API", "14,118 rows",  "30 teams, box scores per game"],
        ["Player bio info",    "NBA API", "530 rows",     "Height, weight, country, birthdate"],
        ["Career stats",       "NBA API", "4,914 rows",   "Season-by-season career totals"],
        ["Player salaries",    "Kaggle",  "15,857 rows",  "Contract values by season"],
        ["BRef advanced",      "Bball Ref","5,438 rows",  "PER, WS, BPM, VORP per team-season"],
    ]
    story += [
        _table(["Data source","Origin","Size","Contents"],
               ds_rows,
               [FW*0.22, FW*0.16, FW*0.18, FW*0.44], S),
        SP(6),
        P("<b>Target variable:</b> PTS (points scored) in a player's next game. We chose "
          "points because it is the most complete measure of offensive contribution, "
          "directly observable, and the primary currency by which players are valued."),
        P("<b>Preprocessing:</b> DNP (Did Not Play) rows were removed (MIN = 0), "
          "duplicate player-game pairs were deduplicated, and players with fewer than "
          "10 games were excluded to ensure meaningful rolling features. The final "
          "dataset contains 149,316 player-game observations."),
        SP(4),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    #  4. FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        SectionRule(FW), SP(4),
        P("4. Feature Engineering", "h1"),
        P("We engineered features across five groups. Every rolling feature uses a "
          "<b>shift(1)-then-roll</b> pattern: the series is shifted one position before "
          "computing the rolling window, ensuring that the prediction for game N only "
          "uses information from games 1 through N-1. This prevents data leakage."),
        SP(6),
    ]

    feat_groups = [
        ("Player Form (21 features)",
         "Rolling means (3/5/10 games), exponentially-weighted moving average (span=5), "
         "rolling standard deviation, recent scoring trend (slope of last 5 games), "
         "season-to-date average, performance volatility (std of last 10 games), "
         "consistency score (1/volatility), hot streak indicator (last-3 avg &gt; season "
         "avg + 3 pts), cold streak indicator (last-3 avg &lt; season avg - 3 pts). "
         "These features capture a player's recent form and baseline level."),
        ("Schedule Context (7 features)",
         "Rest days since last game (capped at 7), back-to-back indicator (rest = 1 day), "
         "3-games-in-4-nights indicator, games played in last 7 days, consecutive home "
         "game streak, consecutive away game streak. These features capture fatigue "
         "and logistical disadvantages of dense scheduling."),
        ("Opponent Context (8 features)",
         "Since DEF_RATING is not available in per-game team logs, we derive defensive "
         "quality proxies: rolling points allowed per game (pts scored minus plus-minus), "
         "rolling opponent shooting efficiency (eFG%), rolling opponent win percentage, "
         "rolling opponent net rating (plus-minus). Higher points allowed = weaker defence "
         "= easier to score against."),
        ("Team Context (5 features)",
         "Rolling team offensive rating, defensive rating, net rating, pace, and assist "
         "percentage derived from team game logs. These capture whether the player's "
         "team is in a good rhythm and playing fast (which inflates individual stats) "
         "or slow and defensive."),
        ("Career Context (5 features)",
         "Player age (derived from birthdate), years of experience (seasons played), "
         "career average points, career average minutes, career eFG%. These are slowly "
         "varying signals that anchor predictions for players whose recent form is "
         "misleadingly high or low."),
    ]
    for name, desc in feat_groups:
        story += [
            P(f"<b>{name}</b>", "h2"),
            P(desc),
            SP(3),
        ]

    story.append(SP(4))

    # ══════════════════════════════════════════════════════════════════════════
    #  5. EXPERIMENTAL DESIGN
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        SectionRule(FW), SP(4),
        P("5. Experimental Design", "h1"),
        P("We use a <b>strict chronological train/test split</b>: rows are sorted by "
          "GAME_DATE, and the first 80% serve as training data, the last 20% as test. "
          "This mirrors real-world deployment — a model trained in October must predict "
          "games in March — and prevents future information from leaking into training. "
          "A random split would be methodologically invalid for time-series data."),
        P("<b>Ablation study:</b> We run six experiments (EXP1-EXP6) using XGBoost, "
          "adding one feature group at a time. This measures the marginal contribution "
          "of each context layer independently of model choice."),
        P("<b>Model comparison:</b> We then fix the full feature set and compare ten "
          "models: Naive Baseline, Ridge, Lasso, ElasticNet, Random Forest, Extra Trees, "
          "Gradient Boosting, XGBoost, LightGBM, and CatBoost. This isolates model "
          "architecture from feature set."),
        SP(6),
    ]

    exp_rows = [
        ["EXP1 — Baseline",   "Player form only",                      "21", "—"],
        ["EXP2 — Schedule",   "EXP1 + schedule context",               "28", "+7"],
        ["EXP3 — Opponent",   "EXP2 + opponent context",               "36", "+8"],
        ["EXP4 — Team",       "EXP3 + team context",                   "41", "+5"],
        ["EXP5 — Career",     "EXP4 + career context",                 "46", "+5"],
        ["EXP6 — All",        "All feature groups combined",           "46", " 0"],
    ]
    story += [
        _table(["Experiment","Feature groups included","# Features","Added"],
               exp_rows,
               [FW*0.22, FW*0.42, FW*0.18, FW*0.18], S),
        SP(8),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    #  6. MODEL RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        SectionRule(FW), SP(4),
        P("6. Results", "h1"),
        P("6.1 The Surprising Truth: Knowing More About the Game Barely Helps", "h2"),
    ]

    if ablation_df is not None and not ablation_df.empty:
        abl_rows = []
        baseline_mae = ablation_df.iloc[0]["mae"]
        for _, row in ablation_df.iterrows():
            imp = baseline_mae - row["mae"]
            pct = imp / baseline_mae * 100
            abl_rows.append([
                row["experiment"].replace("EXP", "Exp ").replace("_", " "),
                str(row["n_features"]),
                f"{row['mae']:.3f}",
                f"{row['rmse']:.3f}",
                f"{row['r2']:.3f}",
                f"−{imp:.3f} ({pct:.1f}%)" if imp > 0.0001 else "—",
            ])
        story += [
            _table(
                ["Experiment","Features","MAE","RMSE","R²","vs Baseline"],
                abl_rows,
                [FW*0.26, FW*0.10, FW*0.12, FW*0.12, FW*0.12, FW*0.28], S,
            ),
            SP(6),
        ]

    story += [
        TakeawayBox(
            "Key finding: Career context (player age, experience, career averages) "
            "provides the largest single improvement beyond player form. Schedule, "
            "opponent, and team context each contribute smaller but meaningful gains. "
            "The full feature set reduces MAE by approximately 0.05 pts vs player "
            "form alone — modest in absolute terms, but consistent across all models.",
            S, FW,
        ),
        SP(10),
        P("6.2 Ten Models Walk In. They All Walk Out With the Same Answer.", "h2"),
        P("<b>Ridge regression, invented in 1970, matches XGBoost within 0.001 points. That result alone tells you more about NBA prediction than any model comparison chart can.</b>"),
        SP(4),
    ]

    if model_df is not None and not model_df.empty:
        best_mae = model_df["mae"].min()
        mod_rows = []
        for _, row in model_df.sort_values("mae").iterrows():
            mod_rows.append([
                row["model"],
                f"{row['mae']:.3f}",
                f"{row['rmse']:.3f}",
                f"{row['r2']:.3f}",
                f"{row['train_time_s']:.1f}s",
            ])
        story += [
            _table(
                ["Model","MAE","RMSE","R²","Train time"],
                mod_rows,
                [FW*0.32, FW*0.14, FW*0.14, FW*0.14, FW*0.26], S,
            ),
            SP(6),
        ]

    story += _fig("Q1_fig02_model_comparison", FW/cm, 5.5,
        "Figure 1. Model comparison on the full feature set (chronological split). "
        "Left: MAE (lower = better). Centre: R² score (higher = better). "
        "Right: training time in seconds.", S)

    story += [
        TakeawayBox(
            "Key finding: All ten models converge to nearly identical MAE (4.54-4.63 "
            "pts) and R² (0.53-0.54). This is the single most important result: "
            "the prediction ceiling is set by the irreducible noise in NBA scoring, "
            "not by model architecture. XGBoost is the recommended model — it matches "
            "Gradient Boosting's accuracy in 2 seconds vs 115 seconds.",
            S, FW,
        ),
        SP(10),
        CondPageBreak(6*cm),
        P("6.3 Prediction Quality", "h2"),
    ]

    story += _fig("Q1_fig03_actual_vs_predicted", FW/cm, 6.0,
        "Figure 2. Left: hexbin scatter of actual vs predicted points for the best model (scale limited to 50 pts). "
        "Points along the diagonal were predicted perfectly. Right: prediction error "
        "distribution — approximately symmetric around zero, indicating no systematic bias.",
        S)
    story.append(SP(8))

    # ══════════════════════════════════════════════════════════════════════════
    #  7. EXPLAINABILITY
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        SectionRule(FW), SP(4),
        P("7. What the Model Actually Learned (It's Simpler Than You Think)", "h1"),
        P("We use SHAP (SHapley Additive exPlanations) to decompose each prediction "
          "into the contribution of individual features. SHAP values are grounded in "
          "cooperative game theory: the SHAP value of a feature is its average marginal "
          "contribution across all possible orderings of the features."),
        SP(4),
    ]

    shap_path = cfg.FIGURES_DIR / "Q1_fig04_shap_summary.png"
    if shap_path.exists():
        story += _fig("Q1_fig04_shap_summary", FW/cm, 7.0,
            "Figure 3. Mean absolute SHAP value per feature — overall importance ranking. "
            "Features with near-zero impact (<0.005) have been excluded.", S)
    else:
        story.append(P("[SHAP figure not generated — re-run main.py]", "caption"))

    story += [
        SP(6),
        TakeawayBox(
            "Key finding: Season average points is the dominant predictor — a player's "
            "baseline level explains most of the variance. Short-term rolling averages "
            "(3-game, 5-game) add the next layer of signal by capturing current form. "
            "Context features (opponent, schedule, team) have smaller but consistent "
            "positive SHAP contributions, confirming they add marginal value beyond "
            "the player's own history.",
            S, FW,
        ),
        SP(10),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    #  8. STATISTICAL ANALYSIS & SUBGROUP ERROR
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        SectionRule(FW), SP(4),
        P("8. Statistical Analysis & Subgroup Error", "h1"),
        P("We conduct four hypothesis tests on the test-set predictions to answer "
          "specific research sub-questions. Each test reports a test statistic, "
          "p-value, effect size, and interpretation."),
        SP(4),
    ]

    if stats_df is not None and not stats_df.empty:
        stat_rows = []
        for _, row in stats_df.iterrows():
            stat_rows.append([
                row["Question"],
                row["Test"],
                row["p-value"],
                row["Significant"],
                row.get("Effect size", "—"),
            ])
        story += [
            _table(
                ["Question","Test","p-value","Sig.","Effect size"],
                stat_rows,
                [FW*0.38, FW*0.14, FW*0.10, FW*0.08, FW*0.30], S,
            ),
            SP(6),
        ]

    story += [
        SP(6),
        P("<b>Fatigue (back-to-back games):</b> NBA players are professionals. One extra night of rest does not move the needle — at least not in the box score. The within-player paired test (321 players, MIN &gt; 20) finds no scoring difference on back-to-back nights whatsoever (p=0.928). What fatigue changes — defensive effort, load management decisions, late-game availability — simply doesn't show up in points. The box score is not a fatigue sensor."),
        P("<b>Home court advantage:</b> Home games are not significantly easier to "
          "predict than away games (p=0.267). Home court does boost scoring (a well-"
          "documented effect), but this boost is already captured in the rolling "
          "averages, so the model's residual error is similar in both venues."),
        P("<b>Scoring tier predictability:</b> ANOVA reveals highly significant "
          "differences in prediction error across scoring tiers (F=609, p&lt;0.001, "
          "η²=0.058). Star players (20+ ppg) have substantially higher prediction "
          "error than role players — they face targeted defensive schemes that create "
          "genuine game-to-game variance that no model can fully capture."),
        SP(6),
    ]

    story += _fig("Q1_fig06_error_by_subgroup", FW/cm, 5.5,
        "Figure 4. Prediction error broken down by scoring tier (left) and position "
        "(right). Stars are harder to predict; position differences are smaller.",
        S)
    story.append(SP(8))

    # ══════════════════════════════════════════════════════════════════════════
    #  9. ROLLING PREDICTIONS
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        SectionRule(FW), SP(4),
        P("9. Individual Player Tracking", "h1"),
        P("To validate the model qualitatively, we plot game-by-game actual vs predicted "
          "points for individual players across the test season. A good model should "
          "track the broad shape of a player's season — rising after hot streaks, "
          "dipping after injuries — without over-fitting to individual game results."),
        SP(4),
    ]

    story += _fig("Q1_fig08_rolling_predictions", FW/cm, 9.0,
        "Figure 5. Actual vs predicted points for three example players across the "
        "test period. Blue line = model predictions, grey line = actual scoring, "
        "red dashed line = season average. The model tracks the broad seasonal arc "
        "but cannot anticipate single-game outliers.", S)

    story += [
        SP(6),
        P("DiVincenzo is a textbook case: a consistent 14–16 point scorer whose game-by-game output swings wildly — a 31-point night followed by a 4-point night. The model tracks his seasonal arc reliably (MAE: 5.05 pts) but cannot anticipate the individual swings. That gap between the blue and grey lines is not model failure — it is the irreducible randomness of basketball."),
        SP(8)
    ]

    # ══════════════════════════════════════════════════════════════════════════
    #  10. DISCUSSION
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        SectionRule(FW), SP(4),
        P("10. Discussion", "h1"),
        P("The central finding of this study is that NBA scoring prediction has a hard "
          "accuracy ceiling of approximately ±4.5 points MAE, regardless of model "
          "sophistication. This ceiling is not a modelling failure — it reflects "
          "genuine irreducible variance in player performance. A player can score "
          "30 points one night and 12 the next for reasons no dataset can fully capture: "
          "a pick-and-roll the defence ran flawlessly, a hot shooting night, a minor "
          "injury that never appears in a box score."),
        P("The convergence of all ten models to nearly identical accuracy is striking. "
          "Ridge regression — a 1970s linear model — matches XGBoost, LightGBM, and "
          "CatBoost within 0.001 MAE points. This tells us the relationship between "
          "our features and next-game scoring is approximately linear: more rolling "
          "average points predicts more next-game points, proportionally. Non-linear "
          "interactions add virtually nothing. This is consistent with prior sports "
          "analytics research showing that simple models often match complex ones on "
          "noisy prediction tasks."),
        P("The ablation study reveals that career context (age, experience, career "
          "averages) provides the most improvement beyond short-term form. This is "
          "intuitive: a player in their prime with a 20-point career average is more "
          "reliably predicted than a young player or a veteran in decline whose recent "
          "form is volatile. Schedule and opponent context add smaller but consistent "
          "improvements — the signal is real, just small relative to the player's "
          "own history."),
        P("The null fatigue finding is counterintuitive but methodologically clean. "
          "When we control for player identity (within-player paired test), back-to-back "
          "games show no significant scoring effect. NBA players are professional "
          "athletes at peak conditioning; one extra night of rest does not measurably "
          "change their performance in the box score. What does change — effort, "
          "defensive intensity, load management decisions — is not captured in points."),
        SP(4),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    #  11. LIMITATIONS & FUTURE WORK
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        SectionRule(FW), SP(4),
        P("11. Limitations and Future Work", "h1"),
        P("<b>Limitations:</b>"),
        P("&nbsp;&nbsp;• <b>No DEF_RATING in team game logs:</b> The per-game team logs "
          "from the NBA API (TeamGameLogs endpoint) contain only box score statistics. "
          "True opponent DEF_RATING requires the LeagueDashTeamStats advanced endpoint "
          "which returns season-level aggregates. We approximate defensive quality via "
          "rolling points allowed, which is a reasonable proxy but not identical.<br/><br/>"
          "&nbsp;&nbsp;• <b>No injury data:</b> Injuries are among the strongest "
          "predictors of next-game performance but are not included in the NBA API. "
          "Adding injury report data (available from ESPN or official NBA injury reports) "
          "would likely be the single largest accuracy improvement available.<br/><br/>"
          "&nbsp;&nbsp;• <b>No shot quality / defense-adjusted metrics:</b> A player "
          "guarded by the opponent's best defender plays a fundamentally different game "
          "than one matched against a backup. Tracking data and lineup-adjusted metrics "
          "would capture this, but require proprietary data access.<br/><br/>"
          "&nbsp;&nbsp;• <b>Single target variable:</b> We predict only points. "
          "Rebounds, assists, and efficiency metrics are equally important in many "
          "contexts (fantasy sports, team building) and may be more predictable."),
        SP(6),
        P("<b>Future work:</b>"),
        P("&nbsp;&nbsp;• Integrate injury report data to handle load management and "
          "returns-from-injury as explicit features.<br/>"
          "&nbsp;&nbsp;• Predict a composite performance metric (fantasy score or "
          "game score) rather than points alone.<br/>"
          "&nbsp;&nbsp;• Explore per-player models — a separate model for each player "
          "calibrated to their individual variance profile.<br/>"
          "&nbsp;&nbsp;• Add minute projections as a feature: predicted minutes played "
          "is a strong upstream predictor of all counting stats."),
        SP(4),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    #  12. CONCLUSION
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        SectionRule(FW), SP(4),
        P("12. Conclusion", "h1"),
        P("We built a modular, reproducible pipeline for predicting NBA player next-game "
          "scoring using five categories of contextual features and ten machine learning "
          "models. Our primary findings are:"),
        P("&nbsp;&nbsp;<b>1.</b> All models converge to approximately 4.54-4.62 MAE — "
          "the prediction ceiling is set by irreducible variance, not by model choice.<br/>"
          "&nbsp;&nbsp;<b>2.</b> Career context features (age, experience, career averages) "
          "provide the largest single improvement beyond player form.<br/>"
          "&nbsp;&nbsp;<b>3.</b> Star players (20+ ppg) are significantly harder to predict "
          "than role players — targeted defensive schemes create genuine unpredictability.<br/>"
          "&nbsp;&nbsp;<b>4.</b> Back-to-back games do not measurably reduce scoring when "
          "controlling for player identity (within-player paired test, p=0.928).<br/>"
          "&nbsp;&nbsp;<b>5.</b> XGBoost is the recommended model: it matches the most "
          "accurate models while training 54× faster than Gradient Boosting."),
        SP(8),
        TakeawayBox(
            "The dominant lesson from this analysis: in predicting NBA scoring, "
            "a player's own history is far more informative than any external context. "
            "Schedule, opponent, and team features add signal, but a player's season "
            "average and recent form explain most of what can be explained. "
            "The remaining ~47% of variance (1 - R²) is intrinsic game-to-game noise "
            "that no model — however sophisticated — can remove.",
            S, FW,
        ),
        SP(12),
        _rule(color=C_RULE),
        SP(6),
        P(f"NBA Data Science Project · Question 1 · "
          f"{datetime.now().strftime('%B %Y')} · "
          f"Generated by report_generator.py",
          "footer"),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    #  BUILD
    # ══════════════════════════════════════════════════════════════════════════
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        leftMargin=MARGIN_L, rightMargin=MARGIN_R,
        topMargin=2.0*cm,    bottomMargin=2.0*cm,
        title="Q1: Predicting NBA Player Next-Game Performance",
        author="Yonatan Gan, Ariel Mersel, Nimrod Segev",
        subject="NBA Data Science Project 2026",
    )
    doc.build(story, onFirstPage=_on_first_page, onLaterPages=_on_page)
    print(f"\n  ✅  Report saved → {OUTPUT_PATH}\n")
    return OUTPUT_PATH


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Building Q1 PDF report...")

    # Load result tables if available
    def _try_load(path: Path) -> Optional[pd.DataFrame]:
        return pd.read_csv(path) if path.exists() else None

    ablation_df = _try_load(THIS_DIR / "results" / "ablation_results.csv")
    model_df    = _try_load(THIS_DIR / "results" / "model_comparison.csv")
    stats_df    = _try_load(THIS_DIR / "results" / "statistical_tests.csv")

    if ablation_df is None:
        print("  ⚠  ablation_results.csv not found — run main.py first")
    if model_df is None:
        print("  ⚠  model_comparison.csv not found — run main.py first")

    build_report(ablation_df, model_df, stats_df)