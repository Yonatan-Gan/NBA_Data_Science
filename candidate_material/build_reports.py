"""Build compact, modular Q1 and Q2 candidate PDFs from generated outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"
PDF = OUTPUT / "pdf"
PDF.mkdir(parents=True, exist_ok=True)

INK = colors.HexColor("#182230")
MUTED = colors.HexColor("#64748B")
BLUE = colors.HexColor("#2563A6")
TEAL = colors.HexColor("#138A7E")
PALE_BLUE = colors.HexColor("#E8F1F8")
PALE_TEAL = colors.HexColor("#E3F3F0")
RULE = colors.HexColor("#DDE3EA")


def styles():
    base = getSampleStyleSheet()
    return {
        "kicker": ParagraphStyle(
            "Kicker", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=8,
            leading=10, textColor=BLUE, spaceAfter=4,
        ),
        "title": ParagraphStyle(
            "Title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=20,
            leading=23, textColor=INK, alignment=TA_LEFT, spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=10.5,
            leading=14, textColor=MUTED, spaceAfter=10,
        ),
        "heading": ParagraphStyle(
            "Heading", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13,
            leading=16, textColor=INK, spaceBefore=4, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5,
            leading=13.2, textColor=INK, spaceAfter=6,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["Normal"], fontName="Helvetica-Oblique", fontSize=8,
            leading=10.5, textColor=MUTED, spaceAfter=6,
        ),
        "callout": ParagraphStyle(
            "Callout", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=10,
            leading=13.5, textColor=colors.HexColor("#0D5F58"), leftIndent=4,
            rightIndent=4, spaceAfter=0,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["Normal"], fontName="Helvetica", fontSize=8.3,
            leading=11.2, textColor=MUTED,
        ),
    }


STYLES = styles()


def header(story, question: str, title: str, subtitle: str):
    story.extend(
        [
            Paragraph("NBA DATA SCIENCE - CANDIDATE MATERIAL", STYLES["kicker"]),
            Paragraph(f"{question}: {title}", STYLES["title"]),
            Paragraph(subtitle, STYLES["subtitle"]),
            HRFlowable(width="100%", thickness=1.2, color=BLUE, spaceAfter=9),
        ]
    )


def figure(path: Path, width: float = 178 * mm) -> Image:
    from PIL import Image as PILImage

    with PILImage.open(path) as image:
        px_w, px_h = image.size
    return Image(str(path), width=width, height=width * px_h / px_w)


def callout(text: str, color=PALE_TEAL):
    table = Table([[Paragraph(text, STYLES["callout"])]], colWidths=[174 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), color),
                ("BOX", (0, 0), (-1, -1), 0.6, TEAL),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def page_chrome(canvas: Canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 8.5 * mm, "Predicting and Analyzing NBA Performance")
    canvas.drawRightString(A4[0] - 18 * mm, 8.5 * mm, f"Candidate page {doc.page}")
    canvas.restoreState()


def document(path: Path, title: str) -> SimpleDocTemplate:
    return SimpleDocTemplate(
        str(path),
        pagesize=A4,
        title=title,
        author="Yonatan Gan, Ariel Mersel, Nimrod Segev",
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=18 * mm,
    )


def build_q1():
    summary = json.loads((TABLES / "q1_summary.json").read_text(encoding="utf-8"))
    results = pd.read_csv(TABLES / "q1_model_results.csv").set_index("model")
    roles = pd.read_csv(TABLES / "q1_error_by_role.csv")
    best = summary["best_model"]
    best_mae = results.loc[best, "mae"]
    baseline_mae = results.loc["Season-to-date average", "mae"]
    gain = baseline_mae - best_mae
    ci_lo = -summary["ci95_high"]
    ci_hi = -summary["ci95_low"]

    story = []
    header(
        story,
        "Q1",
        "How predictable is a player's next game?",
        "We compare the models with a simple player baseline and test them on a later season.",
    )
    story.append(Paragraph("Setup", STYLES["heading"]))
    story.append(
        Paragraph(
            "We trained on 2019-20 through 2023-24 and froze the models before evaluating "
            f"<b>{summary['n_test_games']:,} player-games</b> in 2024-25. Every rolling feature is shifted, "
            "so it only uses information from earlier games. We compare the models with each player's "
            "season-to-date scoring average, which is a stronger baseline than the league average. We only predict "
            "games after the player has made 10 appearances in that season.",
            STYLES["body"],
        )
    )
    story.append(figure(FIGURES / "q1_fig1_forecast_ladder.png", 176 * mm))
    story.append(
        Paragraph(
            "Figure 1. Mean absolute error on the 2024-25 season. Lower is better. The two machine-learning "
            "models have almost the same error and are only slightly better than the player baseline.",
            STYLES["caption"],
        )
    )
    story.append(
        callout(
            f"Main result: {best} reaches {best_mae:.2f} MAE, which is {gain:.2f} points better than the "
            f"season-average baseline (95% CI: {ci_lo:.2f} to {ci_hi:.2f}). The confidence interval does not include "
            "zero, but the difference is too small to call the predictions precise."
        )
    )

    story.append(PageBreak())
    header(
        story,
        "Q1",
        "Who is hardest to predict?",
        "Absolute error and relative error tell different stories about stars and role players.",
    )
    story.append(figure(FIGURES / "q1_fig2_error_by_role.png", 176 * mm))
    story.append(
        Paragraph(
            "Figure 2. Scoring roles are assigned from pre-game season-to-date points. Gray is the running-average "
            f"baseline; green is {best}. Sample sizes are printed under each group.",
            STYLES["caption"],
        )
    )
    star = roles.iloc[-1]
    role = roles.iloc[0]
    story.append(
        Paragraph(
            f"Stars have the largest error in raw points ({star['model_mae']:.2f}), but the smallest error relative "
            f"to their scoring level ({star['model_relative']:.0f}%). Role players reverse that pattern: only "
            f"{role['model_mae']:.2f} points of absolute error, yet {role['model_relative']:.0f}% relative error. "
            "This means the answer depends on whether error is measured in points or as a percentage of scoring.",
            STYLES["body"],
        )
    )
    story.append(
        callout(
            "Main point: Stars miss by more points because they score more. Relative to their scoring average, their "
            "error is smaller than the error for low-volume role players.",
            PALE_BLUE,
        )
    )

    story.append(PageBreak())
    header(
        story,
        "Q1",
        "Which inputs help the model most?",
        "We shuffle one input at a time and measure how much the test error changes.",
    )
    story.append(figure(FIGURES / "q1_fig3_feature_importance.png", 176 * mm))
    story.append(
        Paragraph(
            "Figure 3. Each feature is shuffled in the held-out season. A large increase in MAE means the fitted "
            "model depended on that feature. Correlated features can share importance, so values should not be read "
            "as causal effects.",
            STYLES["caption"],
        )
    )
    story.append(
        Paragraph(
            "The player's season-to-date scoring level dominates. Recent shot volume and longer scoring windows add "
            "some information, while opponent points allowed, rest, home court, and back-to-backs add very little. "
            "Most of the useful information comes from the player's normal scoring level and recent role.",
            STYLES["body"],
        )
    )
    story.append(
        callout(
            "Q1 conclusion: A player's general scoring level is predictable, but a single game still has a lot of "
            "variation. A simple season average captures most of the information used by these models."
        )
    )
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Limitations: the target is conditional on playing and does not model late scratches, injuries, lineup "
            "announcements, betting-market information, or expected minutes. Adding these inputs may improve the "
            "predictions.",
            STYLES["small"],
        )
    )

    doc = document(PDF / "q1_candidate.pdf", "Q1 Candidate - Next-Game Scoring")
    doc.build(story, onFirstPage=page_chrome, onLaterPages=page_chrome)


def build_q2():
    summary = json.loads((TABLES / "q2_summary.json").read_text(encoding="utf-8"))
    results = pd.read_csv(TABLES / "q2_model_results.csv").set_index("model")
    talent_gain = results.loc["Prior team only", "mae"] - results.loc["+ roster talent", "mae"]
    composition_gain = results.loc["+ roster talent", "mae"] - results.loc["+ roster composition", "mae"]
    continuity_gain = results.loc["+ roster composition", "mae"] - results.loc["+ continuity", "mae"]
    continuity_ci_lo = summary["continuity_gain_ci95_low"]
    continuity_ci_hi = summary["continuity_gain_ci95_high"]

    story = []
    header(
        story,
        "Q2",
        "Does roster continuity help explain team success?",
        "We test whether continuity adds information after accounting for prior team strength and player talent.",
    )
    story.append(Paragraph("How continuity is measured", STYLES["heading"]))
    story.append(
        Paragraph(
            "We define <b>continuity</b> as the share of rotation minutes played by players who were on that same "
            "team one year earlier. Rotation players must log at least 150 minutes in the season. Player-team minutes "
            "are reconstructed from game logs, which handles trades correctly. Talent is measured only from current "
            "rotation players' prior-season Player Impact "
            "Estimate (PIE). We evaluate the models by leaving out one complete season at a time. The roster "
            "summaries use the full season, so this is an explanatory test rather than a preseason forecast.",
            STYLES["body"],
        )
    )
    story.append(figure(FIGURES / "q2_fig1_incremental_prediction.png", 176 * mm))
    story.append(
        Paragraph(
            "Figure 1. Out-of-season prediction of current team net rating across 150 team-seasons. Roster "
            "composition includes age, playing-time concentration, and usage spread. These are full-season summaries. "
            "Lower MAE is better, while higher R-squared is better.",
            STYLES["caption"],
        )
    )
    story.append(
        callout(
            f"Main result: Prior roster talent reduces MAE by {talent_gain:.2f} net-rating points, and composition "
            f"adds another {composition_gain:.2f}. Continuity adds only {continuity_gain:.3f} points of improvement "
            f"(95% team-bootstrap CI: {continuity_ci_lo:.3f} to {continuity_ci_hi:.3f}). R-squared changes from "
            f"{results.loc['+ roster composition', 'r2']:.3f} to {results.loc['+ continuity', 'r2']:.3f}."
        )
    )

    story.append(PageBreak())
    header(
        story,
        "Q2",
        "Continuity after controlling for talent",
        "We remove the part explained by season, prior team strength, and prior player performance.",
    )
    story.append(figure(FIGURES / "q2_fig2_continuity_signal.png", 176 * mm))
    story.append(
        Paragraph(
            "Figure 2. The axes show what remains after controlling for season, prior team net rating, average prior "
            "roster PIE, star PIE, and missing prior-season data. The confidence band uses standard errors clustered "
            "by team.",
            STYLES["caption"],
        )
    )
    story.append(
        Paragraph(
            f"An extra 10 percentage points of continuity is associated with {summary['continuity_slope_per_10pp']:+.2f} "
            f"net-rating points, but the estimate is uncertain (p = {summary['continuity_p_value']:.3f}). The result "
            "suggests a small positive relationship, but it is not statistically significant at the 0.05 level.",
            STYLES["body"],
        )
    )
    story.append(
        callout(
            "Main point: Continuity may help, but most of the predictable difference between teams comes from roster "
            "quality and composition.",
            PALE_BLUE,
        )
    )

    story.append(PageBreak())
    header(
        story,
        "Q2",
        "Talent and continuity together",
        "We split the teams into higher and lower groups to make the pattern easier to see.",
    )
    story.append(figure(FIGURES / "q2_fig3_talent_continuity_matrix.png", 170 * mm))
    story.append(
        Paragraph(
            "Figure 3. Average current-season net rating after splitting teams at the sample medians for prior roster "
            "talent and continuity. Counts are shown in each cell.",
            STYLES["caption"],
        )
    )
    story.append(
        Paragraph(
            "Higher-continuity teams have a better average net rating within both talent groups. However, the talent "
            "difference is still clear. Higher-talent teams with lower continuity average +1.8, while lower-talent "
            "teams with higher continuity average -1.2. Teams with both higher talent and higher continuity average +3.9.",
            STYLES["body"],
        )
    )
    story.append(
        callout(
            "Q2 conclusion: In this sample, teams do best when they combine prior talent with continuity. Continuity "
            "is associated with better results, but talent explains much more of the difference between teams."
        )
    )
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Limitations: this is an observational sample of five NBA seasons. End-of-season minutes define the "
            "rotation, so injuries and coaching choices affect the measured exposure. The analysis cannot identify a "
            "causal effect of continuity or directly observe relationships, leadership, or locker-room dynamics.",
            STYLES["small"],
        )
    )

    doc = document(PDF / "q2_candidate.pdf", "Q2 Candidate - Team Chemistry")
    doc.build(story, onFirstPage=page_chrome, onLaterPages=page_chrome)


if __name__ == "__main__":
    build_q1()
    build_q2()
    print(f"Built {PDF / 'q1_candidate.pdf'}")
    print(f"Built {PDF / 'q2_candidate.pdf'}")
