"""Build a three-page PDF of the research-inspired extensions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
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
PALE_TEAL = colors.HexColor("#E3F3F0")
RULE = colors.HexColor("#DDE3EA")


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "kicker": ParagraphStyle(
            "Kicker",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=BLUE,
            spaceAfter=4,
        ),
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=22,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10.2,
            leading=13.5,
            textColor=MUTED,
            spaceAfter=8,
        ),
        "heading": ParagraphStyle(
            "Heading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=14,
            textColor=INK,
            spaceBefore=2,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.6,
            textColor=INK,
            spaceAfter=5,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=7.8,
            leading=10,
            textColor=MUTED,
            spaceAfter=5,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=12.6,
            textColor=colors.HexColor("#0D5F58"),
        ),
        "source": ParagraphStyle(
            "Source",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.6,
            textColor=MUTED,
            spaceBefore=4,
        ),
    }


STYLES = make_styles()


def header(story: list, label: str, title: str, subtitle: str) -> None:
    story.extend(
        [
            Paragraph("NBA DATA SCIENCE - RESEARCH-INSPIRED EXTENSIONS", STYLES["kicker"]),
            Paragraph(f"{label}: {title}", STYLES["title"]),
            Paragraph(subtitle, STYLES["subtitle"]),
            HRFlowable(width="100%", thickness=1.2, color=BLUE, spaceAfter=7),
        ]
    )


def figure(path: Path, width: float = 178 * mm) -> Image:
    from PIL import Image as PILImage

    with PILImage.open(path) as image:
        pixel_width, pixel_height = image.size
    return Image(
        str(path), width=width, height=width * pixel_height / pixel_width
    )


def callout(text: str) -> Table:
    table = Table([[Paragraph(text, STYLES["callout"])]], colWidths=[174 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_TEAL),
                ("BOX", (0, 0), (-1, -1), 0.6, TEAL),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def page_chrome(canvas: Canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 8.5 * mm, "Predicting and Analyzing NBA Performance")
    canvas.drawRightString(A4[0] - 18 * mm, 8.5 * mm, f"Extension page {doc.page}")
    canvas.restoreState()


def build() -> Path:
    summary = json.loads((TABLES / "summary.json").read_text(encoding="utf-8"))
    q1_roles = pd.read_csv(TABLES / "q1_interval_by_role.csv").set_index("scoring_role")
    q2_results = pd.read_csv(TABLES / "q2_continuity_adjustment.csv").set_index("model")
    q3_results = pd.read_csv(TABLES / "q3_scoring_decomposition.csv").set_index("scoring_role")

    story: list = []

    header(
        story,
        "Q1 extension",
        "How wide should a scoring forecast be?",
        "A prediction range shows the uncertainty that a single MAE number hides.",
    )
    story.append(Paragraph("Idea and setup", STYLES["heading"]))
    story.append(
        Paragraph(
            "Published NBA forecast research emphasizes <b>calibration</b>: an 80% forecast should contain the "
            "real outcome about 80% of the time. We trained Ridge on 2019-20 through 2022-23, used only 2023-24 "
            "errors to set the range sizes, and then checked them on 2024-25. This keeps the final season fully "
            "unseen when the ranges are chosen.",
            STYLES["body"],
        )
    )
    story.append(figure(FIGURES / "q1_forecast_ranges.png", 176 * mm))
    story.append(
        Paragraph(
            "Figure 1. Left: the number of points added above and below the prediction for an 80% range. Right: "
            "target coverage compared with observed coverage on 20,914 future games.",
            STYLES["caption"],
        )
    )
    role_width = q1_roles.loc["Role player", "half_width_points"]
    star_width = q1_roles.loc["Star", "half_width_points"]
    story.append(
        callout(
            f"Main result: The global 80% range is +/- {summary['q1']['global_80_half_width']:.2f} points and "
            f"covers {100 * summary['q1']['global_80_coverage']:.1f}% of the next season. Role players need about "
            f"+/- {role_width:.1f} points, while stars need +/- {star_width:.1f}. A 25-point star forecast is "
            "therefore closer to a range of about 15 to 35 points."
        )
    )
    story.append(
        Paragraph(
            "Limitation: these are empirical ranges, not guarantees. They still do not know about late scratches, "
            "injuries, lineup announcements, or expected minutes.",
            STYLES["body"],
        )
    )
    story.append(
        Paragraph(
            "Inspiration: Yeh, Rice, and Dubin, Evaluating real-time probabilistic forecasts with application to "
            "NBA outcome prediction, arxiv.org/abs/2010.00781. Prediction-interval method inspiration: "
            "arxiv.org/abs/1909.07889.",
            STYLES["source"],
        )
    )

    story.append(PageBreak())
    header(
        story,
        "Q2 extension",
        "Why does continuity look so important at first?",
        "We add the likely confounders one group at a time and watch the estimate change.",
    )
    story.append(Paragraph("Idea and setup", STYLES["heading"]))
    story.append(
        Paragraph(
            "NBA continuity rankings use the percentage of minutes played by returning players. They also note "
            "that continuity is related to how good a team already was. We use the same returning-minute idea, "
            "then add prior team net rating, prior roster talent, and roster composition. Every model includes "
            "season indicators, and the uncertainty is clustered by team.",
            STYLES["body"],
        )
    )
    story.append(figure(FIGURES / "q2_continuity_confounding.png", 176 * mm))
    story.append(
        Paragraph(
            "Figure 2. Each dot is the estimated net-rating difference linked to 10 percentage points more "
            "continuity. Lines show 95% confidence intervals. An interval crossing zero means the sample does not "
            "separate the estimate from no association.",
            STYLES["caption"],
        )
    )
    raw = q2_results.loc["Continuity only"]
    full = q2_results.loc["Add roster composition"]
    story.append(
        callout(
            f"Main result: The estimate starts at {raw['effect_per_10pp']:+.2f} net-rating points per 10 percentage "
            f"points of continuity. After the controls, it falls to {full['effect_per_10pp']:+.2f} "
            f"(95% CI: {full['ci95_low']:+.2f} to {full['ci95_high']:+.2f}, p = {full['p_value']:.3f}). That is an "
            f"{summary['q2']['percent_shrink']:.0f}% reduction. Much of the raw continuity advantage is really a "
            "talent advantage."
        )
    )
    story.append(
        Paragraph(
            "Limitation: this is still an observational, full-season measure. It cannot prove that keeping a "
            "roster causes better results, and injuries can affect both continuity and performance.",
            STYLES["body"],
        )
    )
    story.append(
        Paragraph(
            "Inspiration: NBA.com, Continuity Rankings, nba.com/news/2025-continuity-rankings. Related definition "
            "and team-clustered modeling: Wang, Sarker, and Hosoi (2025), doi.org/10.1177/15270025251328264.",
            STYLES["source"],
        )
    )

    story.append(PageBreak())
    header(
        story,
        "Q3 extension",
        "What actually changes in playoff scoring?",
        "Points per game can change because playing time changes or because scoring rate changes.",
    )
    story.append(Paragraph("Idea and setup", STYLES["heading"]))
    story.append(
        Paragraph(
            "Earlier playoff comparisons often use per-minute efficiency measures so that extra playing time is "
            "not mistaken for better performance. We make that idea visual. For every player-season, the change "
            "in points per game is split exactly into a minutes effect and a points-per-minute effect. We keep "
            "players with at least 20 regular-season games and 5 playoff games.",
            STYLES["body"],
        )
    )
    story.append(figure(FIGURES / "q3_playoff_scoring_decomposition.png", 176 * mm))
    story.append(
        Paragraph(
            "Figure 3. The first bar shows the scoring change caused by playoff minutes. The second bar adds the "
            "change in scoring rate. The diamond is the final change in points per game, with a player-bootstrap "
            "95% confidence interval.",
            STYLES["caption"],
        )
    )
    stars = q3_results.loc["Star"]
    story.append(
        callout(
            f"Main result: Across {summary['q3']['n_player_seasons']:,} player-seasons, scoring falls by "
            f"{abs(summary['q3']['mean_net_change']):.2f} points per game on average. Most of the drop comes from "
            f"the scoring-rate effect ({summary['q3']['mean_scoring_rate_effect']:.2f}). Stars are different: "
            f"extra minutes add {stars['minutes_effect']:+.2f} points, while the lower scoring rate removes "
            f"{abs(stars['scoring_rate_effect']):.2f}. The effects almost cancel, leaving only "
            f"{stars['net_change']:+.2f} points per game."
        )
    )
    story.append(
        Paragraph(
            "Limitation: playoff opponents are stronger and playoff samples are shorter. The chart describes the "
            "change but does not identify whether defense, role, health, or coaching caused it.",
            STYLES["body"],
        )
    )
    story.append(
        Paragraph(
            "Inspiration: Game statistics that discriminate winning and losing at the NBA level, "
            "pmc.ncbi.nlm.nih.gov/articles/PMC9390892. Related per-minute playoff comparison: FiveThirtyEight, "
            "LeBron Doesn't Get Better in the Playoffs. He's Always This Good.",
            STYLES["source"],
        )
    )

    output_path = PDF / "research_inspired_extensions.pdf"
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        title="Research-Inspired NBA Extensions",
        author="Yonatan Gan, Ariel Mersel, Nimrod Segev",
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=13 * mm,
        bottomMargin=18 * mm,
    )
    doc.build(story, onFirstPage=page_chrome, onLaterPages=page_chrome)
    print(f"Built {output_path}")
    return output_path


if __name__ == "__main__":
    build()
