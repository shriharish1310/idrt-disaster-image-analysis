"""Build the polished IDRT analysis report PDF from generated results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from PIL import Image as PILImage

NAVY = colors.HexColor("#17324d")
TEAL = colors.HexColor("#1f6f8b")
GOLD = colors.HexColor("#e09f3e")
RED = colors.HexColor("#9e2a2b")
CREAM = colors.HexColor("#f6f4ef")
LIGHT_BLUE = colors.HexColor("#e8f1f5")
TEXT = colors.HexColor("#263238")
MUTED = colors.HexColor("#5f6b73")


def image_at_width(path: Path, width: float) -> Image:
    image = Image(str(path))
    ratio = image.imageHeight / image.imageWidth
    image.drawWidth = width
    image.drawHeight = width * ratio
    return image


def prepare_report_image(source: Path, destination: Path, max_width_px: int = 1800) -> Path:
    """Downsample and JPEG-compress a figure for an email-friendly PDF."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with PILImage.open(source) as image:
        image = image.convert("RGB")
        if image.width > max_width_px:
            height = round(image.height * max_width_px / image.width)
            image = image.resize((max_width_px, height), PILImage.Resampling.LANCZOS)
        image.save(destination, format="JPEG", quality=88, optimize=True, progressive=True)
    return destination


def add_page_decor(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, letter[1] - 0.24 * inch, letter[0], 0.24 * inch, fill=1, stroke=0)
    canvas.setStrokeColor(colors.HexColor("#cbd5dc"))
    canvas.line(0.58 * inch, 0.47 * inch, letter[0] - 0.58 * inch, 0.47 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.58 * inch, 0.29 * inch, "IDRT disaster image analysis | August 25, 2026")
    canvas.drawRightString(letter[0] - 0.58 * inch, 0.29 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCustom",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCustom",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            textColor=MUTED,
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "Heading1Custom",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=NAVY,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "Heading2Custom",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=TEAL,
            spaceBefore=7,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.4,
            leading=13.2,
            textColor=TEXT,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "SmallCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=MUTED,
            spaceAfter=3,
        ),
        "callout": ParagraphStyle(
            "CalloutCustom",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=14,
            textColor=NAVY,
            backColor=LIGHT_BLUE,
            borderColor=TEAL,
            borderWidth=0.8,
            borderPadding=9,
            spaceBefore=5,
            spaceAfter=9,
        ),
        "caption": ParagraphStyle(
            "CaptionCustom",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=7.8,
            leading=10,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=6,
        ),
    }


def metric_table(results: list[dict], styles: dict[str, ParagraphStyle]) -> Table:
    header = ["Pair", "Coverage", "SSIM", "DINOv2", "Candidate area"]
    short_names = {"maxar": "Maxar", "noaa": "NOAA", "sentinel2": "Sentinel-2"}
    rows = [header]
    for result in results:
        rows.append(
            [
                short_names[result["pair"]],
                f"{100 * result['analysis_coverage']:.1f}%",
                f"{result['ssim']:.3f}",
                f"{result['dinov2_cosine']:.3f}",
                f"{100 * result['high_change_fraction']:.1f}%",
            ]
        )
    table = Table(rows, colWidths=[1.45 * inch, 1.15 * inch, 0.85 * inch, 0.95 * inch, 1.25 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.7),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CREAM]),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5dc")),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def pair_page(story: list, result: dict, figure: Path, styles: dict[str, ParagraphStyle]) -> None:
    interpretations = {
        "maxar": (
            "The broadest appearance change occurs in this pair. Differences follow the beach and dune system, "
            "nearshore water, canals, and many built-feature edges. The map is useful for prioritization, but the "
            "11-month interval mixes possible storm effects with season, tide, water turbidity, view geometry, and "
            "processing differences. A closer-date pre-event Maxar image is needed before damage attribution."
        ),
        "noaa": (
            "This is the strongest controlled comparison: both images are NOAA products and only 11 days apart. "
            "Candidates are concentrated along the Gulf shoreline and beach, with sparse detections over developed "
            "areas. Water tone and texture also differ, so manual or labeled validation remains necessary before "
            "assigning changes specifically to Milton."
        ),
        "sentinel2": (
            "Broad structure is stable where both dates are usable, but this is a partial-coverage result. The RGB "
            "screen excludes 55.9% of the canvas and leaves only 10 usable DINOv2 tiles. The low candidate fraction "
            "therefore means limited detected change in the visible subset; it does not imply no damage across the "
            "full scene. Original multispectral bands and the Scene Classification Layer are needed next."
        ),
    }
    story.append(Paragraph(result["label"], styles["h1"]))
    metrics = (
        f"<b>Coverage:</b> {100 * result['analysis_coverage']:.1f}% &nbsp;&nbsp; "
        f"<b>SSIM:</b> {result['ssim']:.3f} &nbsp;&nbsp; "
        f"<b>DINOv2:</b> {result['dinov2_cosine']:.3f} &nbsp;&nbsp; "
        f"<b>Candidate area:</b> {100 * result['high_change_fraction']:.1f}%"
    )
    story.append(Paragraph(metrics, styles["callout"]))
    story.append(image_at_width(figure, 6.25 * inch))
    story.append(
        Paragraph(
            "Figure: supplied previews, continuous change score, and high-confidence candidates. Black or gray "
            "regions in analysis panels are excluded no-data/cloud-screened pixels.",
            styles["caption"],
        )
    )
    story.append(Paragraph("Interpretation", styles["h2"]))
    story.append(Paragraph(interpretations[result["pair"]], styles["body"]))


def build(project_root: Path, output_path: Path) -> None:
    results_path = project_root / "outputs" / "metrics" / "analysis_results.json"
    with results_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload["results"]
    by_key = {result["pair"]: result for result in results}
    figures = project_root / "outputs" / "figures"
    report_assets = project_root / "tmp" / "pdfs" / "report_assets"
    embedded_figures = {
        "overview": prepare_report_image(
            figures / "metrics_overview.png", report_assets / "metrics_overview.jpg"
        ),
        **{
            key: prepare_report_image(
                figures / f"{key}_comparison.png", report_assets / f"{key}_comparison.jpg"
            )
            for key in ("maxar", "noaa", "sentinel2")
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.62 * inch,
        leftMargin=0.62 * inch,
        topMargin=0.48 * inch,
        bottomMargin=0.62 * inch,
        title="Satellite Image Similarity and Change Screening",
        author="Shri Harish Saravanan",
        subject="IDRT research application image comparison",
    )
    styles = build_styles()
    story: list = []

    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph("Satellite Image Similarity and Change Screening", styles["title"]))
    story.append(
        Paragraph(
            "Institute for a Disaster Resilient Texas research application | Treasure Island / Tampa Bay, Florida | August 25, 2026",
            styles["subtitle"],
        )
    )
    story.append(
        Paragraph(
            "I compared three supplied satellite-image pairs with a reproducible, quality-controlled workflow "
            "combining preview masks, registration diagnostics, classical image metrics, DINOv2 perceptual features, "
            "and spatial change-candidate maps.",
            styles["callout"],
        )
    )
    story.append(Paragraph("Executive summary", styles["h1"]))
    story.append(metric_table(results, styles))
    story.append(Spacer(1, 0.12 * inch))
    story.append(image_at_width(embedded_figures["overview"], 7.0 * inch))
    story.append(
        Paragraph(
            "Scores are not directly interchangeable across sensors and scales. Candidate area is the fraction of "
            "analyzed pixels above the empirical 0.65 fused-score threshold.",
            styles["caption"],
        )
    )
    story.append(
        Paragraph(
            "The NOAA pair is the cleanest short-interval comparison. Maxar shows scene-wide appearance change but "
            "has stronger temporal and radiometric confounding. Sentinel-2 appears structurally stable in its visible "
            "subset, but cloud screening limits analysis to 42.8% of the canvas.",
            styles["body"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Method", styles["h1"]))
    story.append(Paragraph("1. Quality control and registration", styles["h2"]))
    story.append(
        Paragraph(
            "Near-black preview borders were excluded. Sentinel-2 received a conservative RGB cloud/cloud-shadow "
            "screen. Phase correlation on Laplacian edge maps found residual translations below one pixel for all "
            "pairs, so the existing common canvas was retained without another warp.",
            styles["body"],
        )
    )
    story.append(Paragraph("2. Similarity measures", styles["h2"]))
    story.append(
        Paragraph(
            "Per-channel median/IQR matching reduces global radiometric bias. MAE, RMSE, PSNR, masked SSIM, and "
            "gradient correlation characterize pixel and edge agreement. DINOv2 ViT-S/14 embeddings are computed "
            "over a 6 x 6 grid; tiles with less than 55% usable coverage are excluded, and matching-tile cosine "
            "similarities are averaged.",
            styles["body"],
        )
    )
    story.append(Paragraph("3. Spatial change score", styles["h2"]))
    story.append(
        Paragraph(
            "The continuous score fuses 55% normalized CIE Lab color difference and 45% normalized SSIM "
            "dissimilarity. Morphologically cleaned pixels above 0.65 are visualized as high-confidence candidates. "
            "The threshold is empirical and has not been calibrated against labeled damage.",
            styles["body"],
        )
    )
    story.append(Paragraph("Critical interpretation limits", styles["h2"]))
    limits = [
        "Candidate change is not confirmed disaster damage.",
        "The PNG previews contain no CRS, pixel size, provider QA band, or full acquisition metadata.",
        "Tide, water turbidity, illumination, season, view geometry, and processing can change similarity scores.",
        "Cross-pair rankings are descriptive because sensor resolution, time interval, and usable coverage differ.",
        "The Sentinel-2 RGB cloud mask is a conservative preview heuristic, not the official Scene Classification Layer.",
    ]
    for item in limits:
        story.append(Paragraph(f"- {item}", styles["body"]))
    story.append(Paragraph("Metric details", styles["h2"]))
    detail_rows = [["Pair", "MAE", "PSNR (dB)", "Gradient corr.", "Residual shift (x, y)"]]
    pair_display_names = {"maxar": "Maxar", "noaa": "NOAA", "sentinel2": "Sentinel-2"}
    for key in ("maxar", "noaa", "sentinel2"):
        result = by_key[key]
        detail_rows.append(
            [
                pair_display_names[key],
                f"{result['mae']:.3f}",
                f"{result['psnr_db']:.2f}",
                f"{result['gradient_correlation']:.3f}",
                f"({result['residual_shift_x_px']:.2f}, {result['residual_shift_y_px']:.2f}) px",
            ]
        )
    detail_table = Table(detail_rows, colWidths=[1.2 * inch, 0.8 * inch, 1.0 * inch, 1.1 * inch, 1.55 * inch])
    detail_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.3),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5dc")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CREAM]),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(detail_table)

    for key in ("maxar", "noaa", "sentinel2"):
        story.append(PageBreak())
        pair_page(story, by_key[key], embedded_figures[key], styles)

    story.append(PageBreak())
    story.append(Paragraph("Conclusions and recommended next steps", styles["h1"]))
    story.append(
        Paragraph(
            "The three signals are complementary: SSIM captures local structure, DINOv2 supplies a learned "
            "perceptual comparison, and the fused maps prioritize spatial review. The NOAA pair is the most "
            "defensible comparison for storm-to-storm change. Maxar supplies valuable object-level detail but needs "
            "a closer-date baseline. Sentinel-2 supplies regional context but is cloud-limited.",
            styles["callout"],
        )
    )
    recommendations = [
        "Acquire original georeferenced imagery, acquisition metadata, and provider QA layers.",
        "Use Sentinel-2 Scene Classification and cloud-free compositing instead of an RGB preview heuristic.",
        "Match date, tide, sun/view geometry, and sensor where possible; normalize resolution before comparison.",
        "Segment buildings, roads, vegetation, water, and shoreline so change thresholds can be class-specific.",
        "Create reviewed damaged/unchanged labels and report precision-recall, F1, spatial cross-validation, and uncertainty.",
        "Compare DINOv2 with CLIP and DreamSim on the labeled validation set rather than selecting a model by raw similarity alone.",
    ]
    for index, recommendation in enumerate(recommendations, start=1):
        story.append(Paragraph(f"<b>{index}.</b> {recommendation}", styles["body"]))

    story.append(Paragraph("References", styles["h1"]))
    references = [
        "Oquab, M. et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. https://arxiv.org/abs/2304.07193",
        "Wang, Z., Bovik, A. C., Sheikh, H. R., and Simoncelli, E. P. (2004). Image Quality Assessment: From Error Visibility to Structural Similarity. IEEE TIP 13(4), 600-612. https://ece.uwaterloo.ca/~z70wang/publications/ssim.html",
        "OpenCV. phaseCorrelate: Motion Analysis and Object Tracking. https://docs.opencv.org/4.x/d7/df3/group__imgproc__motion.html",
        "ESA. Sentinel-2 Level-2A Algorithm Theoretical Basis Document (Sen2Cor). https://step.esa.int/thirdparties/sen2cor/2.10.0/docs/S2-PDGS-MPC-L2A-ATBD-V2.10.0.pdf",
    ]
    for index, reference in enumerate(references, start=1):
        story.append(Paragraph(f"{index}. {reference}", styles["small"]))

    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            "Reproducible repository: https://github.com/shriharish1310/idrt-disaster-image-analysis",
            styles["callout"],
        )
    )

    doc.build(story, onFirstPage=add_page_decor, onLaterPages=add_page_decor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.project_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
