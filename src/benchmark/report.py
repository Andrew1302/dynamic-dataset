"""PDF aggregator: one sample per page with direct and disguise side-by-side."""

from __future__ import annotations

from io import BytesIO
from typing import Iterable

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .base import Sample


_MARGIN = 1.5 * cm
_IMG_W = 8 * cm
_IMG_W_COMPACT = 7 * cm


def build_pdf(
    samples: Iterable[tuple[str, str, int, Sample]],
    output_path: str,
) -> None:
    """Write a PDF report.

    Parameters
    ----------
    samples
        Iterable of ``(task_name, difficulty, seed, sample)`` tuples. The
        order is preserved in the PDF.
    output_path
        Destination path for the PDF.
    """
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "title", parent=styles["Heading2"], spaceAfter=6, textColor=colors.HexColor("#1B2631"),
    )
    label_style = ParagraphStyle(
        "label", parent=styles["BodyText"], fontSize=9, textColor=colors.HexColor("#566573"),
        spaceBefore=4, spaceAfter=2,
    )
    prompt_style = ParagraphStyle(
        "prompt", parent=styles["BodyText"], fontSize=10, leading=13,
    )
    answer_style = ParagraphStyle(
        "answer", parent=styles["BodyText"], fontSize=11, leading=14,
        textColor=colors.HexColor("#1E8449"),
    )

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=_MARGIN,
        rightMargin=_MARGIN,
        topMargin=_MARGIN,
        bottomMargin=_MARGIN,
    )

    story: list = []
    samples = list(samples)
    story.append(Paragraph("Graph-Disguise Benchmark Report", styles["Title"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(f"{len(samples)} samples", styles["BodyText"]))
    story.append(PageBreak())

    for i, (task_name, difficulty, seed, sample) in enumerate(samples, start=1):
        story.append(
            Paragraph(
                f"Sample {i} — task: <b>{task_name}</b> · difficulty: {difficulty} · seed: {seed}",
                title_style,
            )
        )

        direct_img = _pil_to_flowable(sample["direct_image"])
        disguise_img = _pil_to_flowable(sample["disguise_image"])

        img_row = Table(
            [[direct_img, disguise_img]],
            colWidths=[_IMG_W, _IMG_W],
            hAlign="CENTER",
        )
        img_row.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(img_row)

        label_row = Table(
            [
                [
                    Paragraph("<b>Direct prompt</b>", label_style),
                    Paragraph("<b>Disguise prompt</b>", label_style),
                ],
                [
                    Paragraph(_escape(sample["direct_prompt"]), prompt_style),
                    Paragraph(_escape(sample["disguise_prompt"]), prompt_style),
                ],
            ],
            colWidths=[_IMG_W, _IMG_W],
            hAlign="CENTER",
        )
        label_row.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(label_row)
        story.append(Spacer(1, 0.3 * cm))
        story.append(
            Paragraph(
                f"<b>Ground-truth answer:</b> {_escape(sample['answer'])}",
                answer_style,
            )
        )
        story.append(PageBreak())

    doc.build(story)


def _pil_to_flowable(img: PILImage.Image, img_w: float = _IMG_W) -> Image:
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    w, h = img.size
    target_w = img_w - 0.4 * cm
    aspect = h / w
    return Image(buf, width=target_w, height=target_w * aspect)


def build_pdf_compact(
    samples: Iterable[tuple[str, str, int, Sample]],
    output_path: str,
) -> None:
    """Write a PDF report packed two samples per page.

    Same per-sample content as :func:`build_pdf` (title, side-by-side
    images, side-by-side prompts, ground-truth answer) at a smaller scale
    so two samples fit on a single A4 page. A horizontal rule separates
    the two samples on a page.
    """
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "title_compact", parent=styles["Heading3"], spaceAfter=4,
        textColor=colors.HexColor("#1B2631"),
    )
    label_style = ParagraphStyle(
        "label_compact", parent=styles["BodyText"], fontSize=8,
        textColor=colors.HexColor("#566573"), spaceBefore=2, spaceAfter=1,
    )
    prompt_style = ParagraphStyle(
        "prompt_compact", parent=styles["BodyText"], fontSize=8, leading=10,
    )
    answer_style = ParagraphStyle(
        "answer_compact", parent=styles["BodyText"], fontSize=9, leading=11,
        textColor=colors.HexColor("#1E8449"),
    )

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=_MARGIN,
        rightMargin=_MARGIN,
        topMargin=_MARGIN,
        bottomMargin=_MARGIN,
    )

    samples = list(samples)
    story: list = []
    story.append(Paragraph("Graph-Disguise Benchmark Report", styles["Title"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        Paragraph(
            f"{len(samples)} samples · two questions per page",
            styles["BodyText"],
        )
    )
    story.append(PageBreak())

    img_w = _IMG_W_COMPACT
    for i, (task_name, difficulty, seed, sample) in enumerate(samples, start=1):
        story.append(
            Paragraph(
                f"Sample {i} — task: <b>{task_name}</b> · "
                f"difficulty: {difficulty} · seed: {seed}",
                title_style,
            )
        )

        direct_img = _pil_to_flowable(sample["direct_image"], img_w=img_w)
        disguise_img = _pil_to_flowable(sample["disguise_image"], img_w=img_w)
        img_row = Table(
            [[direct_img, disguise_img]],
            colWidths=[img_w, img_w],
            hAlign="CENTER",
        )
        img_row.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(img_row)

        label_row = Table(
            [
                [
                    Paragraph("<b>Direct prompt</b>", label_style),
                    Paragraph("<b>Disguise prompt</b>", label_style),
                ],
                [
                    Paragraph(_escape(sample["direct_prompt"]), prompt_style),
                    Paragraph(_escape(sample["disguise_prompt"]), prompt_style),
                ],
            ],
            colWidths=[img_w, img_w],
            hAlign="CENTER",
        )
        label_row.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(label_row)
        story.append(Spacer(1, 0.15 * cm))
        story.append(
            Paragraph(
                f"<b>Ground-truth answer:</b> {_escape(sample['answer'])}",
                answer_style,
            )
        )

        if i % 2 == 0:
            story.append(PageBreak())
        else:
            # Separator between two samples on the same page.
            story.append(Spacer(1, 0.25 * cm))
            sep = Table(
                [[""]],
                colWidths=[2 * img_w],
                rowHeights=[0.4],
            )
            sep.setStyle(
                TableStyle(
                    [
                        ("LINEBELOW", (0, 0), (-1, -1), 0.4,
                         colors.HexColor("#BDC3C7")),
                    ]
                )
            )
            story.append(sep)
            story.append(Spacer(1, 0.25 * cm))

    doc.build(story)


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )
