"""Render each task under every ablation variant into a side-by-side PDF.

Useful for eyeballing the ablation surface before committing to a long
evaluation run: confirms that lettered labels are readable, that the
"none" condition doesn't break the visual flow, and that the
straight-edge default doesn't introduce label collisions for the
weighted shortest-path task.

Usage:
    PYTHONPATH=. uv run python scripts/preview_ablations.py
    PYTHONPATH=. uv run python scripts/preview_ablations.py --difficulty hard
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
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

from src.benchmark import RenderConfig, get_all_tasks


ABLATIONS: list[tuple[str, RenderConfig, bool]] = [
    ("baseline (numeric, straight, no matrix)",
     RenderConfig(label_style="numeric", edge_style="straight"), False),
    ("letters",
     RenderConfig(label_style="letters", edge_style="straight"), False),
    ("no labels",
     RenderConfig(label_style="none", edge_style="straight"), False),
    ("alt color (#F1948A)",
     RenderConfig(label_style="numeric", edge_style="straight",
                  node_color="#F1948A"), False),
    ("adjacency matrix",
     RenderConfig(label_style="numeric", edge_style="straight"), True),
    ("curved edges (legacy)",
     RenderConfig(label_style="numeric", edge_style="curved"), False),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--difficulty", default="medium")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", default="out/preview_ablations.pdf")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    styles = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=styles["Heading2"], spaceAfter=4)
    cap = ParagraphStyle("c", parent=styles["BodyText"], fontSize=8, leading=10)
    body = ParagraphStyle("b", parent=styles["BodyText"], fontSize=8, leading=10)

    doc = SimpleDocTemplate(
        args.output,
        pagesize=landscape(A4),
        leftMargin=1 * cm, rightMargin=1 * cm,
        topMargin=1 * cm, bottomMargin=1 * cm,
    )

    story: list = []
    tasks = sorted(get_all_tasks().keys())
    for task_name in tasks:
        story.append(Paragraph(f"Task: <b>{task_name}</b> &nbsp; (difficulty={args.difficulty}, seed={args.seed})", title))
        story.append(Spacer(1, 0.2 * cm))

        cells: list[list] = []
        # Lay out two rows × three columns per page.
        ncols = 3
        row: list = []
        cap_row: list = []
        for label, cfg, adj in ABLATIONS:
            task = get_all_tasks()[task_name]()
            sample = task.generate(
                seed=args.seed,
                difficulty=args.difficulty,
                config=cfg,
                include_adjacency_matrix=adj,
            )
            img = _pil_to_flowable(sample["direct_image"], target_w=7 * cm)
            row.append(img)
            note = label
            if adj:
                note += f"<br/><font size='6'>{_escape(sample['direct_prompt'])[:200]}</font>"
            cap_row.append(Paragraph(note, cap))
            if len(row) == ncols:
                cells.append(row)
                cells.append(cap_row)
                row, cap_row = [], []
        if row:
            while len(row) < ncols:
                row.append(Paragraph("", body))
                cap_row.append(Paragraph("", body))
            cells.append(row)
            cells.append(cap_row)

        tbl = Table(cells, colWidths=[8 * cm] * ncols)
        tbl.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(tbl)
        story.append(PageBreak())

    doc.build(story)
    print(f"Wrote {args.output}")
    return 0


def _pil_to_flowable(img: PILImage.Image, target_w: float) -> Image:
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    w, h = img.size
    aspect = h / w
    return Image(buf, width=target_w, height=target_w * aspect)


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace("\n", "<br/>"))


if __name__ == "__main__":
    raise SystemExit(main())
