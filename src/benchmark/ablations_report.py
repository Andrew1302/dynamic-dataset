"""Single-PDF ablation overview report.

For each visual ablation defined in
``dynamic-lmms-eval/remote_execution_scripts/jobs/generate_graph_benchmark_jobs.py``
we re-render the same per-task graph (same seed, same difficulty) and lay
out direct + disguise images side-by-side so the visual effect of the
ablation can be inspected at a glance.

Non-visual ablations (thinking-mode, model-size) only swap the model;
they don't change the rendered images, so the PDF lists them as text
rather than re-rendering. Sweeps over node/edge counts are a separate
axis (graph shape) and are shown as a small filmstrip per task.

Run with::

    uv run python -m benchmark.ablations_report
    uv run python -m benchmark.ablations_report -o out/ablations.pdf
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from io import BytesIO
from typing import Iterable

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

from .base import Sample, get_task
from .rendering import RenderConfig


# Tasks used by the benchmark (matches TASKS in generate_graph_benchmark_jobs.py).
_TASKS: tuple[str, ...] = ("coloring", "directed_connectivity", "shortest_path")

# Same difficulty defaults as the standard run.
_DIFFICULTY = "medium"
_DIFFICULTY_OVERRIDES = {"shortest_path": "easy"}

# Fixed seed so re-runs are deterministic and ablations are comparable.
_SEED = 42

# Sweep filmstrip: a few node counts spanning the documented range 3..14.
_SWEEP_NODE_VALUES: tuple[int, ...] = (4, 7, 10, 13)


@dataclass(frozen=True)
class VisualAblation:
    """A render-side ablation: one RenderConfig + adjacency-matrix flag."""

    key: str
    title: str
    description: str
    config: RenderConfig
    include_adjacency_matrix: bool = False


_BASELINE_CONFIG = RenderConfig(
    label_style="numeric",
    node_color="#AED6F1",
    edge_style="straight",
)

_VISUAL_ABLATIONS: tuple[VisualAblation, ...] = (
    VisualAblation(
        key="standard",
        title="Standard (baseline)",
        description=(
            "Default render settings: numeric labels, light-blue nodes, "
            "straight edges, no adjacency matrix in the prompt."
        ),
        config=_BASELINE_CONFIG,
    ),
    VisualAblation(
        key="labels_letters",
        title="Ablation — labels = letters",
        description="Node labels rendered as letters (A, B, C, …) instead of numbers.",
        config=RenderConfig(
            label_style="letters",
            node_color="#AED6F1",
            edge_style="straight",
        ),
    ),
    VisualAblation(
        key="labels_none",
        title="Ablation — labels = none",
        description="Node labels suppressed on the direct image.",
        config=RenderConfig(
            label_style="none",
            node_color="#AED6F1",
            edge_style="straight",
        ),
    ),
    VisualAblation(
        key="node_color",
        title="Ablation — node colour = #F1948A",
        description="Default node fill swapped from #AED6F1 (blue) to #F1948A (salmon).",
        config=RenderConfig(
            label_style="numeric",
            node_color="#F1948A",
            edge_style="straight",
        ),
    ),
    VisualAblation(
        key="adjmatrix",
        title="Ablation — adjacency matrix in prompt",
        description=(
            "Direct prompt augmented with a text adjacency matrix. The "
            "image is unchanged from baseline — see the prompt snippet."
        ),
        config=_BASELINE_CONFIG,
        include_adjacency_matrix=True,
    ),
)


# Non-visual ablations from generate_graph_benchmark_jobs.py — listed for
# completeness so the report covers every ablation in the suite.
_NON_VISUAL_ABLATIONS: tuple[tuple[str, str], ...] = (
    (
        "thinking-mode",
        "Swaps the eval model to a Qwen3-VL Thinking SKU "
        "(4B + 8B). Render output is unchanged.",
    ),
    (
        "model-size",
        "Runs the 8B panel (Qwen3-VL-8B, InternVL3_5-8B, "
        "LLaVA-OneVision-1.5-8B, MiniCPM-V-2_6, Llama-3.2-11B-Vision, "
        "Qwen3.5-9B). Render output is unchanged.",
    ),
)


# --- Output layout knobs ---------------------------------------------------

_MARGIN = 1.2 * cm
# Three task rows fit comfortably on a landscape A4 page (≈27.7×19 cm usable).
_IMG_W = 6.4 * cm
# Padding for prompt snippets so long prompts wrap rather than overflow.
_PROMPT_W = 6.4 * cm
_PROMPT_MAX_CHARS = 600


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a single PDF showing every benchmark ablation visually."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="out/benchmark/ablations_report.pdf",
        help="Path to the output PDF.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_SEED,
        help="Seed used for the per-task sample (same seed for every ablation).",
    )
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Don't render the node-count filmstrip page.",
    )
    parser.add_argument(
        "--include-prompts",
        action="store_true",
        help=(
            "Include direct-prompt snippets under each image. Off by default "
            "to keep the report compact; turn it on to inspect the "
            "adjacency-matrix ablation."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Generate every (ablation, task) sample upfront. This is bounded and
    # cheap — 5 ablations × 3 tasks = 15 generations.
    print(f"[ablations] seed={args.seed} tasks={_TASKS}")
    rendered: dict[str, dict[str, Sample]] = {}
    for abl in _VISUAL_ABLATIONS:
        rendered[abl.key] = {}
        for task_name in _TASKS:
            difficulty = _DIFFICULTY_OVERRIDES.get(task_name, _DIFFICULTY)
            task = get_task(task_name)()
            sample = task.generate(
                seed=args.seed,
                difficulty=difficulty,
                config=abl.config,
                include_adjacency_matrix=abl.include_adjacency_matrix,
            )
            rendered[abl.key][task_name] = sample
            print(
                f"  [{abl.key:14s}] {task_name:22s} "
                f"v={sample['n_vertices']:>2d} e={sample['n_edges']:>2d} "
                f"answer={sample['answer']!r}"
            )

    sweep_samples: dict[str, list[tuple[int, Sample]]] = {}
    if not args.skip_sweep:
        print(f"[ablations] sweep node counts: {_SWEEP_NODE_VALUES}")
        for task_name in _TASKS:
            sweep_samples[task_name] = []
            task = get_task(task_name)()
            for nc in _SWEEP_NODE_VALUES:
                # Stride seeds so each sweep value gets a distinct graph
                # but the run stays reproducible.
                seed = args.seed + nc * 1009
                sample = task.generate(
                    seed=seed,
                    difficulty=_DIFFICULTY,
                    config=_BASELINE_CONFIG,
                    node_count=nc,
                )
                sweep_samples[task_name].append((nc, sample))
                print(
                    f"  [sweep nodes={nc:2d}] {task_name:22s} "
                    f"v={sample['n_vertices']:>2d} e={sample['n_edges']:>2d}"
                )

    _build_pdf(
        output_path=args.output,
        seed=args.seed,
        rendered=rendered,
        sweep_samples=sweep_samples,
        include_prompts=args.include_prompts,
    )
    print(f"PDF written to {args.output}")
    return 0


def _build_pdf(
    *,
    output_path: str,
    seed: int,
    rendered: dict[str, dict[str, Sample]],
    sweep_samples: dict[str, list[tuple[int, Sample]]],
    include_prompts: bool,
) -> None:
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    h2 = ParagraphStyle(
        "ablation_h2",
        parent=styles["Heading2"],
        spaceAfter=4,
        textColor=colors.HexColor("#1B2631"),
    )
    body = ParagraphStyle(
        "body",
        parent=styles["BodyText"],
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#34495E"),
    )
    caption = ParagraphStyle(
        "caption",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=10,
        textColor=colors.HexColor("#566573"),
        alignment=1,  # center
    )
    answer = ParagraphStyle(
        "answer",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=10,
        textColor=colors.HexColor("#1E8449"),
        alignment=1,
    )
    prompt_style = ParagraphStyle(
        "prompt",
        parent=styles["BodyText"],
        fontSize=7.5,
        leading=9,
        textColor=colors.HexColor("#5D6D7E"),
    )
    col_header = ParagraphStyle(
        "col_header",
        parent=styles["BodyText"],
        fontSize=10,
        leading=12,
        alignment=1,
        textColor=colors.HexColor("#1B2631"),
    )

    doc = SimpleDocTemplate(
        output_path,
        pagesize=landscape(A4),
        leftMargin=_MARGIN,
        rightMargin=_MARGIN,
        topMargin=_MARGIN,
        bottomMargin=_MARGIN,
    )

    story: list = []

    # ---------- Cover page ----------
    story.append(Paragraph("Graph Benchmark — Ablation Visual Report", title_style))
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        Paragraph(
            f"Seed: <b>{seed}</b> · difficulty: <b>medium</b> "
            f"(<b>shortest_path</b>: easy)",
            body,
        )
    )
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Visual ablations covered:", h2))
    bullet_rows = [[Paragraph(f"<b>{a.title}</b>", body),
                    Paragraph(a.description, body)] for a in _VISUAL_ABLATIONS]
    cover_tbl = Table(bullet_rows, colWidths=[6.0 * cm, 18.0 * cm], hAlign="LEFT")
    cover_tbl.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(cover_tbl)
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Non-visual ablations (listed for completeness):", h2))
    nv_rows = [[Paragraph(f"<b>{k}</b>", body), Paragraph(v, body)]
               for k, v in _NON_VISUAL_ABLATIONS]
    nv_tbl = Table(nv_rows, colWidths=[6.0 * cm, 18.0 * cm], hAlign="LEFT")
    nv_tbl.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(nv_tbl)
    story.append(PageBreak())

    # ---------- One page per ablation ----------
    for abl in _VISUAL_ABLATIONS:
        story.append(Paragraph(abl.title, h2))
        story.append(Paragraph(abl.description, body))
        cfg = abl.config
        story.append(
            Paragraph(
                f"<b>RenderConfig:</b> label_style={cfg.label_style} · "
                f"node_color={cfg.node_color} · edge_style={cfg.edge_style} · "
                f"include_adjacency_matrix={abl.include_adjacency_matrix}",
                body,
            )
        )
        story.append(Spacer(1, 0.2 * cm))

        # Header row: blank | direct | disguise
        rows: list[list] = []
        rows.append(
            [
                Paragraph("<b>Task</b>", col_header),
                Paragraph("<b>Direct view</b>", col_header),
                Paragraph("<b>Disguise view</b>", col_header),
            ]
        )
        for task_name in _TASKS:
            sample = rendered[abl.key][task_name]
            direct_img = _pil_to_flowable(sample["direct_image"])
            disguise_img = _pil_to_flowable(sample["disguise_image"])

            task_cell = [
                Paragraph(f"<b>{task_name}</b>", body),
                Spacer(1, 0.1 * cm),
                Paragraph(
                    f"v={sample['n_vertices']} · e={sample['n_edges']}",
                    body,
                ),
                Spacer(1, 0.1 * cm),
                Paragraph(
                    f"<b>Answer:</b> {_escape(sample['answer'])}",
                    answer,
                ),
            ]
            direct_cell: list = [direct_img]
            disguise_cell: list = [disguise_img]
            if include_prompts:
                direct_cell.append(Spacer(1, 0.1 * cm))
                direct_cell.append(
                    Paragraph(
                        _escape(_truncate(sample["direct_prompt"])),
                        prompt_style,
                    )
                )
                disguise_cell.append(Spacer(1, 0.1 * cm))
                disguise_cell.append(
                    Paragraph(
                        _escape(_truncate(sample["disguise_prompt"])),
                        prompt_style,
                    )
                )
            rows.append([task_cell, direct_cell, disguise_cell])

        tbl = Table(
            rows,
            colWidths=[5.0 * cm, _IMG_W + 0.4 * cm, _IMG_W + 0.4 * cm],
            hAlign="CENTER",
        )
        tbl.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D5D8DC")),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F2F4F4")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(tbl)
        story.append(PageBreak())

    # ---------- Sweep page (node-count filmstrip per task) ----------
    if sweep_samples:
        story.append(Paragraph("Sweep ablation — node count", h2))
        story.append(
            Paragraph(
                "Same baseline render settings; the sample's "
                "<i>node_count</i> argument is varied across "
                f"{list(_SWEEP_NODE_VALUES)}. The benchmark sweep job runs "
                "the full range 3..14 at 250 samples per value.",
                body,
            )
        )
        story.append(Spacer(1, 0.25 * cm))

        # Each task gets one row of direct-only thumbnails (disguise is
        # the same idea, omitted for space).
        n_cols = len(_SWEEP_NODE_VALUES)
        thumb_w = (27.0 - 5.0) * cm / n_cols  # fit landscape page minus label col
        for task_name in _TASKS:
            header_row = [Paragraph(f"<b>{task_name}</b>", col_header)] + [
                Paragraph(f"<b>n={nc}</b>", col_header) for nc, _ in sweep_samples[task_name]
            ]
            img_row: list = [Paragraph("direct", body)]
            disguise_row: list = [Paragraph("disguise", body)]
            for _, sample in sweep_samples[task_name]:
                img_row.append(_pil_to_flowable(sample["direct_image"], width=thumb_w))
                disguise_row.append(
                    _pil_to_flowable(sample["disguise_image"], width=thumb_w)
                )

            tbl = Table(
                [header_row, img_row, disguise_row],
                colWidths=[5.0 * cm] + [thumb_w + 0.2 * cm] * n_cols,
                hAlign="CENTER",
            )
            tbl.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D5D8DC")),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F2F4F4")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            story.append(tbl)
            story.append(Spacer(1, 0.25 * cm))

    doc.build(story)


def _pil_to_flowable(img: PILImage.Image, width: float = _IMG_W) -> Image:
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    w, h = img.size
    aspect = h / w
    return Image(buf, width=width, height=width * aspect)


def _truncate(text: str, limit: int = _PROMPT_MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


if __name__ == "__main__":
    sys.exit(main())
