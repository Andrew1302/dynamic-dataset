"""CLI entry point for the dynamic dataset generator.

Usage::

    python src/dataset-generator/multimodal.py          # 10 samples, all tasks
    python src/dataset-generator/multimodal.py -n 50    # 50 samples
    python src/dataset-generator/multimodal.py --tasks mst shortest_path
    python src/dataset-generator/multimodal.py --pdf --one-per-type  # PDF with one of each task
"""

from __future__ import annotations

import argparse
import io
import os
import random
import sys

import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# Support running as a script: python src/dataset-generator/multimodal.py
if __name__ == "__main__" and __package__ is None:
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))
    __package__ = "src.dataset-generator"
    # Hyphens in package names aren't importable; use importlib to bootstrap
    import importlib

    importlib.import_module(__package__)

from . import graph_generator as gg  # noqa: E402
from .tasks import get_all_tasks  # noqa: E402

# ---------------------------------------------------------------------------
# Which graph generator to use per task
# ---------------------------------------------------------------------------
_TASK_GRAPH_BUILDERS: dict[str, callable] = {
    # tasks requiring weighted connected graphs
    "mst": gg.random_weighted_connected_graph,
    "shortest_path": lambda n: gg.add_random_weights(gg.random_connected_graph(n)),
    # tasks that benefit from possibly-disconnected graphs
    "connectivity_check": gg.random_possibly_disconnected,
    "connected_components": gg.random_possibly_disconnected,
    "reachability": gg.random_possibly_disconnected,
    # max-flow needs a directed weighted graph
    "maximum_flow": gg.random_directed_weighted_graph,
}

# Default builder for tasks not listed above
_DEFAULT_GRAPH_BUILDER = gg.random_graph


def _build_graph_for_task(task_name: str, n: int):
    builder = _TASK_GRAPH_BUILDERS.get(task_name, _DEFAULT_GRAPH_BUILDER)
    return builder(n)


def _sample_flowables(
    idx: int,
    task_name: str,
    prompt: str,
    image: object,
    answer: str,
    styles: object,
    img_width: float,
) -> list:
    """Build flowables for one sample (title, image, prompt, answer)."""
    flowables = []
    flowables.append(
        Paragraph(
            f"<b>Sample {idx} — {task_name}</b>",
            styles["Heading3"],
        )
    )
    flowables.append(Spacer(1, 0.08 * inch))

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    img_height = img_width * image.height / image.width
    flowables.append(RLImage(buf, width=img_width, height=img_height))
    flowables.append(Spacer(1, 0.08 * inch))

    flowables.append(Paragraph("<b>Prompt</b>", styles["Normal"]))
    prompt_escaped = prompt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
    flowables.append(Paragraph(f"<font size='8'>{prompt_escaped}</font>", styles["Normal"]))
    flowables.append(Spacer(1, 0.05 * inch))
    flowables.append(Paragraph("<b>Answer</b>", styles["Normal"]))
    answer_escaped = str(answer).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    flowables.append(Paragraph(f"<font size='8'>{answer_escaped}</font>", styles["Normal"]))
    return flowables


def _write_pdf(
    samples: list[tuple[str, str, object, str]],
    output_path: str,
    img_width: float = 2.6 * inch,
    samples_per_page: int = 2,
) -> None:
    """Write a PDF with image, prompt, answer; two samples per page, smaller images."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )
    styles = getSampleStyleSheet()
    story = []

    # Two columns per page; ~3.35 inch each to fit A4 with margins
    col_width_pt = 3.35 * inch

    for i in range(0, len(samples), samples_per_page):
        pair = samples[i : i + samples_per_page]
        cells = []
        for j, (task_name, prompt, image, answer) in enumerate(pair):
            idx = i + j + 1
            flowables = _sample_flowables(
                idx, task_name, prompt, image, answer, styles, img_width=img_width
            )
            cells.append(flowables)

        if len(cells) == 1:
            cells.append([Spacer(1, 0.1 * inch)])  # empty cell so table has 2 columns

        t = Table(
            [cells],
            colWidths=[col_width_pt, col_width_pt],
        )
        t.setStyle(
            TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (0, -1), 6),
                ("RIGHTPADDING", (1, 0), (1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        story.append(t)
        story.append(Spacer(1, 0.2 * inch))

    doc.build(story)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multimodal graph QA samples."
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to generate.",
    )
    parser.add_argument(
        "-o", "--output-dir", default="output_samples", help="Output directory."
    )
    parser.add_argument("--seed", type=int, default=12227, help="Random seed.")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Subset of task names to use (default: all).",
    )
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large", "all"],
        default="small",
        help="Graph size preset. Use 'all' to randomize size per sample.",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Generate a single PDF with image, prompt, and answer for each sample.",
    )
    parser.add_argument(
        "--one-per-type",
        action="store_true",
        help="Generate one sample per task type (ignores -n). Use with --pdf for one PDF with one of each graph problem.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    all_tasks = get_all_tasks()

    if args.tasks:
        task_names = args.tasks
        for t in task_names:
            if t not in all_tasks:
                raise ValueError(f"Unknown task {t!r}. Available: {sorted(all_tasks)}")
    else:
        task_names = list(all_tasks.keys())

    if args.one_per_type:
        num_samples = len(task_names)
        task_sequence = list(task_names)
        print(f"Generating one sample per task ({num_samples} samples) …")
    else:
        num_samples = args.num_samples
        task_sequence = [random.choice(task_names) for _ in range(num_samples)]
        print(f"Generating {num_samples} samples across {len(task_names)} tasks …")

    print(f"Tasks: {', '.join(sorted(task_names))}")
    print(f"Size: {args.size}")

    pdf_samples: list[tuple[str, str, object, str]] = []

    for i in range(num_samples):
        task_name = task_sequence[i]
        task = all_tasks[task_name]()

        n = gg.random_node_count(args.size)
        G = _build_graph_for_task(task_name, n)

        sample = task.generate(G)

        prompt = sample["prompt"]
        image = sample["image"]
        answer = sample["answer"]

        img_path = os.path.join(args.output_dir, f"sample_{i + 1}.png")
        image.save(img_path, dpi=(120, 120))

        if args.pdf:
            pdf_samples.append((task_name, prompt, image, answer))

        print(f"\n--- Sample {i + 1} [{task_name}] ---")
        print(prompt)
        print(f" {answer}")

    if args.pdf and pdf_samples:
        pdf_path = os.path.join(args.output_dir, "samples.pdf")
        _write_pdf(pdf_samples, pdf_path)
        print(f"\nPDF written to {pdf_path}")


if __name__ == "__main__":
    main()
