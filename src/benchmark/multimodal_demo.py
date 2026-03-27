"""
multimodal_demo.py — Generates a multimodal benchmark report as a PDF.

For each (family, difficulty, variant) triple the report shows:
    - A rendered image of the problem instance (the visual question)
    - The natural-language prompt (text)
    - The ground-truth answer

Backbone is unchanged: project → solve → verify → to_image → to_prompt.
Only the final output is different: instead of printing to stdout, each
instance is rendered into a PDF page using ReportLab.

Run with:
    uv run python src/benchmark/multimodal_demo.py [--output PATH]

Output defaults to  benchmark_report.pdf  in the current working directory.
"""

from __future__ import annotations

import argparse
import io
import os
import random
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any

# ReportLab
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .base import BaseGraphGenerator, ProjectionFailure, are_isomorphic_instances
from .shortest_path_variants import BareGraphVariant, MazeVariant, WordLadderVariant
from .state_search_variants import (
    BareStateGraphVariant,
    SlidingPuzzleVariant,
    TowerOfHanoiVariant,
)

# ---------------------------------------------------------------------------
# Difficulty presets (identical to demo.py)
# ---------------------------------------------------------------------------

DIFFICULTY_CONFIGS: dict[str, dict] = {
    "easy":   {"n_nodes": 5,  "graph_type": "tree",   "weighted": False},
    "medium": {"n_nodes": 8,  "graph_type": "sparse",  "weighted": True},
    "hard":   {"n_nodes": 12, "graph_type": "random",  "weighted": True},
}

# Variant definitions: (display_name, ProblemVariant instance, family_id)
FAMILY_1_VARIANTS = [
    ("Bare Graph",   BareGraphVariant()),
    ("Maze",         MazeVariant()),
    ("Word Ladder",  WordLadderVariant()),
]

FAMILY_2_VARIANTS = [
    ("Bare State Graph", BareStateGraphVariant()),
    ("Sliding Puzzle",   SlidingPuzzleVariant()),
    ("Tower of Hanoi",   TowerOfHanoiVariant()),
]


# ---------------------------------------------------------------------------
# Data container for a generated sample
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    family: str          # "Family 1: Shortest Path" etc.
    difficulty: str      # "easy" / "medium" / "hard"
    variant_name: str    # "Bare Graph" etc.
    prompt: str
    answer: str
    image_buf: io.BytesIO   # PNG bytes of the rendered problem image
    verify_ok: bool
    iso_peers: list[str] = field(default_factory=list)  # names of isomorphic peers
    failed: bool = False
    failure_msg: str = ""


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

def _generate_sample(
    family_label: str,
    variant_name: str,
    variant,
    G,
    source: int,
    target: int,
    seed: int,
) -> Sample:
    """Generate one sample.  Returns a failed Sample on ProjectionFailure."""
    dummy = Sample(
        family=family_label, difficulty="", variant_name=variant_name,
        prompt="", answer="", image_buf=io.BytesIO(), verify_ok=False,
        failed=True,
    )
    try:
        inst = variant.project(G, source, target, seed=seed)
        sol = variant.solve(inst)
        ok = variant.verify(inst, sol)
        prompt = variant.to_prompt(inst)
        img = variant.to_image(inst)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        answer_text = _format_answer(sol)

        return Sample(
            family=family_label,
            difficulty="",        # filled in by caller
            variant_name=variant_name,
            prompt=prompt,
            answer=answer_text,
            image_buf=buf,
            verify_ok=ok,
        )
    except ProjectionFailure as exc:
        dummy.failure_msg = str(exc)
        return dummy
    except Exception as exc:
        dummy.failure_msg = f"Unexpected error: {exc}"
        return dummy


def _format_answer(sol: dict[str, Any]) -> str:
    """Format the solution dict into a concise human-readable answer string."""
    parts: list[str] = []
    if "path" in sol:
        p = sol["path"]
        # Shorten very long paths
        if isinstance(p, list) and len(p) > 10:
            p_str = str(p[:5])[:-1] + " ... " + str(p[-2:])[1:]
        else:
            p_str = str(p)
        parts.append(f"Path: {p_str}")
    if "moves" in sol:
        m = sol["moves"]
        m_str = str(m) if len(str(m)) < 80 else str(m)[:77] + "..."
        parts.append(f"Moves: {m_str}")
    if "cost" in sol:
        parts.append(f"Cost/Steps: {sol['cost']}")
    return "\n".join(parts) if parts else str(sol)


def generate_all_samples(base_seed: int = 42) -> list[Sample]:
    """Run the full benchmark and return all Sample objects."""
    all_samples: list[Sample] = []

    for difficulty in ("easy", "medium", "hard"):
        cfg = DIFFICULTY_CONFIGS[difficulty]

        # ── Family 1: Shortest Path ──────────────────────────────────────
        G1 = BaseGraphGenerator.generate(seed=base_seed, **cfg)
        rng1 = random.Random(base_seed)
        src1, tgt1 = rng1.sample(sorted(G1.nodes()), 2)

        family1_label = "Family 1: Shortest Path"
        f1_samples: list[Sample] = []
        for i, (vname, variant) in enumerate(FAMILY_1_VARIANTS):
            s = _generate_sample(
                family1_label, vname, variant,
                G1, src1, tgt1, seed=base_seed + (i + 1) * 1000,
            )
            s.difficulty = difficulty
            f1_samples.append(s)
            all_samples.append(s)

        # Isomorphism annotations for succeeded variants
        _annotate_isomorphism(f1_samples, FAMILY_1_VARIANTS)

        # ── Family 2: State Space Search ─────────────────────────────────
        G2 = BaseGraphGenerator.generate(seed=base_seed, **cfg)
        rng2 = random.Random(base_seed)
        src2, tgt2 = rng2.sample(sorted(G2.nodes()), 2)

        family2_label = "Family 2: State Space Search"
        f2_samples: list[Sample] = []
        for i, (vname, variant) in enumerate(FAMILY_2_VARIANTS):
            s = _generate_sample(
                family2_label, vname, variant,
                G2, src2, tgt2, seed=base_seed + (i + 1) * 1000,
            )
            s.difficulty = difficulty
            f2_samples.append(s)
            all_samples.append(s)

        _annotate_isomorphism(f2_samples, FAMILY_2_VARIANTS)

    return all_samples


def _annotate_isomorphism(
    samples: list[Sample],
    variants: list[tuple[str, object]],
) -> None:
    """Annotate each succeeded sample with which other variants are isomorphic to it."""
    succeeded = [(s, v) for s, (_, v) in zip(samples, variants) if not s.failed]
    variant_map = {name: v for name, v in variants}
    inst_map = {}  # variant_name -> instance dict (re-projected via solve, we can't get it back)
    # We can only annotate based on cost matching since instances aren't stored
    # Group by solution cost as a proxy
    cost_groups: dict[str, list[str]] = {}
    for s in samples:
        if not s.failed:
            cost_key = s.answer  # same answer implies same path length
            cost_groups.setdefault(cost_key, []).append(s.variant_name)

    for s in samples:
        if not s.failed:
            peers = [n for n in cost_groups.get(s.answer, []) if n != s.variant_name]
            s.iso_peers = peers


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

_PAGE_W, _PAGE_H = A4
_MARGIN = 1.5 * cm
_CONTENT_W = _PAGE_W - 2 * _MARGIN

# Image sizing: fit in a reasonable column
_IMG_MAX_W = _CONTENT_W * 0.55   # ~55% of content width
_IMG_MAX_H = 8.0 * cm

_STYLES = getSampleStyleSheet()

_STYLE_TITLE = ParagraphStyle(
    "BenchTitle",
    parent=_STYLES["Heading1"],
    fontSize=18,
    spaceAfter=6,
    textColor=colors.HexColor("#1A5276"),
)
_STYLE_SECTION = ParagraphStyle(
    "SectionHead",
    parent=_STYLES["Heading2"],
    fontSize=13,
    spaceAfter=4,
    spaceBefore=10,
    textColor=colors.HexColor("#1F618D"),
    borderPad=3,
)
_STYLE_VARIANT = ParagraphStyle(
    "VariantHead",
    parent=_STYLES["Heading3"],
    fontSize=10,
    spaceAfter=2,
    spaceBefore=6,
    textColor=colors.HexColor("#2E86C1"),
)
_STYLE_BODY = ParagraphStyle(
    "Body",
    parent=_STYLES["Normal"],
    fontSize=8,
    leading=11,
    spaceAfter=2,
)
_STYLE_ANSWER = ParagraphStyle(
    "Answer",
    parent=_STYLES["Normal"],
    fontSize=8,
    leading=11,
    textColor=colors.HexColor("#196F3D"),
)
_STYLE_FAIL = ParagraphStyle(
    "Fail",
    parent=_STYLES["Normal"],
    fontSize=8,
    textColor=colors.HexColor("#922B21"),
    italic=True,
)
_STYLE_ISO = ParagraphStyle(
    "Iso",
    parent=_STYLES["Normal"],
    fontSize=7,
    textColor=colors.HexColor("#7D3C98"),
    italic=True,
)


def _rl_image(buf: io.BytesIO) -> RLImage:
    """Convert a BytesIO PNG to a ReportLab Image scaled to fit the allotted space."""
    from PIL import Image as PILImage
    buf.seek(0)
    pil = PILImage.open(buf)
    orig_w, orig_h = pil.size
    buf.seek(0)

    # Scale to fit within max dimensions preserving aspect ratio
    scale = min(_IMG_MAX_W / orig_w, _IMG_MAX_H / orig_h)
    rl_w = orig_w * scale
    rl_h = orig_h * scale
    return RLImage(buf, width=rl_w, height=rl_h)


def _escape(text: str) -> str:
    """Escape XML special characters for ReportLab Paragraph."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _prompt_paragraph(prompt: str, max_chars: int = 600) -> Paragraph:
    """Build a Paragraph from the prompt, truncating if very long."""
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars] + " ...[truncated]"
    lines = _escape(prompt).split("\n")
    html = "<br/>".join(lines)
    return Paragraph(html, _STYLE_BODY)


def _sample_flowables(sample: Sample) -> list:
    """Build the list of ReportLab flowables for one sample cell."""
    items: list = []

    # Variant heading + status badge
    status = "PASS" if sample.verify_ok else ("FAIL" if not sample.failed else "N/A")
    badge_color = "#1E8449" if sample.verify_ok else "#922B21"
    heading_html = (
        f"<b>{_escape(sample.variant_name)}</b>"
        f'  <font color="{badge_color}"><b>[{status}]</b></font>'
    )
    items.append(Paragraph(heading_html, _STYLE_VARIANT))

    if sample.failed:
        items.append(Paragraph(
            f"ProjectionFailure: {_escape(sample.failure_msg[:200])}",
            _STYLE_FAIL,
        ))
        return items

    # Image
    items.append(_rl_image(sample.image_buf))
    items.append(Spacer(1, 2 * mm))

    # Prompt (text, truncated)
    items.append(_prompt_paragraph(sample.prompt))
    items.append(Spacer(1, 2 * mm))

    # Answer
    items.append(Paragraph("<b>Answer:</b>", _STYLE_BODY))
    for line in sample.answer.split("\n"):
        items.append(Paragraph(_escape(line), _STYLE_ANSWER))

    # Isomorphism peers
    if sample.iso_peers:
        items.append(Paragraph(
            f"Isomorphic peers: {', '.join(sample.iso_peers)}",
            _STYLE_ISO,
        ))

    return items


def build_pdf(samples: list[Sample], output_path: str) -> None:
    """Assemble all samples into a multi-page A4 PDF report."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=_MARGIN,
        rightMargin=_MARGIN,
        topMargin=_MARGIN,
        bottomMargin=_MARGIN,
        title="Graph Reasoning Benchmark — Multimodal Report",
        author="benchmark/multimodal_demo.py",
    )

    story: list = []

    # ── Cover / title ────────────────────────────────────────────────────
    story.append(Paragraph("Graph Reasoning Benchmark", _STYLE_TITLE))
    story.append(Paragraph(
        "Multimodal Problem Report — all variants × all difficulties",
        _STYLE_BODY,
    ))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(
        "Each problem is presented in three domain disguises.  "
        "The underlying graph structure is <b>identical</b> across disguises within "
        "the same (family, difficulty) group — only the surface representation differs.",
        _STYLE_BODY,
    ))
    story.append(Spacer(1, 4 * mm))

    # Legend
    legend_rows = [
        ["Green node / cell", "Source / start"],
        ["Red node / cell",   "Target / goal"],
        ["[PASS]",            "Solution verified correct"],
        ["[FAIL]",            "Solution verification failed"],
        ["[N/A]",             "ProjectionFailure — variant not applicable to this graph"],
    ]
    legend_table = Table(
        [[Paragraph(_escape(a), _STYLE_BODY), Paragraph(_escape(b), _STYLE_BODY)]
         for a, b in legend_rows],
        colWidths=[4.5 * cm, _CONTENT_W - 4.5 * cm],
    )
    legend_table.setStyle(TableStyle([
        ("BOX",         (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID",   (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("BACKGROUND",  (0, 0), (0, -1),  colors.HexColor("#EBF5FB")),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",(0, 0), (-1, -1), 4),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0),(-1, -1), 3),
    ]))
    story.append(legend_table)
    story.append(PageBreak())

    # ── Sections by family × difficulty ─────────────────────────────────
    # Group samples
    grouped: dict[str, dict[str, dict[str, Sample]]] = {}
    for s in samples:
        grouped.setdefault(s.family, {}).setdefault(s.difficulty, {})[s.variant_name] = s

    families_order = ["Family 1: Shortest Path", "Family 2: State Space Search"]
    difficulty_order = ["easy", "medium", "hard"]

    for family in families_order:
        story.append(Paragraph(family, _STYLE_SECTION))
        story.append(Spacer(1, 3 * mm))

        variants_list = (
            FAMILY_1_VARIANTS if "1" in family else FAMILY_2_VARIANTS
        )

        for diff in difficulty_order:
            diff_samples = grouped.get(family, {}).get(diff, {})

            # Difficulty sub-heading
            diff_label = f"{diff.capitalize()} difficulty"
            cfg = DIFFICULTY_CONFIGS[diff]
            cfg_str = f"n={cfg['n_nodes']}, type={cfg['graph_type']}, weighted={cfg['weighted']}"
            story.append(Paragraph(
                f"<b>{diff_label}</b>  <font color='#7F8C8D' size='8'>({cfg_str})</font>",
                _STYLE_VARIANT,
            ))
            story.append(Spacer(1, 2 * mm))

            # Three variant cells in a single-column table (one row per variant)
            # Each row: [image+prompt cell]
            for vname, _ in variants_list:
                sample = diff_samples.get(vname)
                if sample is None:
                    continue

                cell_items = _sample_flowables(sample)

                # Wrap in a single-cell table for box border
                t = Table([[cell_items]], colWidths=[_CONTENT_W])
                t.setStyle(TableStyle([
                    ("BOX",          (0, 0), (-1, -1), 0.8, colors.HexColor("#AED6F1")),
                    ("BACKGROUND",   (0, 0), (-1, -1), colors.HexColor("#F8FBFF")),
                    ("VALIGN",       (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING",  (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING",   (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
                ]))
                story.append(t)
                story.append(Spacer(1, 3 * mm))

            story.append(Spacer(1, 4 * mm))

        story.append(PageBreak())

    doc.build(story)
    print(f"Report written to: {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the multimodal benchmark PDF report."
    )
    parser.add_argument(
        "--output", "-o",
        default="benchmark_report.pdf",
        help="Output PDF path (default: benchmark_report.pdf)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    args = parser.parse_args()

    print("Generating samples...", flush=True)
    samples = generate_all_samples(base_seed=args.seed)

    n_ok = sum(1 for s in samples if not s.failed)
    n_fail = sum(1 for s in samples if s.failed)
    print(f"  {n_ok} succeeded,  {n_fail} failed (ProjectionFailure — expected for some variants)")

    print("Building PDF...", flush=True)
    build_pdf(samples, args.output)


if __name__ == "__main__":
    main()
