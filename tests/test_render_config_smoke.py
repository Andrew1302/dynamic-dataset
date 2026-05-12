"""Plain-assertion smoke test for the RenderConfig + ablation surface.

Run with ``PYTHONPATH=. uv run python tests/test_render_config_smoke.py``.

Exits non-zero on the first failure with a clear message. Doesn't
require pytest — kept minimal so it can run on the bare uv-managed
environment used for dataset generation.
"""

from __future__ import annotations

import sys

from PIL import Image

from src.benchmark import RenderConfig, get_all_tasks


def _check_sample(task_name: str, style: str, include_adj: bool) -> None:
    cfg = RenderConfig(label_style=style)
    task = get_all_tasks()[task_name]()
    sample = task.generate(
        seed=123,
        difficulty="easy",
        config=cfg,
        include_adjacency_matrix=include_adj,
    )

    for key in (
        "direct_prompt",
        "direct_image",
        "disguise_prompt",
        "disguise_image",
        "answer",
        "n_vertices",
        "n_edges",
    ):
        assert key in sample, f"{task_name}/{style}: Sample missing key {key!r}"

    assert isinstance(sample["direct_image"], Image.Image)
    assert sample["direct_image"].size[0] > 100, (
        f"{task_name}/{style}: direct image suspiciously small"
    )
    assert sample["direct_prompt"].rstrip().endswith("A:"), (
        f"{task_name}/{style}: direct_prompt should end with 'A:'"
    )
    assert sample["answer"], f"{task_name}/{style}: empty answer"
    assert isinstance(sample["n_vertices"], int) and sample["n_vertices"] > 0
    assert isinstance(sample["n_edges"], int) and sample["n_edges"] >= 0

    if include_adj:
        assert "Adjacency matrix:" in sample["direct_prompt"], (
            f"{task_name}/{style}: adjacency matrix missing from prompt"
        )
    else:
        assert "Adjacency matrix:" not in sample["direct_prompt"]


def _check_node_count(task_name: str, node_count: int) -> None:
    task = get_all_tasks()[task_name]()
    sample = task.generate(seed=7, difficulty="easy", node_count=node_count)
    achieved = sample["n_vertices"]
    if task_name in ("shortest_path", "coloring"):
        assert achieved == node_count, (
            f"{task_name}: requested {node_count} nodes, got {achieved}"
        )
    else:
        # Lattice-based tasks may differ slightly after sparsification.
        assert abs(achieved - node_count) <= 2, (
            f"{task_name}: requested {node_count} nodes, got {achieved} "
            f"(connectivity tasks tolerate ±2)"
        )


def _check_edge_styles_render() -> None:
    """Just ensure both edge_style values yield a renderable image."""
    task = get_all_tasks()["shortest_path"]()
    for edge_style in ("straight", "curved"):
        cfg = RenderConfig(edge_style=edge_style)
        sample = task.generate(seed=11, difficulty="medium", config=cfg)
        assert sample["direct_image"].size[0] > 100


def main() -> int:
    tasks = sorted(get_all_tasks().keys())
    styles = ("numeric", "letters", "none")

    print(f"Running smoke checks across {len(tasks)} tasks × {len(styles)} label styles ...")
    for t in tasks:
        for s in styles:
            for adj in (False, True):
                _check_sample(t, s, adj)
        print(f"  [{t}] sample-shape OK")

    print("Running node-count override checks ...")
    for t in tasks:
        for nc in (4, 7, 10):
            try:
                _check_node_count(t, nc)
            except ValueError as e:
                # E.g. shortest_path with nc<3 — should not fire here.
                raise AssertionError(f"{t}/nc={nc}: {e}") from e
        print(f"  [{t}] node_count override OK")

    print("Running edge-style checks ...")
    _check_edge_styles_render()
    print("  edge styles OK")

    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
