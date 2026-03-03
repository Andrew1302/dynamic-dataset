"""CLI entry point for the dynamic dataset generator.

Usage::

    python src/dataset-generator/multimodal.py          # 10 samples, all tasks
    python src/dataset-generator/multimodal.py -n 50    # 50 samples
    python src/dataset-generator/multimodal.py --tasks mst shortest_path
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np

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
        choices=["small", "medium", "large"],
        default="small",
        help="Graph size preset.",
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

    print(f"Generating {args.num_samples} samples across {len(task_names)} tasks …")
    print(f"Tasks: {', '.join(sorted(task_names))}")

    for i in range(args.num_samples):
        task_name = random.choice(task_names)
        task = all_tasks[task_name]()

        n = gg.random_node_count(args.size)
        G = _build_graph_for_task(task_name, n)

        sample = task.generate(G)

        prompt = sample["prompt"]
        image = sample["image"]
        answer = sample["answer"]

        img_path = os.path.join(args.output_dir, f"sample_{i + 1}.png")
        image.save(img_path, dpi=(120, 120))

        print(f"\n--- Sample {i + 1} [{task_name}] ---")
        print(prompt)
        print(f" {answer}")


if __name__ == "__main__":
    main()
