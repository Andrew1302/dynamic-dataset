"""CLI entry point: generate benchmark samples and write five artifacts each.

Each sample produces five files in ``--output-dir``:

- ``sample_{i}_{task}_direct.png``
- ``sample_{i}_{task}_direct_prompt.txt``
- ``sample_{i}_{task}_disguise.png``
- ``sample_{i}_{task}_disguise_prompt.txt``
- ``sample_{i}_{task}_answer.txt``

Pass ``--pdf`` to also emit a combined ``report.pdf``.
"""

from __future__ import annotations

import argparse
import os
import sys

from .base import Sample, get_all_tasks, get_task
from .report import build_pdf


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    all_task_names = sorted(get_all_tasks().keys())
    parser = argparse.ArgumentParser(description="Graph-disguise benchmark generator.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=all_task_names,
        choices=all_task_names,
        help="Subset of tasks to run. Default: all.",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default="easy",
    )
    parser.add_argument("-n", "--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output-dir", default="out/benchmark")
    parser.add_argument("--pdf", action="store_true", help="Also produce report.pdf.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    collected: list[tuple[str, str, int, Sample]] = []
    for i in range(args.num_samples):
        for task_name in args.tasks:
            task = get_task(task_name)()
            seed = args.seed + i * len(args.tasks) + hash(task_name) % 1000
            sample = task.generate(seed=seed, difficulty=args.difficulty)
            _write_sample(args.output_dir, i + 1, task_name, sample)
            collected.append((task_name, args.difficulty, seed, sample))
            print(f"[sample {i + 1}] task={task_name} answer={sample['answer']}")

    if args.pdf:
        pdf_path = os.path.join(args.output_dir, "report.pdf")
        build_pdf(collected, pdf_path)
        print(f"PDF written to {pdf_path}")

    return 0


def _write_sample(out_dir: str, idx: int, task: str, sample: Sample) -> None:
    stem = f"sample_{idx}_{task}"
    sample["direct_image"].save(os.path.join(out_dir, f"{stem}_direct.png"))
    sample["disguise_image"].save(os.path.join(out_dir, f"{stem}_disguise.png"))
    _write_text(out_dir, f"{stem}_direct_prompt.txt", sample["direct_prompt"])
    _write_text(out_dir, f"{stem}_disguise_prompt.txt", sample["disguise_prompt"])
    _write_text(out_dir, f"{stem}_answer.txt", sample["answer"])


def _write_text(out_dir: str, name: str, content: str) -> None:
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    sys.exit(main())
