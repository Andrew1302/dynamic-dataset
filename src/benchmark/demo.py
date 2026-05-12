"""CLI entry point: generate benchmark samples and write five artifacts each.

Each sample produces five files in ``--output-dir``:

- ``sample_{i}_{task}_direct.png``
- ``sample_{i}_{task}_direct_prompt.txt``
- ``sample_{i}_{task}_disguise.png``
- ``sample_{i}_{task}_disguise_prompt.txt``
- ``sample_{i}_{task}_answer.txt``

Pass ``--pdf`` to also emit a combined ``report.pdf``.

Sweep mode (``--constraint nodes|edges``) generates many samples per
target value with the other axis left free; per-sample metadata
(``n_vertices``, ``n_edges``, requested value) is written alongside the
five artifacts and dumped to ``manifest.jsonl`` for downstream binning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .base import Sample, get_all_tasks, get_task
from .rendering import RenderConfig
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

    # Rendering knobs (ablation axes).
    parser.add_argument(
        "--label-style",
        choices=["numeric", "letters", "none"],
        default="numeric",
        help="Node-label style on the direct-view image.",
    )
    parser.add_argument(
        "--node-color",
        default="#AED6F1",
        help="Default node fill colour (hex). Highlights are unaffected.",
    )
    parser.add_argument(
        "--edge-style",
        choices=["straight", "curved"],
        default="straight",
        help="Edge style for weighted graphs. Unweighted graphs are always straight.",
    )
    parser.add_argument(
        "--include-adjacency-matrix",
        action="store_true",
        help="Append a text adjacency matrix to the direct-view prompt.",
    )

    # Sweep mode.
    parser.add_argument(
        "--constraint",
        choices=["nodes", "edges"],
        default=None,
        help="Sweep over this axis. When set, --difficulty and -n are ignored.",
    )
    parser.add_argument(
        "--constraint-values",
        default=None,
        help=(
            "Values to sweep. Either LO..HI[:STEP] (e.g. '5..14' or "
            "'4..20:2') or a comma-separated list (e.g. '5,8,11,14')."
        ),
    )
    parser.add_argument(
        "--samples-per-value",
        type=int,
        default=None,
        help=(
            "Samples per constraint value. Defaults: 250 for nodes, 100 "
            "for edges."
        ),
    )
    parser.add_argument(
        "--edge-tolerance",
        type=float,
        default=0.15,
        help=(
            "For --constraint edges: accept samples whose edge count is "
            "within (1±tol) of the target. Larger tol = faster, less precise."
        ),
    )
    parser.add_argument(
        "--edge-max-attempts",
        type=int,
        default=200,
        help="Max rejection-sampling attempts per (task, edge target, sample).",
    )

    args = parser.parse_args(argv)

    if args.constraint is not None and args.constraint_values is None:
        parser.error("--constraint requires --constraint-values")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = RenderConfig(
        label_style=args.label_style,
        node_color=args.node_color,
        edge_style=args.edge_style,
    )

    if args.constraint is not None:
        return _run_sweep(args, cfg)

    collected: list[tuple[str, str, int, Sample]] = []
    for i in range(args.num_samples):
        for task_name in args.tasks:
            task = get_task(task_name)()
            seed = args.seed + i * len(args.tasks) + hash(task_name) % 1000
            sample = task.generate(
                seed=seed,
                difficulty=args.difficulty,
                config=cfg,
                include_adjacency_matrix=args.include_adjacency_matrix,
            )
            _write_sample(args.output_dir, i + 1, task_name, sample)
            collected.append((task_name, args.difficulty, seed, sample))
            print(f"[sample {i + 1}] task={task_name} answer={sample['answer']}")

    if args.pdf:
        pdf_path = os.path.join(args.output_dir, "report.pdf")
        build_pdf(collected, pdf_path)
        print(f"PDF written to {pdf_path}")

    return 0


def _run_sweep(args: argparse.Namespace, cfg: RenderConfig) -> int:
    """Generate samples while sweeping over node count or edge count.

    Samples-per-value defaults: 250 (nodes), 100 (edges). For the edges
    sweep we rejection-sample by varying the achieved node count; if
    the target can't be hit within tolerance after
    ``--edge-max-attempts``, the closest-edge sample is kept.
    """
    values = _parse_constraint_values(args.constraint_values)
    spv = args.samples_per_value or (250 if args.constraint == "nodes" else 100)
    manifest_path = os.path.join(args.output_dir, "manifest.jsonl")
    print(
        f"[sweep] constraint={args.constraint} values={values} "
        f"samples_per_value={spv} tasks={args.tasks}"
    )

    sample_idx = 0
    collected: list[tuple[str, str, int, Sample]] = []
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for task_name in args.tasks:
            task = get_task(task_name)()
            for value in values:
                for s in range(spv):
                    seed = args.seed + sample_idx * 7919  # any odd stride
                    sample_idx += 1
                    if args.constraint == "nodes":
                        sample = task.generate(
                            seed=seed,
                            difficulty="medium",
                            config=cfg,
                            include_adjacency_matrix=args.include_adjacency_matrix,
                            node_count=value,
                        )
                        achieved_v = _extract_meta(sample, "n_vertices")
                        achieved_e = _extract_meta(sample, "n_edges")
                    else:
                        sample, achieved_v, achieved_e = _sample_for_edge_target(
                            task=task,
                            target_edges=value,
                            cfg=cfg,
                            include_adj=args.include_adjacency_matrix,
                            base_seed=seed,
                            tol=args.edge_tolerance,
                            max_attempts=args.edge_max_attempts,
                        )
                    idx = sample_idx
                    _write_sample(
                        args.output_dir, idx, task_name, sample,
                        meta={
                            "task": task_name,
                            "constraint": args.constraint,
                            "requested": value,
                            "n_vertices": achieved_v,
                            "n_edges": achieved_e,
                            "seed": seed,
                        },
                    )
                    manifest.write(json.dumps({
                        "idx": idx,
                        "task": task_name,
                        "constraint": args.constraint,
                        "requested": value,
                        "n_vertices": achieved_v,
                        "n_edges": achieved_e,
                        "seed": seed,
                        "answer": sample["answer"],
                    }) + "\n")
                    collected.append(
                        (task_name, f"{args.constraint}={value}", seed, sample)
                    )
                print(f"  [{task_name}] value={value}: {spv} samples done")

    if args.pdf:
        pdf_path = os.path.join(args.output_dir, "report.pdf")
        build_pdf(collected, pdf_path)
        print(f"PDF written to {pdf_path}")

    print(f"[sweep] manifest written to {manifest_path}")
    return 0


def _sample_for_edge_target(
    task,
    target_edges: int,
    cfg: RenderConfig,
    include_adj: bool,
    base_seed: int,
    tol: float,
    max_attempts: int,
) -> tuple[Sample, int, int]:
    """Rejection-sample by varying node count to hit ``target_edges``.

    Each attempt picks a random node count and generates one sample.
    Accept if the edge count is within ``tol`` of target; otherwise
    track the closest sample and return it as a fallback.
    """
    import math
    import random

    # A rough node-count range that can plausibly produce the target
    # edge count for any of the four tasks: trees give ~V-1 edges, dense
    # graphs scale ~3V. So V in [target/3 - 2, target + 4] is generous.
    v_lo = max(3, target_edges // 3 - 2)
    v_hi = max(v_lo + 2, target_edges + 4)
    lo, hi = tol_lo, tol_hi = (1 - tol) * target_edges, (1 + tol) * target_edges

    rng = random.Random(base_seed)
    best_sample = None
    best_v = 0
    best_e = 0
    best_diff = math.inf
    for attempt in range(max_attempts):
        nc = rng.randint(v_lo, v_hi)
        seed = base_seed + attempt * 101
        sample = task.generate(
            seed=seed,
            difficulty="medium",
            config=cfg,
            include_adjacency_matrix=include_adj,
            node_count=nc,
        )
        v = _extract_meta(sample, "n_vertices")
        e = _extract_meta(sample, "n_edges")
        diff = abs(e - target_edges)
        if diff < best_diff:
            best_sample = sample
            best_v, best_e, best_diff = v, e, diff
        if tol_lo <= e <= tol_hi:
            return sample, v, e
    assert best_sample is not None
    return best_sample, best_v, best_e


def _extract_meta(sample: Sample, key: str) -> int:
    return int(sample[key])  # type: ignore[literal-required]


def _write_sample(
    out_dir: str, idx: int, task: str, sample: Sample, meta: dict | None = None
) -> None:
    stem = f"sample_{idx}_{task}"
    sample["direct_image"].save(os.path.join(out_dir, f"{stem}_direct.png"))
    sample["disguise_image"].save(os.path.join(out_dir, f"{stem}_disguise.png"))
    _write_text(out_dir, f"{stem}_direct_prompt.txt", sample["direct_prompt"])
    _write_text(out_dir, f"{stem}_disguise_prompt.txt", sample["disguise_prompt"])
    _write_text(out_dir, f"{stem}_answer.txt", sample["answer"])
    if meta is not None:
        with open(os.path.join(out_dir, f"{stem}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def _write_text(out_dir: str, name: str, content: str) -> None:
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        f.write(content)


def _parse_constraint_values(spec: str) -> list[int]:
    """Parse ``LO..HI[:STEP]`` or ``v1,v2,v3``.

    Step defaults to 1 for the range form. The range is inclusive of HI
    when the step lands on it.
    """
    if ".." in spec:
        body, _, step_s = spec.partition(":")
        step = int(step_s) if step_s else 1
        lo_s, _, hi_s = body.partition("..")
        lo, hi = int(lo_s), int(hi_s)
        if lo > hi:
            raise ValueError(f"constraint range lo > hi: {spec!r}")
        return list(range(lo, hi + 1, step))
    if "," in spec:
        return [int(x.strip()) for x in spec.split(",") if x.strip()]
    return [int(spec)]


if __name__ == "__main__":
    sys.exit(main())
