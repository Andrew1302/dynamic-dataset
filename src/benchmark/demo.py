"""
demo.py — End-to-end demonstration of the graph-disguise benchmark.

For each difficulty level (easy / medium / hard), this script:
    1. Generates a base graph G.
    2. Projects G into three variants per family.
    3. Solves each projected instance.
    4. Verifies each solution.
    5. Runs pairwise isomorphism checks between projections.
    6. Prints the LLM-facing prompts.

Run with:
    uv run python src/benchmark/demo.py

Expected: all isomorphism checks print ISOMORPHIC and all verifications pass.
"""

from __future__ import annotations

import random
import textwrap

from .base import BaseGraphGenerator, ProjectionFailure, are_isomorphic_instances
from .shortest_path_variants import BareGraphVariant, MazeVariant, WordLadderVariant
from .state_search_variants import (
    BareStateGraphVariant,
    SlidingPuzzleVariant,
    TowerOfHanoiVariant,
)

# ---------------------------------------------------------------------------
# Difficulty presets (mirrors the spec's suggested values)
# ---------------------------------------------------------------------------

DIFFICULTY_CONFIGS: dict[str, dict] = {
    "easy":   {"n_nodes": 5,  "graph_type": "tree",   "weighted": False},
    "medium": {"n_nodes": 8,  "graph_type": "sparse",  "weighted": True},
    "hard":   {"n_nodes": 12, "graph_type": "random",  "weighted": True},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEP = "=" * 68


def _header(text: str) -> None:
    print(f"\n{_SEP}\n{text}\n{_SEP}")


def _subheader(text: str) -> None:
    print(f"\n--- {text} ---")


def _print_prompt(prompt: str, max_chars: int = 800) -> None:
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars] + " ...[truncated]"
    print(textwrap.indent(prompt, "  "))


# ---------------------------------------------------------------------------
# Family 1: Shortest Path
# ---------------------------------------------------------------------------


def demo_family1(difficulty: str, base_seed: int = 42) -> None:
    """Generate one shortest-path triple and verify structural isomorphism."""
    cfg = DIFFICULTY_CONFIGS[difficulty]
    _header(f"FAMILY 1 — SHORTEST PATH — {difficulty.upper()}")
    print(f"Config: {cfg}  seed={base_seed}")

    G = BaseGraphGenerator.generate(seed=base_seed, **cfg)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    rng = random.Random(base_seed)
    source, target = rng.sample(sorted(G.nodes()), 2)
    print(f"Source: {source}  Target: {target}")

    variants: list[tuple[str, object]] = [
        ("BareGraph",   BareGraphVariant()),
        ("Maze",        MazeVariant()),
        ("WordLadder",  WordLadderVariant()),
    ]

    instances: dict[str, dict] = {}
    solutions: dict[str, dict] = {}

    for i, (name, variant) in enumerate(variants):
        _subheader(name)
        v_seed = base_seed + (i + 1) * 1000
        try:
            inst = variant.project(G, source, target, seed=v_seed)  # type: ignore[union-attr]
            sol = variant.solve(inst)                                 # type: ignore[union-attr]
            ok = variant.verify(inst, sol)                           # type: ignore[union-attr]
            instances[name] = inst
            solutions[name] = sol
            print(f"Solution cost: {sol['cost']}   verify: {'PASS' if ok else 'FAIL'}")
            print(f"Base path: {sol.get('base_path', '—')}")
            _subheader(f"{name} prompt")
            _print_prompt(variant.to_prompt(inst))                   # type: ignore[union-attr]
        except ProjectionFailure as exc:
            print(f"ProjectionFailure: {exc}")

    # Pairwise isomorphism checks
    _subheader("Isomorphism verification")
    names = list(instances.keys())
    variant_map = dict(variants)
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            try:
                iso = are_isomorphic_instances(
                    instances[na], variant_map[na],   # type: ignore[arg-type]
                    instances[nb], variant_map[nb],   # type: ignore[arg-type]
                )
                symbol = "ISOMORPHIC [OK]" if iso else "NOT ISOMORPHIC [FAIL]"
                print(f"  {na} <-> {nb}: {symbol}")
            except Exception as exc:
                print(f"  {na} <-> {nb}: ERROR — {exc}")


# ---------------------------------------------------------------------------
# Family 2: State Space Search
# ---------------------------------------------------------------------------


def demo_family2(difficulty: str, base_seed: int = 200) -> None:
    """Generate one state-search triple and verify structural isomorphism."""
    cfg = DIFFICULTY_CONFIGS[difficulty]
    _header(f"FAMILY 2 — STATE SPACE SEARCH — {difficulty.upper()}")
    print(f"Config: {cfg}  seed={base_seed}")

    G = BaseGraphGenerator.generate(seed=base_seed, **cfg)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    rng = random.Random(base_seed)
    source, target = rng.sample(sorted(G.nodes()), 2)
    print(f"Source: {source}  Target: {target}")

    variants: list[tuple[str, object]] = [
        ("BareState",     BareStateGraphVariant()),
        ("SlidingPuzzle", SlidingPuzzleVariant()),
        ("TowerOfHanoi",  TowerOfHanoiVariant()),
    ]

    instances: dict[str, dict] = {}
    solutions: dict[str, dict] = {}

    for i, (name, variant) in enumerate(variants):
        _subheader(name)
        v_seed = base_seed + (i + 1) * 1000
        try:
            inst = variant.project(G, source, target, seed=v_seed)  # type: ignore[union-attr]
            sol = variant.solve(inst)                                 # type: ignore[union-attr]
            ok = variant.verify(inst, sol)                           # type: ignore[union-attr]
            instances[name] = inst
            solutions[name] = sol
            print(f"Solution cost: {sol['cost']}   verify: {'PASS' if ok else 'FAIL'}")
            print(f"Base path: {sol.get('base_path', '—')}")
            _subheader(f"{name} prompt")
            _print_prompt(variant.to_prompt(inst))                   # type: ignore[union-attr]
        except ProjectionFailure as exc:
            print(f"ProjectionFailure: {exc}")

    # Pairwise isomorphism checks
    _subheader("Isomorphism verification")
    names = list(instances.keys())
    variant_map = dict(variants)
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            try:
                iso = are_isomorphic_instances(
                    instances[na], variant_map[na],   # type: ignore[arg-type]
                    instances[nb], variant_map[nb],   # type: ignore[arg-type]
                )
                symbol = "ISOMORPHIC [OK]" if iso else "NOT ISOMORPHIC [FAIL]"
                print(f"  {na} <-> {nb}: {symbol}")
            except Exception as exc:
                print(f"  {na} <-> {nb}: ERROR — {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    for difficulty in ["easy", "medium", "hard"]:
        demo_family1(difficulty, base_seed=42)
        demo_family2(difficulty, base_seed=42)


if __name__ == "__main__":
    main()
