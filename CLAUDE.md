# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Academic research project investigating dynamic multimodal LLM dataset generation for graph theory problems. The key idea is that problems are generated on-the-fly so the model always faces brand-new instances, avoiding dataset contamination.

The project generates text prompts, graph visualizations (PNG), and ground-truth answers for training/evaluating multimodal language models on algorithmic reasoning tasks.

**Planned evolution:**
1. ~~Add more graph tasks beyond the current four~~ — **Done**: 14 tasks (11 GraphQA + 3 original)
2. Add an evaluation script that benchmarks LLMs (via provider APIs or local inference) on dynamically generated graphs
3. ~~Refactor the generator into a library with a clean API~~ — **Done**: package with `GraphTask` base class + registry

## Commands

```bash
uv sync                                           # Install dependencies
uv run python src/dataset-generator/multimodal.py  # Run the generator (10 samples, all tasks)
uv run python src/dataset-generator/multimodal.py -n 50 --size medium  # 50 samples, medium graphs
uv run python src/dataset-generator/multimodal.py --tasks mst shortest_path  # Specific tasks only
```

No test suite or linter is configured yet.

## Architecture

Package at `src/dataset-generator/` with a `GraphTask` base class and auto-registration via `__init_subclass__`.

```
src/dataset-generator/
├── __init__.py              # Public API: GraphTask, get_all_tasks, get_task, graph2img
├── __main__.py              # python -m entry point
├── tasks/
│   ├── __init__.py          # GraphTask base class + registry
│   ├── basic.py             # NodeCount, EdgeCount, NodeDegree, CycleCheck
│   ├── connectivity.py      # EdgeExistence, ConnectedNodes, DisconnectedNodes, Reachability, ConnectivityCheck, ConnectedComponents
│   ├── pathfinding.py       # ShortestPath, MST
│   └── advanced.py          # TriangleCounting, MaximumFlow
├── graph_generator.py       # Graph generation utilities (ER, BA, WS, SBM, SFN, complete, star, path, tree)
├── visualization.py         # graph2img → PIL.Image
└── multimodal.py            # CLI entry point
```

- **`GraphTask`** — base class; subclasses set `name` and implement `generate(G) → {prompt, image, answer}`
- **Auto-registration** — any `GraphTask` subclass with a non-empty `name` is added to the registry automatically
- **`graph_generator`** — graph creation functions, size presets (small/medium/large), weight helpers
- **`graph2img(G, weighted)`** — renders a networkx graph to a `PIL.Image` via matplotlib
- **`multimodal.py`** — CLI that picks random tasks, generates graphs, and saves PNG + text output

Graph generation uses networkx (Watts-Strogatz, Erdős-Rényi, Barabási-Albert, random trees, etc.). Default seed is `12227`.

## Key Details

- Python 3.12 only (>=3.12, <3.13), managed with **uv**
- New graph tasks: subclass `GraphTask`, set `name`, implement `generate(G)` returning `{prompt, image, answer}`
- All prompts in English, following GraphQA's "Q: …\nA:" format
- The package is importable: `importlib.import_module('src.dataset-generator')` exposes `GraphTask`, `get_all_tasks`, `graph2img`, etc.
