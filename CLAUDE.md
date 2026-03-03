# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Academic research project investigating dynamic multimodal LLM dataset generation for graph theory problems. The key idea is that problems are generated on-the-fly so the model always faces brand-new instances, avoiding dataset contamination.

The project generates text prompts, graph visualizations (PNG), and ground-truth answers for training/evaluating multimodal language models on algorithmic reasoning tasks.

**Planned evolution:**
1. Add more graph tasks beyond the current four (MST, connectivity, connected components, shortest path)
2. Add an evaluation script that benchmarks LLMs (via provider APIs or local inference) on dynamically generated graphs
3. Refactor the generator into a library with a clean API so the evaluation script can import and call it directly

## Commands

```bash
uv sync                                      # Install dependencies
python src/dataset-generator/multimodal.py   # Run the generator
```

No test suite or linter is configured yet.

## Architecture

Currently a single-file application (`src/dataset-generator/multimodal.py`) with a decorator-based algorithm registry. This will evolve into a library (importable API) + separate evaluation script.

- **`@register` decorator** — adds algorithm functions to a global `algorithms` list
- **Algorithm functions** (`mst`, `connected`, `connected_components`, `shortest_path`) — each generates a random graph, a prompt, a visualization, and computes the answer
- **`graph2img(G, weighted)`** — converts a networkx graph to a matplotlib PNG image
- **`main()`** — loops over samples, picks a random algorithm, and saves output to `output_samples/`

Graph generation uses networkx (Watts-Strogatz, Erdős-Rényi, random trees). Seeds are fixed (`12227`) for reproducibility.

## Key Details

- Python 3.12 only (>=3.12, <3.13), managed with **uv**
- New graph tasks should follow the `@register` pattern: take `n` (node count), return `(prompt, image, answer)`
- The generator needs to be usable both as a standalone script and as an importable library for the evaluation pipeline
