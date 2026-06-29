# `src/benchmark/demo.py`

CLI for generating graph-disguise benchmark samples — text prompts, PNG
images, optional combined PDF report, and (new) per-image PDF assets
ready to drop into a LaTeX presentation.

## Quick start

```bash
# Default: 3 samples × all registered tasks, PNGs + text artifacts.
uv run python -m src.benchmark.demo

# Standalone PDF report (one sample per page, side-by-side direct / disguise).
uv run python -m src.benchmark.demo --pdf -o out/run

# LaTeX-ready bundle: 18 samples + 2-per-page PDF + assets/ folder of per-image PDFs.
uv run python -m src.benchmark.demo --demo -o out/demo
```

## Output layout

Every sample writes five flat files into `--output-dir`:

```
sample_{idx}_{task}_direct.png
sample_{idx}_{task}_direct_prompt.txt
sample_{idx}_{task}_disguise.png
sample_{idx}_{task}_disguise_prompt.txt
sample_{idx}_{task}_answer.txt
```

With `--pdf` (or `--two-per-page` or `--demo`):

```
report.pdf
```

With `--assets` (or `--demo`):

```
assets/
  sample_{idx:02d}_{task}_{difficulty}_direct.pdf
  sample_{idx:02d}_{task}_{difficulty}_disguise.pdf
```

## Flag reference

| Flag | Purpose |
| --- | --- |
| `--tasks ...` | Subset of tasks to run. Default: every registered task. |
| `--difficulty {easy,medium,hard}` | Difficulty for the classic loop. Ignored when `--samples-per-difficulty` is set. |
| `-n / --num-samples N` | Samples per task for the classic loop. Ignored when `--samples-per-difficulty` is set. |
| `--samples-per-difficulty N` | Generate N samples for **each** of easy / medium / hard, for every selected task. Total = `N × 3 × len(tasks)`. |
| `-o / --output-dir DIR` | Output directory. Default `out/benchmark`. |
| `--seed S` | Base RNG seed. |
| `--pdf` | Also write `report.pdf` (one sample per page). |
| `--two-per-page` | Switch `report.pdf` to a 2-questions-per-page layout. Implies `--pdf`. |
| `--assets` | Also write per-image PDFs to `<output-dir>/assets/`. |
| `--asset-dpi DPI` | Raster DPI for the directed-maze PDF (raster wrapped in PDF). Default 220. |
| `--demo` | Preset: 3 tasks (coloring, directed_connectivity, shortest_path), 2 samples per difficulty, `--pdf --two-per-page --assets`. |
| `--label-style`, `--node-color`, `--edge-style` | Rendering ablation knobs. |
| `--include-adjacency-list` | Append an adjacency-list text block (with an encoding legend) to the prompts. |
| `--constraint`, `--constraint-values`, `--samples-per-value`, `--edge-tolerance`, `--edge-max-attempts` | Sweep mode — orthogonal to the demo / asset path. See module docstring. |
| `--special-coloring` | Balanced coloring mode: force a uniform distribution of the chromatic number (the answer) by construction. Restricts the run to the `coloring` task. See below. |
| `--chromatic-values` | Comma-separated chromatic targets for `--special-coloring`, assigned round-robin. Allowed: `2,3,4`. Default `2,3,4`. |

## The `--demo` preset

`--demo` is the easiest way to produce a LaTeX-ready bundle. It is
equivalent to:

```bash
uv run python -m src.benchmark.demo \
    --tasks coloring directed_connectivity shortest_path \
    --samples-per-difficulty 2 \
    --two-per-page \
    --assets \
    --pdf
```

This generates **18 questions** (3 tasks × 3 difficulties × 2 samples),
arranged 2 per page in `report.pdf` (9 pages of content), with 36 PDFs
in `assets/`.

## Special / balanced coloring mode

```bash
# 120 coloring samples, answer uniform over {2, 3, 4} (40 each).
uv run python -m src.benchmark.demo --special-coloring -n 120 --difficulty medium -o out/balanced
```

By default the `coloring` task draws a full Delaunay triangulation. Such a
graph always contains triangles (so χ ≥ 3) and is planar (so χ ≤ 4 by the
**four-color theorem**), and on medium sizes it concentrates almost entirely
on **4**. The answer therefore carries little information — a model scores
well by always guessing 4.

`--special-coloring` fixes this by **constraining graph generation** so the
chromatic number (the answer) is uniform over a target set. It is built by
construction, never by rejection-sampling graphs until χ matches:

- The ambient graph is still a Delaunay triangulation **T** (so the Voronoi
  map stays faithful). We keep a *subgraph* G ⊆ T whose chromatic number is
  *exactly* the target `k`:
  - **k = 4** — keep all of T (a random triangulation is 4-chromatic almost
    always; resampled until χ(T) = 4).
  - **k ∈ {2, 3}** — keep a planted *k-partite* subset of T's edges (a proper
    k-coloring derived from a BFS spanning tree, so G is connected and
    χ(G) ≤ k; for k = 3 one triangle is preserved to force χ(G) ≥ 3).
- χ(G) is asserted exactly with the backtracking solver as a safety net.
- The result is **fully seed-deterministic**: the same `--seed` always yields
  the same samples.

In the map disguise, adjacent regions share a **black land border** while
non-adjacent regions are pulled apart by an **open-water gap** on a pale-sea
backdrop, so two regions share a border *iff* their nodes are adjacent in G —
the disguise stays exactly faithful to the ground truth. For k = 4 there are
no gaps and the map matches the classic full-triangulation rendering.

**Why no 5?** By the four-color theorem, no faithful planar map ever requires
five colors, so the target set is capped at `{2, 3, 4}`.

**Known property (not a bug).** Fewer colors inherently allow fewer planar
edges (a bipartite planar graph has ≤ 2n−4 edges vs ≤ 3n−6 for a
triangulation), so the *edge count* weakly correlates with the answer. The
construction maximizes kept edges per target to dampen this, and node count is
drawn from the same preset for every k (so node count does **not** leak χ).
The residual edge-count signal is a fundamental property of planar coloring,
not an artifact of the generator.

## Vector vs. raster PDFs

Three of the four renderers are matplotlib-based and produce **true
vector PDFs**:

- `coloring` direct view (`render_graph`) and disguise (`render_planar_map`)
- `shortest_path` direct view (`render_graph`) and disguise (`render_latin_america_map`)
- `directed_connectivity` direct view (`render_graph`)

One renderer — the `directed_connectivity` **disguise** (maze) — uses
pure-PIL drawing primitives and cannot produce a true vector PDF. Its
asset is a high-DPI PNG (default 220 DPI, see `--asset-dpi`) wrapped in
a PDF container. It still works with `\includegraphics{}` and prints
cleanly at slide resolution, but text is not selectable.

## LaTeX usage

```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=0.46\linewidth]{assets/sample_01_shortest_path_easy_direct.pdf}\hfill
  \includegraphics[width=0.46\linewidth]{assets/sample_01_shortest_path_easy_disguise.pdf}
  \caption{Direct view (left) and the Latin-America road-trip disguise (right).}
\end{figure}
```

## Deprecated tasks

The undirected `connectivity` task is **deprecated** in favour of
`directed_connectivity`. It remains registered for backward
compatibility but emits a `DeprecationWarning` when instantiated and is
not included in the `--demo` task set. If you need to reproduce older
results, pass it explicitly: `--tasks connectivity`.
