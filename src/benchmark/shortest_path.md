# Task 2: Shortest Path in DAG ↔ Latin America Road Trip

## Overview

This task tests whether a multimodal LLM can solve a **shortest path problem on a weighted DAG** when it is disguised as a **road trip across Latin America** with cities and one-way driving connections.

The model receives two versions of the same problem:
1. **Direct (graph theory):** A weighted DAG with labeled vertices and edge weights. "Find the minimum-weight path from source to sink."
2. **Disguised (road trip):** A real geographic map of Latin America with cities marked at real coordinates and one-way driving routes annotated with travel hours. "What is the minimum total driving time from the start city to the destination city?"

Both versions are structurally identical. The road trip IS the DAG, with vertices mapped to real cities (sorted north-to-south by latitude so the topological order matches a visual flow direction) and edge weights mapped to driving hours.

---

## Graph → Road Trip Mapping

| Graph element | Disguise element |
|---|---|
| Source vertex (in-degree = 0) | Northernmost city (the "start" point) |
| Sink vertex (out-degree = 0) | Southernmost city (the "destination" point) |
| Intermediate vertex | A Latin American city |
| Directed edge (u,v) with weight w | One-way road annotated "w h" (driving hours) |
| Minimum-weight path s→t | Minimum total driving time, start → destination |

**Key context:** A road trip across Latin America is a domain in which **any topology is plausible** (any traveler can pick any sequence of intermediate cities) and **any edge-weight value is plausible** (driving times across this region span 1 to 60+ hours depending on terrain, road quality, and border crossings). The disguise carries no specific structural prior the model can use to second-guess the labeled values.

---

## Constraints for Random Graph Generation

The graph G = (V, E) must satisfy:

1. **Type:** DAG (directed acyclic graph), weighted, with positive integer weights.
2. **|V|:** between 5 and 20 (capped at 20 because the city pool has 21 entries).
3. **Weights:** w ∈ {1, 2, ..., 10} (integers, so the answer is always an integer; presented as "h" for hours).
4. **Source:** exactly 1 vertex s with in-degree = 0.
5. **Sink:** exactly 1 vertex t with out-degree = 0.
6. **Paths s→t:** ≥ 2 distinct simple paths from s to t.

### DAG Generation Recipe

Since we index vertices as 0, 1, ..., n-1 and only add edges i→j where i < j, the graph is **automatically a DAG** (no cycle detection needed). We add a deterministic orphan-fix step so the "exactly one source / one sink" constraint is met by construction rather than by rejection sampling (which has near-zero acceptance at n ≥ 15).

```python
import random
import networkx as nx


def gen_shortest_path(n, p=0.18, seed=0, max_attempts=200,
                      weight_lo=1, weight_hi=10):
    """
    Generate a random weighted DAG instance for the shortest path task.

    Args:
        n: number of vertices (5 to 20)
        p: edge probability (0.10 - 0.20 works well)
        seed: RNG seed
        max_attempts: rejection-sampling cap
        weight_lo / weight_hi: integer weight range

    Returns:
        dict with V, E, source, sink, shortest, shortest_path, graph
        or None if no valid instance was found.
    """
    rng = random.Random(seed)
    for attempt in range(max_attempts):
        V = list(range(n))
        edge_set = set()

        # 1. Erdős–Rényi DAG (i→j only when i<j)
        for i in V:
            for j in V:
                if i < j and rng.random() < p:
                    edge_set.add((i, j))

        # 2. Force vertex 0 to be the only source: every v != 0 must
        # have at least one incoming edge from some u < v.
        for v in range(1, n):
            if not any((u, v) in edge_set for u in range(v)):
                edge_set.add((rng.randrange(v), v))

        # 3. Force vertex n-1 to be the only sink (symmetric).
        for v in range(n - 1):
            if not any((v, w) in edge_set for w in range(v + 1, n)):
                edge_set.add((v, rng.randrange(v + 1, n)))

        # 4. Assign integer weights.
        weights = {e: rng.randint(weight_lo, weight_hi) for e in edge_set}

        G = nx.DiGraph()
        G.add_nodes_from(V)
        for (u, v) in edge_set:
            G.add_edge(u, v, weight=weights[(u, v)])

        s, t = 0, n - 1

        # Sanity (should always hold by construction).
        sources = [v for v in V if G.in_degree(v) == 0]
        sinks = [v for v in V if G.out_degree(v) == 0]
        if len(sources) != 1 or len(sinks) != 1:
            continue

        try:
            sp = nx.shortest_path(G, s, t, weight="weight")
            sp_len = nx.shortest_path_length(G, s, t, weight="weight")
        except nx.NetworkXNoPath:
            continue

        # ≥ 2 distinct simple paths
        paths_iter = nx.all_simple_paths(G, s, t)
        if next(paths_iter, None) is None or next(paths_iter, None) is None:
            continue

        return {
            "n": n,
            "V": V,
            "E": [(u, v, weights[(u, v)]) for (u, v) in sorted(edge_set)],
            "source": s,
            "sink": t,
            "shortest": sp_len,
            "shortest_path": sp,
            "graph": G,
            "seed": seed,
            "attempt": attempt,
        }
    return None
```

---

## City Pool

A curated list of **21 Latin American cities** spanning all sub-regions (Mexico, Central America, Caribbean, Andes, Amazon, Brazilian east coast, southern cone, Patagonia). Pool size > maximum n so farthest-first sampling actually has a choice when n < 21.

```python
# (name, lat, lon)
CITIES = [
    ("Tijuana",         32.50, -117.00),
    ("Monterrey",       25.69, -100.32),
    ("Mexico City",     19.43,  -99.13),
    ("Mérida",          20.97,  -89.62),
    ("Havana",          23.13,  -82.39),
    ("Guatemala City",  14.63,  -90.51),
    ("Santo Domingo",   18.47,  -69.90),
    ("Caracas",         10.49,  -66.90),
    ("Bogotá",           4.71,  -74.07),
    ("Quito",           -0.18,  -78.47),
    ("Manaus",          -3.10,  -60.03),
    ("Recife",          -8.05,  -34.88),
    ("Lima",           -12.05,  -77.04),
    ("La Paz",         -16.50,  -68.15),
    ("Brasília",       -15.79,  -47.88),
    ("Salvador",       -12.97,  -38.51),
    ("São Paulo",      -23.55,  -46.63),
    ("Asunción",       -25.30,  -57.63),
    ("Santiago",       -33.45,  -70.66),
    ("Buenos Aires",   -34.61,  -58.38),
    ("Punta Arenas",   -53.16,  -70.92),
]
```

The pool can be extended (e.g. to 30+ cities) without any other code change; farthest-first sampling will still pick the most-spread n.

---

## Map Configuration

Cartopy rendering parameters used for every disguise instance:

| Parameter | Value |
|---|---|
| Projection | `cartopy.crs.PlateCarree()` |
| Extent (lon_min, lon_max, lat_min, lat_max) | `[-122, -30, -58, 36]` |
| Figure size (inches) | `(15, 15)` |
| DPI on save | `170` |
| Land color | `#F4EFE3` |
| Ocean color | `#DCE9F2` |
| Coastline color / width | `#7a8693` / `0.6` |
| Border color / width | `#a0aab8` / `0.7` |
| Lakes face / edge | `#DCE9F2` / `#7a8693` (width 0.3) |
| Source marker | green `#1D9E75`, size 380 |
| Sink marker | magenta `#C2185B`, size 380 |
| Intermediate marker | navy `#374B7A`, size 240 |
| Edge color / width | orange `#C24A0E` / `1.8` |
| Edge weight bbox | `#FFF7E5` fill, `#C24A0E` border |
| Font sizes | city name 11, marker number 11, edge weight 11, title 12 |

---

## Disguise Generation Pipeline

```python
import math
import random
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from adjustText import adjust_text


def farthest_first(pool, n, seed=0):
    """
    Greedy farthest-first sampling. Start from a random pool entry,
    then iteratively add the entry with the largest minimum-distance
    to the already-chosen set. Guarantees high pairwise spread.
    """
    rng = random.Random(seed)
    chosen = [rng.randrange(len(pool))]
    while len(chosen) < n:
        best_idx, best_d = -1, -1.0
        for k in range(len(pool)):
            if k in chosen:
                continue
            d_min = min(
                math.hypot(pool[k][1] - pool[c][1],
                           pool[k][2] - pool[c][2])
                for c in chosen
            )
            if d_min > best_d:
                best_d = d_min
                best_idx = k
        chosen.append(best_idx)
    return [pool[i] for i in chosen]


def render_disguise(inst, cities, filename, seed=0):
    """
    Render a Latin America road trip image for a given DAG instance.

    Pipeline:
      1. Pick n cities from the pool by farthest-first sampling.
      2. Sort them by latitude DESCENDING (N -> S) so graph node 0 maps
         to the northernmost chosen city, and graph node n-1 maps to
         the southernmost.
      3. Render cartopy backdrop, edges, and markers.
      4. Run adjustText on every label so nothing overlaps.
    """
    n = inst["n"]
    chosen = farthest_first(cities, n, seed=seed)
    chosen.sort(key=lambda r: -r[1])  # latitude DESC
    coords = {i: (chosen[i][1], chosen[i][2]) for i in range(n)}
    names  = {i: chosen[i][0] for i in range(n)}

    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-122, -30, -58, 36], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND,      facecolor="#F4EFE3")
    ax.add_feature(cfeature.OCEAN,     facecolor="#DCE9F2")
    ax.add_feature(cfeature.COASTLINE, edgecolor="#7a8693", linewidth=0.6)
    ax.add_feature(cfeature.BORDERS,   edgecolor="#a0aab8", linewidth=0.7)
    ax.add_feature(cfeature.LAKES,     facecolor="#DCE9F2",
                                       edgecolor="#7a8693", linewidth=0.3)

    all_texts = []
    marker_xs, marker_ys = [], []

    # Edges (drawn before markers)
    for (u, v, w) in inst["E"]:
        lat_u, lon_u = coords[u]
        lat_v, lon_v = coords[v]
        rad = 0.16 if (u + v) % 2 else -0.16
        arr = FancyArrowPatch(
            (lon_u, lat_u), (lon_v, lat_v),
            arrowstyle="-|>", mutation_scale=18,
            connectionstyle=f"arc3,rad={rad}",
            color="#C24A0E", lw=1.8, alpha=0.85,
            shrinkA=14, shrinkB=14,
            transform=ccrs.PlateCarree()._as_mpl_transform(ax),
            zorder=4,
        )
        ax.add_patch(arr)
        # Initial weight position: edge midpoint. adjustText refines.
        mid_lon = 0.5 * (lon_u + lon_v)
        mid_lat = 0.5 * (lat_u + lat_v)
        t = ax.text(mid_lon, mid_lat, f"{w}h",
                    ha="center", va="center", fontsize=11,
                    color="#7a3010", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.22", fc="#FFF7E5",
                              ec="#C24A0E", alpha=0.97, lw=0.8),
                    transform=ccrs.PlateCarree(), zorder=5)
        all_texts.append(t)

    # Markers + city labels
    for i in range(n):
        lat, lon = coords[i]
        if i == inst["source"]:
            color, ms, extra = "#1D9E75", 380, "  (start)"
        elif i == inst["sink"]:
            color, ms, extra = "#C2185B", 380, "  (end)"
        else:
            color, ms, extra = "#374B7A", 240, ""
        ax.scatter(lon, lat, s=ms, color=color,
                   edgecolors="white", linewidths=2.2,
                   transform=ccrs.PlateCarree(), zorder=6)
        ax.text(lon, lat, str(i),
                ha="center", va="center", color="white",
                fontweight="bold", fontsize=11,
                transform=ccrs.PlateCarree(), zorder=7)
        marker_xs.append(lon)
        marker_ys.append(lat)
        t = ax.text(lon, lat, f"{names[i]}{extra}",
                    ha="center", va="center", fontsize=11,
                    color="#1f1a4e", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec=color, alpha=0.96, lw=0.9),
                    transform=ccrs.PlateCarree(), zorder=8)
        all_texts.append(t)

    # Deconflict every label so none overlaps any other label or marker
    adjust_text(
        all_texts,
        x=marker_xs, y=marker_ys,
        ax=ax,
        expand=(1.4, 1.6),
        force_text=(0.7, 0.9),
        force_static=(0.5, 0.7),
        arrowprops=dict(arrowstyle="-", color="#666666", lw=0.6, alpha=0.6),
    )

    ax.set_title(
        f"Latin America road trip (n={n}): minimum total driving hours, "
        f"{names[inst['source']]} -> {names[inst['sink']]}",
        fontsize=12, pad=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
```

### Why each step matters

- **Farthest-first sampling** guarantees that even when n approaches the pool size, the chosen cities are maximally spread. Random sampling would frequently pick clusters (Mexico City + Guatemala City + Havana), causing visual crowding.
- **Sorting by latitude (N → S)** maps graph node 0 to the northernmost chosen city and node n-1 to the southernmost. Because graph edges always run from a low-index vertex to a higher-index one (i < j by construction), every drawn arrow tends to point southward. This gives the rendering a coherent visual flow direction.
- **adjustText** is the headline guarantee for readability. After all labels are placed at their initial anchor positions, `adjust_text` runs a force-directed solver that pushes labels apart until no two overlap. When a label has to move far from its anchor, a thin gray connector line is drawn so the association stays clear. The result: every city name and every weight label is unambiguously readable, even at the maximum n.

---

## Graph Image Generation (direct version)

The direct (graph theory) version is drawn with `networkx` + `matplotlib`, using a layered topological layout for clarity:

```python
import networkx as nx
import matplotlib.pyplot as plt


def render_direct(inst, filename):
    """Render the abstract weighted DAG, optionally with shortest path
    highlighted. The shortest_path edges are drawn in green, the rest
    in lavender."""
    G = inst["graph"]
    sp_edges = set(zip(inst["shortest_path"][:-1], inst["shortest_path"][1:]))

    # Layered (topological) layout, left-to-right
    layers = list(nx.topological_generations(G))
    pos = {}
    for x, layer in enumerate(layers):
        ys = ([0.0] if len(layer) == 1
              else [-(len(layer) - 1) / 2 + k for k in range(len(layer))])
        for v, y in zip(sorted(layer), ys):
            pos[v] = (x * 1.6, y * 1.0)

    fig, ax = plt.subplots(figsize=(13, 7))

    edge_colors, widths = [], []
    for u, v in G.edges():
        if (u, v) in sp_edges:
            edge_colors.append("#1D9E75"); widths.append(2.6)
        else:
            edge_colors.append("#9089D0"); widths.append(1.0)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                           width=widths, arrows=True, arrowsize=14,
                           arrowstyle="->", connectionstyle="arc3,rad=0.06",
                           min_source_margin=14, min_target_margin=14,
                           alpha=0.85)

    node_colors = ["#1D9E75" if v in (inst["source"], inst["sink"])
                   else "#CECBF6" for v in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=520, edgecolors="#3C3489",
                           linewidths=1.2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9,
                            font_weight="bold", font_color="#1f1a4e")

    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax,
                                 font_size=7, font_color="#3C3489",
                                 bbox=dict(boxstyle="round,pad=0.08",
                                           fc="white", ec="none", alpha=0.7))

    ax.margins(0.06); ax.axis("off")
    fig.tight_layout()
    fig.savefig(filename, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
```

For the model-facing version, omit the green path highlight (set all edge colors to `#9089D0`); for the ground-truth visualization (debugging / paper figures), keep it.

---

## Prompt Templates

### Direct (Graph Theory) prompt

```
Given the weighted directed acyclic graph G = (V, E) shown in the image,
calculate the minimum-weight path from vertex 0 to vertex N-1.
```

(Replace `N-1` with the actual sink index in the rendered instance.)

### Disguised (Road Trip) prompt

```
The map of Latin America below shows several cities and the available
driving routes between them. Each arrow indicates a one-way driving
connection from one city to another, and is labeled with the typical
driving time in hours. What is the minimum total driving time, in hours,
from the start city to the end city?
```

The disguised prompt does **not** describe the graph topology. The model must infer it from the image.

---

## Answer Verification

```python
def verify_answer(model_answer: int, ground_truth: int) -> bool:
    """Exact match on the shortest path total weight (driving hours)."""
    return int(model_answer) == int(ground_truth)
```

---

## Concrete Example (n = 18, seed = 3)

**Graph:**
- 18 vertices (0..17), 31 edges generated by `gen_shortest_path(n=18, p=0.10, seed=3)`.
- Source = vertex 0, sink = vertex 17.

**Shortest path:** `[0, 1, 10, 13, 17]` with total weight **17**.

**Cities (after farthest-first sampling and N→S sort):**
| Index | City | Lat | Lon |
|---|---|---|---|
| 0 | Tijuana (start) | 32.50 | -117.00 |
| 1 | Monterrey | 25.69 | -100.32 |
| 2 | Havana | 23.13 | -82.39 |
| 3 | Mérida | 20.97 | -89.62 |
| 4 | Santo Domingo | 18.47 | -69.90 |
| 5 | Caracas | 10.49 | -66.90 |
| 6 | Bogotá | 4.71 | -74.07 |
| 7 | Quito | -0.18 | -78.47 |
| 8 | Manaus | -3.10 | -60.03 |
| 9 | Recife | -8.05 | -34.88 |
| 10 | Lima | -12.05 | -77.04 |
| 11 | La Paz | -16.50 | -68.15 |
| 12 | Brasília | -15.79 | -47.88 |
| 13 | São Paulo | -23.55 | -46.63 |
| 14 | Asunción | -25.30 | -57.63 |
| 15 | Santiago | -33.45 | -70.66 |
| 16 | Buenos Aires | -34.61 | -58.38 |
| 17 | Punta Arenas (end) | -53.16 | -70.92 |

**Ground-truth answer:** minimum total driving time = **17 h** along Tijuana → Monterrey → Lima → São Paulo → Punta Arenas.

---

## Dependencies

```
pip install networkx matplotlib cartopy adjustText
```

Cartopy will download Natural Earth shapefiles on first run (one-time, a few MB).

## File Structure

```
task2_shortest_path/
├── gen_dag.py            # gen_shortest_path: random DAG instance with ground truth
├── render_direct.py      # draws the abstract weighted DAG (networkx + matplotlib)
├── render_disguise.py    # draws the Latin America road trip (cartopy + adjustText)
├── prompts.py            # prompt templates for both versions
├── verifier.py           # exact-match answer verification
└── cities.py             # the 21-city pool, farthest_first() helper
```

---

## Notes for Adapting to Other Sizes / Regions

- **Smaller n:** The pipeline supports any n from 5 to 20. With smaller n, farthest-first picks the most-extreme spread of cities, making the visual even cleaner.
- **Other regions:** Replace `CITIES` with a different curated pool of 20+ coordinates and update `set_extent` accordingly. The rest of the pipeline (`farthest_first`, `render_disguise`, `adjust_text`) is region-agnostic.
- **Other transport modes:** Edit edge weight units in the rendering and the prompt template. "Hours" works for driving; "days" works for sailing or postal transit; "minutes" works for city-scale walking or public transit.
- **Ground-truth path overlay:** For debugging, pass a `shortest_path` argument to `render_disguise` and color those edges green (same convention as `render_direct`). Omit for the model-facing image.