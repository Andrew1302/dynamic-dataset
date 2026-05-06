"""Plain graph rendering (direct view of the underlying task graph)."""

from __future__ import annotations

from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image


_DEFAULT_COLOR = "#AED6F1"

# Node visual size. Kept small enough that a 15-node component fits in
# the default figure without label overlap.
_NODE_SIZE = 360
_FONT_SIZE = 9

# Minimum centre-to-centre distance between any two nodes after layout,
# in unit-cube coords (layouts are normalised to roughly [-1, 1]). A
# direct repel pass enforces this — bumping ``k`` in the spring layout
# is ineffective because networkx rescales the result to a fixed box,
# which collapses any extra spacing earned during iteration.
#
# Weighted graphs need extra room because each edge carries a label
# whose bounding box would collide with neighbours at the default
# spacing.
_MIN_NODE_DIST = 0.18
_MIN_NODE_DIST_WEIGHTED = 0.34
_REPEL_ITERATIONS = 60

# Per-mode figure size. Weighted graphs use a larger canvas so the
# extra inter-node room translates into actual on-screen pixels rather
# than being scaled away.
_FIGSIZE = (6, 6)
_FIGSIZE_WEIGHTED = (9, 9)

# Two distinct curvatures (different sign AND magnitude) so fan-in /
# fan-out edges with shared endpoints stop sharing a single arc and
# their weight labels separate vertically.
_EDGE_RADS: tuple[float, float] = (0.18, -0.10)


def render_graph(
    G: nx.Graph,
    highlights: dict[int, str] | None = None,
    with_labels: bool = True,
    arrowsize: int = 10,
    weighted: bool = False,
) -> Image.Image:
    """Render *G* to a square PIL image.

    Single-component graphs use Kamada–Kawai with a repel pass that
    enforces a minimum pairwise node distance; disconnected graphs are
    laid out per-component and packed side-by-side (see ``_layout``).

    Parameters
    ----------
    G
        NetworkX graph. ``DiGraph`` instances are drawn with arrowheads.
    highlights
        Optional ``{node: color}`` mapping. Highlighted nodes are drawn in
        the given color; unhighlighted nodes use the default palette.
    with_labels
        Whether to draw numeric node labels.
    arrowsize
        Arrowhead size for directed graphs. Ignored for undirected.
    weighted
        If ``True``, draw the integer ``weight`` attribute as a label on
        each edge. Edges without a ``weight`` attribute are skipped.
    """
    highlights = highlights or {}

    min_dist = _MIN_NODE_DIST_WEIGHTED if weighted else _MIN_NODE_DIST
    figsize = _FIGSIZE_WEIGHTED if weighted else _FIGSIZE

    pos = _layout(G, min_dist=min_dist)
    node_colors = [highlights.get(n, _DEFAULT_COLOR) for n in G.nodes()]

    fig, ax = plt.subplots(figsize=figsize)
    if weighted:
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors, node_size=_NODE_SIZE,
        )
        if with_labels:
            nx.draw_networkx_labels(
                G, pos, ax=ax,
                font_size=_FONT_SIZE, font_weight="bold",
            )
        # Two curvature buckets: edges fanning into or out of a shared
        # node land on visibly different arcs, and their weight labels
        # stop piling on top of each other.
        edge_labels = {
            (u, v): str(d["weight"])
            for u, v, d in G.edges(data=True)
            if "weight" in d
        }
        for bucket, rad in enumerate(_EDGE_RADS):
            bucket_edges = [
                (u, v) for u, v in G.edges()
                if (u + v) % len(_EDGE_RADS) == bucket
            ]
            if not bucket_edges:
                continue
            connectionstyle = f"arc3,rad={rad}"
            nx.draw_networkx_edges(
                G, pos, ax=ax, edgelist=bucket_edges,
                edge_color="#2C3E50", width=1.4,
                arrowsize=arrowsize,
                connectionstyle=connectionstyle,
                node_size=_NODE_SIZE,
            )
            bucket_labels = {
                e: edge_labels[e] for e in bucket_edges if e in edge_labels
            }
            if bucket_labels:
                nx.draw_networkx_edge_labels(
                    G, pos, edge_labels=bucket_labels, ax=ax,
                    font_size=_FONT_SIZE, font_color="#1f1a4e",
                    connectionstyle=connectionstyle,
                    rotate=False,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="#2C3E50", alpha=0.85, lw=0.6),
                )
    else:
        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=with_labels,
            node_color=node_colors,
            node_size=_NODE_SIZE,
            font_size=_FONT_SIZE,
            font_weight="bold",
            edge_color="#2C3E50",
            width=1.4,
            arrowsize=arrowsize,
        )
    ax.margins(0.12 if weighted else 0.08)
    ax.set_axis_off()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def _layout(
    G: nx.Graph,
    seed: int = 42,
    gap: float = 0.6,
    min_dist: float = _MIN_NODE_DIST,
) -> dict:
    """Layout for the direct view.

    Single-component graphs use Kamada-Kawai, which spreads nodes more
    uniformly than spring layout for small tree-ish graphs. Disconnected
    graphs lay out each component independently and pack them side by
    side with a fixed gap. Edges may cross between components, which is
    fine — the goal is to keep components visually close so the viewer
    can compare them at once.
    """
    components = list(
        nx.weakly_connected_components(G) if G.is_directed()
        else nx.connected_components(G)
    )
    if len(components) <= 1:
        return _component_layout(G, seed, min_dist)

    components.sort(key=len, reverse=True)
    sublayouts: list[tuple[dict, float, float]] = []
    for comp in components:
        H = G.subgraph(comp)
        sub = _component_layout(H, seed, min_dist)
        xs = np.array([p[0] for p in sub.values()])
        ys = np.array([p[1] for p in sub.values()])
        cx, cy = float(xs.mean()), float(ys.mean())
        # Floor on width/height keeps singleton and 2-node components from
        # stacking on top of their neighbours when packed side-by-side.
        width = max(float(xs.max() - xs.min()), 0.8)
        height = max(float(ys.max() - ys.min()), 0.8)
        sub = {n: (float(p[0]) - cx, float(p[1]) - cy) for n, p in sub.items()}
        sublayouts.append((sub, width, height))

    pos: dict = {}
    x_cursor = 0.0
    for sub, width, _height in sublayouts:
        half_w = width / 2
        x_cursor += half_w
        for n, (x, y) in sub.items():
            pos[n] = np.array([x_cursor + x, y])
        x_cursor += half_w + gap
    return pos


def _component_layout(H: nx.Graph, seed: int, min_dist: float) -> dict:
    """Per-component layout. DAGs use a longest-path layered layout so
    sources sit at the top and sinks at the bottom, matching the visual
    flow of edges. Everything else uses Kamada-Kawai for uniform spacing.
    A repel pass enforces a minimum pairwise distance after layout."""
    if len(H) < 2:
        return nx.spring_layout(H, seed=seed)
    if H.is_directed() and nx.is_directed_acyclic_graph(H):
        pos = _layered_dag_layout(H)
    else:
        try:
            pos = nx.kamada_kawai_layout(H)
        except (nx.NetworkXError, ValueError):
            pos = nx.spring_layout(H, seed=seed)
    return _repel_overlaps(pos, min_dist)


def _layered_dag_layout(H: nx.DiGraph) -> dict:
    """Longest-path layering: layer[v] = max(layer[u]+1) over predecessors.

    Stamps the layer onto a shallow copy (so the caller's graph is
    untouched) and hands it to ``multipartite_layout`` with horizontal
    align — sources end up on the top row, sinks on the bottom.
    """
    layers: dict = {}
    for v in nx.topological_sort(H):
        preds = list(H.predecessors(v))
        layers[v] = max((layers[u] + 1 for u in preds), default=0)
    # Negate so multipartite_layout places the source (layer 0, now the
    # largest key) at the top and the sink (most negative key) at the
    # bottom — matplotlib's y axis points up.
    H_copy = H.copy()
    for v, layer in layers.items():
        H_copy.nodes[v]["_layer"] = -layer
    return nx.multipartite_layout(H_copy, subset_key="_layer", align="horizontal")


def _repel_overlaps(pos: dict, min_dist: float) -> dict:
    """Iteratively push pairs of nodes apart until every pair sits at
    least ``min_dist`` apart. Coordinates are 2D numpy arrays."""
    if not pos:
        return pos
    nodes = list(pos.keys())
    arr = np.array([pos[n] for n in nodes], dtype=float)
    rng = np.random.default_rng(0)
    for _ in range(_REPEL_ITERATIONS):
        moved = False
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                delta = arr[j] - arr[i]
                dist = float(np.linalg.norm(delta))
                if dist >= min_dist:
                    continue
                if dist < 1e-9:
                    # Same point: pick a deterministic-ish random axis.
                    delta = rng.normal(size=2)
                    dist = float(np.linalg.norm(delta))
                direction = delta / dist
                push = (min_dist - dist) / 2 + 1e-3
                arr[i] -= direction * push
                arr[j] += direction * push
                moved = True
        if not moved:
            break
    return {n: arr[i] for i, n in enumerate(nodes)}
