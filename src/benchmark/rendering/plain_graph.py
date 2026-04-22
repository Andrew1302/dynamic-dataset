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


def render_graph(
    G: nx.Graph,
    highlights: dict[int, str] | None = None,
    with_labels: bool = True,
) -> Image.Image:
    """Render *G* to a square PIL image using a spring layout.

    Parameters
    ----------
    G
        NetworkX graph.
    highlights
        Optional ``{node: color}`` mapping. Highlighted nodes are drawn in
        the given color; unhighlighted nodes use the default palette.
    with_labels
        Whether to draw numeric node labels.
    """
    highlights = highlights or {}

    pos = _layout(G)
    node_colors = [highlights.get(n, _DEFAULT_COLOR) for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=with_labels,
        node_color=node_colors,
        node_size=500,
        font_size=10,
        font_weight="bold",
        edge_color="#2C3E50",
        width=1.4,
    )
    ax.set_axis_off()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def _layout(G: nx.Graph, seed: int = 42, gap: float = 0.25) -> dict:
    """Spring layout for connected graphs; for disconnected graphs, lay out
    each component independently and pack them side by side with a small
    gap. Edges may cross between components, which is fine — the goal is to
    keep components visually close so the viewer can compare them at once.
    """
    components = list(nx.connected_components(G))
    if len(components) <= 1:
        return nx.spring_layout(G, seed=seed)

    components.sort(key=len, reverse=True)
    sublayouts: list[tuple[dict, float, float]] = []
    for comp in components:
        H = G.subgraph(comp)
        sub = nx.spring_layout(H, seed=seed)
        xs = np.array([p[0] for p in sub.values()])
        ys = np.array([p[1] for p in sub.values()])
        cx, cy = float(xs.mean()), float(ys.mean())
        width = max(float(xs.max() - xs.min()), 0.2)
        height = max(float(ys.max() - ys.min()), 0.2)
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
