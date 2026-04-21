"""Plain graph rendering (direct view of the underlying task graph)."""

from __future__ import annotations

from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
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

    pos = nx.spring_layout(G, seed=42)
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
