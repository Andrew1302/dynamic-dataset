import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image


def graph2img(G: nx.Graph, weighted: bool = False, with_labels: bool = True) -> Image.Image:
    """Render a networkx graph to a PIL Image.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        The graph to draw.
    weighted : bool
        If True, draw edge weight labels.
    with_labels : bool
        If True (default), draw node index labels inside each node.
        Pass False to hide node indices (e.g. for node-counting tasks).

    Returns
    -------
    PIL.Image.Image
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=42)

    is_directed = G.is_directed()

    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=with_labels,
        node_color="lightblue",
        node_size=400,
        font_size=9,
        arrows=is_directed,
    )

    if weighted:
        labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax, font_size=7)

    ax.set_title("")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
