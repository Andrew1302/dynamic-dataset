"""Text-format an adjacency list to append to a direct-view prompt.

The list mirrors whatever node-label style is used on the image so that
prompt and image refer to the same names. For ``label_style="none"`` (no
labels rendered on the image), numeric indices are used as the legend —
there's no way to refer to a node textually otherwise, and the user
explicitly opted to send the list in this case.

The literature-standard adjacency-list notation is used: ``Adj[u]`` lists
the neighbors of node ``u``. A one-line "Encoding" legend describing only
the form actually emitted is prepended so the model knows how to read it.
"""

from __future__ import annotations

import networkx as nx

from .config import LabelStyle, node_label


def format_adjacency_list(G: nx.Graph, label_style: LabelStyle) -> str:
    """Return a text adjacency list for *G*, prefixed with an encoding legend.

    - Unweighted: ``Adj[u] = [v1, v2, ...]`` lists the neighbors of ``u``.
    - Weighted (any edge carries a ``weight`` attribute): ``Adj[u] =
      [(v1, w1), (v2, w2), ...]`` pairs each neighbor with its edge weight.
    - Directed graphs list out-neighbors only (``G.successors``).

    Nodes and each neighbor list are sorted by node id for determinism.
    """
    nodes = sorted(G.nodes())
    label_for = {n: node_label(n, label_style) for n in nodes}
    if label_style == "none":
        # The image has no labels; use numeric indices in the list so the
        # list is still interpretable.
        label_for = {n: str(int(n)) for n in nodes}

    weighted = any("weight" in d for _, _, d in G.edges(data=True))

    lines = [_legend(weighted)]
    for u in nodes:
        neighbors = sorted(G.neighbors(u))
        if weighted:
            entries = [
                f"({label_for[v]}, {int(G[u][v].get('weight', 1))})" for v in neighbors
            ]
        else:
            entries = [label_for[v] for v in neighbors]
        lines.append(f"Adj[{label_for[u]}] = [{', '.join(entries)}]")
    return "\n".join(lines)


def _legend(weighted: bool) -> str:
    """One-line description of the adjacency-list encoding actually emitted."""
    if weighted:
        return (
            "Encoding: adjacency list. Adj[u] = [(v, w), ...] lists each "
            "neighbor v of node u and the weight w of edge (u, v)."
        )
    return (
        "Provided graph adjacency list."
        "Encoding: adjacency list. Adj[u] = [v1, v2, ...] lists the "
        "neighbors of node u."
    )
