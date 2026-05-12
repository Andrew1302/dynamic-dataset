"""Text-format an adjacency matrix to append to a direct-view prompt.

The matrix mirrors whatever node-label style is used on the image so
that prompt and image refer to the same names. For ``label_style="none"``
(no labels rendered on the image), numeric indices are used as the
matrix legend — there's no way to refer to a node textually otherwise,
and the user explicitly opted to send the matrix in this case.
"""

from __future__ import annotations

import networkx as nx

from .config import LabelStyle, node_label


def format_adjacency(G: nx.Graph, label_style: LabelStyle) -> str:
    """Return a text adjacency matrix for *G*.

    - Unweighted: cells are ``1`` (edge) or ``0`` (no edge).
    - Weighted (any edge carries a ``weight`` attribute): cells are the
      integer weight, with ``-`` for missing edges so 0 isn't confused
      with "no edge". Directed graphs produce an asymmetric matrix.
    """
    nodes = list(G.nodes())
    label_for = {n: node_label(n, label_style) for n in nodes}
    if label_style == "none":
        # The image has no labels; use numeric indices in the matrix so
        # at least the matrix is interpretable.
        label_for = {n: str(int(n)) for n in nodes}

    weighted = any("weight" in d for _, _, d in G.edges(data=True))

    col_w = max(1, max(len(label_for[n]) for n in nodes))
    if weighted:
        max_w = max((int(d["weight"]) for _, _, d in G.edges(data=True)), default=0)
        col_w = max(col_w, len(str(max_w)), 1)

    header = " " * (col_w + 2) + " ".join(f"{label_for[n]:>{col_w}}" for n in nodes)
    lines = [header]
    for u in nodes:
        row = [f"{label_for[u]:>{col_w}} "]
        for v in nodes:
            if G.has_edge(u, v):
                if weighted:
                    w = G[u][v].get("weight")
                    cell = str(int(w)) if w is not None else "1"
                else:
                    cell = "1"
            else:
                cell = "-" if weighted else "0"
            row.append(f"{cell:>{col_w}}")
        lines.append(" ".join(row))
    return "\n".join(lines)
