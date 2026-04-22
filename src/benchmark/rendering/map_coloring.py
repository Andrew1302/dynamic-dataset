"""Map disguise: graph → uncolored political map with black borders.

Each node becomes a geographic region. Every region is filled with pure
white — no color hints — and separated from its neighbors by a uniform
black border. The LLM is asked how many colors the map needs, which
equals the graph's chromatic number.

Regions are built as the radial dual of the graph's straight-line planar
embedding, so two regions share a border iff the corresponding nodes are
adjacent in G (exact adjacency, by construction).
"""

from __future__ import annotations

import networkx as nx
from PIL import Image

from ._planar_map import render_planar_map


def render_map(
    G: nx.Graph,
    seed: int = 42,  # retained for API compatibility; unused
    show_labels: bool = False,
    pos: dict | None = None,
) -> Image.Image:
    """Render *G* as a black-and-white map. Requires each node in ``G`` to
    carry a ``pos`` attribute or ``pos`` to be passed in explicitly."""
    if pos is None:
        pos = {n: G.nodes[n]["pos"] for n in G.nodes()}
    return render_planar_map(G, pos, show_labels=show_labels)
