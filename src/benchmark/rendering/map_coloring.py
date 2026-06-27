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

from dataclasses import dataclass

import networkx as nx
from PIL import Image

from ._planar_map import render_planar_map
from .config import LabelStyle


@dataclass(frozen=True)
class Map:
    """Graph rendered as a map. Result of ``build_map``."""

    G: nx.Graph
    pos: dict
    show_labels: bool = False
    label_style: LabelStyle = "numeric"

    def render(self, pdf_path: str | None = None) -> Image.Image:
        return render_planar_map(
            self.G,
            self.pos,
            show_labels=self.show_labels,
            label_style=self.label_style,
            pdf_path=pdf_path,
        )


def build_map(
    G: nx.Graph,
    seed: int = 42,  # retained for API compatibility; unused
    show_labels: bool = False,
    pos: dict | None = None,
    label_style: LabelStyle = "numeric",
) -> Map:
    if pos is None:
        pos = {n: G.nodes[n]["pos"] for n in G.nodes()}
    return Map(G=G, pos=pos, show_labels=show_labels, label_style=label_style)


def render_map(
    G: nx.Graph,
    seed: int = 42,
    show_labels: bool = False,
    pos: dict | None = None,
    label_style: LabelStyle = "numeric",
    pdf_path: str | None = None,
) -> Image.Image:
    """Convenience: ``build_map(...).render()``."""
    return build_map(
        G, seed=seed, show_labels=show_labels, pos=pos, label_style=label_style
    ).render(pdf_path=pdf_path)
