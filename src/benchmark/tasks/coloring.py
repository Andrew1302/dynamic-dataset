"""Graph coloring (chromatic number) task paired with the map disguise."""

from __future__ import annotations

import networkx as nx
import numpy as np
from PIL import Image

from ..base import BenchmarkTask
from ..graph_sampling import chromatic_number, coloring_graph
from ..rendering import RenderConfig, build_map, render_graph
from ..rendering.map_coloring import Map


class ColoringTask(BenchmarkTask):
    name = "coloring"

    def sample_graph(
        self,
        rng: np.random.Generator,
        difficulty: str,
        node_count: int | None = None,
        target_chromatic: int | None = None,
    ) -> nx.Graph:
        return coloring_graph(
            rng,
            difficulty,
            node_count=node_count,
            target_chromatic=target_chromatic,
        )

    def solve(self, G: nx.Graph) -> str:
        return str(chromatic_number(G))

    def direct_prompt(self, G: nx.Graph, config: RenderConfig | None = None) -> str:
        return (
            "Q: What is the minimum number of colors needed to color this graph "
            "so that no two adjacent nodes share a color?\nA:"
        )

    def render_direct(
        self,
        G: nx.Graph,
        config: RenderConfig | None = None,
        pdf_path: str | None = None,
    ) -> Image.Image:
        return render_graph(G, config=config, pdf_path=pdf_path)

    def disguise_prompt(self) -> str:
        return (
            "Q: How many colors are needed to color this map so that no two "
            "neighboring regions share the same color?\nA:"
        )

    def disguise(
        self,
        G: nx.Graph,
        seed: int,
        config: RenderConfig | None = None,
    ) -> Map:
        cfg = config if config is not None else RenderConfig()
        # label_style="none" → hide region labels. That removes the nodes
        # from the disguise (the existing show_labels flag is the
        # mechanism: regions stay, but no node markers / text).
        show_labels = cfg.label_style != "none"
        pos = {n: G.nodes[n].get("pos") for n in G.nodes()}
        # In special / balanced mode G is a subgraph of a triangulation, so the
        # map must know the real adjacencies: adjacent regions share a border,
        # non-adjacent regions are separated by an open-water gap. The default
        # full-triangulation path leaves ``edges=None`` and renders as before.
        edges = None
        if G.graph.get("chromatic_target") is not None:
            edges = {tuple(sorted(e)) for e in G.edges()}
        if all(p is not None for p in pos.values()):
            return build_map(
                G,
                seed=seed,
                pos=pos,
                show_labels=show_labels,
                label_style=cfg.label_style,
                edges=edges,
            )
        return build_map(
            G,
            seed=seed,
            show_labels=show_labels,
            label_style=cfg.label_style,
            edges=edges,
        )
