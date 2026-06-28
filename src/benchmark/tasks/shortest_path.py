"""Shortest-path task paired with the Latin America road-trip disguise."""

from __future__ import annotations

import networkx as nx
import numpy as np
from PIL import Image

from ..base import BenchmarkTask
from ..graph_sampling import shortest_path_graph
from ..rendering import RenderConfig, build_latin_america_map, node_label, render_graph
from ..rendering.latin_america_map import LatinAmericaMap


class ShortestPathTask(BenchmarkTask):
    name = "shortest_path"

    def sample_graph(
        self,
        rng: np.random.Generator,
        difficulty: str,
        node_count: int | None = None,
        **kwargs,  # task-specific knobs (e.g. target_chromatic) are not supported
    ) -> nx.DiGraph:
        self._warn_unsupported_kwargs(kwargs)
        return shortest_path_graph(rng, difficulty, node_count=node_count)

    def solve(self, G: nx.DiGraph) -> str:
        s, t = G.graph["source"], G.graph["sink"]
        return str(nx.shortest_path_length(G, s, t, weight="weight"))

    def direct_prompt(self, G: nx.DiGraph, config: RenderConfig | None = None) -> str:
        cfg = config if config is not None else RenderConfig()
        s, t = G.graph["source"], G.graph["sink"]
        if cfg.label_style == "none":
            return (
                "Q: In the weighted directed acyclic graph shown, what is the "
                "minimum-weight path total from the green vertex (source) to "
                "the pink vertex (sink)?\nA:"
            )
        s_lbl = node_label(s, cfg.label_style)
        t_lbl = node_label(t, cfg.label_style)
        return (
            f"Q: In the weighted directed acyclic graph shown, what is the "
            f"minimum-weight path total from vertex {s_lbl} to vertex {t_lbl}?\nA:"
        )

    def render_direct(
        self,
        G: nx.DiGraph,
        config: RenderConfig | None = None,
        pdf_path: str | None = None,
    ) -> Image.Image:
        highlights = {
            G.graph["source"]: "#1D9E75",
            G.graph["sink"]: "#C2185B",
        }
        return render_graph(
            G, highlights=highlights, arrowsize=22, weighted=True,
            config=config, pdf_path=pdf_path,
        )

    def disguise_prompt(self) -> str:
        return (
            "Q: The map of Latin America below shows several cities and the "
            "available driving routes between them. Each arrow indicates a "
            "one-way driving connection from one city to another, and is "
            "labeled with the typical driving time in hours. What is the "
            "minimum total driving time, in hours, from the start city to "
            "the end city?\nA:"
        )

    def disguise(
        self,
        G: nx.DiGraph,
        seed: int,
        config: RenderConfig | None = None,
    ) -> LatinAmericaMap:
        cfg = config if config is not None else RenderConfig()
        # label_style="none" → hide intermediate city markers and their
        # integer ID labels. Endpoints (start/end) stay because the
        # prompt explicitly references them.
        show_intermediate = cfg.label_style != "none"
        return build_latin_america_map(
            G,
            seed=seed,
            label_style=cfg.label_style,
            show_intermediate_nodes=show_intermediate,
        )
