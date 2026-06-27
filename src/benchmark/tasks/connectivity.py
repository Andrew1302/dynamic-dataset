"""Connectivity-check task paired with the maze disguise.

.. deprecated::
    The undirected ``connectivity`` task is deprecated and no longer part
    of the default demo task set. The directed variant
    (:mod:`directed_connectivity`) covers the same evaluation surface on
    a directed graph and should be used instead.
"""

from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
from PIL import Image

from ..base import BenchmarkTask
from ..graph_sampling import connectivity_graph
from ..rendering import RenderConfig, build_maze, node_label, render_graph
from ..rendering.maze import Maze


class ConnectivityTask(BenchmarkTask):
    name = "connectivity"

    def __init__(self) -> None:
        warnings.warn(
            "The 'connectivity' benchmark task is deprecated; use "
            "'directed_connectivity' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def sample_graph(
        self,
        rng: np.random.Generator,
        difficulty: str,
        node_count: int | None = None,
    ) -> nx.Graph:
        G, entrance, exit_ = connectivity_graph(rng, difficulty, node_count=node_count)
        G.graph["entrance"] = entrance
        G.graph["exit"] = exit_
        return G

    def solve(self, G: nx.Graph) -> str:
        u, v = G.graph["entrance"], G.graph["exit"]
        return "Yes" if nx.has_path(G, u, v) else "No"

    def direct_prompt(self, G: nx.Graph, config: RenderConfig | None = None) -> str:
        cfg = config if config is not None else RenderConfig()
        u, v = G.graph["entrance"], G.graph["exit"]
        if cfg.label_style == "none":
            return "Q: Is there a path from the green node to the red node?\nA:"
        u_lbl = node_label(u, cfg.label_style)
        v_lbl = node_label(v, cfg.label_style)
        return f"Q: Is there a path from node {u_lbl} to node {v_lbl}?\nA:"

    def render_direct(
        self,
        G: nx.Graph,
        config: RenderConfig | None = None,
        pdf_path: str | None = None,
    ) -> Image.Image:
        highlights = {
            G.graph["entrance"]: "#58CF76",
            G.graph["exit"]: "#EB5E5E",
        }
        return render_graph(
            G, highlights=highlights, config=config, pdf_path=pdf_path,
        )

    def disguise_prompt(self) -> str:
        return (
            "Q: Is it possible to travel from the green cell to the red cell "
            "through the corridors of this maze?\nA:"
        )

    def disguise(
        self,
        G: nx.Graph,
        seed: int,
        config: RenderConfig | None = None,
    ) -> Maze:
        cfg = config if config is not None else RenderConfig()
        # label_style="none" → no decoy node markers in the maze. The
        # entrance and exit are always painted, since the prompt asks
        # for "green cell" → "red cell".
        highlight = cfg.label_style != "none"
        return build_maze(
            G,
            seed=seed,
            entrance=G.graph["entrance"],
            exit=G.graph["exit"],
            highlight_all_nodes=highlight,
            label_style=cfg.label_style,
        )
