"""Directed connectivity task paired with the directed maze disguise."""

from __future__ import annotations

import networkx as nx
import numpy as np
from PIL import Image

from ..base import BenchmarkTask
from ..graph_sampling import directed_connectivity_graph
from ..rendering import RenderConfig, build_directed_maze, node_label, render_graph
from ..rendering.directed_maze import DirectedMaze


class DirectedConnectivityTask(BenchmarkTask):
    name = "directed_connectivity"

    def sample_graph(
        self,
        rng: np.random.Generator,
        difficulty: str,
        node_count: int | None = None,
    ) -> nx.DiGraph:
        G, entrance, exit_ = directed_connectivity_graph(
            rng, difficulty, node_count=node_count
        )
        G.graph["entrance"] = entrance
        G.graph["exit"] = exit_
        return G

    def solve(self, G: nx.DiGraph) -> str:
        return "Yes" if nx.has_path(G, G.graph["entrance"], G.graph["exit"]) else "No"

    def direct_prompt(self, G: nx.DiGraph, config: RenderConfig | None = None) -> str:
        cfg = config if config is not None else RenderConfig()
        u, v = G.graph["entrance"], G.graph["exit"]
        if cfg.label_style == "none":
            return (
                "Q: Following the arrow directions, is there a directed path "
                "from the green node to the red node?\nA:"
            )
        u_lbl = node_label(u, cfg.label_style)
        v_lbl = node_label(v, cfg.label_style)
        return (
            f"Q: Following the arrow directions, is there a directed path "
            f"from node {u_lbl} to node {v_lbl}?\nA:"
        )

    def render_direct(
        self,
        G: nx.DiGraph,
        config: RenderConfig | None = None,
        pdf_path: str | None = None,
    ) -> Image.Image:
        highlights = {
            G.graph["entrance"]: "#58CF76",
            G.graph["exit"]: "#EB5E5E",
        }
        return render_graph(
            G, highlights=highlights, arrowsize=22, config=config, pdf_path=pdf_path,
        )

    def disguise_prompt(self) -> str:
        return (
            "Q: Each corridor passage in this maze has an arrow showing the "
            "only allowed direction of travel. Following the arrows, is it "
            "possible to travel from the green cell to the red cell?\nA:"
        )

    def disguise(
        self,
        G: nx.DiGraph,
        seed: int,
        config: RenderConfig | None = None,
    ) -> DirectedMaze:
        cfg = config if config is not None else RenderConfig()
        highlight = cfg.label_style != "none"
        return build_directed_maze(
            G,
            seed=seed,
            entrance=G.graph["entrance"],
            exit=G.graph["exit"],
            highlight_all_nodes=highlight,
            label_style=cfg.label_style,
        )
