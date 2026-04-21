"""Connectivity-check task paired with the maze disguise."""

from __future__ import annotations

import networkx as nx
import numpy as np
from PIL import Image

from ..base import BenchmarkTask
from ..graph_sampling import connectivity_graph
from ..rendering import render_graph, render_maze


_DIRECT_HIGHLIGHTS = {"_entrance_color": "#58CF76", "_exit_color": "#EB5E5E"}


class ConnectivityTask(BenchmarkTask):
    name = "connectivity"

    def sample_graph(self, rng: np.random.Generator, difficulty: str) -> nx.Graph:
        G, entrance, exit_ = connectivity_graph(rng, difficulty)
        G.graph["entrance"] = entrance
        G.graph["exit"] = exit_
        return G

    def solve(self, G: nx.Graph) -> str:
        u, v = G.graph["entrance"], G.graph["exit"]
        return "Yes" if nx.has_path(G, u, v) else "No"

    def direct_prompt(self, G: nx.Graph) -> str:
        u, v = G.graph["entrance"], G.graph["exit"]
        return f"Q: Is there a path from node {u} to node {v}?\nA:"

    def render_direct(self, G: nx.Graph) -> Image.Image:
        highlights = {
            G.graph["entrance"]: _DIRECT_HIGHLIGHTS["_entrance_color"],
            G.graph["exit"]: _DIRECT_HIGHLIGHTS["_exit_color"],
        }
        return render_graph(G, highlights=highlights)

    def disguise_prompt(self) -> str:
        return (
            "Q: Is it possible to travel from the green cell to the red cell "
            "through the corridors of this maze?\nA:"
        )

    def render_disguise(self, G: nx.Graph, seed: int) -> Image.Image:
        return render_maze(
            G,
            seed=seed,
            entrance=G.graph["entrance"],
            exit=G.graph["exit"],
        )
