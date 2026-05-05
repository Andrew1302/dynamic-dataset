"""Shortest-path task paired with the Latin America road-trip disguise."""

from __future__ import annotations

import networkx as nx
import numpy as np
from PIL import Image

from ..base import BenchmarkTask
from ..graph_sampling import shortest_path_graph
from ..rendering import build_latin_america_map, render_graph
from ..rendering.latin_america_map import LatinAmericaMap


class ShortestPathTask(BenchmarkTask):
    name = "shortest_path"

    def sample_graph(self, rng: np.random.Generator, difficulty: str) -> nx.DiGraph:
        return shortest_path_graph(rng, difficulty)

    def solve(self, G: nx.DiGraph) -> str:
        s, t = G.graph["source"], G.graph["sink"]
        return str(nx.shortest_path_length(G, s, t, weight="weight"))

    def direct_prompt(self, G: nx.DiGraph) -> str:
        s, t = G.graph["source"], G.graph["sink"]
        return (
            f"Q: In the weighted directed acyclic graph shown, what is the "
            f"minimum-weight path total from vertex {s} to vertex {t}?\nA:"
        )

    def render_direct(self, G: nx.DiGraph) -> Image.Image:
        highlights = {
            G.graph["source"]: "#1D9E75",
            G.graph["sink"]: "#C2185B",
        }
        return render_graph(G, highlights=highlights, arrowsize=22, weighted=True)

    def disguise_prompt(self) -> str:
        return (
            "Q: The map of Latin America below shows several cities and the "
            "available driving routes between them. Each arrow indicates a "
            "one-way driving connection from one city to another, and is "
            "labeled with the typical driving time in hours. What is the "
            "minimum total driving time, in hours, from the start city to "
            "the end city?\nA:"
        )

    def disguise(self, G: nx.DiGraph, seed: int) -> LatinAmericaMap:
        return build_latin_america_map(G, seed=seed)
