"""Graph coloring (chromatic number) task paired with the map disguise."""

from __future__ import annotations

import networkx as nx
import numpy as np
from PIL import Image

from ..base import BenchmarkTask
from ..graph_sampling import coloring_graph
from ..rendering import render_graph, render_map


class ColoringTask(BenchmarkTask):
    name = "coloring"

    def sample_graph(self, rng: np.random.Generator, difficulty: str) -> nx.Graph:
        return coloring_graph(rng, difficulty)

    def solve(self, G: nx.Graph) -> str:
        return str(_chromatic_number(G))

    def direct_prompt(self, G: nx.Graph) -> str:
        return (
            "Q: What is the minimum number of colors needed to color this graph "
            "so that no two adjacent nodes share a color?\nA:"
        )

    def render_direct(self, G: nx.Graph) -> Image.Image:
        return render_graph(G)

    def disguise_prompt(self) -> str:
        return (
            "Q: How many colors are needed to color this map so that no two "
            "neighboring regions share the same color?\nA:"
        )

    def render_disguise(self, G: nx.Graph, seed: int) -> Image.Image:
        pos = {n: G.nodes[n].get("pos") for n in G.nodes()}
        if all(p is not None for p in pos.values()):
            return render_map(G, seed=seed, pos=pos)
        return render_map(G, seed=seed)


def _chromatic_number(G: nx.Graph) -> int:
    """Exact chromatic number via simple k-coloring backtracking.

    Iterates ``k = 1, 2, …`` and returns the smallest ``k`` for which a
    valid coloring exists. Fast enough for the benchmark's graph sizes
    (n ≤ 14).
    """
    if G.number_of_nodes() == 0:
        return 0
    if G.number_of_edges() == 0:
        return 1

    nodes = sorted(G.nodes(), key=lambda v: -G.degree(v))
    adj = {v: set(G.neighbors(v)) for v in G.nodes()}

    for k in range(1, len(nodes) + 1):
        color: dict[int, int] = {}
        if _try_color(nodes, adj, color, k, 0):
            return k
    return len(nodes)


def _try_color(
    nodes: list,
    adj: dict,
    color: dict,
    k: int,
    idx: int,
) -> bool:
    if idx == len(nodes):
        return True
    v = nodes[idx]
    used = {color[u] for u in adj[v] if u in color}
    for c in range(k):
        if c in used:
            continue
        color[v] = c
        if _try_color(nodes, adj, color, k, idx + 1):
            return True
        del color[v]
    return False
