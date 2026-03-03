"""Pathfinding tasks: ShortestPath, MST."""

from __future__ import annotations

import random

import networkx as nx

from . import GraphTask
from ..visualization import graph2img


class ShortestPath(GraphTask):
    name = "shortest_path"

    def generate(self, G, **kw):
        nodes = list(G.nodes())
        u, v = random.sample(nodes, 2)

        weighted = any("weight" in d for _, _, d in G.edges(data=True))
        image = graph2img(G, weighted=weighted)

        if weighted:
            length = nx.dijkstra_path_length(G, source=u, target=v, weight="weight")
        else:
            length = nx.shortest_path_length(G, source=u, target=v)

        prompt = (
            f"Q: What is the length of the shortest path from node {u} to node {v}?\nA:"
        )
        return {"prompt": prompt, "image": image, "answer": str(length)}


class MST(GraphTask):
    name = "mst"

    def generate(self, G, **kw):
        image = graph2img(G, weighted=True)
        mst = nx.minimum_spanning_tree(G, algorithm="kruskal", weight="weight")
        weight = mst.size(weight="weight")
        prompt = "Q: What is the weight of the minimum spanning tree of this graph?\nA:"
        return {"prompt": prompt, "image": image, "answer": str(weight)}
