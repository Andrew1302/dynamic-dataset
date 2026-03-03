"""Advanced tasks: TriangleCounting, MaximumFlow."""

from __future__ import annotations

import random

import networkx as nx

from . import GraphTask
from ..visualization import graph2img


class TriangleCounting(GraphTask):
    name = "triangle_counting"

    def generate(self, G, **kw):
        image = graph2img(G)
        # sum of per-node triangle counts, each triangle counted 3 times
        triangles = sum(nx.triangles(G).values()) // 3
        prompt = "Q: How many triangles are in this graph?\nA:"
        return {"prompt": prompt, "image": image, "answer": str(triangles)}


class MaximumFlow(GraphTask):
    name = "maximum_flow"

    def generate(self, G, **kw):
        """G should be a directed graph with 'weight' as capacity."""
        nodes = list(G.nodes())
        u, v = random.sample(nodes, 2)
        image = graph2img(G, weighted=True)

        try:
            flow_value, _ = nx.maximum_flow(G, u, v, capacity="weight")
        except nx.NetworkXError:
            # No path exists
            flow_value = 0

        prompt = f"Q: What is the maximum capacity of the flow from node {u} to node {v}?\nA:"
        return {"prompt": prompt, "image": image, "answer": str(flow_value)}
