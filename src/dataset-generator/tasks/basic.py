"""Basic graph property tasks: NodeCount, EdgeCount, NodeDegree, CycleCheck."""

from __future__ import annotations

import random

import networkx as nx

from . import GraphTask
from ..visualization import graph2img


class NodeCount(GraphTask):
    name = "node_count"

    def generate(self, G, **kw):
        image = graph2img(G, with_labels=False)
        answer = str(G.number_of_nodes())
        prompt = "Q: How many nodes are in this graph?\nA:"
        return {"prompt": prompt, "image": image, "answer": answer}


class EdgeCount(GraphTask):
    name = "edge_count"

    def generate(self, G, **kw):
        image = graph2img(G)
        answer = str(G.number_of_edges())
        prompt = "Q: How many edges are in this graph?\nA:"
        return {"prompt": prompt, "image": image, "answer": answer}


class NodeDegree(GraphTask):
    name = "node_degree"

    def generate(self, G, **kw):
        node = random.choice(list(G.nodes()))
        image = graph2img(G)
        degree = G.degree(node)
        prompt = f"Q: What is the degree of node {node}?\nA:"
        return {"prompt": prompt, "image": image, "answer": str(degree)}


class CycleCheck(GraphTask):
    name = "cycle_check"

    def generate(self, G, **kw):
        image = graph2img(G)
        has_cycle = True
        try:
            nx.find_cycle(G)
        except nx.NetworkXNoCycle:
            has_cycle = False
        answer = "Yes" if has_cycle else "No"
        prompt = "Q: Is there a cycle in this graph?\nA:"
        return {"prompt": prompt, "image": image, "answer": answer}
