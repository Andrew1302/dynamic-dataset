"""Connectivity-related tasks."""

from __future__ import annotations

import random

import networkx as nx

from . import GraphTask
from ..visualization import graph2img


class EdgeExistence(GraphTask):
    name = "edge_existence"

    def generate(self, G, **kw):
        nodes = list(G.nodes())
        u, v = random.sample(nodes, 2)
        image = graph2img(G)
        exists = G.has_edge(u, v)
        answer = "Yes" if exists else "No"
        prompt = f"Q: Is node {u} connected to node {v}?\nA:"
        return {"prompt": prompt, "image": image, "answer": answer}


class ConnectedNodes(GraphTask):
    name = "connected_nodes"

    def generate(self, G, **kw):
        node = random.choice(list(G.nodes()))
        image = graph2img(G)
        neighbors = sorted(G.neighbors(node))
        answer = ", ".join(str(n) for n in neighbors) if neighbors else "None"
        prompt = f"Q: List all the nodes connected to {node} in ascending order.\nA:"
        return {"prompt": prompt, "image": image, "answer": answer}


class DisconnectedNodes(GraphTask):
    name = "disconnected_nodes"

    def generate(self, G, **kw):
        node = random.choice(list(G.nodes()))
        image = graph2img(G)
        neighbors = set(G.neighbors(node))
        disconnected = sorted(n for n in G.nodes() if n != node and n not in neighbors)
        answer = ", ".join(str(n) for n in disconnected) if disconnected else "None"
        prompt = f"Q: List all the nodes that are not connected to {node} in ascending order.\nA:"
        return {"prompt": prompt, "image": image, "answer": answer}


class Reachability(GraphTask):
    name = "reachability"

    def generate(self, G, **kw):
        nodes = list(G.nodes())
        u, v = random.sample(nodes, 2)
        image = graph2img(G)
        reachable = nx.has_path(G, u, v)
        answer = "Yes" if reachable else "No"
        prompt = f"Q: Is there a path from node {u} to node {v}?\nA:"
        return {"prompt": prompt, "image": image, "answer": answer}


class ConnectivityCheck(GraphTask):
    name = "connectivity_check"

    def generate(self, G, **kw):
        image = graph2img(G)
        answer = "Yes" if nx.is_connected(G) else "No"
        prompt = "Q: Is this graph connected?\nA:"
        return {"prompt": prompt, "image": image, "answer": answer}


class ConnectedComponents(GraphTask):
    name = "connected_components"

    def generate(self, G, **kw):
        image = graph2img(G)
        answer = str(nx.number_connected_components(G))
        prompt = "Q: How many connected components does this graph have?\nA:"
        return {"prompt": prompt, "image": image, "answer": answer}
