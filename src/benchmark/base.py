"""
base.py — Core abstractions for the graph-disguise benchmark.

Architecture: Generate-Then-Project
    1. BaseGraphGenerator produces a connected nx.Graph (optionally weighted).
    2. ProblemVariant subclasses each project() that graph into a domain surface.
    3. are_isomorphic_instances() verifies structural identity across projections.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class ProjectionFailure(Exception):
    """Raised when a variant cannot produce a valid instance for a given graph.

    Callers should catch this and retry with a different seed.
    """


# ---------------------------------------------------------------------------
# Base graph generator
# ---------------------------------------------------------------------------


class BaseGraphGenerator:
    """Generates connected undirected graphs suitable for projection into any variant."""

    @staticmethod
    def generate(
        n_nodes: int,
        graph_type: str = "random",
        weighted: bool = False,
        seed: int | None = None,
    ) -> nx.Graph:
        """Generate a connected undirected graph.

        Parameters
        ----------
        n_nodes:
            Number of nodes.  Node labels are integers 0 .. n_nodes-1.
        graph_type:
            "random"  — Erdős–Rényi with p tuned for likely connectivity.
            "grid"    — 2-D grid; n_nodes must be a perfect square.
            "tree"    — Random labeled spanning tree (always connected).
            "sparse"  — Random spanning tree plus edges until |E| ≥ 1.2 * n_nodes.
        weighted:
            If True, assign integer weights drawn uniformly from [1, 10] to every edge.
        seed:
            Integer seed for full reproducibility.

        Returns
        -------
        nx.Graph with integer node labels 0 .. n_nodes-1.  Always connected.
        Edge attribute "weight" is present on every edge iff weighted=True.
        """
        rng_py = random.Random(seed)

        if graph_type == "random":
            G = BaseGraphGenerator._make_random(n_nodes, seed)
        elif graph_type == "grid":
            G = BaseGraphGenerator._make_grid(n_nodes)
        elif graph_type == "tree":
            G = nx.random_labeled_tree(n_nodes, seed=seed)
        elif graph_type == "sparse":
            G = BaseGraphGenerator._make_sparse(n_nodes, seed, rng_py)
        else:
            raise ValueError(f"Unknown graph_type: {graph_type!r}")

        # Ensure connectivity (needed for "random" which may be disconnected)
        G = BaseGraphGenerator._ensure_connected(G, rng_py)

        if weighted:
            rng_np = np.random.default_rng(seed)
            for u, v in G.edges():
                G[u][v]["weight"] = int(rng_np.integers(1, 11))

        return G

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_random(n: int, seed: int | None) -> nx.Graph:
        """Erdős–Rényi graph with p biased toward connectivity."""
        if n <= 1:
            return nx.Graph([(0,)] if n == 1 else [])
        p = min(0.99, math.log(max(n, 2)) / n + 0.25)
        return nx.erdos_renyi_graph(n, p, seed=seed)

    @staticmethod
    def _make_grid(n: int) -> nx.Graph:
        """2-D grid graph.  n must be a perfect square."""
        r = int(math.isqrt(n))
        if r * r != n:
            raise ValueError(
                f"graph_type='grid' requires n_nodes to be a perfect square, got {n}"
            )
        G = nx.grid_2d_graph(r, r)
        return nx.convert_node_labels_to_integers(G)

    @staticmethod
    def _make_sparse(n: int, seed: int | None, rng_py: random.Random) -> nx.Graph:
        """Random spanning tree plus extra random edges until |E| ≥ 1.2 * n."""
        G = nx.random_labeled_tree(n, seed=seed)
        target = max(n - 1, int(1.2 * n))
        nodes = list(G.nodes())
        attempts = 0
        while G.number_of_edges() < target and attempts < 10 * n:
            u, v = rng_py.sample(nodes, 2)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
            attempts += 1
        return G

    @staticmethod
    def _ensure_connected(G: nx.Graph, rng_py: random.Random) -> nx.Graph:
        """Bridge disconnected components by adding one edge per consecutive pair."""
        if nx.is_connected(G):
            return G
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        for i in range(len(components) - 1):
            u = rng_py.choice(sorted(components[i]))
            v = rng_py.choice(sorted(components[i + 1]))
            G.add_edge(u, v)
        return G


# ---------------------------------------------------------------------------
# Abstract variant interface
# ---------------------------------------------------------------------------


class ProblemVariant(ABC):
    """Abstract base for all domain variants.

    Every variant must be stateless after construction.  All state lives in
    the instance dict returned by project().
    """

    @abstractmethod
    def project(
        self,
        G: nx.Graph,
        source: int,
        target: int,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Project base graph G into this domain.

        Parameters
        ----------
        G:
            Connected undirected graph produced by BaseGraphGenerator.
        source, target:
            Node IDs in G designating start and end of the task.
        seed:
            Seed for any stochastic choices in the projection.

        Returns
        -------
        JSON-serializable dict fully describing the domain instance.
        Must not contain nx.Graph objects or numpy scalars.

        Raises
        ------
        ProjectionFailure if the variant cannot produce a valid instance.
        """

    @abstractmethod
    def solve(self, instance: dict[str, Any]) -> dict[str, Any]:
        """Solve the instance.

        Returns a JSON-serializable dict with at minimum:
            "path"      — domain-specific path representation
            "cost"      — total cost (steps or sum of weights)
            "base_path" — path expressed as node IDs in the original base graph
        """

    @abstractmethod
    def to_base_graph(self, instance: dict[str, Any]) -> nx.Graph:
        """Reconstruct the base graph purely from the domain instance.

        The reconstruction must not rely on the original G stored internally.
        Returns an nx.Graph with optional "weight" edge attributes.
        Node labels may differ from the original; isomorphism is checked
        structurally by are_isomorphic_instances().
        """

    @abstractmethod
    def verify(self, instance: dict[str, Any], solution: dict[str, Any]) -> bool:
        """Check correctness and optimality of a solution."""

    @abstractmethod
    def to_prompt(self, instance: dict[str, Any]) -> str:
        """Render the instance as a natural-language LLM prompt.

        Must expose only the domain surface — never the raw graph structure.
        """

    @abstractmethod
    def to_image(self, instance: dict[str, Any]) -> "PILImage.Image":
        """Render the instance as a PIL image for multimodal LLM input.

        The image should convey the same information as to_prompt() but visually.
        Domain-specific: graphs drawn as node-edge diagrams, mazes as pixel grids,
        puzzles as board illustrations, etc.
        """


# ---------------------------------------------------------------------------
# Isomorphism verifier
# ---------------------------------------------------------------------------


def are_isomorphic_instances(
    instance_a: dict[str, Any],
    variant_a: ProblemVariant,
    instance_b: dict[str, Any],
    variant_b: ProblemVariant,
) -> bool:
    """Ground-truth structural verifier for the benchmark.

    Converts both instances to base graphs via their respective to_base_graph()
    methods, then checks graph isomorphism using NetworkX's VF2 algorithm.
    Edge weights are compared as attributes when present.

    Returns True iff the two domain instances encode the same underlying graph.
    """
    G_a = variant_a.to_base_graph(instance_a)
    G_b = variant_b.to_base_graph(instance_b)

    def edge_match(d1: dict, d2: dict) -> bool:
        return d1.get("weight", 1) == d2.get("weight", 1)

    return nx.is_isomorphic(G_a, G_b, edge_match=edge_match)
