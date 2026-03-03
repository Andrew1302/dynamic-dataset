"""Graph generation utilities.

Provides functions to create random graphs of various types and sizes.
"""

import random

import networkx as nx


# ---------------------------------------------------------------------------
# Size presets (node counts) following GraphQA conventions
# ---------------------------------------------------------------------------
SIZE_RANGES = {
    "small": (5, 9),
    "medium": (10, 14),
    "large": (15, 19),
}


def random_node_count(size: str = "small") -> int:
    """Return a random node count for the given size preset.

    Pass ``size="all"`` to pick uniformly from small, medium, and large.
    """
    if size == "all":
        size = random.choice(list(SIZE_RANGES.keys()))
    lo, hi = SIZE_RANGES[size]
    return random.randint(lo, hi)


# ---------------------------------------------------------------------------
# Graph generators
# ---------------------------------------------------------------------------


def erdos_renyi(n: int, p: float | None = None) -> nx.Graph:
    if p is None:
        p = random.uniform(0.2, 0.5)
    return nx.erdos_renyi_graph(n, p)


def barabasi_albert(n: int, m: int | None = None) -> nx.Graph:
    if m is None:
        m = random.randint(1, max(1, n // 4))
    return nx.barabasi_albert_graph(n, m)


def watts_strogatz(n: int, k: int | None = None, p: float = 0.3) -> nx.Graph:
    if k is None:
        k = min(n - 2, max(2, n // 3))
    # k must be even for watts_strogatz_graph
    k = max(2, k if k % 2 == 0 else k - 1)
    return nx.watts_strogatz_graph(n, k, p)


def connected_watts_strogatz(n: int, k: int | None = None, p: float = 0.3) -> nx.Graph:
    if k is None:
        k = min(n - 2, max(2, n // 3))
    k = max(2, k if k % 2 == 0 else k - 1)
    return nx.connected_watts_strogatz_graph(n, k, p)


def complete_graph(n: int) -> nx.Graph:
    return nx.complete_graph(n)


def star_graph(n: int) -> nx.Graph:
    # nx.star_graph(k) gives k+1 nodes; we want n nodes total
    return nx.star_graph(n - 1)


def path_graph(n: int) -> nx.Graph:
    return nx.path_graph(n)


def random_tree(n: int) -> nx.Graph:
    return nx.random_labeled_tree(n)


def scale_free(n: int) -> nx.Graph:
    return nx.scale_free_graph(n).to_undirected()


def stochastic_block_model(
    n: int,
    num_blocks: int = 2,
    p_in: float = 0.5,
    p_out: float = 0.05,
) -> nx.Graph:
    sizes = []
    remaining = n
    for i in range(num_blocks - 1):
        s = remaining // (num_blocks - i)
        sizes.append(s)
        remaining -= s
    sizes.append(remaining)
    probs = [
        [p_in if i == j else p_out for j in range(num_blocks)]
        for i in range(num_blocks)
    ]
    return nx.stochastic_block_model(sizes, probs)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

_UNWEIGHTED_GENERATORS = [
    erdos_renyi,
    barabasi_albert,
    complete_graph,
    star_graph,
    path_graph,
    random_tree,
]
_CONNECTED_GENERATORS = [
    connected_watts_strogatz,
    complete_graph,
    star_graph,
    path_graph,
    random_tree,
]


def random_graph(n: int) -> nx.Graph:
    """Return a random unweighted graph with *n* nodes."""
    gen = random.choice(_UNWEIGHTED_GENERATORS)
    return gen(n)


def random_connected_graph(n: int) -> nx.Graph:
    """Return a random connected unweighted graph with *n* nodes."""
    gen = random.choice(_CONNECTED_GENERATORS)
    return gen(n)


def add_random_weights(G: nx.Graph, lo: int = 1, hi: int | None = None) -> nx.Graph:
    """Add random integer weights to all edges (in-place). Returns *G*."""
    if hi is None:
        hi = max(10, G.number_of_nodes() // 2)
    for u, v in G.edges():
        G[u][v]["weight"] = random.randint(lo, hi)
    return G


def random_weighted_connected_graph(n: int) -> nx.Graph:
    """Return a random connected graph with random integer edge weights."""
    G = random_connected_graph(n)
    return add_random_weights(G)


def random_possibly_disconnected(n: int) -> nx.Graph:
    """Return a graph that may or may not be connected (useful for connectivity tasks)."""
    if random.random() < 0.5:
        return erdos_renyi(n, p=random.uniform(0.05, 0.2))
    else:
        return random_connected_graph(n)


def random_directed_weighted_graph(n: int) -> nx.DiGraph:
    """Return a random directed graph with capacity weights (for max-flow)."""
    p = random.uniform(0.3, 0.6)
    G = nx.erdos_renyi_graph(n, p, directed=True)
    for u, v in G.edges():
        G[u][v]["weight"] = random.randint(1, 20)
    return G
