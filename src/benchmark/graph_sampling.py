"""Random graph factories for benchmark tasks.

Each task has its own size preset because the maze and map disguises have
very different visual budgets: a legible maze tolerates more nodes than a
legible map.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay, Voronoi


_CONNECTIVITY_SIZES: dict[str, tuple[int, int]] = {
    "easy": (6, 8),
    "medium": (9, 12),
    "hard": (13, 16),
}

_COLORING_SIZES: dict[str, tuple[int, int]] = {
    "easy": (5, 7),
    "medium": (8, 10),
    "hard": (11, 14),
}


def connectivity_graph(
    rng: np.random.Generator,
    difficulty: str,
) -> tuple[nx.Graph, int, int]:
    """Return (G, entrance, exit) with balanced Yes/No answer distribution.

    Draws a sparse Erdős-Rényi graph likely to contain 1-3 components, then
    picks entrance/exit with a 50/50 coin flip between same-component (Yes)
    and different-component (No). If the draw turns out fully connected,
    retries with a slightly lower edge probability.
    """
    lo, hi = _CONNECTIVITY_SIZES[difficulty]
    n = int(rng.integers(lo, hi + 1))
    want_yes = bool(rng.integers(0, 2))

    p = 1.2 / n
    for _ in range(30):
        G = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(0, 2**31 - 1)))
        components = [c for c in nx.connected_components(G) if len(c) >= 1]
        if want_yes:
            big = [c for c in components if len(c) >= 2]
            if big:
                chosen = list(big[int(rng.integers(0, len(big)))])
                u, v = rng.choice(chosen, size=2, replace=False)
                return G, int(u), int(v)
        else:
            if len(components) >= 2:
                idx = rng.choice(len(components), size=2, replace=False)
                ca = list(components[int(idx[0])])
                cb = list(components[int(idx[1])])
                u = int(ca[int(rng.integers(0, len(ca)))])
                v = int(cb[int(rng.integers(0, len(cb)))])
                return G, u, v
        p *= 0.85 if want_yes else 1.0 / 0.85

    nodes = list(G.nodes())
    u, v = rng.choice(nodes, size=2, replace=False)
    return G, int(u), int(v)


def coloring_graph(rng: np.random.Generator, difficulty: str) -> nx.Graph:
    """Return a Delaunay triangulation over random points in the unit square.

    Keeping the full triangulation (no edge drops) lets the map-disguise
    renderer use Voronoi cells directly — adjacency in the Voronoi diagram
    equals adjacency in the Delaunay graph by duality, so the disguise
    matches G exactly. Chromatic number is ≤ 4 by the four-color theorem
    (typically 3 or 4 on small random point sets).
    """
    lo, hi = _COLORING_SIZES[difficulty]
    target = int(rng.integers(lo, hi + 1))

    for _ in range(80):
        pts = rng.random((target, 2))
        if not _voronoi_well_behaved(pts):
            continue
        try:
            tri = Delaunay(pts)
        except Exception:
            continue
        edges: set[tuple[int, int]] = set()
        for simplex in tri.simplices:
            for i, j in ((0, 1), (1, 2), (2, 0)):
                a, b = int(simplex[i]), int(simplex[j])
                edges.add((min(a, b), max(a, b)))

        G = nx.Graph()
        G.add_nodes_from(range(target))
        G.add_edges_from(edges)
        for i in range(target):
            G.nodes[i]["pos"] = (float(pts[i, 0]), float(pts[i, 1]))

        if nx.is_connected(G):
            assert nx.check_planarity(G)[0], "Delaunay triangulation must be planar"
            return G

    return nx.cycle_graph(target)


def _voronoi_well_behaved(pts: np.ndarray, max_ratio: float = 1.2) -> bool:
    """Reject point sets whose Voronoi diagram has vertices far outside the
    point cloud. Near-collinear triples produce distant circumcenters that
    stretch the map beyond recognition in the disguise renderer."""
    try:
        vor = Voronoi(pts)
    except Exception:
        return False
    center = pts.mean(axis=0)
    pt_radius = float(np.linalg.norm(pts - center, axis=1).max())
    if pt_radius < 1e-9:
        return False
    vor_radius = float(np.linalg.norm(vor.vertices - center, axis=1).max())
    return vor_radius <= max_ratio * pt_radius
