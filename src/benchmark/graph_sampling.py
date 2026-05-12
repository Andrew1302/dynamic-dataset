"""Random graph factories for benchmark tasks.

Each task has its own size preset because the maze and map disguises have
very different visual budgets: a legible maze tolerates more nodes than a
legible map.

Connectivity invariant
----------------------
Graphs produced by :func:`connectivity_graph` are always **subgraphs of a
2D lattice**: every node ``n`` carries ``G.nodes[n]["lattice"] = (lr, lc)``
and every edge ``(u, v)`` connects lattice-adjacent positions (Manhattan
distance 1). This lets the maze disguise render correctly by construction
— every graph edge maps to a straight corridor between two block centers,
with no routing search that could silently drop an edge.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay, Voronoi


_CONNECTIVITY_LATTICE: dict[str, tuple[int, int]] = {
    "easy": (3, 3),
    "medium": (4, 4),
    "hard": (5, 5),
}

_CONNECTIVITY_FILL: dict[str, tuple[float, float]] = {
    "easy": (0.7, 0.9),
    "medium": (0.55, 0.8),
    "hard": (0.5, 0.75),
}

# Fraction of the *extra* (non-spanning-tree) lattice edges kept. A random
# spanning tree is always kept — ``extra=0`` makes the blob a tree, ``1.0``
# gives the full lattice-induced subgraph. Intermediate values give a
# branchy-but-not-perfectly-grid look.
_CONNECTIVITY_EXTRA_EDGES: tuple[float, float] = (0.25, 0.6)

_COLORING_SIZES: dict[str, tuple[int, int]] = {
    "easy": (5, 7),
    "medium": (8, 10),
    "hard": (11, 14),
}

_SHORTEST_PATH_SIZES: dict[str, tuple[int, int]] = {
    "easy": (3, 5),
    "medium": (6, 8),
    "hard": (9, 12),
}
_SHORTEST_PATH_P = 0.18
_SHORTEST_PATH_WEIGHT = (1, 10)
_SHORTEST_PATH_MAX_ATTEMPTS = 200


_NBRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def connectivity_graph(
    rng: np.random.Generator,
    difficulty: str,
    node_count: int | None = None,
) -> tuple[nx.Graph, int, int]:
    """Return ``(G, entrance, exit)`` for the connectivity task.

    The pair is balanced 50/50 between reachable (Yes) and unreachable (No).
    For Yes, entrance/exit are non-adjacent within the same component. For
    No, both sit inside components of size ≥ 2 (neither endpoint is a
    lone dot). ``G`` is always a lattice subgraph — see module docstring.

    Parameters
    ----------
    node_count
        If set, the returned graph has exactly ``node_count`` nodes
        (lattice size and fill range are overridden accordingly). Used
        by the vertex-sweep CLI; for ``node_count < 4`` falls back to
        the smallest viable graph.
    """
    H, W = _CONNECTIVITY_LATTICE[difficulty]
    fill_lo, fill_hi = _CONNECTIVITY_FILL[difficulty]
    if node_count is not None:
        H, W = _lattice_for_node_count(node_count)
    want_yes = bool(rng.integers(0, 2))

    if want_yes:
        return _connected_lattice_pair(rng, H, W, fill_lo, fill_hi, node_count)
    return _disconnected_lattice_pair(rng, H, W, fill_lo, fill_hi, node_count)


def _lattice_for_node_count(node_count: int) -> tuple[int, int]:
    """Square-ish lattice with at least 1.6× headroom over ``node_count``.

    Headroom is needed because the blob-growth step picks from a
    connected subset and may not be able to hit ``node_count`` exactly
    in a tight lattice (the disconnected variant also splits cells
    along an empty corridor).
    """
    import math
    side = max(2, int(math.ceil(math.sqrt(max(node_count, 1) * 1.8))))
    return side, side


def _connected_lattice_pair(
    rng: np.random.Generator,
    H: int,
    W: int,
    fill_lo: float,
    fill_hi: float,
    node_count: int | None = None,
) -> tuple[nx.Graph, int, int]:
    """One lattice-connected blob. Pick a non-adjacent pair inside it."""
    if node_count is not None:
        n_target = max(3, min(node_count, H * W))
    else:
        fill = float(rng.uniform(fill_lo, fill_hi))
        n_target = max(3, int(round(H * W * fill)))
        n_target = min(n_target, H * W)

    allowed = [(r, c) for r in range(H) for c in range(W)]
    blob = _grow_lattice_blob(rng, allowed, n_target)
    G = _sparsify_lattice(_lattice_subgraph(blob), rng)

    # Lattice is bipartite (no triangles), so any connected blob of size ≥ 3
    # contains a non-adjacent pair.
    non_adj = [(u, v) for u, v in nx.non_edges(G) if u < v]
    u, v = non_adj[int(rng.integers(0, len(non_adj)))]
    return G, int(u), int(v)


def _disconnected_lattice_pair(
    rng: np.random.Generator,
    H: int,
    W: int,
    fill_lo: float,
    fill_hi: float,
    node_count: int | None = None,
) -> tuple[nx.Graph, int, int]:
    """Split the lattice along a row or column, grow a blob on each side.

    The split line is kept empty so the two blobs can't be lattice-adjacent
    — the graph is disconnected by construction. Each side has ≥ H or ≥ W
    cells available (depending on split axis), so both blobs are ≥ 2.
    """
    if bool(rng.integers(0, 2)) and W >= 3:
        split = int(rng.integers(1, W - 1))  # col in [1, W-2]
        left = [(r, c) for r in range(H) for c in range(W) if c < split]
        right = [(r, c) for r in range(H) for c in range(W) if c > split]
    else:
        split = int(rng.integers(1, H - 1))  # row in [1, H-2]
        left = [(r, c) for r in range(H) for c in range(W) if r < split]
        right = [(r, c) for r in range(H) for c in range(W) if r > split]

    if node_count is not None:
        # Split the total node budget between the two sides, biased
        # toward whichever side has more allowed cells.
        budget = max(4, min(node_count, len(left) + len(right)))
        # At least 2 nodes per side.
        n_a = max(2, min(len(left), budget // 2))
        n_b = max(2, min(len(right), budget - n_a))
        # If one side is too small, push the leftover onto the other.
        if n_a + n_b < budget:
            extra = budget - n_a - n_b
            if n_a + extra <= len(left):
                n_a += extra
            else:
                n_b += extra
    else:
        fill = float(rng.uniform(fill_lo, fill_hi))
        n_a = max(2, int(round(len(left) * fill)))
        n_b = max(2, int(round(len(right) * fill)))

    blob_a = _grow_lattice_blob(rng, left, n_a)
    blob_b = _grow_lattice_blob(rng, right, n_b)

    G = _sparsify_lattice(_lattice_subgraph(blob_a + blob_b), rng)
    pos_to_id = {G.nodes[n]["lattice"]: n for n in G.nodes()}
    u = pos_to_id[blob_a[int(rng.integers(0, len(blob_a)))]]
    v = pos_to_id[blob_b[int(rng.integers(0, len(blob_b)))]]
    return G, int(u), int(v)


def directed_connectivity_graph(
    rng: np.random.Generator,
    difficulty: str,
    node_count: int | None = None,
) -> tuple[nx.DiGraph, int, int]:
    """Return ``(D, entrance, exit)`` for the directed connectivity task.

    A single connected lattice blob is grown, sparsified, and randomly
    oriented (one direction per edge). Endpoints are chosen so the
    answer is balanced 50/50 between reachable (Yes) and unreachable
    (No) under directed reachability — no two-blob hack needed.

    When ``node_count`` is set, the lattice size is overridden to fit
    exactly that many cells in the blob (subject to a minimum of 3).
    """
    H, W = _CONNECTIVITY_LATTICE[difficulty]
    fill_lo, fill_hi = _CONNECTIVITY_FILL[difficulty]
    if node_count is not None:
        H, W = _lattice_for_node_count(node_count)
    want_yes = bool(rng.integers(0, 2))
    allowed = [(r, c) for r in range(H) for c in range(W)]

    D: nx.DiGraph | None = None
    for _ in range(32):
        if node_count is not None:
            n_target = max(3, min(node_count, H * W))
        else:
            fill = float(rng.uniform(fill_lo, fill_hi))
            n_target = max(3, min(int(round(H * W * fill)), H * W))
        blob = _grow_lattice_blob(rng, allowed, n_target)
        U = _sparsify_lattice(_lattice_subgraph(blob), rng)
        D = _orient_edges(U, rng)
        pick = _pick_directed_endpoints(D, rng, want_yes)
        if pick is not None:
            return D, pick[0], pick[1]

    # Fallback: with a connected blob of size ≥ 3 a graph almost always has
    # at least one directed pair on each side, but flipping want_yes lets
    # us return the last sample without retrying graph generation again.
    assert D is not None
    pick = _pick_directed_endpoints(D, rng, not want_yes)
    assert pick is not None
    return D, pick[0], pick[1]


def _orient_edges(U: nx.Graph, rng: np.random.Generator) -> nx.DiGraph:
    """One random direction per undirected edge. Carries node attrs over."""
    D = nx.DiGraph()
    for n, data in U.nodes(data=True):
        D.add_node(n, **data)
    for u, v in U.edges():
        if int(rng.integers(0, 2)):
            D.add_edge(u, v)
        else:
            D.add_edge(v, u)
    return D


def _pick_directed_endpoints(
    D: nx.DiGraph, rng: np.random.Generator, want_yes: bool
) -> tuple[int, int] | None:
    """Pick a balanced (entrance, exit) pair. Excludes lattice-adjacent
    pairs (direct edge in either direction) so the visual puzzle isn't
    trivialised by entrance and exit sitting one room apart.

    Walks every ``u`` in a shuffled order before giving up — returning
    ``None`` means no valid pair exists in ``D`` for this ``want_yes``
    polarity, not that we got unlucky. Candidates are filtered by
    iterating ``nodes`` (insertion-ordered) rather than the descendants
    set, so the random pick doesn't depend on Python set iteration order.
    """
    nodes = list(D.nodes())
    order = rng.permutation(len(nodes))
    for i in order:
        u = nodes[int(i)]
        reach = nx.descendants(D, u)
        if want_yes:
            cand = [n for n in nodes if n in reach]
        else:
            cand = [n for n in nodes if n != u and n not in reach]
        cand = [n for n in cand if not (D.has_edge(u, n) or D.has_edge(n, u))]
        if cand:
            v = cand[int(rng.integers(0, len(cand)))]
            return int(u), int(v)
    return None


def _grow_lattice_blob(
    rng: np.random.Generator,
    allowed: list[tuple[int, int]],
    n_target: int,
) -> list[tuple[int, int]]:
    """Grow a connected blob of exactly ``n_target`` cells inside ``allowed``.

    ``allowed`` must be a lattice-connected set with ``len(allowed) ≥
    n_target``. The result is a BFS flood from a random seed, picking
    frontier cells uniformly at random for organic, non-rectangular shapes.
    """
    allowed_set = set(allowed)
    start = allowed[int(rng.integers(0, len(allowed)))]
    selected: set[tuple[int, int]] = {start}
    frontier: list[tuple[int, int]] = []
    for dr, dc in _NBRS:
        nb = (start[0] + dr, start[1] + dc)
        if nb in allowed_set and nb not in selected:
            frontier.append(nb)

    while len(selected) < n_target and frontier:
        idx = int(rng.integers(0, len(frontier)))
        pick = frontier.pop(idx)
        if pick in selected:
            continue
        selected.add(pick)
        for dr, dc in _NBRS:
            nb = (pick[0] + dr, pick[1] + dc)
            if nb in allowed_set and nb not in selected:
                frontier.append(nb)
    return list(selected)


def _lattice_subgraph(positions: list[tuple[int, int]]) -> nx.Graph:
    """Build a graph whose edges are all lattice-adjacent pairs in ``positions``."""
    G = nx.Graph()
    pos_to_id: dict[tuple[int, int], int] = {}
    for i, pos in enumerate(positions):
        G.add_node(i, lattice=pos)
        pos_to_id[pos] = i
    for (r, c), node_id in pos_to_id.items():
        for dr, dc in ((1, 0), (0, 1)):
            nb = (r + dr, c + dc)
            if nb in pos_to_id:
                G.add_edge(node_id, pos_to_id[nb])
    return G


def _sparsify_lattice(G_full: nx.Graph, rng: np.random.Generator) -> nx.Graph:
    """Keep a random spanning tree per component, plus a fraction of extras.

    Preserves the component structure of ``G_full`` (Yes/No invariant) while
    breaking the "perfectly rectangular grid" look in the direct rendering.
    """
    extra_lo, extra_hi = _CONNECTIVITY_EXTRA_EDGES
    extra_keep = float(rng.uniform(extra_lo, extra_hi))

    result = nx.Graph()
    for n, data in G_full.nodes(data=True):
        result.add_node(n, **data)

    for comp in nx.connected_components(G_full):
        sub = G_full.subgraph(comp)
        tree_edges, extra_edges = _random_spanning_tree(sub, rng)
        result.add_edges_from(tree_edges)
        n_keep = int(round(len(extra_edges) * extra_keep))
        idx = rng.permutation(len(extra_edges))
        for i in idx[:n_keep]:
            result.add_edge(*extra_edges[int(i)])
    return result


def _random_spanning_tree(
    G: nx.Graph, rng: np.random.Generator
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Kruskal with random edge weights. Returns (tree_edges, non_tree_edges)."""
    all_edges = list(G.edges())
    order = rng.permutation(len(all_edges))
    parent = {n: n for n in G.nodes()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree: list[tuple[int, int]] = []
    extras: list[tuple[int, int]] = []
    for i in order:
        u, v = all_edges[int(i)]
        pu, pv = find(u), find(v)
        if pu == pv:
            extras.append((u, v))
        else:
            parent[pu] = pv
            tree.append((u, v))
    return tree, extras


def shortest_path_graph(
    rng: np.random.Generator,
    difficulty: str,
    node_count: int | None = None,
) -> nx.DiGraph:
    """Return a weighted DAG for the shortest-path task.

    Vertices are 0..n-1, edges only run i→j with i<j (so the graph is
    automatically acyclic). Vertex 0 is the unique source (in-degree 0)
    and vertex n-1 is the unique sink (out-degree 0); both are forced
    by an orphan-fix step rather than rejection sampling because the
    accept rate of the pure ER recipe collapses for n ≥ 15. Each edge
    carries an integer ``weight`` in [1, 10]. The graph admits at least
    two distinct simple s→t paths, which keeps the task non-trivial.

    ``G.graph["source"]`` and ``G.graph["sink"]`` carry the endpoints.

    When ``node_count`` is set, ``n`` is forced to that value rather
    than sampled from the difficulty's size preset. Must be ≥ 3.
    """
    lo, hi = _SHORTEST_PATH_SIZES[difficulty]
    if node_count is not None:
        if node_count < 3:
            raise ValueError(
                f"shortest_path_graph: node_count must be ≥ 3, got {node_count}"
            )
        lo = hi = node_count
    w_lo, w_hi = _SHORTEST_PATH_WEIGHT

    for _ in range(_SHORTEST_PATH_MAX_ATTEMPTS):
        n = int(rng.integers(lo, hi + 1))
        edge_set: set[tuple[int, int]] = set()

        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < _SHORTEST_PATH_P:
                    edge_set.add((i, j))

        # Force vertex 0 to be the only source.
        for v in range(1, n):
            if not any((u, v) in edge_set for u in range(v)):
                edge_set.add((int(rng.integers(0, v)), v))

        # Force vertex n-1 to be the only sink.
        for v in range(n - 1):
            if not any((v, w) in edge_set for w in range(v + 1, n)):
                edge_set.add((v, int(rng.integers(v + 1, n))))

        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for u, v in edge_set:
            G.add_edge(u, v, weight=int(rng.integers(w_lo, w_hi + 1)))

        s, t = 0, n - 1

        # By construction; assert as a tripwire.
        if any(G.in_degree(v) == 0 for v in range(1, n)):
            continue
        if any(G.out_degree(v) == 0 for v in range(n - 1)):
            continue

        try:
            nx.shortest_path_length(G, s, t, weight="weight")
        except nx.NetworkXNoPath:
            continue

        paths = nx.all_simple_paths(G, s, t)
        if next(paths, None) is None or next(paths, None) is None:
            continue

        G.graph["source"] = s
        G.graph["sink"] = t
        return G

    raise RuntimeError(
        f"shortest_path_graph: no valid instance after "
        f"{_SHORTEST_PATH_MAX_ATTEMPTS} attempts for difficulty={difficulty!r}"
    )


def coloring_graph(
    rng: np.random.Generator,
    difficulty: str,
    node_count: int | None = None,
) -> nx.Graph:
    """Return a Delaunay triangulation over random points in the unit square.

    Keeping the full triangulation (no edge drops) lets the map-disguise
    renderer use Voronoi cells directly — adjacency in the Voronoi diagram
    equals adjacency in the Delaunay graph by duality, so the disguise
    matches G exactly. Chromatic number is ≤ 4 by the four-color theorem
    (typically 3 or 4 on small random point sets).

    When ``node_count`` is set, exactly that many points are placed
    (must be ≥ 3 for a valid triangulation).
    """
    lo, hi = _COLORING_SIZES[difficulty]
    if node_count is not None:
        if node_count < 3:
            raise ValueError(
                f"coloring_graph: node_count must be ≥ 3, got {node_count}"
            )
        target = node_count
    else:
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

    # Fallback: a deterministic cycle on the requested node count. The map
    # disguise reads ``G.nodes[n]["pos"]`` unconditionally, so we stamp
    # coordinates onto each node here (evenly spaced on a unit circle)
    # rather than returning the raw cycle_graph result.
    fallback = nx.cycle_graph(target)
    for i in range(target):
        angle = 2.0 * float(np.pi) * i / max(target, 1)
        fallback.nodes[i]["pos"] = (
            0.5 + 0.4 * float(np.cos(angle)),
            0.5 + 0.4 * float(np.sin(angle)),
        )
    return fallback


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
