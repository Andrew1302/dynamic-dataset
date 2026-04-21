"""Graph-constrained BFS Voronoi partition on a 2D grid.

Each graph node seeds a region; BFS expansion is blocked whenever it
would create adjacency between non-adjacent nodes. A gaussian noise
field is added to the BFS distances so region boundaries look organic.

Used by both the map and maze disguises: both need a grid partition
where cells of two different nodes touch only if the underlying graph
has that edge.
"""

from __future__ import annotations

import heapq

import networkx as nx
import numpy as np
from scipy.ndimage import gaussian_filter


UNCLAIMED = -1
WALL = -2
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def build_voronoi(
    G: nx.Graph,
    seed: int,
    pos: dict | None = None,
    grid_size: int | None = None,
) -> tuple[np.ndarray, dict, list, int]:
    """Return ``(owner, node_pos, nodes, grid_size)``.

    ``owner`` is a ``(grid_size, grid_size)`` int16 array with entries in
    ``{UNCLAIMED, WALL, 0..n-1}`` (the index into ``nodes``).
    ``node_pos`` maps each node to its seed ``(col, row)`` cell.
    """
    np.random.seed(seed)
    n = G.number_of_nodes()
    if grid_size is None:
        grid_size = max(200, n * 22)
    nodes = list(G.nodes())

    if pos is None:
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=seed, k=2.0)

    coords = np.array([pos[nd] for nd in nodes], dtype=np.float64)
    mn = coords.min(axis=0)
    mx = coords.max(axis=0)
    span = np.where(mx > mn, mx - mn, 1.0)
    pad = 0.12 * span
    coords = (coords - mn + pad) / (span + 2 * pad)
    coords = np.clip((coords * (grid_size - 1)).astype(int), 0, grid_size - 1)
    node_pos = {nd: (int(coords[i][0]), int(coords[i][1])) for i, nd in enumerate(nodes)}

    all_dists = [
        np.linalg.norm(np.array(node_pos[u]) - np.array(node_pos[v]))
        for u, v in G.edges()
    ] or [grid_size / max(n, 1)]
    min_dist = min(all_dists)
    avg_dist = float(np.mean(all_dists))
    noise_strength = min_dist * 0.08
    sigma = max(3, int(avg_dist * 0.3))
    raw = gaussian_filter(
        np.random.randn(grid_size, grid_size).astype(np.float32), sigma=sigma
    )
    noise = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8) * noise_strength

    owner = np.full((grid_size, grid_size), UNCLAIMED, dtype=np.int16)
    dist = np.full((grid_size, grid_size), 1e9, dtype=np.float32)
    pq: list = []
    for i, nd in enumerate(nodes):
        c, r = node_pos[nd]
        owner[r, c] = i
        dist[r, c] = 0.0
        heapq.heappush(pq, (0.0, i, r, c))

    nbr_idx = {
        i: frozenset(nodes.index(nb) for nb in G.neighbors(nd))
        for i, nd in enumerate(nodes)
    }

    while pq:
        d, lbl, r, c = heapq.heappop(pq)
        if d > dist[r, c] + 1e-6:
            continue
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < grid_size and 0 <= nc < grid_size):
                continue
            ex = int(owner[nr, nc])
            if ex == WALL or (ex >= 0 and ex != lbl):
                continue
            conflict = False
            for dr2, dc2 in _DIRS:
                nr2, nc2 = nr + dr2, nc + dc2
                if 0 <= nr2 < grid_size and 0 <= nc2 < grid_size:
                    o = int(owner[nr2, nc2])
                    if o >= 0 and o != lbl and o not in nbr_idx[lbl]:
                        conflict = True
                        break
            if conflict:
                owner[nr, nc] = WALL
                continue
            nd2 = d + 1.0 + float(noise[nr, nc])
            if nd2 < dist[nr, nc]:
                dist[nr, nc] = nd2
                owner[nr, nc] = lbl
                heapq.heappush(pq, (nd2, lbl, nr, nc))

    return owner, node_pos, nodes, grid_size
