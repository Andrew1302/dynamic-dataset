"""Maze disguise: graph → 2D maze with entrance and exit.

Algorithm (graph-constrained Voronoi + backbone-first carver):

1. Partition a square grid into one Voronoi region per graph node. The
   partition is graph-constrained: two cells of different regions touch
   only if the corresponding nodes share a graph edge (a single-cell
   wall is inserted otherwise).
2. Phase 1 — edge backbones. For every edge (u, v) of G, find a
   shortest 2-step path from ``seed[u]`` to ``seed[v]`` restricted to
   cells owned by u or v, and carve it. Each graph edge becomes a
   short, direct corridor between its two node markers, so the maze's
   primary topology mirrors G.
3. Phase 2 — bounded dead-end branches. Starting from already-carved
   backbone cells, run a recursive-backtracker that adds dead-end
   sprouts, capped by ``branch_budget_ratio * backbone_cells`` extra
   cells. Keeps the maze aesthetic without drowning the backbones.
4. Paint the entrance (green) and exit (red) at their region seeds.
   When ``highlight_all_nodes`` is set, every other node seed is blue.

Correctness: both phases only traverse cells owned by adjacent-in-G
regions (the Voronoi's conflict rule guarantees that cross-region
4-adjacency implies a G edge), so corridor reachability matches
connected components of G — a path from entrance to exit in the maze
exists iff ``nx.has_path(G, entrance, exit)``.
"""

from __future__ import annotations

from io import BytesIO

import networkx as nx
import numpy as np
from PIL import Image, ImageDraw

from ._voronoi import build_voronoi


_WALL = np.uint8(0)
_CORRIDOR = np.uint8(1)

_C_WALL = (20, 20, 20)
_C_CORRIDOR = (245, 241, 232)
_C_ENTRANCE = (88, 207, 118)
_C_EXIT = (235, 94, 94)
_C_NODE = (76, 142, 222)
_C_BORDER = (10, 10, 10)

_DIRS_2 = ((-2, 0), (2, 0), (0, -2), (0, 2))


def render_maze(
    G: nx.Graph,
    seed: int,
    entrance: int,
    exit: int,
    cell_px: int = 14,
    grid_size: int | None = None,
    highlight_all_nodes: bool = True,
    branch_budget_ratio: float | None = None,
) -> Image.Image:
    """Render *G* as a 2D maze and return it as a PIL image.

    Parameters
    ----------
    G
        NetworkX graph.
    seed
        RNG seed for the Voronoi layout noise and the carver.
    entrance, exit
        Node ids whose region seeds are painted green and red.
    cell_px
        Pixel size of one grid cell.
    grid_size
        Override the default grid side length. Forced odd to keep the
        2-step carver aligned.
    highlight_all_nodes
        When true, paint every non-endpoint node's seed in blue in
        addition to the green entrance and red exit.
    branch_budget_ratio
        Fraction of backbone cells to allow as extra dead-end cells in
        phase 2. ``None`` (default) = unlimited, i.e. the carver runs
        until no more 2-step moves are possible (fully filled maze with
        dead-ends in every region). ``0.0`` = pure skeleton (edges only,
        no dead-ends).
    """
    n = G.number_of_nodes()
    if grid_size is None:
        grid_size = max(11, int(n * 1.5) + 5)
    if grid_size % 2 == 0:
        grid_size += 1

    owner, _node_pos, nodes, _ = build_voronoi(G, seed, grid_size=grid_size)
    node_idx = {nd: i for i, nd in enumerate(nodes)}
    rng = np.random.default_rng(seed + 17)

    maze = np.full(owner.shape, _WALL, dtype=np.uint8)

    seeds: dict[int, tuple[int, int]] = {}
    for i in range(len(nodes)):
        cells = np.argwhere(owner == i)
        if len(cells) == 0:
            continue
        seeds[i] = _region_seed(cells)

    for s in seeds.values():
        maze[s] = _CORRIDOR

    for u, v in G.edges():
        iu, iv = node_idx[u], node_idx[v]
        if iu not in seeds or iv not in seeds:
            continue
        path = _find_backbone(owner, seeds[iu], seeds[iv], {iu, iv})
        if path is None:
            continue
        _carve_path(maze, path)

    if branch_budget_ratio is None:
        _carve_branches(maze, owner, rng, None)
    else:
        backbone_cells = int(np.sum(maze == _CORRIDOR))
        budget = int(backbone_cells * branch_budget_ratio)
        if budget > 0:
            _carve_branches(maze, owner, rng, budget)

    return _render_image(
        maze,
        cell_px,
        seeds[node_idx[entrance]],
        seeds[node_idx[exit]],
        _other_seeds(seeds, node_idx, entrance, exit) if highlight_all_nodes else [],
    )


def _region_seed(cells: np.ndarray) -> tuple[int, int]:
    """Pick an (even, even) cell in the region closest to its centroid.

    The recursive-backtracker carver advances in 2-step jumps, so every
    carved cell inherits the seed's parity. Using a consistent
    (even, even) parity across all region seeds keeps walls and
    corridors cleanly alternating; mixing parities across seeds causes
    midpoints from one carve to overlap corridors from another,
    producing open swiss-cheese regions instead of a maze.
    """
    cr, cc = cells.mean(axis=0)
    mask = (cells[:, 0] % 2 == 0) & (cells[:, 1] % 2 == 0)
    candidates = cells[mask] if mask.any() else cells
    dists = (candidates[:, 0] - cr) ** 2 + (candidates[:, 1] - cc) ** 2
    r, c = candidates[int(np.argmin(dists))]
    return int(r), int(c)


def _other_seeds(
    seeds: dict[int, tuple[int, int]],
    node_idx: dict,
    entrance,
    exit,
) -> list[tuple[int, int]]:
    skip = {node_idx[entrance], node_idx[exit]}
    return [c for i, c in seeds.items() if i not in skip]


def _find_backbone(
    owner: np.ndarray,
    src: tuple[int, int],
    dst: tuple[int, int],
    allowed: set[int],
) -> list[tuple[int, int]] | None:
    """Shortest 2-step path from *src* to *dst*, restricted to cells whose
    ``owner`` is in *allowed*. Returns the cell list inclusive, or None.
    """
    if src == dst:
        return [src]
    h, w = owner.shape
    parent: dict[tuple[int, int], tuple[int, int] | None] = {src: None}
    frontier: list[tuple[int, int]] = [src]
    while frontier:
        nxt: list[tuple[int, int]] = []
        for r, c in frontier:
            for dr, dc in _DIRS_2:
                nr, nc = r + dr, c + dc
                mr, mc = r + dr // 2, c + dc // 2
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(owner[nr, nc]) not in allowed:
                    continue
                if int(owner[mr, mc]) not in allowed:
                    continue
                if (nr, nc) in parent:
                    continue
                parent[(nr, nc)] = (r, c)
                if (nr, nc) == dst:
                    path: list[tuple[int, int]] = []
                    cur: tuple[int, int] | None = dst
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    return path
                nxt.append((nr, nc))
        frontier = nxt
    return None


def _carve_path(maze: np.ndarray, path: list[tuple[int, int]]) -> None:
    """Carve every cell in *path* plus the midpoint between consecutive pairs."""
    for i, (r, c) in enumerate(path):
        maze[r, c] = _CORRIDOR
        if i > 0:
            pr, pc = path[i - 1]
            maze[(r + pr) // 2, (c + pc) // 2] = _CORRIDOR


def _carve_branches(
    maze: np.ndarray,
    owner: np.ndarray,
    rng: np.random.Generator,
    budget: int | None,
) -> None:
    """Sprout dead-end branches off existing corridor cells.

    Recursive-backtracker style walk seeded at every existing corridor
    cell. When *budget* is an int, caps total new corridor cells so
    backbones stay dominant. When ``None``, runs to completion — every
    2-step-reachable owned cell is eventually carved, producing a fully
    dense maze with dead-ends in every region.
    """
    h, w = maze.shape
    corridor = np.argwhere(maze == _CORRIDOR)
    if len(corridor) == 0:
        return
    order = rng.permutation(len(corridor))
    starts = [(int(corridor[i, 0]), int(corridor[i, 1])) for i in order]

    remaining = budget
    for start in starts:
        if remaining is not None and remaining <= 0:
            break
        stack: list[tuple[int, int]] = [start]
        while stack and (remaining is None or remaining > 0):
            r, c = stack[-1]
            dir_order = rng.permutation(4)
            advanced = False
            for k in dir_order:
                dr, dc = _DIRS_2[int(k)]
                nr, nc = r + dr, c + dc
                mr, mc = r + dr // 2, c + dc // 2
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if owner[nr, nc] < 0 or owner[mr, mc] < 0:
                    continue
                if maze[nr, nc] == _CORRIDOR:
                    continue
                delta = int(maze[mr, mc] != _CORRIDOR) + 1
                maze[mr, mc] = _CORRIDOR
                maze[nr, nc] = _CORRIDOR
                stack.append((nr, nc))
                if remaining is not None:
                    remaining -= delta
                advanced = True
                break
            if not advanced:
                stack.pop()


def _render_image(
    maze: np.ndarray,
    cell_px: int,
    entrance: tuple[int, int],
    exit_: tuple[int, int],
    extra_nodes: list[tuple[int, int]],
) -> Image.Image:
    h, w = maze.shape
    img = Image.new("RGB", (w * cell_px, h * cell_px), _C_WALL)
    draw = ImageDraw.Draw(img)

    for r in range(h):
        for c in range(w):
            if maze[r, c] == _CORRIDOR:
                _fill_cell(draw, (r, c), cell_px, _C_CORRIDOR)

    for cell in extra_nodes:
        _fill_cell(draw, cell, cell_px, _C_NODE)
    _fill_cell(draw, entrance, cell_px, _C_ENTRANCE)
    _fill_cell(draw, exit_, cell_px, _C_EXIT)

    draw.rectangle(
        (0, 0, w * cell_px - 1, h * cell_px - 1),
        outline=_C_BORDER,
        width=2,
    )

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return Image.open(buf).copy()


def _fill_cell(
    draw: ImageDraw.ImageDraw,
    cell: tuple[int, int],
    cell_px: int,
    color: tuple[int, int, int],
) -> None:
    r, c = cell
    x0, y0 = c * cell_px, r * cell_px
    draw.rectangle(
        (x0, y0, x0 + cell_px - 1, y0 + cell_px - 1),
        fill=color,
    )
