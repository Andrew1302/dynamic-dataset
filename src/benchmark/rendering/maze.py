"""Maze disguise: graph → 2D maze with entrance and exit.

Algorithm (Voronoi-region carve + per-room recursive backtracker):

1. Partition a square grid into one Voronoi region per graph node.
   The partition is graph-constrained: two cells of different regions
   touch only if the corresponding nodes share a graph edge (a wall
   buffer is inserted otherwise).
2. Within each region, run a recursive-backtracker maze carver that
   turns the region into a network of corridors and dead-ends. Every
   cell is either corridor or wall.
3. For each graph edge (u, v), ensure a "door" exists between the two
   regions — a pair of touching corridor cells. Non-adjacent region
   boundaries stay walled.
4. Mark the entrance and exit cells (green and red).

Correctness: because region boundaries are walled (or have doors only
where graph edges exist), a path from entrance to exit in the maze
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
_C_BORDER = (10, 10, 10)


def render_maze(
    G: nx.Graph,
    seed: int,
    entrance: int,
    exit: int,
    cell_px: int = 8,
    grid_size: int | None = None,
) -> Image.Image:
    """Render *G* as a 2D maze and return it as a PIL image."""
    n = G.number_of_nodes()
    if grid_size is None:
        grid_size = max(41, n * 7 + 5)
    if grid_size % 2 == 0:
        grid_size += 1

    owner, node_pos, nodes, _ = build_voronoi(G, seed, grid_size=grid_size)
    node_idx = {nd: i for i, nd in enumerate(nodes)}
    rng = np.random.default_rng(seed + 17)

    maze = np.full(owner.shape, _WALL, dtype=np.uint8)
    starts: dict[int, tuple[int, int]] = {}

    for i in range(len(nodes)):
        cells = np.argwhere(owner == i)
        if len(cells) == 0:
            continue
        start = _region_start(cells)
        starts[i] = start
        _carve_region(maze, owner, i, start, rng)

    for u, v in G.edges():
        _ensure_door(maze, owner, node_idx[u], node_idx[v], starts, rng)

    entr_cell = _mark_endpoint(maze, starts[node_idx[entrance]])
    exit_cell = _mark_endpoint(maze, starts[node_idx[exit]])

    return _render_image(maze, entr_cell, exit_cell, cell_px)


def _region_start(cells: np.ndarray) -> tuple[int, int]:
    cr, cc = cells.mean(axis=0)
    dists = (cells[:, 0] - cr) ** 2 + (cells[:, 1] - cc) ** 2
    r, c = cells[int(np.argmin(dists))]
    return int(r), int(c)


def _carve_region(
    maze: np.ndarray,
    owner: np.ndarray,
    label: int,
    start: tuple[int, int],
    rng: np.random.Generator,
) -> None:
    """Recursive-backtracker maze carver constrained to a Voronoi region."""
    h, w = maze.shape
    maze[start] = _CORRIDOR
    stack = [start]
    dirs = np.array([(-2, 0), (2, 0), (0, -2), (0, 2)])

    while stack:
        r, c = stack[-1]
        order = rng.permutation(4)
        advanced = False
        for k in order:
            dr, dc = int(dirs[k, 0]), int(dirs[k, 1])
            nr, nc = r + dr, c + dc
            mr, mc = r + dr // 2, c + dc // 2
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if owner[nr, nc] != label or owner[mr, mc] != label:
                continue
            if maze[nr, nc] == _CORRIDOR:
                continue
            maze[mr, mc] = _CORRIDOR
            maze[nr, nc] = _CORRIDOR
            stack.append((nr, nc))
            advanced = True
            break
        if not advanced:
            stack.pop()


def _ensure_door(
    maze: np.ndarray,
    owner: np.ndarray,
    iu: int,
    iv: int,
    starts: dict[int, tuple[int, int]],
    rng: np.random.Generator,
) -> None:
    """Guarantee at least one corridor-corridor touch between regions iu and iv.

    Finds all boundary pairs (a cell in iu adjacent to a cell in iv), prefers
    pairs where at least one side is already corridor, then forces both sides
    to corridor and carves a short straight path back to each region's start
    if necessary. Non-adjacent regions already have wall buffers from
    ``build_voronoi``, so doors can only form where the graph has an edge.
    """
    h, w = owner.shape
    pairs: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if owner[r, c] != iu:
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and owner[nr, nc] == iv:
                    pairs.append((r, c, nr, nc))
    if not pairs:
        return

    def score(pair: tuple[int, int, int, int]) -> int:
        r, c, nr, nc = pair
        s = 0
        if maze[r, c] == _CORRIDOR:
            s += 1
        if maze[nr, nc] == _CORRIDOR:
            s += 1
        return s

    pairs.sort(key=score, reverse=True)
    top = [p for p in pairs if score(p) == score(pairs[0])]
    r, c, nr, nc = top[int(rng.integers(0, len(top)))]
    maze[r, c] = _CORRIDOR
    maze[nr, nc] = _CORRIDOR

    _connect_to_start(maze, owner, iu, (r, c), starts[iu])
    _connect_to_start(maze, owner, iv, (nr, nc), starts[iv])


def _connect_to_start(
    maze: np.ndarray,
    owner: np.ndarray,
    label: int,
    cell: tuple[int, int],
    start: tuple[int, int],
) -> None:
    """BFS within ``owner == label`` to find the shortest path from ``cell``
    to any existing corridor connected to ``start``; carve it.
    """
    if maze[cell] != _CORRIDOR:
        maze[cell] = _CORRIDOR

    h, w = owner.shape
    visited = np.zeros(owner.shape, dtype=bool)
    visited[cell] = True
    parent: dict[tuple[int, int], tuple[int, int] | None] = {cell: None}
    q = [cell]
    target: tuple[int, int] | None = None
    while q:
        nxt: list[tuple[int, int]] = []
        for r, c in q:
            if maze[r, c] == _CORRIDOR and _connected(maze, owner, label, (r, c), start):
                target = (r, c)
                break
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < h
                    and 0 <= nc < w
                    and owner[nr, nc] == label
                    and not visited[nr, nc]
                ):
                    visited[nr, nc] = True
                    parent[(nr, nc)] = (r, c)
                    nxt.append((nr, nc))
        if target is not None:
            break
        q = nxt

    if target is None:
        return
    node: tuple[int, int] | None = target
    while node is not None:
        maze[node] = _CORRIDOR
        node = parent[node]


def _connected(
    maze: np.ndarray,
    owner: np.ndarray,
    label: int,
    a: tuple[int, int],
    b: tuple[int, int],
) -> bool:
    """Check whether ``a`` and ``b`` share a corridor path within region ``label``."""
    if maze[a] != _CORRIDOR or maze[b] != _CORRIDOR:
        return False
    h, w = maze.shape
    seen = {a}
    stack = [a]
    while stack:
        r, c = stack.pop()
        if (r, c) == b:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < h
                and 0 <= nc < w
                and owner[nr, nc] == label
                and maze[nr, nc] == _CORRIDOR
                and (nr, nc) not in seen
            ):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return False


def _mark_endpoint(maze: np.ndarray, start: tuple[int, int]) -> tuple[int, int]:
    maze[start] = _CORRIDOR
    return start


def _render_image(
    maze: np.ndarray,
    entrance: tuple[int, int],
    exit_: tuple[int, int],
    cell_px: int,
) -> Image.Image:
    h, w = maze.shape
    img = Image.new("RGB", (w * cell_px, h * cell_px), _C_WALL)
    draw = ImageDraw.Draw(img)

    for r in range(h):
        for c in range(w):
            if maze[r, c] == _CORRIDOR:
                x0, y0 = c * cell_px, r * cell_px
                draw.rectangle(
                    (x0, y0, x0 + cell_px - 1, y0 + cell_px - 1),
                    fill=_C_CORRIDOR,
                )

    def paint(cell: tuple[int, int], color: tuple[int, int, int]) -> None:
        r, c = cell
        x0, y0 = c * cell_px, r * cell_px
        draw.rectangle(
            (x0, y0, x0 + cell_px - 1, y0 + cell_px - 1),
            fill=color,
        )

    paint(entrance, _C_ENTRANCE)
    paint(exit_, _C_EXIT)

    draw.rectangle(
        (0, 0, w * cell_px - 1, h * cell_px - 1),
        outline=_C_BORDER,
        width=2,
    )

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return Image.open(buf).copy()
