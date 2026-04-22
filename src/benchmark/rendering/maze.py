"""Maze disguise: lattice graph → 2D maze with entrance and exit.

Ownership grid
--------------
The fine grid tiles one ``block × block`` territory per lattice slot
(``block`` odd, default 7), with 1-cell walls between blocks:

- block (lr, lc) interior cells, if a node lives there → ``owner = node_idx``
- block (lr, lc) interior cells, if the slot is empty → ``owner = FILLER``
  (a dummy id past the last node index)
- **shared wall cells between edge-connected node neighbors** → owner is
  set to one of the two nodes (carver can cross freely)
- **shared wall cells between two empty neighbors** → ``owner = FILLER``
  (filler corridors merge across empty slots)
- shared wall cells between a node and an empty slot, or between
  non-edge node neighbors → ``-1`` (hard wall)
- intersection cells at ``(k*step, j*step)`` → always ``-1``

Correctness by construction
---------------------------
The unified recursive-backtracker dead-end carver advances only via
2-step jumps whose midpoint and landing both have ``owner ≥ 0``. So:

- Corridors flow freely through edge shared-walls (multiple crossing
  points, organic look — no rigid "single door per edge")
- Non-edge walls stay intact (owner = -1 everywhere on them)
- Intersection cells are always -1, blocking diagonal leaks
- Filler corridors live in their own connected region, isolated from
  every node's corridor network by ``-1`` node↔empty shared walls —
  they're decorative dead-ends that visually fill what would otherwise
  be giant black gaps, without affecting entrance/exit reachability

Invariant: corridor reachability in the rendered maze between any two
**node** seeds is **identical** to connected components of ``G``. No BFS
routing, no fallbacks, no retries.
"""

from __future__ import annotations

from io import BytesIO

import networkx as nx
import numpy as np
from PIL import Image, ImageDraw


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
    block: int = 7,
    highlight_all_nodes: bool = True,
) -> Image.Image:
    """Render *G* (a lattice subgraph) as a 2D maze.

    Parameters
    ----------
    G
        NetworkX graph with ``G.nodes[n]["lattice"] = (row, col)`` on every
        node. Edges must only connect lattice-adjacent pairs.
    seed
        RNG seed for seed placement jitter and dead-end branching.
    entrance, exit
        Node ids whose seed cells are painted green and red.
    cell_px
        Pixel size of one fine-grid cell.
    block
        Fine cells per side of one node's block (odd, default 7).
    highlight_all_nodes
        When true, paint every non-endpoint node's seed blue.
    """
    positions = {n: G.nodes[n]["lattice"] for n in G.nodes()}
    H = max(p[0] for p in positions.values()) + 1
    W = max(p[1] for p in positions.values()) + 1

    step = block + 1
    fh = H * step + 1
    fw = W * step + 1

    rng = np.random.default_rng(seed)
    node_ids = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(node_ids)}
    pos_to_node = {positions[n]: n for n in node_ids}

    FILLER = len(node_ids)

    owner = np.full((fh, fw), -1, dtype=np.int32)

    for lr in range(H):
        for lc in range(W):
            r0 = 1 + lr * step
            c0 = 1 + lc * step
            if (lr, lc) in pos_to_node:
                owner[r0 : r0 + block, c0 : c0 + block] = node_idx[pos_to_node[(lr, lc)]]
            else:
                owner[r0 : r0 + block, c0 : c0 + block] = FILLER

    # Vertical shared walls.
    for lc in range(W - 1):
        wall_c = (lc + 1) * step
        for lr in range(H):
            pa, pb = (lr, lc), (lr, lc + 1)
            a_node = pa in pos_to_node
            b_node = pb in pos_to_node
            r0 = 1 + lr * step
            if a_node and b_node:
                a, b = pos_to_node[pa], pos_to_node[pb]
                if G.has_edge(a, b):
                    owner[r0 : r0 + block, wall_c] = node_idx[a]
            elif not a_node and not b_node:
                owner[r0 : r0 + block, wall_c] = FILLER
            # node ↔ empty stays -1 so filler never leaks into a node network

    # Horizontal shared walls.
    for lr in range(H - 1):
        wall_r = (lr + 1) * step
        for lc in range(W):
            pa, pb = (lr, lc), (lr + 1, lc)
            a_node = pa in pos_to_node
            b_node = pb in pos_to_node
            c0 = 1 + lc * step
            if a_node and b_node:
                a, b = pos_to_node[pa], pos_to_node[pb]
                if G.has_edge(a, b):
                    owner[wall_r, c0 : c0 + block] = node_idx[a]
            elif not a_node and not b_node:
                owner[wall_r, c0 : c0 + block] = FILLER

    # Seeds at random even-offsets so entrance/exit aren't always centered.
    offsets = list(range(0, block, 2))
    seeds: dict[int, tuple[int, int]] = {}
    for n, (lr, lc) in positions.items():
        r0 = 1 + lr * step
        c0 = 1 + lc * step
        dr = offsets[int(rng.integers(0, len(offsets)))]
        dc = offsets[int(rng.integers(0, len(offsets)))]
        seeds[n] = (r0 + dr, c0 + dc)

    # One filler seed per empty slot. Since all filler interiors and
    # filler↔filler walls share owner=FILLER, the carver will merge these
    # into one connected filler region (per lattice-connected blob of
    # empty slots) — never touching any node region.
    filler_seeds: list[tuple[int, int]] = []
    for lr in range(H):
        for lc in range(W):
            if (lr, lc) in pos_to_node:
                continue
            r0 = 1 + lr * step
            c0 = 1 + lc * step
            dr = offsets[int(rng.integers(0, len(offsets)))]
            dc = offsets[int(rng.integers(0, len(offsets)))]
            filler_seeds.append((r0 + dr, c0 + dc))

    maze = np.full((fh, fw), _WALL, dtype=np.uint8)
    for s in seeds.values():
        maze[s] = _CORRIDOR
    for s in filler_seeds:
        maze[s] = _CORRIDOR

    _carve_branches(maze, owner, rng)

    other_seeds = (
        [s for n, s in seeds.items() if n != entrance and n != exit]
        if highlight_all_nodes
        else []
    )
    return _render_image(maze, cell_px, seeds[entrance], seeds[exit], other_seeds)


def _carve_branches(
    maze: np.ndarray, owner: np.ndarray, rng: np.random.Generator
) -> None:
    """Recursive-backtracker flood carver, 2-step jumps, owner-aware.

    Every jump's midpoint and landing cell must have ``owner ≥ 0``.
    Consequence: corridors flow across edge shared-walls freely, never
    cross non-edge shared-walls or intersection cells (both owner = -1).
    Seeds of connected-in-G nodes end up in the same corridor network.

    Each seed first receives one guaranteed 2-step extension (carving the
    midpoint and, if not already, the landing). Without this, a seed
    processed late by the main backtracker could find all four of its
    2-step landings already corridor — at which point the backtracker
    pops immediately, leaving the seed's midpoint neighbors as walls and
    the seed itself a disconnected island.
    """
    h, w = maze.shape
    seeds = np.argwhere(maze == _CORRIDOR)
    if len(seeds) == 0:
        return

    for i in rng.permutation(len(seeds)):
        r, c = int(seeds[i, 0]), int(seeds[i, 1])
        for k in rng.permutation(4):
            dr, dc = _DIRS_2[int(k)]
            nr, nc = r + dr, c + dc
            mr, mc = r + dr // 2, c + dc // 2
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if owner[nr, nc] < 0 or owner[mr, mc] < 0:
                continue
            maze[mr, mc] = _CORRIDOR
            maze[nr, nc] = _CORRIDOR
            break

    corridor = np.argwhere(maze == _CORRIDOR)
    order = rng.permutation(len(corridor))
    starts = [(int(corridor[i, 0]), int(corridor[i, 1])) for i in order]

    for start in starts:
        stack = [start]
        while stack:
            r, c = stack[-1]
            advanced = False
            for k in rng.permutation(4):
                dr, dc = _DIRS_2[int(k)]
                nr, nc = r + dr, c + dc
                mr, mc = r + dr // 2, c + dc // 2
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if owner[nr, nc] < 0 or owner[mr, mc] < 0:
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


def _render_image(
    maze: np.ndarray,
    cell_px: int,
    entrance: tuple[int, int],
    exit_: tuple[int, int],
    other_seeds: list[tuple[int, int]],
) -> Image.Image:
    h, w = maze.shape
    img = Image.new("RGB", (w * cell_px, h * cell_px), _C_WALL)
    draw = ImageDraw.Draw(img)
    for r in range(h):
        for c in range(w):
            if maze[r, c] == _CORRIDOR:
                x0, y0 = c * cell_px, r * cell_px
                draw.rectangle(
                    [x0, y0, x0 + cell_px - 1, y0 + cell_px - 1], fill=_C_CORRIDOR
                )
    for r, c in other_seeds:
        _paint_cell(draw, r, c, cell_px, _C_NODE)
    _paint_cell(draw, entrance[0], entrance[1], cell_px, _C_ENTRANCE)
    _paint_cell(draw, exit_[0], exit_[1], cell_px, _C_EXIT)
    draw.rectangle([0, 0, w * cell_px - 1, h * cell_px - 1], outline=_C_BORDER, width=1)

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return Image.open(buf).copy()


def _paint_cell(
    draw: ImageDraw.ImageDraw,
    r: int,
    c: int,
    cell_px: int,
    color: tuple[int, int, int],
) -> None:
    x0, y0 = c * cell_px, r * cell_px
    draw.rectangle([x0, y0, x0 + cell_px - 1, y0 + cell_px - 1], fill=color)
