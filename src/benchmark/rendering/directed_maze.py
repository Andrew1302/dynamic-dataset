"""Directed maze disguise: each shared-wall door carries a direction arrow.

Built on top of :mod:`maze` — the ownership grid, seed placement, and
recursive-backtracker carver are reused. Two changes:

1. **One door per shared wall.** :func:`_restrict_walls_to_doors` closes
   every shared-wall cell except the one at the aligned center. The
   aligned center sits on an even offset within the block so it lines up
   with the carver's 2-step pattern, avoiding the loose "off-pattern"
   corridors that would otherwise widen the maze.
2. **Pre-opened directed doors.** :func:`_open_directed_doors` marks the
   door cell of every directed edge as corridor before carving. Without
   this, the spanning-tree carver may route the connection between two
   adjacent blocks through a third block, leaving a directed edge with
   no visible door. The door cell is 1-cell adjacent to interior pattern
   cells (which are always corridor), so the connection on each side is
   automatic.

Arrows: one per directed edge in *G* at the door position; one per
carved filler↔filler door at random direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO

import networkx as nx
import numpy as np
from PIL import Image, ImageDraw

from .maze import (
    _CORRIDOR,
    _C_BORDER,
    _C_CORRIDOR,
    _C_ENTRANCE,
    _C_EXIT,
    _C_NODE,
    _C_WALL,
    _WALL,
    _build_ownership_grid,
    _carve_branches,
    _paint_cell,
    _place_seeds,
)

_C_ARROW = (40, 40, 40)


@dataclass(frozen=True)
class Arrow:
    r: int
    c: int
    direction: tuple[int, int]


@dataclass(frozen=True)
class DirectedMaze:
    maze: np.ndarray
    seeds: dict[int, tuple[int, int]]
    entrance: int
    exit: int
    arrows: tuple[Arrow, ...] = field(default_factory=tuple)
    cell_px: int = 14
    highlight_all_nodes: bool = True

    def render(self) -> Image.Image:
        other = (
            [s for n, s in self.seeds.items() if n != self.entrance and n != self.exit]
            if self.highlight_all_nodes
            else []
        )
        return _render_image(
            self.maze,
            self.cell_px,
            self.seeds[self.entrance],
            self.seeds[self.exit],
            other,
            self.arrows,
        )


def build_directed_maze(
    G: nx.DiGraph,
    seed: int,
    entrance: int,
    exit: int,
    cell_px: int = 18,
    block: int = 7,
    highlight_all_nodes: bool = True,
    filler_arrows: bool = True,
) -> DirectedMaze:
    """Carve a directed maze. Topology mirrors the underlying undirected
    adjacency; arrows encode direction."""
    positions = {n: G.nodes[n]["lattice"] for n in G.nodes()}
    H = max(p[0] for p in positions.values()) + 1
    W = max(p[1] for p in positions.values()) + 1

    step = block + 1
    rng = np.random.default_rng(seed)
    node_ids = list(G.nodes())
    door_offset = _aligned_door_offset(block)

    owner = _build_ownership_grid(
        G, positions, node_ids, H, W, step, block,
        edge_passable=_directed_passable,
    )
    _restrict_walls_to_doors(owner, H, W, step, block, door_offset)

    seeds, filler_seeds = _place_seeds(positions, H, W, step, block, rng)

    fh = H * step + 1
    fw = W * step + 1
    maze = np.full((fh, fw), _WALL, dtype=np.uint8)
    for s in seeds.values():
        maze[s] = _CORRIDOR
    for s in filler_seeds:
        maze[s] = _CORRIDOR

    _open_directed_doors(maze, G, positions, step, door_offset)
    _carve_branches(maze, owner, rng)

    arrows = _compute_arrows(
        G, positions, maze, H, W, step, block, door_offset, rng, filler_arrows
    )

    return DirectedMaze(
        maze=maze,
        seeds=seeds,
        entrance=entrance,
        exit=exit,
        arrows=tuple(arrows),
        cell_px=cell_px,
        highlight_all_nodes=highlight_all_nodes,
    )


def render_directed_maze(
    G: nx.DiGraph,
    seed: int,
    entrance: int,
    exit: int,
    cell_px: int = 18,
    block: int = 7,
    highlight_all_nodes: bool = True,
    filler_arrows: bool = True,
) -> Image.Image:
    return build_directed_maze(
        G, seed, entrance, exit, cell_px, block, highlight_all_nodes, filler_arrows
    ).render()


def _aligned_door_offset(block: int) -> int:
    """Even offset within a block, closest to the center. Aligning the
    door cell with the carver's 2-step pattern keeps the maze tight."""
    center = block // 2
    return center if center % 2 == 0 else center + 1


def _directed_passable(G: nx.DiGraph, a: int, b: int) -> bool:
    return G.has_edge(a, b) or G.has_edge(b, a)


def _restrict_walls_to_doors(
    owner: np.ndarray, H: int, W: int, step: int, block: int, door_offset: int
) -> None:
    """Force every shared wall to expose at most one passable cell."""
    for lc in range(W - 1):
        wall_c = (lc + 1) * step
        for lr in range(H):
            r0 = 1 + lr * step
            keep = r0 + door_offset
            for r in range(r0, r0 + block):
                if r != keep and owner[r, wall_c] >= 0:
                    owner[r, wall_c] = -1

    for lr in range(H - 1):
        wall_r = (lr + 1) * step
        for lc in range(W):
            c0 = 1 + lc * step
            keep = c0 + door_offset
            for c in range(c0, c0 + block):
                if c != keep and owner[wall_r, c] >= 0:
                    owner[wall_r, c] = -1


def _open_directed_doors(
    maze: np.ndarray,
    G: nx.DiGraph,
    positions: dict[int, tuple[int, int]],
    step: int,
    door_offset: int,
) -> None:
    """Pre-open the door cell of every directed edge so the spanning-tree
    carver always exposes a passage between adjacent rooms."""
    for u, v in G.edges():
        r, c = _door_cell(positions[u], positions[v], step, door_offset)
        maze[r, c] = _CORRIDOR


def _door_cell(
    pu: tuple[int, int], pv: tuple[int, int], step: int, door_offset: int
) -> tuple[int, int]:
    pa, pb = (pu, pv) if pu < pv else (pv, pu)
    if pb[1] - pa[1] == 1:  # vertical wall (east-west neighbors)
        return 1 + pa[0] * step + door_offset, (pa[1] + 1) * step
    return (pa[0] + 1) * step, 1 + pa[1] * step + door_offset


def _compute_arrows(
    G: nx.DiGraph,
    positions: dict[int, tuple[int, int]],
    maze: np.ndarray,
    H: int,
    W: int,
    step: int,
    block: int,
    door_offset: int,
    rng: np.random.Generator,
    filler_arrows: bool,
) -> list[Arrow]:
    arrows: list[Arrow] = []
    pos_to_node = {p: n for n, p in positions.items()}

    for u, v in G.edges():
        pu, pv = positions[u], positions[v]
        r, c = _door_cell(pu, pv, step, door_offset)
        forward = pu < pv
        if pv[1] != pu[1]:
            direction = (0, 1) if forward else (0, -1)
        else:
            direction = (1, 0) if forward else (-1, 0)
        arrows.append(Arrow(r, c, direction))

    if not filler_arrows:
        return arrows

    for lc in range(W - 1):
        wall_c = (lc + 1) * step
        for lr in range(H):
            pa, pb = (lr, lc), (lr, lc + 1)
            if pa in pos_to_node or pb in pos_to_node:
                continue
            r = 1 + lr * step + door_offset
            if maze[r, wall_c] != _CORRIDOR:
                continue
            d = (0, 1) if int(rng.integers(0, 2)) else (0, -1)
            arrows.append(Arrow(r, wall_c, d))

    for lr in range(H - 1):
        wall_r = (lr + 1) * step
        for lc in range(W):
            pa, pb = (lr, lc), (lr + 1, lc)
            if pa in pos_to_node or pb in pos_to_node:
                continue
            c = 1 + lc * step + door_offset
            if maze[wall_r, c] != _CORRIDOR:
                continue
            d = (1, 0) if int(rng.integers(0, 2)) else (-1, 0)
            arrows.append(Arrow(wall_r, c, d))

    return arrows


def _render_image(
    maze: np.ndarray,
    cell_px: int,
    entrance: tuple[int, int],
    exit_: tuple[int, int],
    other_seeds: list[tuple[int, int]],
    arrows: tuple[Arrow, ...],
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
    for arrow in arrows:
        _draw_arrow(draw, arrow, cell_px)
    for r, c in other_seeds:
        _paint_cell(draw, r, c, cell_px, _C_NODE)
    _paint_endpoint(draw, entrance[0], entrance[1], cell_px, _C_ENTRANCE)
    _paint_endpoint(draw, exit_[0], exit_[1], cell_px, _C_EXIT)
    draw.rectangle([0, 0, w * cell_px - 1, h * cell_px - 1], outline=_C_BORDER, width=1)

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return Image.open(buf).copy()


def _paint_endpoint(
    draw: ImageDraw.ImageDraw,
    r: int,
    c: int,
    cell_px: int,
    color: tuple[int, int, int],
) -> None:
    """Endpoint marker: filled cell + thick dark outline + small white
    halo around it so it pops against blue decoys."""
    x0, y0 = c * cell_px, r * cell_px
    x1, y1 = x0 + cell_px - 1, y0 + cell_px - 1
    halo = max(2, cell_px // 5)
    draw.rectangle(
        [x0 - halo, y0 - halo, x1 + halo, y1 + halo],
        outline=color,
        width=halo,
    )
    draw.rectangle([x0, y0, x1, y1], fill=color)
    draw.rectangle([x0, y0, x1, y1], outline=_C_BORDER, width=2)


def _draw_arrow(
    draw: ImageDraw.ImageDraw, arrow: Arrow, cell_px: int
) -> None:
    """Filled triangle pointing along ``arrow.direction`` (axis-aligned)."""
    cx = arrow.c * cell_px + cell_px / 2
    cy = arrow.r * cell_px + cell_px / 2
    s = cell_px * 0.55
    dr, dc = arrow.direction
    tip = (cx + dc * s, cy + dr * s)
    base = (cx - dc * s, cy - dr * s)
    perp_r, perp_c = -dc, dr
    half = cell_px * 0.45
    b1 = (base[0] + perp_c * half, base[1] + perp_r * half)
    b2 = (base[0] - perp_c * half, base[1] - perp_r * half)
    draw.polygon([tip, b1, b2], fill=_C_ARROW)
