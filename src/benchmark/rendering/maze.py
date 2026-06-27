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

Helpers ``_carve_branches``, ``_paint_cell``, the color constants, and
``_DIRS_2`` are also imported by ``directed_maze.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import networkx as nx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import LabelStyle, node_label

_WALL = np.uint8(0)
_CORRIDOR = np.uint8(1)

_C_WALL = (20, 20, 20)
_C_CORRIDOR = (245, 241, 232)
_C_ENTRANCE = (88, 207, 118)
_C_EXIT = (235, 94, 94)
_C_NODE = (76, 142, 222)
_C_BORDER = (10, 10, 10)

_DIRS_2 = ((-2, 0), (2, 0), (0, -2), (0, 2))


@dataclass(frozen=True)
class Maze:
    """Carved maze ready to be drawn. Result of ``build_maze``."""

    maze: np.ndarray
    seeds: dict[int, tuple[int, int]]
    entrance: int
    exit: int
    cell_px: int = 14
    highlight_all_nodes: bool = True
    label_style: LabelStyle = "numeric"

    def render(
        self, pdf_path: str | None = None, pdf_dpi: int = 220
    ) -> Image.Image:
        other = (
            [s for n, s in self.seeds.items() if n != self.entrance and n != self.exit]
            if self.highlight_all_nodes
            else []
        )
        labels: dict[tuple[int, int], str] = {}
        if self.highlight_all_nodes and self.label_style != "none":
            for node_id, cell in self.seeds.items():
                labels[cell] = node_label(node_id, self.label_style)
        img = _render_image(
            self.maze,
            self.cell_px,
            self.seeds[self.entrance],
            self.seeds[self.exit],
            other,
            labels,
        )
        if pdf_path is not None:
            img.save(pdf_path, format="PDF", resolution=float(pdf_dpi))
        return img


def build_maze(
    G: nx.Graph,
    seed: int,
    entrance: int,
    exit: int,
    cell_px: int = 14,
    block: int = 7,
    highlight_all_nodes: bool = True,
    label_style: LabelStyle = "numeric",
) -> Maze:
    """Carve a maze from a lattice subgraph *G*.

    *G* must carry ``G.nodes[n]["lattice"] = (row, col)`` on every node and
    only have edges between lattice-adjacent pairs.
    """
    positions = {n: G.nodes[n]["lattice"] for n in G.nodes()}
    H = max(p[0] for p in positions.values()) + 1
    W = max(p[1] for p in positions.values()) + 1

    step = block + 1
    rng = np.random.default_rng(seed)
    node_ids = list(G.nodes())

    owner, _filler_id = _build_ownership_grid(
        G, positions, node_ids, H, W, step, block, edge_passable=_undirected_passable
    )

    seeds, filler_seeds = _place_seeds(positions, H, W, step, block, rng)

    fh = H * step + 1
    fw = W * step + 1
    maze = np.full((fh, fw), _WALL, dtype=np.uint8)
    for s in seeds.values():
        maze[s] = _CORRIDOR
    for s in filler_seeds:
        maze[s] = _CORRIDOR

    _carve_branches(maze, owner, rng)

    return Maze(
        maze=maze,
        seeds=seeds,
        entrance=entrance,
        exit=exit,
        cell_px=cell_px,
        highlight_all_nodes=highlight_all_nodes,
        label_style=label_style,
    )


def render_maze(
    G: nx.Graph,
    seed: int,
    entrance: int,
    exit: int,
    cell_px: int = 14,
    block: int = 7,
    highlight_all_nodes: bool = True,
    label_style: LabelStyle = "numeric",
    pdf_path: str | None = None,
    pdf_dpi: int = 220,
) -> Image.Image:
    """Convenience: ``build_maze(...).render()``."""
    return build_maze(
        G, seed, entrance, exit, cell_px, block, highlight_all_nodes, label_style
    ).render(pdf_path=pdf_path, pdf_dpi=pdf_dpi)


def _undirected_passable(G: nx.Graph, a: int, b: int) -> bool:
    return G.has_edge(a, b)


def _pick_axis_offsets(
    H: int, W: int, block: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row and per-column even offsets used for aligned seed placement.

    All blocks in lattice row ``lr`` share the same fine-grid row
    ``1 + lr*step + row_off[lr]``; same for columns. With this alignment
    the corridor crossing any passable wall is a single straight line on
    a fixed fine row (vertical walls) or column (horizontal walls), and
    the seed becomes the only branching point inside each block.
    """
    offsets = np.arange(0, block, 2, dtype=np.int32)
    row_off = offsets[rng.integers(0, len(offsets), size=H)]
    col_off = offsets[rng.integers(0, len(offsets), size=W)]
    return row_off, col_off


def _build_ownership_grid(
    G: nx.Graph,
    positions: dict[int, tuple[int, int]],
    node_ids: list[int],
    H: int,
    W: int,
    step: int,
    block: int,
    edge_passable,
    row_off: np.ndarray | None = None,
    col_off: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Build the fine ownership grid.

    ``edge_passable(G, a, b)`` decides whether the shared wall between two
    adjacent node blocks is open. Returns ``(owner, filler_id)`` — the
    sentinel value used for empty-slot cells so downstream code can
    identify filler regions without reaching into this builder.

    When ``row_off`` and ``col_off`` are provided, each passable
    node↔node wall opens at exactly **one** aligned cell instead of the
    full block-length stripe. This is the precondition for the
    seed-as-junction invariant used by the directed maze: a single
    crossing per edge means each in-block stroke contains the wall cell
    and gets a non-tied direction vote.
    """
    fh = H * step + 1
    fw = W * step + 1
    node_idx = {n: i for i, n in enumerate(node_ids)}
    pos_to_node = {positions[n]: n for n in node_ids}
    FILLER = len(node_ids)
    aligned = row_off is not None and col_off is not None

    owner = np.full((fh, fw), -1, dtype=np.int32)

    for lr in range(H):
        for lc in range(W):
            r0 = 1 + lr * step
            c0 = 1 + lc * step
            if (lr, lc) in pos_to_node:
                owner[r0 : r0 + block, c0 : c0 + block] = node_idx[pos_to_node[(lr, lc)]]
            else:
                owner[r0 : r0 + block, c0 : c0 + block] = FILLER

    for lc in range(W - 1):
        wall_c = (lc + 1) * step
        for lr in range(H):
            pa, pb = (lr, lc), (lr, lc + 1)
            a_node = pa in pos_to_node
            b_node = pb in pos_to_node
            r0 = 1 + lr * step
            if a_node and b_node:
                a, b = pos_to_node[pa], pos_to_node[pb]
                if edge_passable(G, a, b):
                    if aligned:
                        owner[r0 + int(row_off[lr]), wall_c] = node_idx[a]
                    else:
                        owner[r0 : r0 + block, wall_c] = node_idx[a]
            elif not a_node and not b_node:
                owner[r0 : r0 + block, wall_c] = FILLER

    for lr in range(H - 1):
        wall_r = (lr + 1) * step
        for lc in range(W):
            pa, pb = (lr, lc), (lr + 1, lc)
            a_node = pa in pos_to_node
            b_node = pb in pos_to_node
            c0 = 1 + lc * step
            if a_node and b_node:
                a, b = pos_to_node[pa], pos_to_node[pb]
                if edge_passable(G, a, b):
                    if aligned:
                        owner[wall_r, c0 + int(col_off[lc])] = node_idx[a]
                    else:
                        owner[wall_r, c0 : c0 + block] = node_idx[a]
            elif not a_node and not b_node:
                owner[wall_r, c0 : c0 + block] = FILLER

    return owner, FILLER


def _place_seeds(
    positions: dict[int, tuple[int, int]],
    H: int,
    W: int,
    step: int,
    block: int,
    rng: np.random.Generator,
    row_off: np.ndarray | None = None,
    col_off: np.ndarray | None = None,
) -> tuple[dict[int, tuple[int, int]], list[tuple[int, int]]]:
    """One node seed per occupied lattice cell, one filler seed per empty
    cell.

    When ``row_off``/``col_off`` are supplied, node seeds use those
    aligned offsets so adjacent blocks share a fine row (vertical walls)
    or column (horizontal walls) — a corridor crossing the wall is then
    a straight line directly into both seeds. Without them, offsets are
    random per-block (legacy behaviour).

    Filler seeds are always random — filler regions are isolated from
    node regions by hard walls, so their positions don't need to align.
    """
    offsets = list(range(0, block, 2))
    pos_to_node = {p: n for n, p in positions.items()}
    aligned = row_off is not None and col_off is not None

    seeds: dict[int, tuple[int, int]] = {}
    for n, (lr, lc) in positions.items():
        r0 = 1 + lr * step
        c0 = 1 + lc * step
        if aligned:
            dr = int(row_off[lr])
            dc = int(col_off[lc])
        else:
            dr = offsets[int(rng.integers(0, len(offsets)))]
            dc = offsets[int(rng.integers(0, len(offsets)))]
        seeds[n] = (r0 + dr, c0 + dc)

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

    return seeds, filler_seeds


def _carve_branches(
    maze: np.ndarray, owner: np.ndarray, rng: np.random.Generator
) -> None:
    """Recursive-backtracker flood carver, 2-step jumps, owner-aware.

    Every jump's midpoint and landing cell must have ``owner ≥ 0``.
    Consequence: corridors flow across edge shared-walls freely, never
    cross non-edge shared-walls or intersection cells (both owner = -1).
    Seeds of connected-in-G nodes end up in the same corridor network.

    Each seed first receives one guaranteed 2-step extension. Without
    this, a seed processed late by the main backtracker could find all
    four of its 2-step landings already corridor — the backtracker would
    pop immediately, leaving the seed an isolated island.
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
    labels: dict[tuple[int, int], str] | None = None,
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
    if labels:
        font = _label_font(cell_px)
        for (r, c), text in labels.items():
            _paint_label(draw, r, c, cell_px, text, font)
    draw.rectangle([0, 0, w * cell_px - 1, h * cell_px - 1], outline=_C_BORDER, width=1)

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return Image.open(buf).copy()


def _label_font(cell_px: int) -> ImageFont.ImageFont:
    """Pick a font size that fits a 2-character label inside a single cell."""
    size = max(8, int(cell_px * 0.55))
    for name in ("arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _paint_label(
    draw: ImageDraw.ImageDraw,
    r: int,
    c: int,
    cell_px: int,
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    cx = c * cell_px + cell_px / 2
    cy = r * cell_px + cell_px / 2
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(
        (cx - tw / 2 - bbox[0], cy - th / 2 - bbox[1]),
        text,
        fill=(0, 0, 0),
        font=font,
    )


def _paint_cell(
    draw: ImageDraw.ImageDraw,
    r: int,
    c: int,
    cell_px: int,
    color: tuple[int, int, int],
) -> None:
    x0, y0 = c * cell_px, r * cell_px
    draw.rectangle([x0, y0, x0 + cell_px - 1, y0 + cell_px - 1], fill=color)
