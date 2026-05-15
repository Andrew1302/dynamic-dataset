"""Directed maze disguise: seed-as-junction node carving + braided filler.

Node-owned regions are carved by drilling **straight arms** from each
node's seed cell to the unique passable wall cell on every side that
has an outgoing or incoming edge (see ``_carve_node_arms``). The seed
is therefore the only cell of corridor-degree ≥ 3 inside any node
block — every other in-block corridor cell is degree-2 and lies on a
single stroke whose body crosses the wall cell of the matching edge.

Why the seed-as-junction invariant matters: the arrow pipeline votes
each stroke's direction from the wall-cell directions in its body
(``_compute_wall_directions`` sets one direction per passable wall).
With the seed as the only junction, every node-block stroke contains
exactly one wall cell and gets a decisive forward-vs-backward vote —
the tie-breaker that previously mis-oriented internal strokes when
the seed was a tree-leaf is never exercised. End-to-end directed
reachability between any two node seeds in the rendered maze matches
``nx.has_path(G, u, v)`` by construction.

Filler regions (empty lattice slots) are still carved with the
recursive-backtracker spanning-tree carver and then **braided** to
close some short loops. Filler is isolated from node regions by hard
walls (``-1`` in the ownership grid), so it cannot leak directed
reachability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO

import networkx as nx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import LabelStyle, node_label
from .maze import (
    _CORRIDOR,
    _C_BORDER,
    _C_CORRIDOR,
    _C_ENTRANCE,
    _C_EXIT,
    _C_NODE,
    _C_WALL,
    _DIRS_2,
    _WALL,
    _build_ownership_grid,
    _paint_cell,
    _pick_axis_offsets,
    _place_seeds,
)

_C_ARROW = (170, 170, 170)
_DIRS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

_BRAID_PROBABILITY = 0.65

Cell = tuple[int, int]
Direction = tuple[int, int]
Stroke = list[Cell]


@dataclass(frozen=True)
class Arrow:
    r: int
    c: int
    direction: Direction


@dataclass(frozen=True)
class DirectedMaze:
    maze: np.ndarray
    seeds: dict[int, Cell]
    entrance: int
    exit: int
    arrows: tuple[Arrow, ...] = field(default_factory=tuple)
    cell_px: int = 14
    highlight_all_nodes: bool = True
    label_style: LabelStyle = "numeric"

    def render(self) -> Image.Image:
        decoy_seeds: list[Cell] = []
        labels: dict[Cell, str] = {}
        if self.highlight_all_nodes:
            for node_id, cell in self.seeds.items():
                if node_id != self.entrance and node_id != self.exit:
                    decoy_seeds.append(cell)
                if self.label_style != "none":
                    labels[cell] = node_label(node_id, self.label_style)
        return _render_image(
            self.maze,
            self.cell_px,
            self.seeds[self.entrance],
            self.seeds[self.exit],
            decoy_seeds,
            self.arrows,
            labels,
        )


@dataclass(frozen=True)
class WallDirections:
    """Direction assigned to each shared-wall column/row, looked up by
    the fine cell coordinates that fall on that wall.

    Vertical walls run between lattice columns ``lc-1`` and ``lc``;
    horizontal walls run between lattice rows ``lr-1`` and ``lr``.
    """

    vertical: dict[tuple[int, int], Direction]
    horizontal: dict[tuple[int, int], Direction]
    H: int
    W: int
    step: int

    def at(self, r: int, c: int) -> Direction | None:
        if 0 < c < self.W * self.step and c % self.step == 0:
            return self.vertical.get(((r - 1) // self.step, c // self.step))
        if 0 < r < self.H * self.step and r % self.step == 0:
            return self.horizontal.get((r // self.step, (c - 1) // self.step))
        return None


def build_directed_maze(
    G: nx.DiGraph,
    seed: int,
    entrance: int,
    exit: int,
    cell_px: int = 18,
    block: int = 7,
    highlight_all_nodes: bool = True,
    label_style: LabelStyle = "numeric",
) -> DirectedMaze:
    """Carve a directed maze. Topology mirrors the underlying undirected
    adjacency; arrows encode direction.

    Node blocks: seed-as-junction carving (``_carve_node_arms``) — the
    seed is the only branching point, with one straight corridor arm
    per incident edge running to the aligned wall cell.

    Filler regions: spanning-tree forest + light braiding, isolated
    from node regions by hard walls.
    """
    positions = {n: G.nodes[n]["lattice"] for n in G.nodes()}
    H = max(p[0] for p in positions.values()) + 1
    W = max(p[1] for p in positions.values()) + 1

    step = block + 1
    rng = np.random.default_rng(seed)
    node_ids = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(node_ids)}

    row_off, col_off = _pick_axis_offsets(H, W, block, rng)

    owner, filler_id = _build_ownership_grid(
        G, positions, node_ids, H, W, step, block,
        edge_passable=_directed_passable,
        row_off=row_off, col_off=col_off,
    )

    seeds, filler_seeds = _place_seeds(
        positions, H, W, step, block, rng,
        row_off=row_off, col_off=col_off,
    )

    fh = H * step + 1
    fw = W * step + 1
    maze = np.full((fh, fw), _WALL, dtype=np.uint8)

    _carve_node_arms(maze, owner, seeds, positions, H, W, step)
    _carve_block_decoys(maze, owner, rng, seeds, node_idx)
    _carve_tree_forest(maze, owner, rng, filler_seeds)
    _braid_filler(maze, owner, rng, filler_id, set(filler_seeds))

    wall_dirs = _compute_wall_directions(G, positions, H, W, step, rng)
    node_seeds = set(seeds.values())
    all_seeds = list(seeds.values()) + filler_seeds
    arrows = _compute_arrows(maze, all_seeds, node_seeds, wall_dirs, rng)

    return DirectedMaze(
        maze=maze,
        seeds=seeds,
        entrance=entrance,
        exit=exit,
        arrows=tuple(arrows),
        cell_px=cell_px,
        highlight_all_nodes=highlight_all_nodes,
        label_style=label_style,
    )


def _carve_node_arms(
    maze: np.ndarray,
    owner: np.ndarray,
    seeds: dict[int, Cell],
    positions: dict[int, Cell],
    H: int,
    W: int,
    step: int,
) -> None:
    """For every occupied lattice block, drill a straight 1-cell-wide
    corridor from the node's seed to the unique passable wall cell on
    each side that has an incident edge.

    The aligned ownership grid guarantees that each passable wall has a
    single passable cell on the same fine row (vertical walls) or column
    (horizontal walls) as the two adjacent seeds, so each arm is a pure
    horizontal or vertical line — no bends, no junctions other than the
    seed itself.
    """
    for n, (lr, lc) in positions.items():
        cr, cc = seeds[n]
        maze[cr, cc] = _CORRIDOR

        if lc > 0:
            wall_c = lc * step
            if owner[cr, wall_c] >= 0:
                maze[cr, wall_c : cc + 1] = _CORRIDOR
        if lc < W - 1:
            wall_c = (lc + 1) * step
            if owner[cr, wall_c] >= 0:
                maze[cr, cc : wall_c + 1] = _CORRIDOR
        if lr > 0:
            wall_r = lr * step
            if owner[wall_r, cc] >= 0:
                maze[wall_r : cr + 1, cc] = _CORRIDOR
        if lr < H - 1:
            wall_r = (lr + 1) * step
            if owner[wall_r, cc] >= 0:
                maze[cr : wall_r + 1, cc] = _CORRIDOR


def _carve_block_decoys(
    maze: np.ndarray,
    owner: np.ndarray,
    rng: np.random.Generator,
    seeds: dict[int, Cell],
    node_idx: dict[int, int],
) -> None:
    """Recursive-backtracker dead-end branches inside each occupied block.

    The carver advances by 2-step jumps from the seed, restricted to
    cells owned by the same node. Both the midpoint and the landing must
    be uncarved before a step is taken — without the midpoint guard a
    decoy could carve through a pre-existing arm cell and create a new
    corridor junction off the seed, breaking the seed-as-only-junction
    invariant the arrow pipeline depends on.
    """
    h, w = maze.shape
    for n, root in seeds.items():
        nid = node_idx[n]
        stack = [root]
        while stack:
            r, c = stack[-1]
            advanced = False
            for k in rng.permutation(4):
                dr, dc = _DIRS_2[int(k)]
                nr, nc = r + dr, c + dc
                mr, mc = r + dr // 2, c + dc // 2
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if owner[nr, nc] != nid or owner[mr, mc] != nid:
                    continue
                if maze[nr, nc] == _CORRIDOR or maze[mr, mc] == _CORRIDOR:
                    continue
                maze[mr, mc] = _CORRIDOR
                maze[nr, nc] = _CORRIDOR
                stack.append((nr, nc))
                advanced = True
                break
            if not advanced:
                stack.pop()


def render_directed_maze(
    G: nx.DiGraph,
    seed: int,
    entrance: int,
    exit: int,
    cell_px: int = 18,
    block: int = 7,
    highlight_all_nodes: bool = True,
    label_style: LabelStyle = "numeric",
) -> Image.Image:
    return build_directed_maze(
        G, seed, entrance, exit, cell_px, block, highlight_all_nodes, label_style
    ).render()


def _directed_passable(G: nx.DiGraph, a: int, b: int) -> bool:
    return G.has_edge(a, b) or G.has_edge(b, a)


# ---------------------------------------------------------------------------
# Carving
# ---------------------------------------------------------------------------


def _carve_tree_forest(
    maze: np.ndarray,
    owner: np.ndarray,
    rng: np.random.Generator,
    seeds: list[Cell],
) -> None:
    """One spanning tree per ownership component.

    Each seed not yet covered by an existing tree becomes the root of a
    recursive-backtracker carve. The ``maze[nr, nc] == _CORRIDOR`` guard
    inside the carve keeps trees from merging — the corridor graph is
    acyclic, so every cell has degree 1–3 and no 2×2 squares form.
    """
    h, w = maze.shape
    for root in seeds:
        if maze[root] == _CORRIDOR:
            continue
        maze[root] = _CORRIDOR
        stack = [root]
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


def _braid_filler(
    maze: np.ndarray,
    owner: np.ndarray,
    rng: np.random.Generator,
    filler_id: int,
    filler_seeds: set[Cell],
) -> None:
    """Close cycles in filler-owned regions to reduce dead-ends.

    For each degree-1 corridor cell that lives in a filler region (and
    isn't a filler seed), with probability ``_BRAID_PROBABILITY``,
    attempt one 2-step jump that lands on an existing filler corridor
    via a wall cell that is currently uncarved. The jump must respect
    the same midpoint/landing ``owner ≥ 0`` invariant the carver uses.

    Filler regions are isolated from node-owned regions by hard walls
    (``owner == -1``), so the only landings reachable from a filler
    cell are themselves filler cells — braiding cannot create new
    corridor connections between graph nodes.
    """
    h, w = maze.shape
    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if maze[r, c] == _CORRIDOR
        and owner[r, c] == filler_id
        and (r, c) not in filler_seeds
        and _corridor_degree(maze, r, c) == 1
    ]

    order = rng.permutation(len(candidates))
    for i in order:
        r, c = candidates[int(i)]
        if _corridor_degree(maze, r, c) != 1:
            continue  # an earlier braid already touched this cell
        if rng.random() > _BRAID_PROBABILITY:
            continue
        for k in rng.permutation(4):
            dr, dc = _DIRS_2[int(k)]
            nr, nc = r + dr, c + dc
            mr, mc = r + dr // 2, c + dc // 2
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if owner[nr, nc] < 0 or owner[mr, mc] < 0:
                continue
            if maze[nr, nc] != _CORRIDOR or maze[mr, mc] == _CORRIDOR:
                continue
            maze[mr, mc] = _CORRIDOR
            break


def _corridor_degree(maze: np.ndarray, r: int, c: int) -> int:
    h, w = maze.shape
    count = 0
    for dr, dc in _DIRS_4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and maze[nr, nc] == _CORRIDOR:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Arrows
# ---------------------------------------------------------------------------


def _compute_wall_directions(
    G: nx.DiGraph,
    positions: dict[int, Cell],
    H: int,
    W: int,
    step: int,
    rng: np.random.Generator,
) -> WallDirections:
    """Per-wall direction map. Graph-edge walls take the edge direction;
    filler↔filler walls take a random sign along the wall's axis."""
    pos_to_node = {p: n for n, p in positions.items()}
    vertical = _build_axis_wall_dirs(
        G, pos_to_node, H, W, axis="vertical", rng=rng
    )
    horizontal = _build_axis_wall_dirs(
        G, pos_to_node, H, W, axis="horizontal", rng=rng
    )
    return WallDirections(vertical, horizontal, H=H, W=W, step=step)


def _build_axis_wall_dirs(
    G: nx.DiGraph,
    pos_to_node: dict[Cell, int],
    H: int,
    W: int,
    axis: str,
    rng: np.random.Generator,
) -> dict[tuple[int, int], Direction]:
    """Build the wall-direction map for one axis.

    For ``axis == "vertical"`` the wall sits between lattice columns
    ``lc-1`` (call it *left*) and ``lc`` (*right*); the natural forward
    direction is ``(0, 1)``. For ``axis == "horizontal"`` the wall sits
    between rows ``lr-1`` (*above*) and ``lr`` (*below*); forward is
    ``(1, 0)``.
    """
    if axis == "vertical":
        outer_range, inner_range = range(H), range(1, W)
        forward: Direction = (0, 1)
        side_a = lambda lo, li: (lo, li - 1)  # left / above
        side_b = lambda lo, li: (lo, li)       # right / below
    else:
        outer_range, inner_range = range(W), range(1, H)
        forward = (1, 0)
        side_a = lambda lo, li: (li - 1, lo)
        side_b = lambda lo, li: (li, lo)

    backward: Direction = (-forward[0], -forward[1])
    out: dict[tuple[int, int], Direction] = {}
    for outer in outer_range:
        for inner in inner_range:
            pa, pb = side_a(outer, inner), side_b(outer, inner)
            a = pos_to_node.get(pa)
            b = pos_to_node.get(pb)
            key = (outer, inner) if axis == "vertical" else (inner, outer)
            if a is not None and b is not None:
                if G.has_edge(a, b):
                    out[key] = forward
                elif G.has_edge(b, a):
                    out[key] = backward
            elif a is None and b is None:
                out[key] = forward if rng.integers(0, 2) else backward
    return out


def _corridor_neighbours(maze: np.ndarray) -> dict[Cell, list[Cell]]:
    h, w = maze.shape
    out: dict[Cell, list[Cell]] = {}
    for r in range(h):
        for c in range(w):
            if maze[r, c] != _CORRIDOR:
                continue
            nbrs: list[Cell] = []
            for dr, dc in _DIRS_4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and maze[nr, nc] == _CORRIDOR:
                    nbrs.append((nr, nc))
            out[(r, c)] = nbrs
    return out


def _extract_strokes(
    neighbours: dict[Cell, list[Cell]],
    is_break: set[Cell],
) -> list[Stroke]:
    """Decompose corridors into strokes.

    A *body* cell has exactly two corridor neighbours and is not in
    ``is_break``. Each stroke is the maximal path that walks through
    body cells between two non-body endpoints — junctions, dead-ends,
    or break cells (typically seeds).

    Pure cycles (every cell on the loop is a body cell, no endpoint
    exists) only arise inside braided filler regions. They're emitted
    as closed strokes with ``stroke[0] == stroke[-1]`` so the caller
    can still place arrows along them.
    """
    body = {
        cell for cell, nbrs in neighbours.items()
        if len(nbrs) == 2 and cell not in is_break
    }
    strokes: list[Stroke] = []
    visited: set[Cell] = set()

    for cell, nbrs in neighbours.items():
        if cell in body:
            continue
        for nbr in nbrs:
            if nbr not in body or nbr in visited:
                continue
            strokes.append(_walk_stroke(neighbours, body, visited, cell, nbr))

    for cell in neighbours:
        if cell in visited or cell not in body:
            continue
        strokes.append(_walk_cycle(neighbours, visited, cell))

    return strokes


def _walk_stroke(
    neighbours: dict[Cell, list[Cell]],
    body: set[Cell],
    visited: set[Cell],
    start_endpoint: Cell,
    first_body: Cell,
) -> Stroke:
    stroke: Stroke = [start_endpoint, first_body]
    visited.add(first_body)
    prev, curr = start_endpoint, first_body
    while True:
        next_cells = [n for n in neighbours[curr] if n != prev]
        if not next_cells:
            break
        nxt = next_cells[0]
        stroke.append(nxt)
        if nxt not in body or nxt in visited:
            break
        visited.add(nxt)
        prev, curr = curr, nxt
    return stroke


def _walk_cycle(
    neighbours: dict[Cell, list[Cell]],
    visited: set[Cell],
    start: Cell,
) -> Stroke:
    """Walk a pure body-cell cycle, returning a stroke that closes back
    on ``start`` (``stroke[0] == stroke[-1]``)."""
    visited.add(start)
    stroke: Stroke = [start]
    prev, curr = start, neighbours[start][0]
    while curr != start:
        visited.add(curr)
        stroke.append(curr)
        next_cells = [n for n in neighbours[curr] if n != prev]
        if not next_cells:
            break
        prev, curr = curr, next_cells[0]
    stroke.append(start)
    return stroke


def _vote_stroke_orientation(
    stroke: Stroke,
    wall_dirs: WallDirections,
    node_seeds: set[Cell],
    rng: np.random.Generator,
) -> Stroke:
    """Return the stroke (possibly reversed) whose forward direction
    agrees with the majority of wall-cell votes along its body.

    On a tie (typical for strokes that never cross a wall — i.e. live
    entirely inside one node's territory), prefer to orient *away* from
    a node seed: arrows inside a node's block should suggest "you can
    leave here", never "you can't move from this direction".
    """
    forward = backward = 0
    for i in range(1, len(stroke) - 1):
        r, c = stroke[i]
        wdir = wall_dirs.at(r, c)
        if wdir is None:
            continue
        nr, nc = stroke[i + 1]
        local = (nr - r, nc - c)
        if local == wdir:
            forward += 1
        elif (-local[0], -local[1]) == wdir:
            backward += 1

    if forward > backward:
        return stroke
    if backward > forward:
        return list(reversed(stroke))

    head_is_seed = stroke[0] in node_seeds
    tail_is_seed = stroke[-1] in node_seeds
    if head_is_seed and not tail_is_seed:
        return stroke
    if tail_is_seed and not head_is_seed:
        return list(reversed(stroke))
    return stroke if rng.integers(0, 2) else list(reversed(stroke))


def _arrow_count(body_len: int) -> int:
    if body_len < 6:
        return 1
    if body_len < 14:
        return 2
    return 3


def _arrow_indices(body_len: int, count: int) -> list[int]:
    """Evenly-spaced indices into ``stroke[1:-1]`` (1-based on the
    stroke array). Returns indices in ``[1, stroke_len - 2]``."""
    indices = []
    for k in range(1, count + 1):
        idx = 1 + (body_len * k - 1) // (count + 1)
        idx = max(1, min(idx, body_len))
        indices.append(idx)
    return indices


def _compute_arrows(
    maze: np.ndarray,
    all_seeds: list[Cell],
    node_seeds: set[Cell],
    wall_dirs: WallDirections,
    rng: np.random.Generator,
) -> list[Arrow]:
    """One to three arrows per corridor stroke, pointing along its
    voted forward direction."""
    seed_set = set(all_seeds)
    neighbours = _corridor_neighbours(maze)
    strokes = _extract_strokes(neighbours, is_break=seed_set)

    arrows: list[Arrow] = []
    for stroke in strokes:
        n = len(stroke)
        if n < 3:
            continue

        oriented = _vote_stroke_orientation(stroke, wall_dirs, node_seeds, rng)
        body_len = n - 2
        for idx in _arrow_indices(body_len, _arrow_count(body_len)):
            r, c = oriented[idx]
            if (r, c) in seed_set:
                continue
            nr, nc = oriented[idx + 1]
            arrows.append(Arrow(r, c, (nr - r, nc - c)))

    return arrows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_image(
    maze: np.ndarray,
    cell_px: int,
    entrance: Cell,
    exit_: Cell,
    decoy_seeds: list[Cell],
    arrows: tuple[Arrow, ...],
    labels: dict[Cell, str] | None = None,
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
    for r, c in decoy_seeds:
        _paint_cell(draw, r, c, cell_px, _C_NODE)
    _paint_endpoint(draw, entrance[0], entrance[1], cell_px, _C_ENTRANCE)
    _paint_endpoint(draw, exit_[0], exit_[1], cell_px, _C_EXIT)
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
    """Pick a font size that fits a 2-digit label inside a single cell."""
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
    """Draw ``text`` centred on the cell at (r, c). Black ink — readable
    on green/red/blue highlight backgrounds."""
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


def _paint_endpoint(
    draw: ImageDraw.ImageDraw,
    r: int,
    c: int,
    cell_px: int,
    color: tuple[int, int, int],
) -> None:
    """Filled cell + thick coloured halo + dark inner outline. Pops
    against the blue decoy-node markers."""
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
    s = cell_px * 0.35
    dr, dc = arrow.direction
    tip = (cx + dc * s, cy + dr * s)
    base = (cx - dc * s, cy - dr * s)
    perp_r, perp_c = -dc, dr
    half = cell_px * 0.28
    b1 = (base[0] + perp_c * half, base[1] + perp_r * half)
    b2 = (base[0] - perp_c * half, base[1] - perp_r * half)
    draw.polygon([tip, b1, b2], fill=_C_ARROW)
