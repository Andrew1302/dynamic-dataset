"""
visualization.py — PIL image renderers for each benchmark domain.

Each function takes domain-specific data and returns a PIL.Image.Image.
All rendering uses only matplotlib + Pillow (already in pyproject dependencies).
"""

from __future__ import annotations

import io
import math
from typing import TYPE_CHECKING, Sequence

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    pass  # keep import block clean


# ---------------------------------------------------------------------------
# Colour palette (consistent across all renderers)
# ---------------------------------------------------------------------------

_C_NODE      = "#AED6F1"   # default node — light blue
_C_SOURCE    = "#82E0AA"   # source/start — green
_C_TARGET    = "#F1948A"   # target/end   — red
_C_WALL      = "#2C3E50"   # maze wall    — dark navy
_C_OPEN      = "#F8F9F9"   # maze open    — near-white
_C_TILE      = "#5DADE2"   # puzzle tile  — blue
_C_BLANK     = "#D5D8DC"   # blank tile   — light grey
_C_PEG       = "#6E2F1A"   # Hanoi peg    — brown
_C_BASE      = "#884C2B"   # Hanoi base   — brown
_C_BG        = "#FDFEFE"   # figure background


# ---------------------------------------------------------------------------
# 1. Graph image (networkx → PIL)
# ---------------------------------------------------------------------------

def draw_graph(
    G: nx.Graph,
    source: int | None = None,
    target: int | None = None,
    weighted: bool = False,
    title: str = "",
    figsize: tuple[float, float] = (5, 4),
) -> Image.Image:
    """Render a NetworkX graph as a PIL Image.

    source node → green,  target node → red,  others → light blue.
    Edge weights are shown as labels when weighted=True.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=_C_BG)
    ax.set_facecolor(_C_BG)
    ax.set_axis_off()

    if title:
        ax.set_title(title, fontsize=10, pad=6)

    if G.number_of_nodes() == 0:
        _save_and_close(fig)
        return Image.new("RGB", (int(figsize[0] * 120), int(figsize[1] * 120)), _C_BG)

    pos = nx.spring_layout(G, seed=42)

    node_colors = [
        _C_SOURCE if n == source else (_C_TARGET if n == target else _C_NODE)
        for n in G.nodes()
    ]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=500)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.5, alpha=0.8,
                           arrows=isinstance(G, nx.DiGraph))

    if weighted:
        edge_labels = {(u, v): d.get("weight", "") for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                     font_size=7, label_pos=0.35)

    # Legend
    handles = []
    if source is not None:
        handles.append(mpatches.Patch(color=_C_SOURCE, label=f"Source: {source}"))
    if target is not None:
        handles.append(mpatches.Patch(color=_C_TARGET, label=f"Target: {target}"))
    if handles:
        ax.legend(handles=handles, fontsize=7, loc="upper left",
                  framealpha=0.7, borderpad=0.4)

    return _save_and_close(fig)


# ---------------------------------------------------------------------------
# 2. Maze image (2-D char grid → PIL)
# ---------------------------------------------------------------------------

def draw_maze(
    grid: list[list[str]],
    start: list[int],
    end: list[int],
    cell_px: int = 12,
    title: str = "",
) -> Image.Image:
    """Render a character maze grid as a pixel image.

    '#' → dark wall, '.' → open, start → green, end → red.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    img_w = cols * cell_px
    img_h = rows * cell_px
    title_h = 22 if title else 0
    total_h = img_h + title_h

    img = Image.new("RGB", (img_w, total_h), color=_C_WALL)
    draw = ImageDraw.Draw(img)

    if title:
        draw.rectangle([0, 0, img_w, title_h - 1], fill="#EAECEE")
        draw.text((4, 3), title, fill="#2C3E50", font=_tiny_font())

    for r in range(rows):
        for c in range(cols):
            x1 = c * cell_px
            y1 = r * cell_px + title_h
            x2 = x1 + cell_px - 1
            y2 = y1 + cell_px - 1
            ch = grid[r][c]
            if [r, c] == start:
                color = _C_SOURCE
            elif [r, c] == end:
                color = _C_TARGET
            elif ch == ".":
                color = _C_OPEN
            else:
                color = _C_WALL
            draw.rectangle([x1, y1, x2, y2], fill=color)

    return img


# ---------------------------------------------------------------------------
# 3. Word-ladder image (vocabulary word-graph → PIL)
# ---------------------------------------------------------------------------

def draw_word_ladder(
    vocabulary: list[str],
    source_word: str,
    target_word: str,
    title: str = "",
) -> Image.Image:
    """Render the word-ladder vocabulary as a graph where Hamming-1 pairs are edges.

    source_word → green,  target_word → red,  others → light blue.
    """
    # Build word graph
    G_w: nx.Graph = nx.Graph()
    G_w.add_nodes_from(vocabulary)
    for i, w1 in enumerate(vocabulary):
        for w2 in vocabulary[i + 1:]:
            if _hamming(w1, w2) == 1:
                G_w.add_edge(w1, w2)

    fig, ax = plt.subplots(figsize=(5, 4), facecolor=_C_BG)
    ax.set_facecolor(_C_BG)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10, pad=6)

    pos = nx.spring_layout(G_w, seed=42)
    node_colors = [
        _C_SOURCE if n == source_word else (_C_TARGET if n == target_word else _C_NODE)
        for n in G_w.nodes()
    ]
    nx.draw_networkx_nodes(G_w, pos, ax=ax, node_color=node_colors, node_size=800)
    nx.draw_networkx_labels(G_w, pos, ax=ax, font_size=7, font_weight="bold")
    nx.draw_networkx_edges(G_w, pos, ax=ax, width=1.2, alpha=0.7)

    handles = [
        mpatches.Patch(color=_C_SOURCE, label=f"Start: {source_word}"),
        mpatches.Patch(color=_C_TARGET, label=f"Goal: {target_word}"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper left",
              framealpha=0.7, borderpad=0.4)

    return _save_and_close(fig)


# ---------------------------------------------------------------------------
# 4. Sliding puzzle image (flat tuple → N×N grid PIL)
# ---------------------------------------------------------------------------

def draw_sliding_puzzle(
    board: list[int],
    n: int,
    goal: list[int] | None = None,
    title: str = "",
) -> Image.Image:
    """Render an N×N sliding puzzle board as a PIL image.

    '0' = blank tile (grey).  Optionally shows the goal board to the right.
    """
    cell = 64
    pad = 8
    board_px = n * cell + 2 * pad

    panels = [("Initial", board)]
    if goal is not None:
        panels.append(("Goal", goal))

    total_w = board_px * len(panels) + pad * (len(panels) - 1)
    title_h = 22 if title else 0
    panel_title_h = 18
    total_h = board_px + title_h + panel_title_h

    img = Image.new("RGB", (total_w, total_h), color=_C_BG)
    draw = ImageDraw.Draw(img)

    if title:
        draw.rectangle([0, 0, total_w, title_h - 1], fill="#EAECEE")
        draw.text((4, 3), title, fill="#2C3E50", font=_tiny_font())

    font_big = _tile_font()
    font_sm = _tiny_font()

    for p_idx, (panel_label, flat) in enumerate(panels):
        x_off = p_idx * (board_px + pad)
        y_off = title_h

        # Panel label
        draw.text((x_off + pad, y_off + 2), panel_label, fill="#2C3E50", font=font_sm)
        y_off += panel_title_h

        # Background
        draw.rectangle([x_off, y_off, x_off + board_px, y_off + board_px],
                       fill="#BDC3C7")

        for i, val in enumerate(flat):
            r, c = divmod(i, n)
            x1 = x_off + pad + c * cell
            y1 = y_off + pad + r * cell
            x2 = x1 + cell - 4
            y2 = y1 + cell - 4

            if val == 0:
                color = _C_BLANK
                label = ""
            else:
                color = _C_TILE
                label = str(val)

            draw.rounded_rectangle([x1, y1, x2, y2], radius=6, fill=color,
                                   outline="#ECF0F1", width=2)
            if label:
                # Centre the text
                bbox = draw.textbbox((0, 0), label, font=font_big)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                tx = x1 + (cell - 4 - tw) // 2
                ty = y1 + (cell - 4 - th) // 2
                draw.text((tx, ty), label, fill="white", font=font_big)

    return img


# ---------------------------------------------------------------------------
# 5. Tower of Hanoi image
# ---------------------------------------------------------------------------

def draw_hanoi(
    state: list[int],
    n_disks: int,
    n_pegs: int,
    goal: list[int] | None = None,
    title: str = "",
) -> Image.Image:
    """Render a Tower of Hanoi configuration as a PIL image.

    state[i] = peg index of disk i (disk 0 = smallest).
    Optionally renders the goal state below with a divider.
    """
    panels: list[tuple[str, list[int]]] = [("Initial", list(state))]
    if goal is not None:
        panels.append(("Goal", list(goal)))

    W = max(320, n_pegs * 100 + 40)
    peg_area_h = 30 + n_disks * 20 + 30   # base + disks + top gap
    panel_h = peg_area_h + 22              # + label row

    total_h = panel_h * len(panels) + (22 if title else 0) + 8
    img = Image.new("RGB", (W, total_h), color=_C_BG)
    draw = ImageDraw.Draw(img)

    font_sm = _tiny_font()

    y0 = 0
    if title:
        draw.rectangle([0, 0, W, 21], fill="#EAECEE")
        draw.text((4, 3), title, fill="#2C3E50", font=font_sm)
        y0 = 22

    peg_xs = [W * (i + 1) // (n_pegs + 1) for i in range(n_pegs)]
    peg_top_y = 18   # relative to panel top (after label row)

    DISK_H = 18
    MAX_DISK_W = min(80, W // n_pegs - 10)

    # Disk colours (rainbow-ish, indexed by disk number)
    disk_colors_hex = [
        "#E74C3C", "#E67E22", "#F1C40F",
        "#2ECC71", "#3498DB", "#9B59B6",
        "#1ABC9C", "#E91E63",
    ]

    for p_idx, (panel_label, cfg) in enumerate(panels):
        y_panel = y0 + p_idx * panel_h

        # Divider
        if p_idx > 0:
            draw.line([(0, y_panel - 1), (W, y_panel - 1)], fill="#BDC3C7", width=1)

        # Panel label
        draw.text((4, y_panel + 3), panel_label, fill="#555", font=font_sm)

        base_y = y_panel + 22 + peg_area_h - 30  # Y of the base line

        # Draw base bar
        draw.line([(peg_xs[0] - 50, base_y), (peg_xs[-1] + 50, base_y)],
                  fill=_C_BASE, width=5)

        # Draw pegs
        for px in peg_xs:
            draw.line([(px, base_y), (px, y_panel + 22 + peg_top_y)],
                      fill=_C_PEG, width=4)

        # Build stack for each peg (bottom-up)
        stacks: dict[int, list[int]] = {p: [] for p in range(n_pegs)}
        for disk in range(n_disks - 1, -1, -1):  # largest → smallest
            stacks[cfg[disk]].append(disk)
        # stacks[p] is now ordered: largest first = bottom of stack

        for peg_i, stack in stacks.items():
            px = peg_xs[peg_i]
            for level, disk in enumerate(reversed(stack)):
                # disk 0 = smallest → thinnest width
                disk_w = int(MAX_DISK_W * (n_disks - disk) / n_disks)
                disk_w = max(disk_w, 14)
                x1 = px - disk_w // 2
                x2 = px + disk_w // 2
                y1 = base_y - (level + 1) * DISK_H
                y2 = y1 + DISK_H - 2
                color = disk_colors_hex[disk % len(disk_colors_hex)]
                draw.rounded_rectangle([x1, y1, x2, y2], radius=3,
                                       fill=color, outline="#ECF0F1", width=1)
                # Disk number label
                label = str(disk)
                bbox = draw.textbbox((0, 0), label, font=font_sm)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text((px - tw // 2, y1 + (DISK_H - 2 - th) // 2),
                          label, fill="white", font=font_sm)

        # Peg labels (0, 1, 2, ...)
        for i, px in enumerate(peg_xs):
            draw.text((px - 4, base_y + 6), str(i), fill="#555", font=font_sm)

    return img


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_and_close(fig: "plt.Figure") -> Image.Image:
    """Save matplotlib figure to a PNG in memory and return PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def _tiny_font() -> ImageFont.ImageFont:
    """Return a small PIL font (falls back to default if no TTF available)."""
    try:
        return ImageFont.truetype("arial.ttf", 11)
    except OSError:
        return ImageFont.load_default()


def _tile_font() -> ImageFont.ImageFont:
    """Larger font for puzzle tile numbers."""
    try:
        return ImageFont.truetype("arial.ttf", 22)
    except OSError:
        return ImageFont.load_default()
