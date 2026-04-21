"""Map disguise: graph → uncolored political map with black borders.

Each node becomes a geographic region. Every region is filled with pure
white — no color hints — and separated from its neighbors by a uniform
black border. The LLM is asked how many colors the map needs, which
equals the graph's chromatic number.

Derived from ``PoC_graph_disguise.py`` adapted to return a ``PIL.Image``.
"""

from __future__ import annotations

from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt

from ._voronoi import build_voronoi


_BG = np.array([1.0, 1.0, 1.0], dtype=np.float32)
_BORDER = np.array([0.05, 0.05, 0.05], dtype=np.float32)


def render_map(
    G: nx.Graph,
    seed: int = 42,
    show_labels: bool = False,
    pos: dict | None = None,
) -> Image.Image:
    """Render *G* as a black-and-white map and return it as a PIL image."""
    owner, _, nodes, grid_size = build_voronoi(G, seed, pos=pos)
    n = len(nodes)

    owner = _fill_unclaimed(owner)

    img = np.tile(_BG, (grid_size, grid_size, 1))

    border_mask = _border_mask(owner)
    border_mask = binary_dilation(border_mask, iterations=1)

    frame = 3
    border_mask[:frame, :] = True
    border_mask[-frame:, :] = True
    border_mask[:, :frame] = True
    border_mask[:, -frame:] = True

    img[border_mask] = _BORDER

    dpi = 150
    size_in = max(5.0, grid_size / dpi)
    fig, ax = plt.subplots(figsize=(size_in, size_in), facecolor="white")
    ax.imshow(img, interpolation="nearest")

    if show_labels:
        fs = max(6, min(12, 80 // max(n, 1)))
        for i, nd in enumerate(nodes):
            region = np.argwhere(owner == i)
            if len(region):
                cr, cc = region.mean(axis=0)
                ax.text(
                    cc, cr, str(nd),
                    ha="center", va="center",
                    fontsize=fs, fontweight="bold", color="#1a1a1a",
                    bbox=dict(
                        boxstyle="round,pad=0.2", fc="white",
                        ec="#999", alpha=0.88, lw=0.7,
                    ),
                )

    ax.axis("off")
    plt.tight_layout(pad=0.3)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def _border_mask(owner: np.ndarray) -> np.ndarray:
    """Boolean mask of all cells on a region boundary."""
    mask = np.zeros_like(owner, dtype=bool)
    right = owner[:, 1:] != owner[:, :-1]
    down = owner[1:, :] != owner[:-1, :]
    mask[:, :-1] |= right
    mask[:, 1:] |= right
    mask[:-1, :] |= down
    mask[1:, :] |= down
    return mask


def _fill_unclaimed(owner: np.ndarray) -> np.ndarray:
    """Assign every UNCLAIMED / WALL cell to the nearest region label."""
    has_region = owner >= 0
    if has_region.all():
        return owner
    _, (ri, ci) = distance_transform_edt(~has_region, return_indices=True)
    return owner[ri, ci]
