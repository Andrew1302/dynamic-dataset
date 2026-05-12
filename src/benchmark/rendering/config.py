"""Rendering configuration for the plain (direct-view) graph image.

A single :class:`RenderConfig` is threaded through ``BenchmarkTask`` so
that ablations (lettered/no labels, alternate node color, straight vs.
curved edges) can be swept without rewriting per-task render methods.

The defaults match what an ablation matrix should treat as the
*baseline* condition: numeric labels, the original light-blue palette,
and straight edges. The previous curved-edge weighted rendering is
still available behind ``edge_style="curved"`` for backward visual
comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LabelStyle = Literal["numeric", "letters", "none"]
EdgeStyle = Literal["straight", "curved"]


DEFAULT_NODE_COLOR = "#AED6F1"


@dataclass(frozen=True)
class RenderConfig:
    label_style: LabelStyle = "numeric"
    node_color: str = DEFAULT_NODE_COLOR
    edge_style: EdgeStyle = "straight"


def node_label(node_id: int, style: LabelStyle) -> str:
    """Render *node_id* as a human-facing label per *style*.

    ``"none"`` falls back to numeric — used when something still has to
    refer to a node textually (e.g. in the adjacency-matrix header) even
    though the on-image labels are suppressed.
    """
    if style == "letters":
        return _letter_label(int(node_id))
    return str(int(node_id))


def _letter_label(n: int) -> str:
    """0→A, 25→Z, 26→AA, 27→AB, …"""
    if n < 0:
        raise ValueError(f"node id must be non-negative, got {n}")
    s = ""
    n += 1
    while n > 0:
        n, rem = divmod(n - 1, 26)
        s = chr(ord("A") + rem) + s
    return s
