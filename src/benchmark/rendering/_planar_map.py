"""Voronoi-cell map renderer.

Builds one Voronoi cell per graph node and clips it to the node
convex-hull expanded outward. Because ``coloring_graph`` returns a full
Delaunay triangulation, two Voronoi cells share a boundary iff the two
nodes share an edge in G (Delaunay–Voronoi duality). Adjacency is exact
by construction; cells are convex and each node lies well inside its
own cell.

In special / balanced mode the graph is a *subgraph* of the triangulation
(see ``edges`` on :func:`render_planar_map`). There, a cell is eroded inward
along every shared ridge whose two nodes are not adjacent in G, so adjacent
regions share a border while non-adjacent regions are separated by an
open-water gap — keeping border ⇔ adjacency exact for the subgraph too.
"""

from __future__ import annotations

from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image
from scipy.spatial import ConvexHull, Voronoi

from .config import LabelStyle, node_label


SEA_BG = "#DCEBF7"  # sea backdrop; non-adjacent regions are separated by it


def render_planar_map(
    G: nx.Graph,
    pos: dict,
    show_labels: bool = False,
    label_style: LabelStyle = "numeric",
    edges: frozenset | None = None,
    pdf_path: str | None = None,
) -> Image.Image:
    if edges is not None:
        return _render_subgraph_map(
            G, pos, edges, show_labels, label_style, pdf_path
        )
    polygons = build_polygons(G, pos)

    all_pts = np.concatenate([np.asarray(poly) for poly in polygons.values()])
    xlo, ylo = all_pts.min(axis=0)
    xhi, yhi = all_pts.max(axis=0)
    pad = 0.04 * max(xhi - xlo, yhi - ylo)

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
    for v, poly in polygons.items():
        ax.add_patch(MplPolygon(
            poly, closed=True,
            facecolor="white", edgecolor="black",
            linewidth=2.0, joinstyle="miter",
        ))
        if show_labels:
            cx, cy = pos[v]
            ax.text(
                cx, cy, node_label(v, label_style),
                ha="center", va="center",
                fontsize=10, fontweight="bold", color="#1a1a1a",
                bbox=dict(
                    boxstyle="round,pad=0.2", fc="white",
                    ec="#999", alpha=0.88, lw=0.7,
                ),
            )

    ax.set_xlim(xlo - pad, xhi + pad)
    ax.set_ylim(ylo - pad, yhi + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=0.3)

    if pdf_path is not None:
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def build_polygons(G: nx.Graph, pos: dict) -> dict:
    """Return ``{node: [(x, y), ...]}`` — one convex polygon per node.

    Each polygon is the node's Voronoi cell clipped to the expanded
    convex hull of all node positions. Requires a Delaunay triangulation
    as input (every edge of G must be a Delaunay edge between its two
    endpoints) and at least 3 non-collinear nodes.
    """
    nodes = list(G.nodes())
    if len(nodes) < 3:
        raise ValueError("Voronoi map renderer needs ≥3 nodes")

    points = np.array([pos[v] for v in nodes], dtype=float)
    vor = Voronoi(points)
    clip_poly = _clip_hull(points, vor.vertices)

    polygons: dict = {}
    far = 100.0 * float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    for i, v in enumerate(nodes):
        cell = _voronoi_cell(vor, i, far)
        clipped = _clip_polygon(cell, clip_poly)
        if len(clipped) < 3:
            raise RuntimeError(f"empty clipped Voronoi cell for node {v}")
        polygons[v] = clipped
    return polygons


def _clip_hull(points: np.ndarray, vor_vertices: np.ndarray) -> np.ndarray:
    """Clip polygon = point convex hull, scaled outward enough to contain
    every finite Voronoi vertex plus a margin. Sizing off the Voronoi
    vertices guarantees every Voronoi ridge survives clipping intact, so
    adjacency between cells is preserved exactly."""
    hull = ConvexHull(points)
    poly = points[hull.vertices]  # CCW
    center = poly.mean(axis=0)

    factor = 1.0
    for vv in vor_vertices:
        for i in range(len(poly)):
            a = poly[i] - center
            b = poly[(i + 1) % len(poly)] - center
            t = _scale_to_contain(a, b, vv - center)
            if t is not None and t > factor:
                factor = t
    factor *= 1.08  # margin
    return center + (poly - center) * factor


def _scale_to_contain(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float | None:
    """Smallest ``s`` such that the segment ``s*a → s*b`` has ``p`` on its
    inside (left) half-plane. Returns ``None`` when ``p`` is already inside
    for all ``s ≥ 1``."""
    edge = b - a
    normal = np.array([-edge[1], edge[0]])
    denom = np.dot(a, normal)
    num = np.dot(p, normal)
    if denom >= 0 or num >= denom:
        return None
    return num / denom


def _voronoi_cell(vor: Voronoi, point_index: int, far: float) -> list:
    """Return the CCW vertex list of ``point_index``'s Voronoi cell.

    Infinite ridges are extended by ``far`` in their outgoing direction so
    the resulting polygon is finite and contains the cell. It will later
    be clipped to the map's outer hull."""
    region_idx = vor.point_region[point_index]
    region = vor.regions[region_idx]
    if -1 not in region:
        pts = [tuple(vor.vertices[r]) for r in region]
        return _sort_ccw(pts)

    center = vor.points.mean(axis=0)
    vs: list = []
    for ridge_points, ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
        p1, p2 = ridge_points
        if point_index not in (p1, p2):
            continue
        other = p2 if p1 == point_index else p1
        v1, v2 = ridge_vertices
        if v1 >= 0 and v2 >= 0:
            vs.append(tuple(vor.vertices[v1]))
            vs.append(tuple(vor.vertices[v2]))
            continue
        finite_v = v1 if v1 >= 0 else v2
        tangent = vor.points[other] - vor.points[point_index]
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        midpoint = (vor.points[point_index] + vor.points[other]) / 2.0
        if np.dot(midpoint - center, normal) < 0:
            normal = -normal
        far_point = vor.vertices[finite_v] + normal * far
        vs.append(tuple(vor.vertices[finite_v]))
        vs.append(tuple(far_point))
    unique: list = []
    seen: set = set()
    for pt in vs:
        key = (round(pt[0], 9), round(pt[1], 9))
        if key in seen:
            continue
        seen.add(key)
        unique.append(pt)
    return _sort_ccw(unique)


def _sort_ccw(pts: list) -> list:
    arr = np.asarray(pts)
    c = arr.mean(axis=0)
    angles = np.arctan2(arr[:, 1] - c[1], arr[:, 0] - c[0])
    order = np.argsort(angles)
    return [tuple(arr[i]) for i in order]


def _clip_polygon(subject: list, clip: np.ndarray) -> list:
    """Sutherland–Hodgman clip of ``subject`` (CCW list) against convex
    ``clip`` (CCW array). Returns the clipped polygon's vertex list."""
    output = [np.asarray(p, dtype=float) for p in subject]
    k = len(clip)
    for i in range(k):
        if not output:
            return []
        a = clip[i]
        b = clip[(i + 1) % k]
        edge = b - a
        normal = np.array([-edge[1], edge[0]])

        def inside(pt: np.ndarray) -> bool:
            return np.dot(pt - a, normal) >= 0

        next_output: list = []
        n = len(output)
        for j in range(n):
            cur = output[j]
            prv = output[(j - 1) % n]
            if inside(cur):
                if not inside(prv):
                    next_output.append(_intersect(prv, cur, a, b))
                next_output.append(cur)
            elif inside(prv):
                next_output.append(_intersect(prv, cur, a, b))
        output = next_output
    return [tuple(p) for p in output]


def _intersect(p1: np.ndarray, p2: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection of segment p1→p2 with infinite line through a, b."""
    r = p2 - p1
    s = b - a
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-12:
        return p1.copy()
    t = ((a[0] - p1[0]) * s[1] - (a[1] - p1[1]) * s[0]) / denom
    return p1 + t * r


def _render_subgraph_map(
    G: nx.Graph,
    pos: dict,
    edges: frozenset,
    show_labels: bool,
    label_style: LabelStyle,
    pdf_path: str | None,
) -> Image.Image:
    """Faithful map for a subgraph of a triangulation (special / balanced mode).

    Each region is a Voronoi cell, but its boundary is **eroded inward** along
    every shared ridge whose two nodes are *not* adjacent in ``G``. So two
    regions share a (black) land border iff their nodes are adjacent; every
    non-adjacent pair is pulled apart by an open-water gap — no marker lines,
    just sea between them. When ``edges`` covers every ridge (χ = 4, full
    triangulation) nothing is eroded and the result matches the classic map.
    """
    nodes = list(G.nodes())
    points = np.array([pos[v] for v in nodes], dtype=float)
    vor = Voronoi(points)
    clip_poly = _clip_hull(points, vor.vertices)
    diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    gap = 0.02 * diag  # open-water width between two non-adjacent regions

    # Sites that share a Voronoi ridge — the binding half-plane constraints of
    # each cell. Non-ridge bisectors are redundant, so this is exact.
    ridge_nbrs: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    for p1, p2 in vor.ridge_points:
        ridge_nbrs[int(p1)].append(int(p2))
        ridge_nbrs[int(p2)].append(int(p1))

    polygons: dict = {}
    for i, v in enumerate(nodes):
        cell = [np.asarray(p, dtype=float) for p in clip_poly]
        a = points[i]
        for j in ridge_nbrs[i]:
            b = points[j]
            normal = a - b  # half-plane that keeps site i's side of the bisector
            mid = (a + b) / 2.0
            key = (min(v, nodes[j]), max(v, nodes[j]))
            if key not in edges:
                # Recede this cell from the shared bisector by gap/2; the two
                # non-adjacent cells together leave a gap of ``gap`` between them.
                mid = mid + (gap / 2.0) * normal / float(np.linalg.norm(normal))
            cell = _clip_halfplane(cell, mid, normal)
            if len(cell) < 3:
                break
        if len(cell) >= 3:
            polygons[v] = [tuple(p) for p in cell]

    all_pts = np.concatenate([np.asarray(poly) for poly in polygons.values()])
    xlo, ylo = all_pts.min(axis=0)
    xhi, yhi = all_pts.max(axis=0)
    pad = 0.04 * max(xhi - xlo, yhi - ylo)

    fig, ax = plt.subplots(figsize=(6, 6), facecolor=SEA_BG)
    ax.set_facecolor(SEA_BG)

    for v, poly in polygons.items():
        ax.add_patch(MplPolygon(
            poly, closed=True,
            facecolor="white", edgecolor="black",
            linewidth=2.0, joinstyle="miter",
        ))
        if show_labels:
            cx, cy = pos[v]
            ax.text(
                cx, cy, node_label(v, label_style),
                ha="center", va="center",
                fontsize=10, fontweight="bold", color="#1a1a1a",
                bbox=dict(
                    boxstyle="round,pad=0.2", fc="white",
                    ec="#999", alpha=0.88, lw=0.7,
                ),
            )

    ax.set_xlim(xlo - pad, xhi + pad)
    ax.set_ylim(ylo - pad, yhi + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=0.3)

    if pdf_path is not None:
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=SEA_BG)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=SEA_BG)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def _clip_halfplane(poly: list, p0: np.ndarray, normal: np.ndarray) -> list:
    """Sutherland–Hodgman clip of CCW ``poly`` to the half-plane
    ``{x : (x - p0)·normal ≥ 0}``."""
    if not poly:
        return poly
    out: list = []
    n = len(poly)
    for k in range(n):
        cur = poly[k]
        prv = poly[(k - 1) % n]
        cur_in = float(np.dot(cur - p0, normal)) >= 0
        prv_in = float(np.dot(prv - p0, normal)) >= 0
        if cur_in:
            if not prv_in:
                out.append(_line_intersect(prv, cur, p0, normal))
            out.append(cur)
        elif prv_in:
            out.append(_line_intersect(prv, cur, p0, normal))
    return out


def _line_intersect(
    s: np.ndarray, e: np.ndarray, p0: np.ndarray, normal: np.ndarray
) -> np.ndarray:
    """Intersection of segment ``s→e`` with the line ``(x - p0)·normal = 0``."""
    d = e - s
    den = float(np.dot(d, normal))
    if abs(den) < 1e-12:
        return s.copy()
    t = float(np.dot(p0 - s, normal)) / den
    return s + t * d
