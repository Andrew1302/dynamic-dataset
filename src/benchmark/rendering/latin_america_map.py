"""Latin America road-trip disguise for the shortest-path task.

The disguise turns a weighted DAG into a real geographic map of Latin
America. Vertices map to cities (sorted N→S so node 0 is northernmost,
matching the topological order i<j → "edges flow southward"), edge
weights map to driving hours, and the shortest path question becomes
"minimum total driving time, start city → end city". Because graph
edges only run from low-index to high-index vertices, every drawn arrow
tends to point southward — giving the rendering a coherent visual flow.

Cartopy's PlateCarree projection draws coastlines and borders; an
``adjustText`` pass shoves city-name and weight labels until none of
them overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from adjustText import adjust_text
from matplotlib.patches import FancyArrowPatch
from PIL import Image

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from .config import LabelStyle, node_label


# (name, lat, lon) — the pool is sized so that even the largest sweep
# (target_edges=35 in graph_bench_sweep_edges → up to n=39 in the
# rejection-sampler in tools/prepare_dynamic_graph_benchmark.py) still
# leaves at least one unused entry for farthest-first to choose from.
_CITIES: tuple[tuple[str, float, float], ...] = (
    ("Tijuana",                32.50, -117.00),
    ("Monterrey",              25.69, -100.32),
    ("Guadalajara",            20.66, -103.35),
    ("Mexico City",            19.43,  -99.13),
    ("Oaxaca",                 17.07,  -96.72),
    ("Mérida",                 20.97,  -89.62),
    ("Cancún",                 21.16,  -86.85),
    ("Havana",                 23.13,  -82.39),
    ("Kingston",               17.97,  -76.79),
    ("San Juan",               18.45,  -66.07),
    ("Guatemala City",         14.63,  -90.51),
    ("Tegucigalpa",            14.07,  -87.19),
    ("Managua",                12.13,  -86.25),
    ("San José",                9.93,  -84.08),
    ("Panama City",             8.97,  -79.53),
    ("Santo Domingo",          18.47,  -69.90),
    ("Maracaibo",              10.65,  -71.65),
    ("Caracas",                10.49,  -66.90),
    ("Medellín",                6.24,  -75.58),
    ("Bogotá",                  4.71,  -74.07),
    ("Cali",                    3.45,  -76.53),
    ("Quito",                  -0.18,  -78.47),
    ("Guayaquil",              -2.17,  -79.92),
    ("Belém",                  -1.46,  -48.50),
    ("Manaus",                 -3.10,  -60.03),
    ("Fortaleza",              -3.73,  -38.52),
    ("Recife",                 -8.05,  -34.88),
    ("Lima",                  -12.05,  -77.04),
    ("Cusco",                 -13.53,  -71.97),
    ("Salvador",              -12.97,  -38.51),
    ("Brasília",              -15.79,  -47.88),
    ("La Paz",                -16.50,  -68.15),
    ("Santa Cruz de la Sierra",-17.78, -63.18),
    ("Belo Horizonte",        -19.92,  -43.94),
    ("Rio de Janeiro",        -22.91,  -43.17),
    ("São Paulo",             -23.55,  -46.63),
    ("Curitiba",              -25.43,  -49.27),
    ("Asunción",              -25.30,  -57.63),
    ("Porto Alegre",          -30.03,  -51.22),
    ("Córdoba",               -31.42,  -64.18),
    ("Mendoza",               -32.89,  -68.83),
    ("Valparaíso",            -33.05,  -71.62),
    ("Santiago",              -33.45,  -70.66),
    ("Montevideo",            -34.90,  -56.19),
    ("Buenos Aires",          -34.61,  -58.38),
    ("Concepción",            -36.83,  -73.05),
    ("Bariloche",             -41.13,  -71.31),
    ("Punta Arenas",          -53.16,  -70.92),
)

_EXTENT = (-122.0, -30.0, -58.0, 36.0)
_FIGSIZE = (15, 15)
_DPI = 170

_LAND = "#F4EFE3"
_OCEAN = "#DCE9F2"
_COAST = "#7a8693"
_BORDER = "#a0aab8"
_SOURCE_COLOR = "#1D9E75"
_SINK_COLOR = "#C2185B"
_INTERMEDIATE_COLOR = "#374B7A"
_EDGE_COLOR = "#C24A0E"
_WEIGHT_BBOX_FACE = "#FFF7E5"
_WEIGHT_TEXT = "#7a3010"
_LABEL_TEXT = "#1f1a4e"


@dataclass(frozen=True)
class LatinAmericaMap:
    """Frozen disguise instance: cities, edges, endpoints, render()."""

    edges: tuple[tuple[int, int, int], ...]  # (u, v, weight)
    cities: tuple[tuple[str, float, float], ...]  # (name, lat, lon), index = node id
    source: int
    sink: int
    label_style: LabelStyle = "numeric"
    show_intermediate_nodes: bool = True

    def render(self) -> Image.Image:
        return _render_image(
            self.edges,
            self.cities,
            self.source,
            self.sink,
            label_style=self.label_style,
            show_intermediate_nodes=self.show_intermediate_nodes,
        )


def build_latin_america_map(
    G: nx.DiGraph,
    seed: int,
    label_style: LabelStyle = "numeric",
    show_intermediate_nodes: bool = True,
) -> LatinAmericaMap:
    """Sample cities for *G*'s vertices and pack a renderable disguise.

    Cities are picked from the city pool by farthest-first sampling
    (high pairwise spread), then sorted by latitude descending so
    node 0 = northernmost and node n-1 = southernmost. The DAG's
    source/sink end up at the extremes of the map, matching the
    visual N→S flow.
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    if n > len(_CITIES):
        raise ValueError(
            f"shortest_path map disguise needs n ≤ {len(_CITIES)}, got {n}"
        )

    chosen = _farthest_first(_CITIES, n, rng)
    chosen.sort(key=lambda r: -r[1])  # latitude DESC

    edges = tuple(
        (int(u), int(v), int(d["weight"]))
        for u, v, d in G.edges(data=True)
    )
    return LatinAmericaMap(
        edges=edges,
        cities=tuple(chosen),
        source=int(G.graph["source"]),
        sink=int(G.graph["sink"]),
        label_style=label_style,
        show_intermediate_nodes=show_intermediate_nodes,
    )


def _farthest_first(
    pool: tuple[tuple[str, float, float], ...],
    n: int,
    rng: np.random.Generator,
) -> list[tuple[str, float, float]]:
    """Greedy farthest-first sampling on (lat, lon) Euclidean distance.

    Start from a random pool entry, then iteratively add the entry whose
    minimum distance to the already-chosen set is largest. This keeps
    cities spread across the continent even at large n, where uniform
    sampling tends to clump.
    """
    start_idx = int(rng.integers(0, len(pool)))
    chosen_idx = [start_idx]
    while len(chosen_idx) < n:
        best_idx, best_d = -1, -1.0
        for k in range(len(pool)):
            if k in chosen_idx:
                continue
            d_min = min(
                ((pool[k][1] - pool[c][1]) ** 2 + (pool[k][2] - pool[c][2]) ** 2) ** 0.5
                for c in chosen_idx
            )
            if d_min > best_d:
                best_d = d_min
                best_idx = k
        chosen_idx.append(best_idx)
    return [pool[i] for i in chosen_idx]


def _render_image(
    edges: tuple[tuple[int, int, int], ...],
    cities: tuple[tuple[str, float, float], ...],
    source: int,
    sink: int,
    label_style: LabelStyle = "numeric",
    show_intermediate_nodes: bool = True,
) -> Image.Image:
    n = len(cities)
    coords = {i: (cities[i][1], cities[i][2]) for i in range(n)}  # (lat, lon)
    names = {i: cities[i][0] for i in range(n)}

    fig = plt.figure(figsize=_FIGSIZE)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(list(_EXTENT), crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor=_LAND)
    ax.add_feature(cfeature.OCEAN, facecolor=_OCEAN)
    ax.add_feature(cfeature.COASTLINE, edgecolor=_COAST, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, edgecolor=_BORDER, linewidth=0.7)
    ax.add_feature(
        cfeature.LAKES, facecolor=_OCEAN, edgecolor=_COAST, linewidth=0.3,
    )

    city_texts = []
    marker_xs: list[float] = []
    marker_ys: list[float] = []

    for (u, v, w) in edges:
        lat_u, lon_u = coords[u]
        lat_v, lon_v = coords[v]
        rad = 0.16 if (u + v) % 2 else -0.16
        arr = FancyArrowPatch(
            (lon_u, lat_u), (lon_v, lat_v),
            arrowstyle="-|>", mutation_scale=18,
            connectionstyle=f"arc3,rad={rad}",
            color=_EDGE_COLOR, lw=1.8, alpha=0.85,
            shrinkA=14, shrinkB=14,
            transform=ccrs.PlateCarree()._as_mpl_transform(ax),
            zorder=4,
        )
        ax.add_patch(arr)
        # Quadratic Bezier midpoint for matplotlib's "arc3,rad=R":
        # control point = chord midpoint + R·(dy, -dx), so the curve
        # midpoint = chord midpoint + 0.5·R·(dy, -dx). Anchors the
        # weight pill on the actual arc instead of the straight
        # midpoint, which can sit far off the arc for long edges
        # (e.g. Tijuana → Punta Arenas crossing the Pacific).
        dx, dy = lon_v - lon_u, lat_v - lat_u
        mid_lon = 0.5 * (lon_u + lon_v) + 0.5 * rad * dy
        mid_lat = 0.5 * (lat_u + lat_v) - 0.5 * rad * dx
        ax.text(
            mid_lon, mid_lat, f"{w}h",
            ha="center", va="center", fontsize=11,
            color=_WEIGHT_TEXT, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", fc=_WEIGHT_BBOX_FACE,
                      ec=_EDGE_COLOR, alpha=0.97, lw=0.8),
            transform=ccrs.PlateCarree(), zorder=5,
        )

    for i in range(n):
        lat, lon = coords[i]
        is_endpoint = i == source or i == sink
        if i == source:
            color, ms, extra = _SOURCE_COLOR, 380, "  (start)"
        elif i == sink:
            color, ms, extra = _SINK_COLOR, 380, "  (end)"
        else:
            color, ms, extra = _INTERMEDIATE_COLOR, 240, ""

        # When label_style="none" we strip the integer ID label off the
        # markers (they're the node-id equivalent in the direct view).
        # When ``show_intermediate_nodes`` is False we additionally hide
        # the dot+name for non-endpoint cities; endpoints are always
        # kept because the prompt references "start city" / "end city".
        draw_marker = is_endpoint or show_intermediate_nodes
        if draw_marker:
            ax.scatter(
                lon, lat, s=ms, color=color,
                edgecolors="white", linewidths=2.2,
                transform=ccrs.PlateCarree(), zorder=6,
            )
            if label_style != "none":
                ax.text(
                    lon, lat, node_label(i, label_style),
                    ha="center", va="center", color="white",
                    fontweight="bold", fontsize=11,
                    transform=ccrs.PlateCarree(), zorder=7,
                )
            marker_xs.append(lon)
            marker_ys.append(lat)
            t = ax.text(
                lon, lat, f"{names[i]}{extra}",
                ha="center", va="center", fontsize=11,
                color=_LABEL_TEXT, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=color, alpha=0.96, lw=0.9),
                transform=ccrs.PlateCarree(), zorder=8,
            )
            city_texts.append(t)

    # Only city labels are nudged — weight pills stay anchored on the
    # arc midpoint where they were placed, so they can never drift away
    # from their arrow.
    adjust_text(
        city_texts,
        x=marker_xs, y=marker_ys,
        ax=ax,
        expand=(1.4, 1.6),
        force_text=(0.7, 0.9),
        force_static=(0.5, 0.7),
        arrowprops=dict(arrowstyle="-", color="#666666", lw=0.6, alpha=0.6),
    )

    ax.set_title(
        f"Latin America road trip (n={n}): minimum total driving hours, "
        f"{names[source]} -> {names[sink]}",
        fontsize=12, pad=8,
    )
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def render_latin_america_map(
    G: nx.DiGraph,
    seed: int,
    label_style: LabelStyle = "numeric",
    show_intermediate_nodes: bool = True,
) -> Image.Image:
    return build_latin_america_map(
        G,
        seed=seed,
        label_style=label_style,
        show_intermediate_nodes=show_intermediate_nodes,
    ).render()
