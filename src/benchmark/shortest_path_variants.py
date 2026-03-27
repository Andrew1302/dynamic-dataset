"""
shortest_path_variants.py — Family 1: Shortest Path, three domain disguises.

Variant 1A — BareGraphVariant   : trivial projection (G presented as-is)
Variant 1B — MazeVariant        : G embedded in a 2-D character grid
Variant 1C — WordLadderVariant  : G embedded as a Hamming-distance word graph

All three share the same underlying task: find the minimum-cost path between
a designated source node and target node in G.
"""

from __future__ import annotations

import collections
import itertools
import math
import random
from typing import Any

import networkx as nx

from .base import ProblemVariant, ProjectionFailure

# ---------------------------------------------------------------------------
# Variant 1A — Bare Graph
# ---------------------------------------------------------------------------


class BareGraphVariant(ProblemVariant):
    """Trivial projection: G is presented as-is with an explicit node/edge list.

    # domain element → graph element mapping
    # node id        = node in G
    # [u, v, w]      = edge (u, v) with weight w (1 if unweighted)
    """

    def project(
        self,
        G: nx.Graph,
        source: int,
        target: int,
        seed: int | None = None,
    ) -> dict[str, Any]:
        weighted = any("weight" in d for _, _, d in G.edges(data=True))
        edges = [
            [int(u), int(v), int(d.get("weight", 1))]
            for u, v, d in G.edges(data=True)
        ]
        return {
            "nodes": [int(n) for n in sorted(G.nodes())],
            "edges": edges,
            "source": int(source),
            "target": int(target),
            "weighted": weighted,
        }

    def solve(self, instance: dict[str, Any]) -> dict[str, Any]:
        G = self._rebuild(instance)
        src, tgt = instance["source"], instance["target"]
        if instance["weighted"]:
            path = nx.dijkstra_path(G, src, tgt, weight="weight")
            cost = nx.dijkstra_path_length(G, src, tgt, weight="weight")
        else:
            path = nx.shortest_path(G, src, tgt)
            cost = len(path) - 1
        return {
            "path": [int(n) for n in path],
            "cost": int(cost),
            "base_path": [int(n) for n in path],
        }

    def to_base_graph(self, instance: dict[str, Any]) -> nx.Graph:
        return self._rebuild(instance)

    def verify(self, instance: dict[str, Any], solution: dict[str, Any]) -> bool:
        G = self._rebuild(instance)
        path = solution["path"]
        if path[0] != instance["source"] or path[-1] != instance["target"]:
            return False
        total = 0
        for u, v in zip(path, path[1:]):
            if not G.has_edge(u, v):
                return False
            total += G[u][v].get("weight", 1)
        return total == solution["cost"]

    def to_prompt(self, instance: dict[str, Any]) -> str:
        nodes = instance["nodes"]
        edges = instance["edges"]
        src, tgt = instance["source"], instance["target"]
        if instance["weighted"]:
            edge_str = ", ".join(f"({u},{v},w={w})" for u, v, w in edges)
            return (
                f"Q: You are given a weighted undirected graph.\n"
                f"Nodes: {nodes}\n"
                f"Edges (u, v, weight): {edge_str}\n"
                f"Find the shortest (minimum-cost) path from node {src} to node {tgt}.\n"
                f"Give the path as a list of node IDs and the total cost.\nA:"
            )
        else:
            edge_str = ", ".join(f"({u},{v})" for u, v, _ in edges)
            return (
                f"Q: You are given an undirected graph.\n"
                f"Nodes: {nodes}\n"
                f"Edges: {edge_str}\n"
                f"Find the shortest path from node {src} to node {tgt}.\n"
                f"Give the path as a list of node IDs and the number of hops.\nA:"
            )

    # ------------------------------------------------------------------
    def _rebuild(self, instance: dict[str, Any]) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(instance["nodes"])
        for u, v, w in instance["edges"]:
            G.add_edge(u, v, weight=w)
        return G


# ---------------------------------------------------------------------------
# Variant 1B — Maze
# ---------------------------------------------------------------------------

_WALL = "#"
_OPEN = "."
_CHAMBER_SIZE = 3   # each chamber is a CHAMBER_SIZE × CHAMBER_SIZE block
_COL_SPACING = _CHAMBER_SIZE + 1  # horizontal gap between adjacent chamber origins


class MazeVariant(ProblemVariant):
    """Projects G into a 2-D character grid maze.

    # domain element → graph element mapping
    # chamber (3×3 open block)                = one node in G
    # corridor (connected chain of '.' cells) = one edge in G
    # BFS path length through maze            = hop count in G (unweighted)
    #                                           (weighted: BFS steps != Dijkstra cost;
    #                                            weights stored in graph_edges for to_base_graph)

    Grid layout:
        - All N chambers are placed in a SINGLE ROW at the top of the grid.
        - Below the chambers is a ROUTING AREA — one horizontal routing row per edge.
        - Each corridor follows an inverted-U route:
            exit_u → stem down to routing_row_e → horizontal → stem up → exit_v
        - Because all routing rows are distinct and below all chambers, corridors
          never pass through any other chamber.

    The instance dict stores "graph_edges" so that to_base_graph() can reconstruct
    G with correct edge weights without relying on BFS-measured corridor lengths.
    This ensures are_isomorphic_instances() always gives the correct answer even for
    weighted graphs where BFS step count != Dijkstra weighted distance.
    """

    def project(
        self,
        G: nx.Graph,
        source: int,
        target: int,
        seed: int | None = None,
    ) -> dict[str, Any]:
        weighted = any("weight" in d for _, _, d in G.edges(data=True))

        # Assign each node a column index (sorted order → reproducible layout)
        nodes_sorted = sorted(G.nodes())
        node_to_col_idx: dict[int, int] = {n: i for i, n in enumerate(nodes_sorted)}

        # Column origin of chamber i (top-left corner)
        def chamber_origin(node: int) -> tuple[int, int]:
            # Row 1 (leave row 0 as border wall), col = idx * _COL_SPACING + 1
            return (1, node_to_col_idx[node] * _COL_SPACING + 1)

        # Exit point = bottom-centre cell of chamber (lowest row, middle column)
        def exit_pt(node: int) -> tuple[int, int]:
            or_, oc = chamber_origin(node)
            return (or_ + _CHAMBER_SIZE - 1, oc + _CHAMBER_SIZE // 2)

        # Grid dimensions
        n_chambers = len(nodes_sorted)
        grid_cols = n_chambers * _COL_SPACING + 2  # +2 border
        # Routing area starts at row CHAMBER_SIZE+1; one row per edge, +1 wall between
        edge_list = list(G.edges(data=True))
        n_edges = len(edge_list)
        routing_start = _CHAMBER_SIZE + 1 + 1  # +1 buffer row after chambers
        grid_rows = routing_start + n_edges + 2  # +2 bottom border

        grid: list[list[str]] = [[_WALL] * grid_cols for _ in range(grid_rows)]

        # Carve chambers
        for node in nodes_sorted:
            or_, oc = chamber_origin(node)
            for dr in range(_CHAMBER_SIZE):
                for dc in range(_CHAMBER_SIZE):
                    grid[or_ + dr][oc + dc] = _OPEN

        # Carve corridors — one per edge, each in its own routing row
        for e_idx, (u, v, d) in enumerate(edge_list):
            routing_row = routing_start + e_idx  # unique row for this edge
            eu = exit_pt(u)  # (exit_row, col_u)
            ev = exit_pt(v)  # (exit_row, col_v)
            exit_row = eu[0]   # same for all (all chambers in row 0)
            col_u = eu[1]
            col_v = ev[1]

            # Stem down from exit_u to routing_row (exclusive exit_row since it's
            # already carved as part of the chamber)
            for r in range(exit_row + 1, routing_row + 1):
                grid[r][col_u] = _OPEN

            # Horizontal segment at routing_row
            c_lo, c_hi = min(col_u, col_v), max(col_u, col_v)
            for c in range(c_lo, c_hi + 1):
                grid[routing_row][c] = _OPEN

            # Stem up from routing_row to exit_v (exclusive exit_row)
            for r in range(exit_row + 1, routing_row + 1):
                grid[r][col_v] = _OPEN

        start = list(exit_pt(source))
        end = list(exit_pt(target))

        chamber_map = {str(node): list(chamber_origin(node)) for node in nodes_sorted}
        graph_edges = [
            [int(u), int(v), int(d.get("weight", 1))]
            for u, v, d in G.edges(data=True)
        ]

        return {
            "grid": grid,
            "start": start,
            "end": end,
            "chamber_map": chamber_map,
            "chamber_size": _CHAMBER_SIZE,
            "weighted": weighted,
            # Stored graph structure for to_base_graph (pure structural reconstruction)
            "graph_nodes": [int(n) for n in nodes_sorted],
            "graph_edges": graph_edges,
        }

    def solve(self, instance: dict[str, Any]) -> dict[str, Any]:
        grid = instance["grid"]
        start = tuple(instance["start"])
        end = tuple(instance["end"])
        chamber_map = instance["chamber_map"]
        chamber_size = instance.get("chamber_size", _CHAMBER_SIZE)

        path = _bfs_grid(grid, start, end)
        if path is None:
            raise ProjectionFailure("MazeVariant: no path found in maze")

        base_path = _grid_path_to_base(path, chamber_map, chamber_size)

        return {
            "path": [list(c) for c in path],
            "cost": len(path) - 1,
            "base_path": base_path,
        }

    def to_base_graph(self, instance: dict[str, Any]) -> nx.Graph:
        """Reconstruct G from stored graph structure (nodes + edges with weights).

        The maze grid encodes connectivity visually for the LLM; to_base_graph reads
        the stored graph_edges so that isomorphism checks are always exact.
        """
        G = nx.Graph()
        G.add_nodes_from(instance["graph_nodes"])
        for u, v, w in instance["graph_edges"]:
            G.add_edge(u, v, weight=w)
        return G

    def verify(self, instance: dict[str, Any], solution: dict[str, Any]) -> bool:
        grid = instance["grid"]
        path = [tuple(c) for c in solution["path"]]
        if not path:
            return False
        if list(path[0]) != instance["start"] or list(path[-1]) != instance["end"]:
            return False
        for r, c in path:
            if grid[r][c] != _OPEN:
                return False
        for (r1, c1), (r2, c2) in zip(path, path[1:]):
            if abs(r1 - r2) + abs(c1 - c2) != 1:
                return False
        return len(path) - 1 == solution["cost"]

    def to_prompt(self, instance: dict[str, Any]) -> str:
        grid = instance["grid"]
        start = instance["start"]
        end = instance["end"]
        grid_str = "\n".join("".join(row) for row in grid)
        return (
            f"Q: You are given a maze represented as a grid of characters.\n"
            f"'#' is a wall, '.' is open space.\n"
            f"Starting position: row={start[0]}, col={start[1]}.\n"
            f"Goal position: row={end[0]}, col={end[1]}.\n"
            f"Find the shortest path (fewest steps) from start to goal.\n"
            f"Report the path as a list of (row, col) coordinates and the total steps.\n\n"
            f"{grid_str}\nA:"
        )


def _bfs_grid(
    grid: list[list[str]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """BFS shortest path on a 2-D grid.  Returns list of (row, col) or None."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if grid[start[0]][start[1]] != _OPEN or grid[end[0]][end[1]] != _OPEN:
        return None
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue: collections.deque[tuple[int, int]] = collections.deque([start])
    while queue:
        cur = queue.popleft()
        if cur == end:
            path = []
            node: tuple[int, int] | None = cur
            while node is not None:
                path.append(node)
                node = prev[node]
            return list(reversed(path))
        r, c = cur
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            nb = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == _OPEN and nb not in prev:
                prev[nb] = cur
                queue.append(nb)
    return None


def _bfs_distances(
    grid: list[list[str]],
    start: tuple[int, int],
) -> dict[tuple[int, int], int]:
    """BFS from start; returns dict of reachable cell → distance."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    dist: dict[tuple[int, int], int] = {start: 0}
    queue: collections.deque[tuple[int, int]] = collections.deque([start])
    while queue:
        cur = queue.popleft()
        r, c = cur
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            nb = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == _OPEN and nb not in dist:
                dist[nb] = dist[cur] + 1
                queue.append(nb)
    return dist


def _grid_path_to_base(
    path: list[tuple[int, int]],
    chamber_map: dict[str, list[int]],
    chamber_size: int,
) -> list[int]:
    """Extract the sequence of node IDs visited along a grid path."""
    result: list[int] = []
    for r, c in path:
        for node_str, (or_, oc) in chamber_map.items():
            if or_ <= r < or_ + chamber_size and oc <= c < oc + chamber_size:
                nid = int(node_str)
                if not result or result[-1] != nid:
                    result.append(nid)
                break
    return result


# ---------------------------------------------------------------------------
# Variant 1C — Word Ladder
# ---------------------------------------------------------------------------

_ALPHABET = ["a", "b", "c", "d"]
_MAX_RETRIES = 200


class WordLadderVariant(ProblemVariant):
    """Projects G into a word-ladder graph where Hamming distance encodes adjacency.

    # domain element → graph element mapping
    # word (string over {a,b,c,d})     = one node in G
    # Hamming distance == 1            = edge in G
    # Hamming distance >= 2            = non-edge in G
    # BFS path length on word graph    = hop count in G (unweighted)

    Constraint satisfaction problem:
        - For each edge (u,v) in G:     hamming(word[u], word[v]) == 1
        - For each non-edge (u,v):      hamming(word[u], word[v]) >= 2

    Solved via backtracking with MRV heuristic and forward checking.
    """

    def project(
        self,
        G: nx.Graph,
        source: int,
        target: int,
        seed: int | None = None,
    ) -> dict[str, Any]:
        n = G.number_of_nodes()
        # Word length: ceil(log4(n)) + 2 gives enough Hamming-1 neighbours.
        # For denser graphs increase L by 1 to widen the usable word space.
        L = max(3, math.ceil(math.log(max(n, 2), 4)) + 2)
        edge_density = G.number_of_edges() / max(1, n * (n - 1) // 2)
        if edge_density > 0.25 and L < 6:
            L += 1

        rng = random.Random(seed)

        for attempt in range(_MAX_RETRIES):
            all_words = ["".join(c) for c in itertools.product(_ALPHABET, repeat=L)]
            rng.shuffle(all_words)
            domain_size = max(4 * n, 80, min(len(all_words), 6 * n))
            candidates = all_words[:domain_size]
            domains: dict[int, list[str]] = {node: list(candidates) for node in G.nodes()}
            assignment = _wl_backtrack({}, list(G.nodes()), domains, G, rng)
            if assignment is not None:
                words = [assignment[node] for node in sorted(G.nodes())]
                node_to_word = {str(node): assignment[node] for node in G.nodes()}
                return {
                    "words": words,
                    "source_word": assignment[source],
                    "target_word": assignment[target],
                    "vocabulary": sorted(set(assignment.values())),
                    "node_to_word": node_to_word,
                    "word_length": L,
                }
            # Different shuffle for next attempt (advance rng state)
            rng.random()

        raise ProjectionFailure(
            f"WordLadderVariant: could not assign words after {_MAX_RETRIES} attempts "
            f"(n={n}, L={L})"
        )

    def solve(self, instance: dict[str, Any]) -> dict[str, Any]:
        vocab = instance["vocabulary"]
        src = instance["source_word"]
        tgt = instance["target_word"]

        # Build word graph: edge iff Hamming distance == 1
        G_w = nx.Graph()
        G_w.add_nodes_from(vocab)
        for i, w1 in enumerate(vocab):
            for w2 in vocab[i + 1:]:
                if _hamming(w1, w2) == 1:
                    G_w.add_edge(w1, w2)

        path = nx.shortest_path(G_w, src, tgt)
        return {
            "path": list(path),
            "cost": len(path) - 1,
            "base_path": _wl_path_to_base(path, instance["node_to_word"]),
        }

    def to_base_graph(self, instance: dict[str, Any]) -> nx.Graph:
        """Reconstruct G from vocabulary: add edge iff Hamming distance == 1."""
        vocab = instance["vocabulary"]
        G = nx.Graph()
        for i, w1 in enumerate(vocab):
            G.add_node(i)
            for j, w2 in enumerate(vocab[i + 1:], start=i + 1):
                if _hamming(w1, w2) == 1:
                    G.add_edge(i, j)
        return G

    def verify(self, instance: dict[str, Any], solution: dict[str, Any]) -> bool:
        vocab_set = set(instance["vocabulary"])
        path = solution["path"]
        if not path:
            return False
        if path[0] != instance["source_word"] or path[-1] != instance["target_word"]:
            return False
        for w in path:
            if w not in vocab_set:
                return False
        for w1, w2 in zip(path, path[1:]):
            if _hamming(w1, w2) != 1:
                return False
        return len(path) - 1 == solution["cost"]

    def to_prompt(self, instance: dict[str, Any]) -> str:
        src = instance["source_word"]
        tgt = instance["target_word"]
        vocab = instance["vocabulary"]
        return (
            f"Q: You are given a word ladder problem.\n"
            f"Starting word: '{src}'\n"
            f"Target word: '{tgt}'\n"
            f"Vocabulary (allowed words): {vocab}\n"
            f"Transform the starting word into the target word one letter at a time.\n"
            f"Each intermediate word must belong to the vocabulary, and each step must "
            f"change exactly one letter.\n"
            f"Give the transformation sequence and the number of steps.\nA:"
        )


# ------------------------------------------------------------------
# Word ladder helper functions
# ------------------------------------------------------------------


def _hamming(a: str, b: str) -> int:
    """Hamming distance between two equal-length strings."""
    return sum(x != y for x, y in zip(a, b))


def _wl_backtrack(
    assignment: dict[int, str],
    unassigned: list[int],
    domains: dict[int, list[str]],
    G: nx.Graph,
    rng: random.Random,
) -> dict[int, str] | None:
    """Backtracking search with MRV heuristic and forward checking."""
    if not unassigned:
        return assignment

    # MRV: pick the unassigned variable with the smallest domain
    node = min(unassigned, key=lambda n: len(domains[n]))
    remaining = [n for n in unassigned if n != node]

    for word in list(domains[node]):
        # Consistency check against already-assigned nodes
        ok = True
        for other, other_word in assignment.items():
            h = _hamming(word, other_word)
            if G.has_edge(node, other):
                if h != 1:
                    ok = False
                    break
            else:
                if h < 2:
                    ok = False
                    break
        if not ok:
            continue

        # Forward checking: prune domains of unassigned nodes
        new_domains: dict[int, list[str]] = {n: list(d) for n, d in domains.items()}
        new_domains[node] = [word]
        pruned_ok = True
        for nb in remaining:
            new_d = []
            for w in new_domains[nb]:
                h = _hamming(w, word)
                if G.has_edge(nb, node):
                    if h != 1:
                        continue
                else:
                    if h < 2:
                        continue
                new_d.append(w)
            if not new_d:
                pruned_ok = False
                break
            new_domains[nb] = new_d
        if not pruned_ok:
            continue

        assignment[node] = word
        result = _wl_backtrack(assignment, remaining, new_domains, G, rng)
        if result is not None:
            return result
        del assignment[node]

    return None


def _wl_path_to_base(
    word_path: list[str],
    node_to_word: dict[str, str],
) -> list[int]:
    """Convert a word path back to node IDs using the node_to_word mapping."""
    word_to_node = {w: int(n) for n, w in node_to_word.items()}
    return [word_to_node[w] for w in word_path if w in word_to_node]
