"""
state_search_variants.py — Family 2: State Space Search, three domain disguises.

Variant 2A — BareStateGraphVariant : explicit state transition graph
Variant 2B — SlidingPuzzleVariant  : N×N sliding tile puzzle
Variant 2C — TowerOfHanoiVariant   : Tower of Hanoi

All three share the same underlying task: find a path from an initial state s0
to a goal state s* in a state transition graph.

Unlike Family 1, the state space graphs for Puzzle and Hanoi have FIXED structure.
Isomorphic instances are created by finding (s0, s*) pairs whose optimal solution
paths have the same length as the bare-graph path between source and target.
"""

from __future__ import annotations

import collections
import heapq
import itertools
import random
from typing import Any

import networkx as nx
from PIL import Image

from .base import ProblemVariant, ProjectionFailure
from .visualization import draw_graph, draw_hanoi, draw_sliding_puzzle

# ---------------------------------------------------------------------------
# Variant 2A — Bare State Graph
# ---------------------------------------------------------------------------


class BareStateGraphVariant(ProblemVariant):
    """Trivial projection: the state transition graph is presented as-is.

    # domain element → graph element mapping
    # state id (int)     = node in G
    # transition (u, v)  = edge in G
    """

    def project(
        self,
        G: nx.Graph,
        source: int,
        target: int,
        seed: int | None = None,
    ) -> dict[str, Any]:
        adjacency = {str(n): [int(nb) for nb in G.neighbors(n)] for n in sorted(G.nodes())}
        return {
            "adjacency": adjacency,
            "s0": int(source),
            "s_goal": int(target),
        }

    def solve(self, instance: dict[str, Any]) -> dict[str, Any]:
        adj = instance["adjacency"]
        s0, sg = instance["s0"], instance["s_goal"]

        # BFS
        prev: dict[int, int | None] = {s0: None}
        queue: collections.deque[int] = collections.deque([s0])
        while queue:
            cur = queue.popleft()
            if cur == sg:
                break
            for nb in adj.get(str(cur), []):
                if nb not in prev:
                    prev[nb] = cur
                    queue.append(nb)

        if sg not in prev:
            raise ProjectionFailure(f"BareStateGraph: no path from {s0} to {sg}")

        path: list[int] = []
        node: int | None = sg
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()

        return {
            "path": path,
            "cost": len(path) - 1,
            "base_path": path,
        }

    def to_base_graph(self, instance: dict[str, Any]) -> nx.Graph:
        """Reconstruct graph from adjacency dict."""
        adj = instance["adjacency"]
        G = nx.Graph()
        for n_str, neighbors in adj.items():
            n = int(n_str)
            G.add_node(n)
            for nb in neighbors:
                G.add_edge(n, nb)
        return G

    def verify(self, instance: dict[str, Any], solution: dict[str, Any]) -> bool:
        adj = instance["adjacency"]
        path = solution["path"]
        if not path:
            return False
        if path[0] != instance["s0"] or path[-1] != instance["s_goal"]:
            return False
        for u, v in zip(path, path[1:]):
            if v not in adj.get(str(u), []):
                return False
        return len(path) - 1 == solution["cost"]

    def to_prompt(self, instance: dict[str, Any]) -> str:
        adj = instance["adjacency"]
        s0, sg = instance["s0"], instance["s_goal"]
        trans_str = "; ".join(
            f"state {n} -> {nbrs}" for n, nbrs in sorted(adj.items(), key=lambda x: int(x[0]))
        )
        return (
            f"Q: You are given a state transition graph.\n"
            f"Transitions: {trans_str}\n"
            f"Find a path from state {s0} to state {sg}.\n"
            f"Give the path as a list of state IDs and the number of steps.\nA:"
        )

    def to_image(self, instance: dict[str, Any]) -> Image.Image:
        """Draw the state transition graph (= bare graph G) with source/target highlighted."""
        G = self.to_base_graph(instance)
        return draw_graph(
            G,
            source=instance["s0"],
            target=instance["s_goal"],
            weighted=False,
            title=f"State Graph  |  s0={instance['s0']}  goal={instance['s_goal']}",
        )


# ---------------------------------------------------------------------------
# Variant 2B — Sliding Puzzle
# ---------------------------------------------------------------------------

# Maximum BFS distances for each puzzle size (approximate diameter)
_MAX_PUZZLE_DIST = {2: 6, 3: 20, 4: 50}


class SlidingPuzzleVariant(ProblemVariant):
    """Projects the bare-graph path length into an N×N sliding tile puzzle.

    # domain element → graph element mapping
    # board state (flat tuple, 0=blank)    = state node in G
    # valid blank-tile slide (U/D/L/R)     = transition edge in G
    # BFS/A* path length between states    = path length in G

    Projection strategy:
        1. Start from a random solvable configuration via random walk from solved.
        2. BFS up to depth (target_length + buffer) to find a state at exactly
           distance target_length from s0.
        3. Use (s0, s_goal) as the puzzle instance.
    """

    def project(
        self,
        G: nx.Graph,
        source: int,
        target: int,
        seed: int | None = None,
    ) -> dict[str, Any]:
        # Determine path length in the base graph
        try:
            bare_path = nx.shortest_path(G, source, target)
        except nx.NetworkXNoPath:
            raise ProjectionFailure("SlidingPuzzle: no path between source and target in G")
        target_len = len(bare_path) - 1

        # Choose puzzle size based on target_len
        n = 3 if target_len > 6 else 2
        if target_len > _MAX_PUZZLE_DIST[n]:
            raise ProjectionFailure(
                f"SlidingPuzzle: target path length {target_len} exceeds max "
                f"for {n}×{n} puzzle ({_MAX_PUZZLE_DIST[n]})"
            )

        rng = random.Random(seed)
        result = _find_puzzle_states(n, target_len, rng)
        if result is None:
            raise ProjectionFailure(
                f"SlidingPuzzle: could not find states at distance {target_len}"
            )
        initial, goal = result

        return {
            "n": n,
            "initial": list(initial),
            "goal": list(goal),
            "optimal_length": target_len,
            # Stored base graph structure for to_base_graph reconstruction
            "graph_nodes": [int(nd) for nd in sorted(G.nodes())],
            "graph_edges": [[int(u), int(v)] for u, v in G.edges()],
        }

    def solve(self, instance: dict[str, Any]) -> dict[str, Any]:
        n = instance["n"]
        initial = tuple(instance["initial"])
        goal = tuple(instance["goal"])
        path, moves = _astar_puzzle(initial, goal, n)
        return {
            "path": [list(s) for s in path],
            "moves": moves,
            "cost": len(moves),
            "base_path": list(range(len(path))),  # state index sequence
        }

    def to_base_graph(self, instance: dict[str, Any]) -> nx.Graph:
        """Reconstruct the base graph G from stored graph structure.

        The puzzle state space is much larger than G; to_base_graph returns G
        so that are_isomorphic_instances() compares base graphs across all variants.
        """
        G = nx.Graph()
        G.add_nodes_from(instance["graph_nodes"])
        for u, v in instance["graph_edges"]:
            G.add_edge(u, v)
        return G

    def verify(self, instance: dict[str, Any], solution: dict[str, Any]) -> bool:
        n = instance["n"]
        goal = tuple(instance["goal"])
        states = [tuple(s) for s in solution["path"]]
        moves = solution["moves"]
        if not states:
            return False
        if states[0] != tuple(instance["initial"]):
            return False
        if states[-1] != goal:
            return False
        if len(moves) != len(states) - 1:
            return False
        valid_dirs = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
        for (cur, nxt, mv) in zip(states, states[1:], moves):
            blank = cur.index(0)
            br, bc = divmod(blank, n)
            if mv not in valid_dirs:
                return False
            dr, dc = valid_dirs[mv]
            nr, nc = br + dr, bc + dc
            if not (0 <= nr < n and 0 <= nc < n):
                return False
            new_blank = nr * n + nc
            rebuilt = list(cur)
            rebuilt[blank], rebuilt[new_blank] = rebuilt[new_blank], rebuilt[blank]
            if tuple(rebuilt) != nxt:
                return False
        return len(moves) == solution["cost"]

    def to_prompt(self, instance: dict[str, Any]) -> str:
        n = instance["n"]
        initial = instance["initial"]
        goal = instance["goal"]

        def fmt_board(flat: list[int]) -> str:
            rows = []
            for i in range(n):
                rows.append(" ".join(str(x) if x != 0 else "_" for x in flat[i * n:(i + 1) * n]))
            return "\n".join(rows)

        return (
            f"Q: You are given a {n}x{n} sliding puzzle (0 represents the blank tile).\n"
            f"Initial board:\n{fmt_board(initial)}\n\n"
            f"Goal board:\n{fmt_board(goal)}\n\n"
            f"Find the minimum sequence of moves to reach the goal.\n"
            f"Each move slides the blank tile: U=up, D=down, L=left, R=right.\n"
            f"Give the sequence of moves and the total number of moves.\nA:"
        )

    def to_image(self, instance: dict[str, Any]) -> Image.Image:
        """Render the sliding puzzle initial and goal boards side by side."""
        return draw_sliding_puzzle(
            instance["initial"],
            instance["n"],
            goal=instance["goal"],
            title=f"{instance['n']}x{instance['n']} Sliding Puzzle  |  optimal moves = {instance['optimal_length']}",
        )


# ------------------------------------------------------------------
# Sliding puzzle helpers
# ------------------------------------------------------------------


def _puzzle_neighbors(
    state: tuple,
    n: int,
) -> list[tuple[tuple, str]]:
    """Return (new_state, move_char) for all valid blank-tile moves."""
    blank = state.index(0)
    br, bc = divmod(blank, n)
    neighbors = []
    for mv, (dr, dc) in [("U", (-1, 0)), ("D", (1, 0)), ("L", (0, -1)), ("R", (0, 1))]:
        nr, nc = br + dr, bc + dc
        if 0 <= nr < n and 0 <= nc < n:
            new_blank = nr * n + nc
            new_state = list(state)
            new_state[blank], new_state[new_blank] = new_state[new_blank], new_state[blank]
            neighbors.append((tuple(new_state), mv))
    return neighbors


def _find_puzzle_states(
    n: int,
    target_len: int,
    rng: random.Random,
    max_attempts: int = 300,
) -> tuple[tuple, tuple] | None:
    """Find a (initial, goal) pair in the n×n puzzle state graph at BFS distance target_len."""
    size = n * n
    solved = tuple(range(1, size)) + (0,)

    for _ in range(max_attempts):
        # Generate a random reachable state via random walk from solved
        state = solved
        walk_len = max(target_len * 4, 20)
        for _ in range(walk_len):
            nbrs = _puzzle_neighbors(state, n)
            state = rng.choice(nbrs)[0]
        initial = state

        # BFS from initial up to depth target_len + 3
        dist: dict[tuple, int] = {initial: 0}
        queue: collections.deque[tuple] = collections.deque([initial])
        candidates_at_target: list[tuple] = []

        while queue:
            cur = queue.popleft()
            d = dist[cur]
            if d > target_len + 3:
                break
            if d == target_len:
                candidates_at_target.append(cur)
                continue
            for nxt, _ in _puzzle_neighbors(cur, n):
                if nxt not in dist:
                    dist[nxt] = d + 1
                    queue.append(nxt)

        if candidates_at_target:
            goal = rng.choice(candidates_at_target)
            # Verify exact distance
            if dist.get(goal) == target_len:
                return initial, goal

    return None


def _manhattan_heuristic(state: tuple, goal: tuple, n: int) -> int:
    """Sum of Manhattan distances of each tile from its goal position."""
    total = 0
    goal_pos = {tile: divmod(i, n) for i, tile in enumerate(goal)}
    for i, tile in enumerate(state):
        if tile == 0:
            continue
        r1, c1 = divmod(i, n)
        r2, c2 = goal_pos[tile]
        total += abs(r1 - r2) + abs(c1 - c2)
    return total


def _astar_puzzle(
    initial: tuple,
    goal: tuple,
    n: int,
) -> tuple[list[tuple], list[str]]:
    """A* search on the sliding puzzle state space.

    Returns (state_sequence, move_sequence).
    """
    h0 = _manhattan_heuristic(initial, goal, n)
    # heap entries: (f, g, state, path_states, path_moves)
    heap: list = [(h0, 0, initial, [initial], [])]
    visited: dict[tuple, int] = {initial: 0}

    while heap:
        f, g, state, path_states, path_moves = heapq.heappop(heap)
        if state == goal:
            return path_states, path_moves
        if g > visited.get(state, float("inf")):
            continue
        for nxt, mv in _puzzle_neighbors(state, n):
            ng = g + 1
            if ng < visited.get(nxt, float("inf")):
                visited[nxt] = ng
                h = _manhattan_heuristic(nxt, goal, n)
                heapq.heappush(
                    heap,
                    (ng + h, ng, nxt, path_states + [nxt], path_moves + [mv]),
                )

    raise ProjectionFailure("SlidingPuzzle A*: no solution found")


# ---------------------------------------------------------------------------
# Variant 2C — Tower of Hanoi
# ---------------------------------------------------------------------------


class TowerOfHanoiVariant(ProblemVariant):
    """Projects the bare-graph path length into a Tower of Hanoi instance.

    # domain element → graph element mapping
    # Hanoi state (tuple of peg indices, state[i] = peg of disk i) = node
    # legal disk move (topmost disk peg A → peg B)                 = edge
    # BFS path length between states                                = path length in G

    State space: P^N states for P pegs and N disks.
    For P=3: the state graph is the Sierpiński triangle graph of order N.

    Disk indexing: disk 0 = smallest, disk N-1 = largest.
    "Topmost disk of peg p" = smallest-indexed disk currently on peg p.
    """

    N_PEGS: int = 3

    def project(
        self,
        G: nx.Graph,
        source: int,
        target: int,
        seed: int | None = None,
    ) -> dict[str, Any]:
        # Determine required path length
        try:
            bare_path = nx.shortest_path(G, source, target)
        except nx.NetworkXNoPath:
            raise ProjectionFailure("TowerOfHanoi: no path between source and target in G")
        target_len = len(bare_path) - 1

        # Choose n_disks: diameter of Hanoi(n, P=3) = (3^n - 1) / 2
        # n=2 → diameter 3, n=3 → diameter 7, n=4 → diameter 15
        n_disks = _hanoi_disks_for_length(target_len, self.N_PEGS)
        if n_disks is None:
            raise ProjectionFailure(
                f"TowerOfHanoi: target path length {target_len} too large for "
                f"any feasible n_disks (max supported diameter = 15 for n=4)"
            )

        rng = random.Random(seed)
        result = _find_hanoi_states(n_disks, self.N_PEGS, target_len, rng)
        if result is None:
            raise ProjectionFailure(
                f"TowerOfHanoi: could not find states at distance {target_len} "
                f"with {n_disks} disks"
            )
        initial, goal = result

        return {
            "n_disks": n_disks,
            "n_pegs": self.N_PEGS,
            "initial": list(initial),
            "goal": list(goal),
            "optimal_length": target_len,
            # Stored base graph structure for to_base_graph reconstruction
            "graph_nodes": [int(nd) for nd in sorted(G.nodes())],
            "graph_edges": [[int(u), int(v)] for u, v in G.edges()],
        }

    def solve(self, instance: dict[str, Any]) -> dict[str, Any]:
        n_disks = instance["n_disks"]
        n_pegs = instance["n_pegs"]
        initial = tuple(instance["initial"])
        goal = tuple(instance["goal"])

        path, moves = _bfs_hanoi(initial, goal, n_disks, n_pegs)
        return {
            "path": [list(s) for s in path],
            "moves": moves,
            "cost": len(moves),
            "base_path": list(range(len(path))),
        }

    def to_base_graph(self, instance: dict[str, Any]) -> nx.Graph:
        """Reconstruct the base graph G from stored graph structure.

        The Hanoi state space is much larger than G; to_base_graph returns G
        so that are_isomorphic_instances() compares base graphs across all variants.
        """
        G = nx.Graph()
        G.add_nodes_from(instance["graph_nodes"])
        for u, v in instance["graph_edges"]:
            G.add_edge(u, v)
        return G

    def verify(self, instance: dict[str, Any], solution: dict[str, Any]) -> bool:
        n_pegs = instance["n_pegs"]
        goal = tuple(instance["goal"])
        states = [tuple(s) for s in solution["path"]]
        moves = solution["moves"]
        if not states:
            return False
        if states[0] != tuple(instance["initial"]):
            return False
        if states[-1] != goal:
            return False
        if len(moves) != len(states) - 1:
            return False
        for cur, nxt, mv in zip(states, states[1:], moves):
            src_peg, dst_peg = mv
            # Verify the move is valid
            valid_nbrs = {s for s, _ in _hanoi_neighbors(cur, n_pegs)}
            if nxt not in valid_nbrs:
                return False
        return len(moves) == solution["cost"]

    def to_prompt(self, instance: dict[str, Any]) -> str:
        n = instance["n_disks"]
        p = instance["n_pegs"]
        initial = instance["initial"]
        goal = instance["goal"]
        return (
            f"Q: You have {n} disks and {p} pegs (labeled 0 to {p-1}).\n"
            f"Disk 0 is the smallest, disk {n-1} is the largest.\n"
            f"Current state: {initial}  (state[i] = peg that disk i is on)\n"
            f"Goal state: {goal}\n"
            f"Find the minimum sequence of moves to reach the goal.\n"
            f"Each move transfers the topmost (smallest) disk from one peg to another peg "
            f"where the destination is either empty or has a larger top disk.\n"
            f"Give each move as (from_peg, to_peg) and the total number of moves.\nA:"
        )

    def to_image(self, instance: dict[str, Any]) -> Image.Image:
        """Render the Tower of Hanoi initial and goal configurations."""
        return draw_hanoi(
            instance["initial"],
            instance["n_disks"],
            instance["n_pegs"],
            goal=instance["goal"],
            title=f"Tower of Hanoi  |  {instance['n_disks']} disks  {instance['n_pegs']} pegs  |  optimal = {instance['optimal_length']} moves",
        )


# ------------------------------------------------------------------
# Tower of Hanoi helpers
# ------------------------------------------------------------------


def _hanoi_top_disk(state: tuple, peg: int) -> int | None:
    """Return index of topmost (smallest) disk on peg, or None if empty."""
    for disk in range(len(state)):
        if state[disk] == peg:
            return disk
    return None


def _hanoi_neighbors(
    state: tuple,
    n_pegs: int,
) -> list[tuple[tuple, tuple[int, int]]]:
    """Return (new_state, (src_peg, dst_peg)) for all valid moves from state."""
    n_disks = len(state)
    neighbors = []
    for src_peg in range(n_pegs):
        top = _hanoi_top_disk(state, src_peg)
        if top is None:
            continue
        for dst_peg in range(n_pegs):
            if dst_peg == src_peg:
                continue
            dst_top = _hanoi_top_disk(state, dst_peg)
            if dst_top is None or dst_top > top:
                new_state = list(state)
                new_state[top] = dst_peg
                neighbors.append((tuple(new_state), (src_peg, dst_peg)))
    return neighbors


def _hanoi_disks_for_length(target_len: int, n_pegs: int = 3) -> int | None:
    """Return the minimum n_disks such that the Hanoi graph diameter >= target_len."""
    # Diameter of Hanoi(n, 3) = (3^n - 1) / 2
    # n=2 → 4, n=3 → 13, n=4 → 40  (these are the DIAMETER values for the full graph)
    # For finding a pair at exact distance target_len, we just need
    # the state space to be large enough.
    for n in range(2, 6):
        diameter = (3**n - 1) // 2
        if diameter >= target_len:
            return n
    return None


def _find_hanoi_states(
    n_disks: int,
    n_pegs: int,
    target_len: int,
    rng: random.Random,
    max_attempts: int = 100,
) -> tuple[tuple, tuple] | None:
    """Find a (initial, goal) pair in the Hanoi state graph at BFS distance target_len."""
    # Enumerate all states and build BFS distance map
    all_states = list(itertools.product(range(n_pegs), repeat=n_disks))

    for _ in range(max_attempts):
        initial = rng.choice(all_states)

        # BFS from initial
        dist: dict[tuple, int] = {initial: 0}
        queue: collections.deque[tuple] = collections.deque([initial])
        candidates: list[tuple] = []

        while queue:
            cur = queue.popleft()
            d = dist[cur]
            if d == target_len:
                candidates.append(cur)
                continue
            if d > target_len:
                continue
            for nxt, _ in _hanoi_neighbors(cur, n_pegs):
                if nxt not in dist:
                    dist[nxt] = d + 1
                    queue.append(nxt)

        if candidates:
            goal = rng.choice(candidates)
            return initial, goal

    return None


def _bfs_hanoi(
    initial: tuple,
    goal: tuple,
    n_disks: int,
    n_pegs: int,
) -> tuple[list[tuple], list[tuple[int, int]]]:
    """BFS shortest path in the Hanoi state graph."""
    prev: dict[tuple, tuple | None] = {initial: None}
    move_to: dict[tuple, tuple[int, int] | None] = {initial: None}
    queue: collections.deque[tuple] = collections.deque([initial])

    while queue:
        cur = queue.popleft()
        if cur == goal:
            break
        for nxt, mv in _hanoi_neighbors(cur, n_pegs):
            if nxt not in prev:
                prev[nxt] = cur
                move_to[nxt] = mv
                queue.append(nxt)

    if goal not in prev:
        raise ProjectionFailure(f"TowerOfHanoi BFS: no path from {initial} to {goal}")

    path: list[tuple] = []
    moves: list[tuple[int, int]] = []
    node: tuple | None = goal
    while node is not None:
        path.append(node)
        mv = move_to[node]
        if mv is not None:
            moves.append(mv)
        node = prev[node]
    path.reverse()
    moves.reverse()
    return path, moves
