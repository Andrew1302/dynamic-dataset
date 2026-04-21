"""Image renderers for benchmark tasks (plain graph + disguises)."""

from .map_coloring import render_map
from .maze import render_maze
from .plain_graph import render_graph

__all__ = ["render_graph", "render_map", "render_maze"]
