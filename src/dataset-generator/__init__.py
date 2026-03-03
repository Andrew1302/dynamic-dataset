"""Dynamic multimodal dataset generator for graph theory problems.

Public API
----------
- ``GraphTask`` — base class for all tasks
- ``get_all_tasks`` / ``get_task`` — task registry accessors
- ``graph_generator`` — graph creation utilities
- ``graph2img`` — render a networkx graph to a PIL Image
"""

from .tasks import GraphTask, get_all_tasks, get_task
from .visualization import graph2img
from . import graph_generator

__all__ = [
    "GraphTask",
    "get_all_tasks",
    "get_task",
    "graph_generator",
    "graph2img",
]
