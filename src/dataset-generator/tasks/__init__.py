"""Task registry and base class."""

from __future__ import annotations

from typing import Any

import networkx as nx
from PIL import Image


# ---------------------------------------------------------------------------
# Global task registry
# ---------------------------------------------------------------------------
_registry: dict[str, type[GraphTask]] = {}


def get_all_tasks() -> dict[str, type[GraphTask]]:
    """Return a copy of the task registry."""
    return dict(_registry)


def get_task(name: str) -> type[GraphTask]:
    return _registry[name]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class GraphTask:
    """Base class for all graph tasks.

    Subclasses must set ``name`` and implement ``generate``.
    Registering happens automatically via ``__init_subclass__``.
    """

    name: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.name:
            _registry[cls.name] = cls

    def generate(self, G: nx.Graph, **kwargs: Any) -> dict[str, str | Image.Image]:
        """Produce a sample from graph *G*.

        Returns
        -------
        dict with keys ``prompt``, ``image``, ``answer``.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Import task modules so they register themselves
# ---------------------------------------------------------------------------
from . import basic, connectivity, pathfinding, advanced  # noqa: E402, F401
