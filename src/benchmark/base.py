"""Core abstractions for the graph-disguise benchmark.

A ``BenchmarkTask`` pairs a graph-theoretic question with exactly one
visual disguise. Each call to :py:meth:`BenchmarkTask.generate` draws a
fresh random graph and returns five artifacts: a direct prompt, a direct
graph image, a disguise prompt, a disguise image, and the shared
ground-truth answer.

Subclasses register themselves automatically via ``__init_subclass__``
as long as they set the ``name`` class attribute.
"""

from __future__ import annotations

from typing import Any, ClassVar, TypedDict

import networkx as nx
import numpy as np
from PIL import Image


class Sample(TypedDict):
    direct_prompt: str
    direct_image: Image.Image
    disguise_prompt: str
    disguise_image: Image.Image
    answer: str


_registry: dict[str, type["BenchmarkTask"]] = {}


def get_all_tasks() -> dict[str, type["BenchmarkTask"]]:
    """Return a copy of the task registry."""
    return dict(_registry)


def get_task(name: str) -> type["BenchmarkTask"]:
    return _registry[name]


class BenchmarkTask:
    """Pair of (graph task, visual disguise) with shared ground truth."""

    name: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.name:
            _registry[cls.name] = cls

    # --- abstract contract ---------------------------------------------------

    def sample_graph(self, rng: np.random.Generator, difficulty: str) -> nx.Graph:
        raise NotImplementedError

    def solve(self, G: nx.Graph) -> str:
        raise NotImplementedError

    def direct_prompt(self, G: nx.Graph) -> str:
        raise NotImplementedError

    def render_direct(self, G: nx.Graph) -> Image.Image:
        raise NotImplementedError

    def disguise_prompt(self) -> str:
        raise NotImplementedError

    def render_disguise(self, G: nx.Graph, seed: int) -> Image.Image:
        raise NotImplementedError

    # --- concrete entry point ------------------------------------------------

    def generate(self, seed: int, difficulty: str = "easy") -> Sample:
        rng = np.random.default_rng(seed)
        G = self.sample_graph(rng, difficulty)
        return {
            "direct_prompt": self.direct_prompt(G),
            "direct_image": self.render_direct(G),
            "disguise_prompt": self.disguise_prompt(),
            "disguise_image": self.render_disguise(G, seed),
            "answer": self.solve(G),
        }
