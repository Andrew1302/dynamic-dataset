"""Core abstractions for the graph-disguise benchmark.

Each ``BenchmarkTask`` exposes four extension points: graph generation,
direct prompt + image, disguise transformation (``G → Disguise``), and
the disguise's prompt + image (rendered by ``Disguise.render()``).

Subclasses register themselves automatically via ``__init_subclass__``
when they set the ``name`` class attribute.
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, TypedDict, runtime_checkable

import networkx as nx
import numpy as np
from PIL import Image


class Sample(TypedDict):
    direct_prompt: str
    direct_image: Image.Image
    disguise_prompt: str
    disguise_image: Image.Image
    answer: str


@runtime_checkable
class Disguise(Protocol):
    """Anything that knows how to draw itself into a PIL image."""

    def render(self) -> Image.Image: ...


_registry: dict[str, type["BenchmarkTask"]] = {}


def get_all_tasks() -> dict[str, type["BenchmarkTask"]]:
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

    def disguise(self, G: nx.Graph, seed: int) -> Disguise:
        raise NotImplementedError

    # --- concrete entry point ------------------------------------------------

    def generate(self, seed: int, difficulty: str = "easy") -> Sample:
        rng = np.random.default_rng(seed)
        G = self.sample_graph(rng, difficulty)
        return {
            "direct_prompt": self.direct_prompt(G),
            "direct_image": self.render_direct(G),
            "disguise_prompt": self.disguise_prompt(),
            "disguise_image": self.disguise(G, seed).render(),
            "answer": self.solve(G),
        }
