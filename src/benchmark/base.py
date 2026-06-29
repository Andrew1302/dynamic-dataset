"""Core abstractions for the graph-disguise benchmark.

Each ``BenchmarkTask`` exposes four extension points: graph generation,
direct prompt + image, disguise transformation (``G → Disguise``), and
the disguise's prompt + image (rendered by ``Disguise.render()``).

Subclasses register themselves automatically via ``__init_subclass__``
when they set the ``name`` class attribute.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar, Protocol, TypedDict, runtime_checkable

import networkx as nx
import numpy as np
from PIL import Image

from .rendering import RenderConfig, format_adjacency_list


class Sample(TypedDict):
    direct_prompt: str
    direct_image: Image.Image
    disguise_prompt: str
    disguise_image: Image.Image
    answer: str
    n_vertices: int
    n_edges: int


@runtime_checkable
class Disguise(Protocol):
    """Anything that knows how to draw itself into a PIL image.

    Implementations may accept ``pdf_path`` to also persist a vector
    (matplotlib-backed) or raster-wrapped (PIL-backed) PDF copy alongside
    the returned ``PIL.Image``.
    """

    def render(self, pdf_path: str | None = None) -> Image.Image: ...


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

    def sample_graph(
        self,
        rng: np.random.Generator,
        difficulty: str,
        node_count: int | None = None,
        target_chromatic: int | None = None,
    ) -> nx.Graph:
        raise NotImplementedError

    def solve(self, G: nx.Graph) -> str:
        raise NotImplementedError

    def direct_prompt(self, G: nx.Graph, config: RenderConfig | None = None) -> str:
        raise NotImplementedError

    def render_direct(
        self,
        G: nx.Graph,
        config: RenderConfig | None = None,
        pdf_path: str | None = None,
    ) -> Image.Image:
        raise NotImplementedError

    def disguise_prompt(self) -> str:
        raise NotImplementedError

    def disguise(
        self,
        G: nx.Graph,
        seed: int,
        config: RenderConfig | None = None,
    ) -> Disguise:
        raise NotImplementedError

    def _warn_unsupported_kwargs(self, kwargs: dict) -> None:
        """Warn about task-specific ``sample_graph`` knobs this task ignores.

        Every task shares the same ``sample_graph`` signature so ``generate``
        can call them uniformly, but a knob like ``target_chromatic`` only
        applies to ``ColoringTask``. Tasks that do not use such knobs absorb
        them via ``**kwargs`` and pass them here to surface a warning instead
        of silently dropping an option the caller expected to take effect."""
        if kwargs:
            names = ", ".join(sorted(kwargs))
            warnings.warn(
                f"{type(self).__name__}.sample_graph does not support "
                f"argument(s): {names} — ignoring.",
                stacklevel=3,
            )

    # --- concrete entry point ------------------------------------------------

    def generate(
        self,
        seed: int,
        difficulty: str = "easy",
        config: RenderConfig | None = None,
        include_adjacency_list: bool = False,
        node_count: int | None = None,
        target_chromatic: int | None = None,
        direct_pdf_path: str | None = None,
        disguise_pdf_path: str | None = None,
    ) -> Sample:
        cfg = config if config is not None else RenderConfig()
        rng = np.random.default_rng(seed)
        # Forward task-specific knobs only when actually set, so tasks that do
        # not support them (via **kwargs) don't warn on the default no-op case.
        extra: dict[str, Any] = {}
        if target_chromatic is not None:
            extra["target_chromatic"] = target_chromatic
        G = self.sample_graph(
            rng, difficulty, node_count=node_count, **extra
        )
        G.graph["n_vertices"] = int(G.number_of_nodes())
        G.graph["n_edges"] = int(G.number_of_edges())

        direct_prompt = self.direct_prompt(G, cfg)
        disguise_prompt = self.disguise_prompt()
        if include_adjacency_list:
            adj = format_adjacency_list(G, cfg.label_style)
            direct_block = f"{adj}\n"
            disguise_block = (
                f"Adjacency list of the underlying graph:\n{adj}\n"
            )
            direct_prompt = _inject_block(direct_prompt, direct_block)
            disguise_prompt = _inject_block(disguise_prompt, disguise_block)

        return {
            "direct_prompt": direct_prompt,
            "direct_image": self.render_direct(G, cfg, pdf_path=direct_pdf_path),
            "disguise_prompt": disguise_prompt,
            "disguise_image": self.disguise(G, seed, cfg).render(
                pdf_path=disguise_pdf_path
            ),
            "answer": self.solve(G),
            "n_vertices": int(G.graph["n_vertices"]),
            "n_edges": int(G.graph["n_edges"]),
        }


def _inject_block(prompt: str, block: str) -> str:
    """Insert *block* before a trailing ``\\nA:`` (so it stays context,
    not part of the answer). Falls back to appending if the prompt does
    not end in the GraphQA-style ``A:`` marker."""
    if prompt.endswith("\nA:"):
        return prompt[: -len("\nA:")] + f"\n\n{block}\nA:"
    return f"{prompt}\n\n{block}"
