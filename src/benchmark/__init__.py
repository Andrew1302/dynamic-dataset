"""Graph-disguise benchmark package.

Each :class:`BenchmarkTask` pairs a graph-theoretic question with exactly
one visual disguise. Every call to :py:meth:`BenchmarkTask.generate`
returns five artifacts (direct prompt + image, disguise prompt + image,
shared ground-truth answer) for a fresh random graph.
"""

from .base import BenchmarkTask, Sample, get_all_tasks, get_task
from . import tasks  # noqa: F401  (triggers task registration)

__all__ = ["BenchmarkTask", "Sample", "get_all_tasks", "get_task"]
