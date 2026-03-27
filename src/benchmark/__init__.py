"""Graph-disguise benchmark package."""
from .base import BaseGraphGenerator, ProjectionFailure, ProblemVariant, are_isomorphic_instances
from .shortest_path_variants import BareGraphVariant, MazeVariant, WordLadderVariant
from .state_search_variants import BareStateGraphVariant, SlidingPuzzleVariant, TowerOfHanoiVariant

__all__ = [
    "BaseGraphGenerator",
    "ProjectionFailure",
    "ProblemVariant",
    "are_isomorphic_instances",
    "BareGraphVariant",
    "MazeVariant",
    "WordLadderVariant",
    "BareStateGraphVariant",
    "SlidingPuzzleVariant",
    "TowerOfHanoiVariant",
]
