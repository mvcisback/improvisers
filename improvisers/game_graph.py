from typing import Any, Callable, Hashable, Literal, Mapping, Protocol
from typing import Optional, Set, Tuple, Union

import attr


Action = Any
Node = Hashable
NodeKinds = Union[Literal['p1'], Literal['p2'], Literal['env'], bool]


class Distribution(Protocol):
    def logprob(self, elem=None) -> float:
        ...

    def sample(self, seed=None) -> Any:
        ...


class Actions(Protocol):
    size: int
    dist: Optional[Distribution] = None

    def __contains__(self, elem) -> bool:
        ...


@attr.s(frozen=True, auto_attribs=True)
class GameGraph:
    """Adjacency list representation of game graph."""
    label: Callable[[Node], NodeKinds]          # Node -> Player or Reward.
    neighbors: Callable[[Node], Set[Node]]      # Adjacency List.
    actions: Callable[[Node, Node], Actions]    # Edge -> Available Actions.

    def __post_init_attrs__(self):
        """DFS to check acyclic and terminals iff bool label."""
        visited = {}
        for node, children in neighbors.items():
            if isinstance(node, bool) == bool(children):
                raise ValueError('Only terminals can be rewards!')
            if children & visited:
                raise ValueError("Graph contains a cycle!")
            visited |= children


__all__ = ['GameGraph', 'Actions', 'Distribution']
