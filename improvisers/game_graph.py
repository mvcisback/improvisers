"""Define GameGraph Protocol."""

from __future__ import annotations

from typing import Hashable, Literal, Protocol
from typing import Optional, Set, Union, Iterable


import attr
from toposort import toposort_flatten as toposort


class Distribution(Protocol):
    """Protocol for distribution over nodes."""
    def sample(self, seed: Optional[int] = None) -> Node:
        """Returns a sampled node from distribution."""
        ...

    def prob(self, node: Node) -> float:
        """Returns probability of given node."""
        ...

    def support(self) -> Iterable[Node]:
        """Iterate over nodes with non-zero probability."""
        ...


@attr.s(frozen=True, auto_attribs=True)
class Action:
    """Annotated edge in game graph."""
    node: Node
    prob: Optional[float] = None

    @property
    def is_stochastic(self) -> bool:
        return self.prob is not None


class GameGraph(Protocol):
    """Adjacency list representation of game graph."""
    @property
    def root(self) -> Node:
        ...

    def nodes(self) -> Iterable[Node]:
        ...

    def label(self, node: Node) -> NodeKinds:
        ...

    def actions(self, node: Node) -> Set[Action]:
        ...


def dfs_nodes(game_graph: GameGraph) -> Iterable[Node]:
    stack, visited = [game_graph.root], set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue

        yield node
        visited.add(node)
        stack.extend((a.node for a in game_graph.actions(node)))


def validate_game_graph(game_graph: GameGraph) -> None:
    """Validates preconditions on game graph.

    1. Graph should define a DAG.
    2. Only terminal nodes should have rewards (and vice versa).
    3. Environment actions should be stochastic.
    """
    nodes = game_graph.nodes()
    graph = {n: {a.node for a in game_graph.actions(n)} for n in nodes}

    for node in toposort(graph):
        actions = game_graph.actions(node)
        label = game_graph.label(node)

        if isinstance(label, bool) == bool(actions):
            raise ValueError('Terminals <-> label is a reward!')

        if (label == "env") and not all(a.is_stochastic for a in actions):
            raise ValueError("Environment actions must by stochastic!")


Node = Hashable
NodeKinds = Union[Literal['p1'], Literal['p2'], Literal['env'], bool]


__all__ = [
    'GameGraph', 'Distribution', 'Node', 'NodeKinds',
    'dfs_nodes', 'validate_game_graph',
]
