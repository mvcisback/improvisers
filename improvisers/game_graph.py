"""Define GameGraph Protocol."""

from __future__ import annotations

from typing import Hashable, Literal, Protocol
from typing import Optional, Set, Union, Iterable


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

    @property
    def entropy(self) -> float:
        ...


class GameGraph(Protocol):
    """Adjacency list representation of game graph."""
    @property
    def root(self) -> Node:
        ...

    def nodes(self) -> Iterable[Node]:
        ...

    def label(self, node: Node) -> NodeKinds:
        ...

    def moves(self, node: Node) -> Set[Node]:
        ...


def dfs_nodes(game_graph: GameGraph) -> Iterable[Node]:
    stack, visited = [game_graph.root], set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue

        yield node
        visited.add(node)
        stack.extend(game_graph.moves(node))


def validate_game_graph(game_graph: GameGraph) -> None:
    """Validates preconditions on game graph.

    1. Graph should define a DAG.
    2. Only terminal nodes should have rewards (and vice versa).
    3. Environment moves should be stochastic.
    """
    nodes = game_graph.nodes()
    graph = {n: game_graph.moves(n) for n in nodes}

    for node in toposort(graph):
        moves = game_graph.moves(node)
        label = game_graph.label(node)

        if isinstance(label, bool) == bool(moves):
            raise ValueError('Terminals <-> label is a reward!')


Node = Hashable
NodeKinds = Union[Literal['p1'], Literal['p2'], Distribution, bool]


__all__ = [
    'GameGraph', 'Distribution', 'Node', 'NodeKinds',
    'dfs_nodes', 'validate_game_graph',
]
