from typing import Hashable, Literal, Protocol
from typing import Optional, Set, Union, Iterable


import attr
from toposort import toposort_flatten as toposort


Node = Hashable
NodeKinds = Union[Literal['p1'], Literal['p2'], Literal['env'], bool]


@attr.s(frozen=True, auto_attribs=True)
class Action:
    node: Node
    prob: Optional[float] = None
    size: int = 1

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


__all__ = [
    'GameGraph', 'validate_game_graph', 'Action', 'Node', 'NodeKinds'
]
