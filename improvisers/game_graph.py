from typing import Any, Callable, Hashable, Literal, List, Mapping, Protocol
from typing import Optional, Set, Tuple, Union


Action = Any
Node = Hashable
NodeKinds = Union[Literal['p1'], Literal['p2'], Literal['env'], bool]


class Distribution(Protocol):
    def logprob(self, action: Action) -> float:
        ...

    def sample(self, seed: Optional[int]=None) -> Any:
        ...


class Actions(Protocol):
    @property
    def size(self) -> int:
        ...

    @property
    def logprob(self) -> Optional[float]:
        ...

    @property
    def dist(self) -> Optional[Distribution]:
        ...

    def __contains__(self, action: Action) -> bool:
        ...


class GameGraph(Protocol):
    """Adjacency list representation of game graph."""
    @property
    def root(self) -> Node:
        ...

    def label(self, node: Node) -> NodeKinds:
        ...

    def neighbors(self, node: Node) -> Set[Node]:
        ...

    def actions(self, start: Node, end: Node) -> Actions:
        ...


def validate_game_graph(graph: GameGraph) -> None:
    """Validates preconditions on game graph.

    1. Graph should define a DAG.
    2. Only terminal nodes should have rewards (and vice versa).
    3. Environment actions should be stochastic.
    """
    stack, visited= [graph.root], {graph.root}
    while stack:
        node = stack.pop()
        neighbors = graph.neighbors(node)
        label = graph.label(node)

        if neighbors & visited:
            raise ValueError("Graph contains a cycle!")

        if isinstance(label, bool) == bool(neighbors):
            raise ValueError('Terminals <-> label is a reward!')

        if label == "env":
            for node2 in neighbors:
                actions = graph.actions(node, node2)
                if (actions.dist is None) or (actions.logprob is None):
                    raise ValueError("Environment actions must by stochastic.")

        visited |= neighbors
        stack.extend(neighbors)


__all__ = [
    'GameGraph', 'Actions', 'Distribution', 'validate_game_graph',
    'Action', 'Node', 'NodeKinds'
]
