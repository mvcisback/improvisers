"""Example GameGraph implementation based on dictionary adjacency list."""

from __future__ import annotations

import random
from typing import Dict, Literal, Iterable, Set, Tuple, Union, Optional

import attr
import numpy as np

from improvisers.game_graph import Node, NodeKinds, Distribution
from improvisers.game_graph import validate_game_graph


NodeKinds2 = Union[Literal['p1'], Literal['p2'], Literal['env'], bool]
ConcreteActions = Union[Set[Node], Dict[Node, float]]
Graph = Dict[Node, Tuple[NodeKinds2, ConcreteActions]]


# HACK: Force hashing since data is not part of Distribution API.
@attr.s(frozen=True, auto_attribs=True, hash=False)
class ExplicitDist(Distribution):
    data: Dict[Node, float] = attr.ib(factory=dict, converter=dict)

    def __hash__(self) -> int:
        return hash(tuple(self.data.items()))

    @property
    def entropy(self) -> float:
        probs = np.array([v for v in self.data.values() if v > 0])
        return -(probs * np.log(probs)).sum()

    def sample(self, seed: Optional[int] = None) -> Node:
        if seed is not None:
            random.seed(seed)
        return random.choices(*zip(*self.data.items()))[0]

    def prob(self, node: Node) -> float:
        return self.data.get(node, 0)

    def support(self) -> Iterable[Node]:
        return self.data.keys()


@attr.s(frozen=True, auto_attribs=True)
class ExplicitGameGraph:
    root: Node
    graph: Graph

    def __attrs_post_init__(self) -> None:
        validate_game_graph(self)

    def label(self, node: Node) -> NodeKinds:
        label, actions = self.graph[node]
        if label == 'env':
            assert isinstance(actions, dict)
            return ExplicitDist(actions)
        return label

    def moves(self, node: Node) -> Set[Node]:
        return set(self.graph[node][1])


__all__ = ['ExplicitGameGraph', 'ExplicitDist']
