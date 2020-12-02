import math
import random
from typing import Dict, Hashable, Mapping, Set, Optional, Tuple, Union

import attr

from improvisers.game_graph import *


def log(x: float) -> float:
    return -float('inf') if x == 0 else math.log(x)


@attr.s(frozen=True, auto_attribs=True)
class ExplicitDistribution:
    logprobs: Mapping[Action, float]

    def logprob(self, action: Action) -> float:
        return self.logprobs.get(action, -float('inf'))

    def sample(self, seed: Optional[int]=None) -> Action:
        if seed is not None:
            random.seed(seed)
        return random.choices(*zip(*self.logprobs.items()))[0]


@attr.s(frozen=True, auto_attribs=True)
class ExplicitActions:
    actions: Set[Action]
    dist: Optional[Distribution]
    logprob: Optional[float]

    def __contains__(self, action: Action) -> bool:
        return action in self.actions


Graph = Mapping[Node, Tuple[NodeKinds, Mapping[Node, Actions]]]


@attr.s(frozen=True, auto_attribs=True)
class ExplicitGameGraph:
    root: Node
    graph: Graph

    def __attrs_post_init__(self) -> None:
        validate_game_graph(self)

    def label(self, node: Node) -> NodeKinds:
        return self.graph[node][0]

    def neighbors(self, node: Node) -> Set[Node]:
        return set(self.graph[node][1])

    def actions(self, start: Node, end: Node) -> Actions:
        return self.graph[start][1][end]


ExplicitGraph = Mapping[Node, Tuple[
    NodeKinds, 
    Mapping[Node, Union[Set[Action], Mapping[Action, float]]]
]]


def from_dict(root: Node, mapping: ExplicitGraph) -> ExplicitGameGraph:
    pass


__all__ = [
    'from_dict',
    'ExplicitGameGraph', 
    'ExplicitActions', 
    'ExplicitDistribution'
]
