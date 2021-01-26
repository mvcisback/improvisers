from __future__ import annotations

from typing import Mapping, Set, Tuple, Union, Iterable

import attr

from improvisers.game_graph import Action, Node, NodeKinds, validate_game_graph


ConcreteActions = Union[Set[Node], Mapping[Node, float]]
Graph = Mapping[Node, Tuple[NodeKinds, ConcreteActions]]


def lift_actions(actions: ConcreteActions) -> Set[Action]:
    if isinstance(actions, set):
        return {Action(a) for a in actions}
    return {Action(a, p) for a, p in actions.items()}


@attr.s(frozen=True, auto_attribs=True)
class ExplicitGameGraph:
    root: Node
    graph: Graph

    def __attrs_post_init__(self) -> None:
        validate_game_graph(self)

    def label(self, node: Node) -> NodeKinds:
        return self.graph[node][0]

    def actions(self, start: Node) -> Set[Action]:
        return lift_actions(self.graph[start][1])

    def nodes(self) -> Iterable[Node]:
        yield from self.graph


__all__ = ['ExplicitGameGraph']
