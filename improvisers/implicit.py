"""Implicit Timed GameGraph implementation based on callables."""

from __future__ import annotations

from typing import cast, Callable, Optional, Set, Tuple, Iterable

import attr

from improvisers.game_graph import Action, Node, NodeKinds, validate_game_graph


TimedNode = Tuple[int, Node]


@attr.s(frozen=True, auto_attribs=True)
class ImplicitGameGraph:
    """Create game graph from update and labeling rules.

    Notes:
    1. All states are augmented with current time step.
    2. If horizon is provided, then all states reachable
       after horizon steps are considered leafs.
    3. If a leaf label is not `True`, then it consider false.
    """
    _root: Node
    _actions: Callable[[Node], Set[Action]]
    _nodes: Callable[[], Iterable[Node]]
    _label: Callable[[Node], NodeKinds]
    horizon: Optional[int] = None

    def __attrs_post_init__(self) -> None:
        validate_game_graph(self)

    @property
    def root(self) -> TimedNode:
        return (0, self.root)

    def episode_ended(self, timed_node: Node) -> bool:
        time, _ = cast(TimedNode, timed_node)
        return (self.horizon is not None) and (time >= self.horizon)

    def label(self, timed_node: Node) -> NodeKinds:
        _, node = cast(TimedNode, timed_node)
        label = self._label(node)
        return label if not self.episode_ended(timed_node) else (label is True)

    def actions(self, timed_node: Node) -> Set[Action]:
        if self.episode_ended(timed_node):
            return set()
        time, node = cast(TimedNode, timed_node)
        actions = self._actions(node)
        return {attr.evolve(a, node=(time + 1, a.node)) for a in actions}

    def nodes(self) -> Iterable[TimedNode]:
        pass


__all__ = ['ImplicitGameGraph']
