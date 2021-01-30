"""Implicit Timed GameGraph implementation based on callables."""

from __future__ import annotations

from typing import cast, Any, Callable, Optional, Set, Tuple, Iterable

import attr

from improvisers.game_graph import Action, Node, NodeKinds
from improvisers.game_graph import dfs_nodes, validate_game_graph


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
    player: Callable[[Node], NodeKinds]
    accepting: Callable[[Node], bool]
    move: Callable[[Node, Any], Node]
    moves: Callable[[Node], Iterable[Any]]
    move_prob: Callable[[Node, Any], Optional[float]] = lambda *_: None
    move_size: Callable[[Node, Any], int] = lambda *_: 1
    horizon: Optional[int] = None
    validate: bool = True

    def __attrs_post_init__(self) -> None:
        if self.validate:
            validate_game_graph(self)

    @property
    def root(self) -> TimedNode:
        return (0, self._root)

    def episode_ended(self, timed_node: Node) -> bool:
        time, node = cast(TimedNode, timed_node)
        if isinstance(self.player(node), bool):
            return True
        return (self.horizon is not None) and (time >= self.horizon)

    def label(self, timed_node: Node) -> NodeKinds:
        _, node = cast(TimedNode, timed_node)
        player = self.player(node)
        if isinstance(player, bool) or not self.episode_ended(timed_node):
            return player
        return self.accepting(node)

    def actions(self, timed_node: Node) -> Set[Action]:
        actions: Set[Action] = set()
        if self.episode_ended(timed_node):
            return actions
        time, node = cast(TimedNode, timed_node)

        for m in self.moves(node):
            action = Action(
                node=(time + 1, self.move(node, m)),
                prob=self.move_prob(node, m),
                size=self.move_size(node, m),
            )
            actions.add(action)
        return actions

    def nodes(self) -> Iterable[Node]:
        yield from dfs_nodes(self)


__all__ = ['ImplicitGameGraph']
