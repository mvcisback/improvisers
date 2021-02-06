"""Implicit Timed GameGraph implementation based on callables."""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, Set, Tuple, Iterable
from typing import cast

import attr

from improvisers.game_graph import Node, NodeKinds
from improvisers.game_graph import dfs_nodes, validate_game_graph


Action = Any
Actions = Iterable[Action]
TimedNode = Tuple[int, Node]


class Dynamics(Protocol):
    init: Node

    def player(self, node: Node) -> NodeKinds:
        ...

    def actions(self, node: Node) -> Actions:
        ...

    def transition(self, node: Node, action: Action) -> Node:
        ...


@attr.s(frozen=True, auto_attribs=True)
class ImplicitGameGraph:
    """Create game graph from labeled dynamics."""
    dyn: Dynamics
    accepting: Callable[[Node], bool]
    invalid: Optional[Callable[[Node], bool]] = None
    horizon: Optional[int] = None
    validate: bool = True

    def __attrs_post_init__(self) -> None:
        if not self.validate:
            return
        validate_game_graph(self)
        if self.invalid is not None:
            assert not any(self.invalid(node) for node in dfs_nodes(self))

    @property
    def root(self) -> TimedNode:
        return (0, self.dyn.init)

    def episode_ended(self, timed_node: Node) -> bool:
        time, node = cast(TimedNode, timed_node)
        if isinstance(self.dyn.player(node), bool):
            return True
        return (self.horizon is not None) and (time >= self.horizon)

    def label(self, timed_node: Node) -> NodeKinds:
        _, node = cast(TimedNode, timed_node)
        player = self.dyn.player(node)
        if isinstance(player, bool) or not self.episode_ended(timed_node):
            return player
        return self.accepting(node)

    def moves(self, timed_node: Node) -> Set[Node]:
        moves: Set[Node] = set()
        if self.episode_ended(timed_node):
            return moves
        time, node = cast(TimedNode, timed_node)

        for a in self.dyn.actions(node):
            moves.add((time + 1, self.dyn.transition(node, a)))
        return moves

    def nodes(self) -> Iterable[Node]:
        yield from dfs_nodes(self)


__all__ = ['ImplicitGameGraph']
