"""Implicit Timed GameGraph implementation based on Dynamics protocol."""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, Set, Tuple, Iterable
from typing import cast

import attr

from improvisers.game_graph import Node, NodeKinds
from improvisers.game_graph import dfs_nodes, validate_game_graph


Action = Any
Actions = Set[Action]
TimedNode = Tuple[int, Node]


class Dynamics(Protocol):
    start: Node

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
        return (0, self.dyn.start)

    def actions(self, node: Node) -> Actions:
        player = self.dyn.player(node)

        if isinstance(player, bool):
            return set()
        elif player == 'p1' or player == 'p2':
            return self.dyn.actions(node)

        actions = set(player.support())
        assert actions, "Distributions must have non-empty support!"
        return actions

    def episode_ended(self, timed_node: Node) -> bool:
        time, node = cast(TimedNode, timed_node)

        if(self.horizon is not None) and (time >= self.horizon):
            return True

        player = self.dyn.player(node)
        actions = self.actions(node)

        if isinstance(player, bool) or not actions:
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

        for a in self.actions(node):
            moves.add((time + 1, self.dyn.transition(node, a)))
        return moves

    def nodes(self) -> Iterable[Node]:
        yield from dfs_nodes(self)


__all__ = ['ImplicitGameGraph', 'Dynamics']
