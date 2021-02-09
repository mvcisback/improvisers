"""Implicit Timed GameGraph implementation based on Dynamics protocol."""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, Set, Tuple, Iterable
from typing import Union

import attr

from improvisers.game_graph import Node, NodeKinds, Player, Distribution
from improvisers.game_graph import dfs_nodes, validate_game_graph
from improvisers.explicit import ExplicitGameGraph, Graph


Action = Any
Actions = Set[Action]
TimedNode = Union[Tuple[int, Node], Distribution]


@attr.s(frozen=True, auto_attribs=True)
class TimedDist:
    time: int
    dist: Distribution

    def sample(self, seed: Optional[int] = None) -> TimedNode:
        return self.time, self.dist.sample(seed)

    def prob(self, timed_node: TimedNode) -> float:
        """Returns probability of given node."""
        if not isinstance(timed_node, tuple):
            return 0

        time, node = timed_node
        return self.dist.prob(node) if time == self.time else 0

    def support(self) -> Iterable[TimedNode]:
        """Iterate over nodes with non-zero probability."""
        for node in self.dist.support():
            yield self.time, node

    @property
    def entropy(self) -> float:
        return self.dist.entropy


class Dynamics(Protocol):
    @property
    def start(self) -> Node:
        ...

    def player(self, node: Node) -> Player:
        ...

    def actions(self, node: Node) -> Actions:
        ...

    def transition(self, node: Node, action: Action) -> Distribution:
        ...


@attr.s(frozen=True, auto_attribs=True)
class ImplicitGameGraph:
    """Create game graph from labeled dynamics."""
    dyn: Dynamics
    accepting: Callable[[Node], bool]
    horizon: Optional[int] = None
    validate: bool = True
    _invalid: Optional[Callable[[Node], bool]] = None

    def __attrs_post_init__(self) -> None:
        if not self.validate:
            return
        validate_game_graph(self)
        if self._invalid is not None:
            assert not any(self.invalid(node) for node in dfs_nodes(self))

    @property
    def root(self) -> TimedNode:
        return (0, self.dyn.start)

    def episode_ended(self, timed_node: TimedNode) -> bool:
        if not isinstance(timed_node, tuple):
            return False

        time, node = timed_node

        if(self.horizon is not None) and (time >= self.horizon):
            return True

        player = self.dyn.player(node)
        actions = self.dyn.actions(node)

        if isinstance(player, bool) or not actions:
            return True
        return (self.horizon is not None) and (time >= self.horizon)

    def label(self, timed_node: TimedNode) -> NodeKinds:
        if not isinstance(timed_node, tuple):
            return timed_node

        _, node = timed_node
        if not self.episode_ended(timed_node):
            return self.dyn.player(node)

        return self.accepting(node)

    def actions(self, timed_node: TimedNode) -> Actions:
        assert isinstance(timed_node, tuple)
        _, node = timed_node
        return self.dyn.actions(node)

    def transition(self, timed_node: TimedNode, action: Action) -> TimedNode:
        assert isinstance(timed_node, tuple)
        time, node = timed_node
        dist = TimedDist(time + 1, self.dyn.transition(node, action))

        # Optimization. Remove singleton distributions.
        for count, node2 in zip([1, 2], dist.support()):
            pass  # count should increment to 1 or 2.
        return dist if count > 1 else node2

    def invalid(self, timed_node: TimedNode) -> bool:
        if not isinstance(timed_node, tuple):
            return False
        return False

    def moves(self, timed_node: TimedNode) -> Set[TimedNode]:
        if self.episode_ended(timed_node):
            return set()
        elif not isinstance(timed_node, tuple):
            return set(timed_node.support())
        else:
            actions = self.actions(timed_node)
            return {self.transition(timed_node, a) for a in actions}

    def nodes(self) -> Iterable[Node]:
        yield from dfs_nodes(self)

    def make_explicit(self) -> ExplicitGameGraph:
        graph: Graph = {}
        for node in dfs_nodes(self):
            label = self.label(node)
            if isinstance(label, bool):
                graph[node] = (label, set())
            elif label == 'p1' or label == 'p2':
                graph[node] = (label, self.moves(node))
            else:  # label is a distribution.
                dist = {n: label.prob(n) for n in label.support()}
                graph[node] = ('env', dist)

        return ExplicitGameGraph(root=self.root, graph=graph)


__all__ = ['ImplicitGameGraph', 'Dynamics']
