"""Critic Protocol for GameGraph"""

from __future__ import annotations

from typing import Iterable, Protocol, Optional, Union

from improvisers.game_graph import Node, GameGraph


class Distribution(Protocol):
    """Protocol for critic aware distribution over nodes."""
    def sample(self, seed: Optional[int] = None) -> Node:
        """Returns a sampled node from distribution."""
        ...

    def prob(self, node: Node) -> float:
        """Returns probability of given node."""
        ...

    def support(self) -> Iterable[Node]:
        """Iterate over nodes with non-zero probability."""
        ...


DistLike = Union[Node, Distribution]


class Critic(Protocol):
    def value(self, node: Node, rationality: float) -> float:
        """Soft value of node."""
        ...

    def entropy(self, node_dist: DistLike, rationality: float) -> float:
        """Causal Entropy of policy starting at node."""
        ...

    def psat(self, node_dist: DistLike, rationality: float) -> float:
        """Worst case sat probability of max ent policy from node."""
        ...

    def lsat(self, node_dist: DistLike, rationality: float) -> float:
        """Worst case sat log probability of max ent policy from node."""
        ...

    def match_entropy(self, node: Node, target: float) -> float:
        """Rationality induced by target entropy."""
        ...

    def match_psat(self, node: Node, target: float) -> float:
        """Rationality induced by target psat."""
        ...

    def action_dist(self, state: Node, rationality: float) -> Distribution:
        """Predicted action distribution at state."""
        ...

    def state_dist(self, action: Node, rationality: float) -> Distribution:
        """Predicted p1 state distribution after apply action."""
        ...

    @staticmethod
    def from_game_graph(game_graph: GameGraph) -> Critic:
        """Creates a critic from a given game graph."""
        ...


__all__ = ['Critic', 'Distribution', 'DistLike']
