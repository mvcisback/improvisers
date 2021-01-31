"""Critic Protocol for GameGraph"""

from __future__ import annotations

from typing import Protocol, Tuple, Union

from improvisers.game_graph import Node, GameGraph, Distribution


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

    def move_dist(self, state: Node, rationality: float) -> Distribution:
        """Predicted move distribution at state."""
        ...

    def state_dist(self, move: Node, rationality: float) -> Distribution:
        """Predicted p1 state distribution after applying move."""
        ...

    def min_ent_move(self, node: Node, rationality: float) -> Node:
        """Return move which minimizes the (*achievable* entropy, psat)."""
        ...

    def min_psat_move(self, node: Node, rationality: float) -> Tuple[Node, float]:  # noqa: E501
        """Return move which minimizes psat of rationality policy."""
        ...

    @staticmethod
    def from_game_graph(game_graph: GameGraph) -> Critic:
        """Creates a critic from a given game graph."""
        ...


__all__ = ['Critic', 'Distribution', 'DistLike']
