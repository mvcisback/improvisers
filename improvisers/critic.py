from typing import Iterable, Mapping, Tuple, Protocol, Optional

import attr

from improvisers.game_graph import Action, Node, GameGraph


class Distribution(Protocol):
    def entropy(self) -> float:
        ...

    def sample(self, seed: Optional[int] = None) -> Node:
        ...

    def prob(self, node: Node) -> float:
        ...

    def support(self) -> Iterable[Node]:
        ...

    def lsat(self, critic: Critic, rationality: float) -> float:
        ...

    def psat(self, critic: Critic, rationality: float) -> float:
        ...


class Critic(Protocol):
    def value(self, node: Node, rationality: float) -> float:
        """Soft value of node."""
        ...

    def entropy(self, node: Node, rationality: float) -> float:
        """Causal Entropy of policy starting at node."""
        ...

    def psat(self, node: Node, rationality: float) -> float:
        """Worst case sat probability of max ent policy from node."""
        ...

    def lsat(self, node: Node, rationality: float) -> float:
        """Worst case sat log probability of max ent policy from node."""
        ...

    def rationality(self, node: Node, psat: float) -> float:
        """Rationality induced by target satisfaction probability."""
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


__all__ = ['Critic']
