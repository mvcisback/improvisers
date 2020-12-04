from typing import Iterable, Mapping, Tuple, Protocol, Optional

import attr

from improvisers.game_graph import Action, Node, GameGraph


class Distribution(Protocol):
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
        ...

    def entropy(self, node: Node, rationality: float) -> float:
        ...

    def psat(self, node: Node, rationality: float) -> float:
        ...

    def lsat(self, node: Node, rationality: float) -> float:
        ...

    def rationality(self, node: Node, psat: float) -> float:
        ...

    def action_dist(self, state: Node, rationality: float) -> Distribution:
        ...

    def state_dist(self, state: Node, action: Node) -> Distribution:
        ...

    @staticmethod
    def from_game_graph(game_graph: GameGraph) -> Critic:
        ...


__all__ = ['Critic']
