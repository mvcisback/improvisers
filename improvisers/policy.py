"""Module for synthesizing policies from ERCI instances."""

import collections
from typing import Generator, Optional, Tuple, Union, Sequence

import attr
from scipy.optimize import brentq

from improvisers.game_graph import Node, GameGraph
from improvisers.critic import Critic, Distribution
from improvisers.tabular import TabularCritic


Dist = Distribution
Observation = Union[
    Dist,            # Provide next state distribution.
    Sequence[Node],  # Observe player 2 nodes. Worst case counter-factuals.
    None,            # Assume worst case dist compatible with next p1 state.
]


ImprovProtocol = Generator[
    Tuple[Node, Dist],         # Yield p1 action and expected next state dist.
    Tuple[Node, Observation],  # Observe p1 state and observation.
    bool                       # Return whether or not p1 won the game.
]


def replan(coeff: float, critic: Critic, dist1: Dist, dist2: Dist) -> float:
    """Replan based on observed state distribution.

    Args:
    - coeff: Current rationality coefficient.
    - critic: Critic to the current stochastic game.
    - dist1: Conjectured next state distribution used for planning.
    - dist2: Actual next state distribution.

    Returns:
      Rationality coefficient induced by actual state distribution.
    """
    expected_entropy = critic.entropy(dist1, coeff)

    def f(x: float) -> float:
        return critic.entropy(dist2, x) - expected_entropy

    # Binary search for rationality coefficient.
    offset = 1
    for _ in range(100):
        try:
            return brentq(f, coeff, coeff + offset)
        except ValueError:
            offset *= 2
    return float('inf')  # Effectively infinite.


@attr.s(auto_attribs=True, frozen=True)
class Actor:
    """Factory for improvisation co-routine."""
    game: GameGraph
    critic: Critic
    rationality: float

    def improvise(self) -> ImprovProtocol:
        """Improviser for game graph.

        Yields:
          Node to transition to and conjectured next player 1 state
          distribution.

        Sends:
          Current player 1 state and distribution the state was drawn from.

        Returns:
          Whether or not player 1 won the game.
        """
        game, critic, rationality = self.game, self.critic, self.rationality

        state = game.root
        while not isinstance(game.label(state), bool):
            move = critic.move_dist(state, rationality).sample()
            state_dist = critic.state_dist(move, rationality)
            state2, obs = yield move, state_dist
            if obs is None:
                raise NotImplementedError  # TODO!
            elif isinstance(obs, collections.Sequence):
                raise NotImplementedError  # TODO!
            else:
                state_dist2 = obs

            rationality = replan(rationality, critic, state_dist, state_dist2)
            state = state2

        return bool(game.label(state))


def solve(game: GameGraph,
          psat: float = 0,
          entropy: float = 0,
          critic: Optional[Critic] = None) -> Actor:
    """Find player 1 improviser for game.

    Args:
      - game: GameGraph for game to play.
      - psat: Min worst case winning probability of improviser.
      - entropy: Min worst case entropy of improviser.
      - critic: Critic instance to use for synthesis.

    Returns:
      Actor factory for improvisation co-routines.
    """

    state = game.root

    if critic is None:
        critic = TabularCritic(game)

    if critic.psat(state, float('inf')) < psat:
        raise ValueError(
            "No improviser exists. Could not reach psat in this MDP"
        )

    rationality = max(0, critic.match_psat(state, psat))

    if critic.entropy(state, rationality) < entropy:
        raise ValueError(
            "No improviser exists. Entropy constraint unreachable."
        )
    return Actor(game, critic, rationality)


__all__ = ['solve', 'Actor', 'ImprovProtocol']
