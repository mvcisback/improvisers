import math
from typing import Generator, Iterable, Optional, Protocol, Tuple

from scipy.optimize import brentq

from improvisers.game_graph import Node, GameGraph
from improvisers.critic import Critic, Distribution


Dist = Distribution


Improviser = Generator[
    Tuple[Node, Dist],     # Yield p1 action and expected state distribution.
    Tuple[Node, Dist],     # Observe p1 state and actual state distribution.
    bool                   # Return whether or not p1 won the game.
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
    psat = dist1.psat(critic, coeff)

    def f(x: float) -> float:
        return dist2.psat(critic, x) - psat

    # Binary search for rationality coefficient.
    return brentq(f, coeff, coeff + 100)


def policy(game: GameGraph, psat: float = 0, entropy: float = 0) -> Improviser:
    """Find player 1 improviser for game.

    Args:
      - game: GameGraph for game to play.
      - psat: Min worst case winning probability of improviser.
      - entropy: Min worst case entropy of improviser.

    Yields:
      Node to transition to and conjectured next player 1 state distribution.

    Sends:
      Current player 1 state and distribution the state was drawn from.

    Returns:
      Whether or not player 1 won the game.
    """
    state = game.root
    critic = Critic.from_game_graph(game)
    rationality = max(0, critic.rationality(state, psat))

    if critic.entropy(state, rationality) < entropy:
        raise ValueError("No improviser exists.")

    while not isinstance(game.label(state), bool):
        action = critic.action_dist(state, rationality).sample()
        state_dist = critic.state_dist(action, rationality)
        state, state_dist2 = yield action, state_dist
        rationality = replan(rationality, critic, state_dist, state_dist2)

    return bool(game.label(state))


__all__ = ['policy', 'Improviser']
