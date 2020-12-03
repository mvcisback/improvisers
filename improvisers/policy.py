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
    psat = dist1.psat(critic, coeff)

    def f(x: float) -> float:
        return dist2.psat(critic, x) - psat

    # Binary search for rationality coefficient.
    return brentq(f, coeff, coeff + 100)[0]


def policy(game: GameGraph, psat: float = 0, entropy: float = 0) -> Improviser:
    """Find improviser for game with p1.

    Args:
    - game: GameGraph for game to play.
    - psat: Min worst case winning probability of improviser.
    - entropy: Min worst case entropy of improviser.
    """
    state = game.root
    critic = Critic.from_game_graph(game)
    rationality = max(0, critic.rationality(state, psat))

    if critic.entropy(state, rationality) < entropy:
        raise ValueError("No improviser exists.")

    while not isinstance(game.label(state), bool):
        action = critic.action_dist(state, rationality).sample()
        state_dist = critic.state_dist(state, action)
        state, state_dist2 = yield action, state_dist
        rationality = replan(rationality, critic, state_dist, state_dist2)

    return bool(game.label(state))
