"""Module for synthesizing policies from ERCI instances."""

import collections
import math
from typing import Dict, Generator, Optional, Tuple, Union, Sequence

import attr
from scipy.optimize import brentq
from scipy.special import logsumexp

from improvisers.game_graph import Node, GameGraph
from improvisers.critic import Critic, Distribution
from improvisers.tabular import TabularCritic
from improvisers.explicit import ExplicitDist


Game = GameGraph
Dist = Distribution
State = Tuple[Node, float]  # Policy State = current node + rationality.
Path = Sequence[Node]
Observation = Union[
    Dist,            # Provide next state distribution.
    Path,            # Observe player 2 path. Worst case counter-factuals.
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


def from_p2_path(game: Game,
                 critic: Critic,
                 state: State,
                 target: Node,
                 path: Optional[Path]) -> Dist:
    """Returns the worst case state distribution given observed path."""
    node, rationality = state
    dist: Dict[Node, float] = {}

    stack = [(node, path, 0.0)]
    while stack:
        node, path, lprob = stack.pop()
        label = game.label(node)

        if path == [] and node != target:
            raise NotImplementedError("Do not support partial paths yet.")

        if (label == 'p1') or isinstance(label, bool):
            prev_lprob = dist.get(node, 0.0)
            dist[node] = logsumexp([prev_lprob, lprob])
        elif label == 'p2':
            if path and (node == path[0]):  # Conform to observed path.
                node2, *path = path
            else:
                path = None  # Start counter-factual.
                node2 = critic.min_ent_move(node, rationality)

            stack.append((node2, path, lprob))
        else:  # Environment case. label is a distribution.
            for node2 in label.support():
                lprob2 = lprob + math.log(label.prob(node2))
                stack.append((node2, path, lprob2))

    # Convert log probs into probs and return.
    return ExplicitDist({n: math.exp(lprob) for n, lprob in dist.items()})


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

            if isinstance(obs, collections.Sequence):
                # Observed partial p2 path. All unobserved suffixes
                # assume worst case entropy policy!
                pstate = (move, rationality)  # Policy State.
                state_dist2 = from_p2_path(game, critic, pstate, state2, obs)
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
