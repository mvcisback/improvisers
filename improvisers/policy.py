"""Module for synthesizing policies from ERCI instances."""

import collections
import math
import random
from typing import Dict, Generator, Optional, Tuple, Union, Sequence

import attr
from scipy.optimize import brentq
from scipy.special import logsumexp

from improvisers.game_graph import Node, GameGraph
from improvisers.critic import Critic, Distribution
from improvisers.tabular import TabularCritic
from improvisers.explicit import ExplicitDist


oo = float('inf')
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


@attr.s(auto_attribs=True, frozen=True)
class PolicyState:
    node: Node
    coeff1: float
    coeff2: float = 0
    prob: float = 1

    def sample_coeff(self) -> float:
        return self.coeff1 if random.random() <= self.prob else self.coeff2
    
    def pareto_point(self, critic) -> Tuple[float, float]:
        entropy = self.prob * critic.entropy(self.node, self.coeff1)
        entropy += (1 - self.prob) * critic.entropy(self.node, self.coeff2)

        psat = self.prob * critic.psat(self.node, self.coeff1)
        psat += (1 - self.prob) * critic.psat(self.node, self.coeff2)

        return (entropy, psat)


def replan(pstate: PolicyState, critic: Critic, dist2: Dist) -> float:
    """Replan based on observed state distribution.

    Args:
    - coeff: Current rationality coefficient.
    - critic: Critic to the current stochastic game.
    - dist2: Actual next state distribution.

    Returns:
      Rationality coefficient induced by actual state distribution.
    """
    expected_entropy, expected_psat = pstate.pareto_point(critic) 

    observed_psat = 0
    for node in dist2.support():
        observed_psat += dist2.prob(node) * pstate.pareto_point(critic)[1]
        
    # There must exist a deterministic p2 move whose p1 replan suffices. 
    for move in critic.game.moves(pstate.node):
        curve = critic.curve(move)
        left, right = curve.find_edge(entropy=expected_entropy)
        entropy_left = critic.entropy(move, left)
        entropy_right = critic.entropy(move, right)

        if not (entropy_right <= expected_entropy <= entropy_left):
            continue

        if entropy_left == entropy_right:
            prob = 1
        else:
            prob = (expected_entropy - entropy_right) / (entropy_left - entropy_right)
        assert 0 <= prob <= 1
        pstate2 = PolicyState(move, left, right, prob)
        new_entropy, new_psat = pstate2.pareto_point(critic)

        if new_psat < observed_psat:
            continue

        return pstate2

    raise RuntimeError("Replanning Failed! This is a bug, please report.")


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

        if (label == 'p1') or isinstance(label, bool):
            prev_lprob = dist.get(node, 0.0)
            dist[node] = logsumexp([prev_lprob, lprob], b=[0.5, 0.5])
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
        game, critic = self.game, self.critic
        pstate = PolicyState(game.root, self.rationality)

        while not isinstance(game.label(pstate.node), bool):
            rationality = pstate.sample_coeff()
            move = critic.move_dist(pstate.node, rationality).sample()
            pstate = PolicyState(move, rationality) 
            state_dist = critic.state_dist(move, rationality)

            # Replan and update state.
            state2, obs = yield move, state_dist

            if isinstance(obs, collections.Sequence):
                # Observed partial p2 path. All unobserved suffixes
                # assume worst case entropy policy!
                state_dist2 = from_p2_path(
                    game, critic, (move, rationality), state2, obs
                )
            else:
                state_dist2 = obs

            pstate = replan(pstate, critic, state_dist2)
            pstate = attr.evolve(pstate, node=state2)

        return bool(game.label(pstate.node))


def solve(game: GameGraph,
          psat: float = 0,
          percent_entropy: Optional[float] = None,
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

    fake_psat = min(psat + critic.tol, 1)

    if critic.psat(state, oo) < psat:
        raise ValueError(
            "No improviser exists. Could not reach psat in this MDP"
        )
    elif percent_entropy is not None:
        h0, hinf = critic.entropy(state, 0), critic.entropy(state, oo)
        entropy = percent_entropy * (h0 - hinf) + hinf
        rationality = critic.feasible(state, entropy, fake_psat)
        if rationality is None:
            raise ValueError('No improviser exists!')
    else:
        # Maximum entropy.
        rationality = max(0, critic.match_psat(state, fake_psat))
        entropy = critic.entropy(state, rationality)

    if critic.entropy(state, rationality) < entropy:
        raise ValueError(
            "No improviser exists. Entropy constraint unreachable."
        )

    # TODO: adjust for tolerance.
    if critic.psat(state, rationality) < psat:
        raise ValueError(
            "No improviser exists. Could not reach psat in this MDP"
        )

    return Actor(game, critic, rationality)


__all__ = ['solve', 'Actor', 'ImprovProtocol']
