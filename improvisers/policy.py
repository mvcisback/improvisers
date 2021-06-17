"""Module for synthesizing policies from ERCI instances."""

import collections
import math
from typing import Dict, Generator, Optional, Tuple, Union, Sequence

import attr
from scipy.optimize import brentq
from scipy.special import logsumexp

from improvisers.game_graph import Node, GameGraph
from improvisers.critic import Critic, Distribution
from improvisers.tabular import TabularCritic, PolicyState
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
        pstate2 = PolicyState.from_entropy(move, critic, expected_entropy)
        if pstate2 is None:
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
    init: PolicyState

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
        pstate = self.init

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
          critic: Optional[Critic] = None,
          tol: float = 1e-4) -> Actor:
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
        critic = TabularCritic(game, tol=tol)

    fake_psat = min(psat + critic.tol, 1)

    if critic.psat(state, oo) < psat:
        raise ValueError(
            "No improviser exists. Could not reach psat in this MDP"
        )
    elif percent_entropy is not None:
        h0, hinf = critic.entropy(state, 0), critic.entropy(state, oo)
        entropy = percent_entropy * (h0 - hinf) + hinf
        pstate = critic.feasible(state, entropy, fake_psat)
        if pstate is None:
            raise ValueError('No improviser exists!')
    else:
        # Maximum entropy.
        rationality = max(0, critic.match_psat(state, fake_psat))
        entropy = critic.entropy(state, rationality)
        pstate = PolicyState(state, rationality, 0, 1)# TODO: fix!!!!!

    policy_entropy, policy_psat = pstate.pareto_point(critic)

    if policy_entropy < entropy:
        raise ValueError(
            "No improviser exists. Entropy constraint unreachable."
        )

    # TODO: adjust for tolerance.
    if policy_psat < psat:
        raise ValueError(
            "No improviser exists. Could not reach psat in this MDP"
        )

    return Actor(game, critic, pstate)


__all__ = ['solve', 'Actor', 'ImprovProtocol']
