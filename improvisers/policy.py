from typing import Union, Mapping, Tuple, TypeVar, NamedTuple, Optional

import attr
import funcy as fn
import numpy as np
import probabilistic_automata as PA
from scipy.special import logsumexp
from dfa import SupAlphabet

#import jax.numpy as np
#from jax.scipy.special import logsumexp
#from jax.nn import softmax
#from jax import grad


State = TypeVar("State")
Action = TypeVar("Action")
oo = float('inf')


class StateAction(NamedTuple):
    state: State
    action: Optional[Action]


@attr.s(frozen=True, auto_attribs=True)
class Policy:
    coeff: float
    dyn: PA.PDFA

    @property
    def actions(self):
        return self.dyn.inputs

    def valid(self, s: State) -> bool:
        return self.dyn.dfa._label(s)[1]

    def label(self, s: State) -> float:
        return float(self.dyn.dfa._label(s)[0])

    def _next_state_dist(self, s: State, a: Action):
        return self.dyn._probs(s, a)

    @fn.memoize
    def value(self, s: State, a: Optional[Action] = None) -> float:
        if a is None:
            if not self.valid(s):
                return -oo
            elif s.time == 0:
                return self.coeff * self.label(s)
            elif s.time < 0:
                return 0

            return logsumexp([self.value(s, a) for a in self.actions])

        acc = 0
        for s2, p in self._next_state_dist(s, a):
            if not self.valid(s2):
                return -oo
            acc += self.value(s2)

        return acc

    @fn.memoize
    def __call__(self, action: Action, state: State) -> float:
        assert self.valid(state), "action probs undefined for invalid states."
        action_prob = np.exp(self.value(state, action) - self.value(state))
        assert 0 <= action_prob <= 1
        return action_prob

    @fn.memoize
    def psat(self, s: Optional[State] = None) -> float:
        if s is None:
            s = self.dyn.start

        assert self.valid(s), "psat undefined for invalid states."

        if s.time == 0:
            return self.label(s)

        acc = 0
        for a in self.actions:
            prob_a = self(a, s)
            if prob_a == 0:
                continue

            state_probs = self._next_state_dist(s, a)
            acc += prob_a * sum(self.psat(s2) * p for s2, p in state_probs)

        return acc


# Enables implementing in a more clever way for simulated systems.


def parametric_policy(composed):
    def ppolicy(coeff):
        policy = Policy(coeff=coeff, dyn=composed)

        def policy_transition(_, composite_action):
            curr_state, next_action = composite_action
            return StateAction(curr_state, next_action)

        def policy_dist(s, _):
            return {a: prob(a, s) for a in composed.inputs}

        """
        policy_pdfa = PA.pdfa(
            start=StateAction(composed.start, None),
            label=lambda s: s.action,
            transition=policy_transition,
            inputs=SupAlphabet(),            # Surrogate for state observation.
            env_inputs=composed.inputs,
            env_dist=policy_dist,
        )
        """

        return policy

    return ppolicy
