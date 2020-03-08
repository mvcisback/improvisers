from __future__ import annotations

from typing import Callable, NamedTuple, Optional
from typing import Union, Mapping, Tuple, TypeVar

import attr
import funcy as fn
import numpy as np
import probabilistic_automata as PA
from scipy.special import logsumexp
from dfa import SupAlphabet

from improvisers.unrolled import TimedState


State = TypeVar("State")
Action = TypeVar("Action")
oo = float('inf')


class PolicyState(NamedTuple):
    state: TimedState
    prev_action: Optional[Action] = None

    @property
    def time(self) -> int:
        return self.state.time

    def evolve(self, s: State, a: Action) -> PolicyState:
        return PolicyState(state=self.evolve(s), prev_action=a)


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

    def as_pdfa(self) -> PA.PDFA:
        def policy_transition(prev_state, composite_action):
            curr_state, action = composite_action
            return prev_state.evolve(curr_state, action)

        def policy_dist(policy_state, sys_state):
            timed_state = TimedState(val=sys_state, timed=policy_state.time)
            return {a: prob(a, timed_state) for a in self.actions}

        return PA.pdfa(
            start=PolicyState(self.dyn.start),
            label=lambda s: s.prev_action,    
            transition=policy_transition,
            inputs=SupAlphabet(),            # Proxy for state observations.
            outputs=self.actions,
            env_dist=policy_dist,
            env_inputs=self.actions,
        )

    def run(self, start=None, seed=None):
        return self.as_pdfa().run(start=start, seed=seed, label=True)


ParametricPolicy = Callable[[float], Policy]


def parametric_policy(composed: PA.PDFA) -> ParametricPolicy:
    return lambda coeff: Policy(coeff=coeff, dyn=composed)
