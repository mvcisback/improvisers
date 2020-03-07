from typing import Union, Mapping, Tuple, TypeVar

import funcy as fn
import numpy as np
from scipy.special import logsumexp
#import jax.numpy as np
#from jax.scipy.special import logsumexp
#from jax.nn import softmax
#from jax import grad


State = TypeVar("State")
Action = TypeVar("Action")
oo = float('inf')


# TODO: Move state_val and action_val into unrolled object.
# Enables implementing in a more clever way for simulated systems.


def parametric_policy(composed):
    # Recurse through DFA and compute psat, state_value, action_value.

    actions = tuple(composed.inputs)  # Fix order.
    iter_probs = composed._probs      # Iterator over transition probabilities.

    # Remove actions that lead to hard constraint.

    label = lambda s: float(composed.dfa._label(s)[0])
    valid = lambda s: float(composed.dfa._label(s)[1])

    def ppolicy(coeff):
        psat: Mapping[State, float] = {}
        values: Mapping[Union[State, Tuple[State, Action]], float] = {}

        @fn.memoize
        def psat(s: State) -> float:
            assert valid(s), "psat undefined for invalid states"

            if s.time == 0:
                return label(s)

            acc = 0
            for a in actions:
                prob_a = prob(a, s)
                if prob_a == 0:
                    continue

                acc += prob_a * sum(psat(s2) * p for s2, p in iter_probs(s, a))

            return acc

        @fn.memoize
        def prob(a: Action, s: State) -> float:
            assert valid(s), "action probs undefined for invalid states."
            action_prob = np.exp(action_val(s, a) - state_val(s))
            assert 0 <= action_prob <= 1
            return action_prob

        @fn.memoize
        def state_val(s: State) -> float:
            if not valid(s):
                return -oo
            elif s.time == 0:
                return coeff * label(s)
            else:
                return logsumexp([action_val(s, a) for a in actions])

        @fn.memoize
        def action_val(s: State, a: Action) -> float:
            acc = 0
            for s2, p in iter_probs(s, a):
                if not valid(s2):
                    return -oo
                acc += state_val(s2)

            return acc

        return psat(composed.start), (state_val, action_val)

    return ppolicy
