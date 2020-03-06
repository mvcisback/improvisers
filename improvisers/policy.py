from typing import Union, Mapping, Tuple, TypeVar

import funcy as fn
import numpy as np
from scipy.special import logsumexp, softmax
#import jax.numpy as np
#from jax.scipy.special import logsumexp
#from jax.nn import softmax
#from jax import grad


State = TypeVar("State")
Action = TypeVar("Action")


# TODO: Move state_val and action_val into unrolled object.
# Enables implementing in a more clever way for simulated systems.


def parametric_policy(composed):
    # Recurse through DFA and compute psat, state_value, action_value.

    actions = tuple(composed.inputs)  # Fix order.
    iter_probs = composed._probs      # Iterator over transition probabilities.
    label = lambda s: float(composed.dfa._label(s))

    def ppolicy(coeff):
        psat: Mapping[State, float] = {}
        values: Mapping[Union[State, Tuple[State, Action]], float] = {}

        @fn.memoize
        def psat(s: State, action=None) -> float:
            if action is not None:
                return sum(psat(s2) * p for s2, p in iter_probs(s, action))
            elif s.time == 0:
                return label(s)
            else:
                sat_probs = np.array([psat(s, a) for a in actions])
                action_probs = softmax(action_vals(s))
                return sat_probs @ action_probs

        @fn.memoize
        def state_val(s: State) -> float:
            if s.time == 0:
                return coeff * label(s)
            else:
                return logsumexp(action_vals(s))

        def action_vals(s: State):
            # TODO: map => vmap
            return np.array([action_val(s, a) for a in actions])

        @fn.memoize
        def action_val(s: State, a: Action) -> float:
            return sum(state_val(s2) * p for s2, p in iter_probs(s, a))

        return psat(composed.start), (state_val, action_val)

    return ppolicy
