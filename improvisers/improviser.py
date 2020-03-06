import probabilistic_automata as PA
from scipy.optimize import brentq, newton
from dfa import DFA
from probabilistic_automata import PDFA

from improvisers.policy import parametric_policy
from improvisers.unrolled import unroll


UNIVERSAL = PA.lift(DFA(start=True, label=bool, transition=lambda *_: True))
EMPTY = PA.lift(DFA(start=False, label=bool, transition=lambda *_: False))


def to_pdfa(obj):
    if isinstance(obj, PDFA):
        return obj
    elif isinstance(obj, DFA):
        return PA.lift(obj)
    elif obj is True:
        return Universal
    elif obj is False:
        return EMPTY

    # TODO: support PTLTL and AIG circuits.
    raise NotImplementedError


def improviser(
        horizon: int, dyn: PDFA, sat_prob=0.9, 
        soft_constraint=None,
        hard_constraint=None,
        
) -> PDFA:
    soft_constraint = to_pdfa(soft_constraint)
    hard_constraint = to_pdfa(hard_constraint)

    composed = dyn >> (soft_constraint | hard_constraint)
    unrolled = composed.unroll(horizon, composed)

    ppolicy = IP.parametric_policy(unrolled)
    return fit(ppolicy, sat_prob)


def fit(ppolicy, sat_prob, top=100):
    def f(coeff):
        return float(ppolicy(coeff)[0]) - sat_prob

    if f(-top) > 0:
        coeff = 0
    elif f(top) < 0:
        coeff = top
    else:
        coeff = brentq(f, -top, top, xtol=1e-3, rtol=1e-3)

    return ppolicy(coeff)
