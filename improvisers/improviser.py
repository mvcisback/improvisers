import probabilistic_automata as PA
from scipy.optimize import brentq, newton
from dfa import DFA
from dfa.utils import universal, empty
from probabilistic_automata.utils import tee
from probabilistic_automata import PDFA

from improvisers.policy import parametric_policy, ParametricPolicy
from improvisers.unrolled import unroll


def improviser(
        horizon: int, dyn: PDFA, sat_prob=0.9, soft=True, hard=True,
) -> PDFA:
    outputs = dyn.outputs

    def _to_pdfa(obj):
        if isinstance(obj, PDFA):
            return obj
        elif isinstance(obj, DFA):
            return PA.lift(obj)
        elif obj is True:
            return PA.lift(universal(outputs))
        elif obj is False:
            return PA.lift(empty(outputs))

        # TODO: support PTLTL and AIG circuits.
        raise NotImplementedError

    dyn, soft, hard = map(_to_pdfa, [dyn, soft, hard])

    composed = dyn >> tee(soft, hard)
    unrolled = unroll(horizon, composed)

    ppolicy = IP.parametric_policy(unrolled)

    if sat_prob is None:
        return ppolicy

    return fit(ppolicy, sat_prob)


def fit(ppolicy, sat_prob, top=100):
    def f(coeff):
        return ppolicy(coeff).psat() - sat_prob

    if f(-top) > 0:
        coeff = 0
    elif f(top) < 0:
        coeff = top
    else:
        coeff = brentq(f, -top, top, xtol=1e-3, rtol=1e-3)

    return ppolicy(coeff)
