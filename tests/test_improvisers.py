import operator as op

import pytest
import probabilistic_automata as PA
from dfa import DFA
from dfa.utils import tee, universal
from probabilistic_automata.utils import tee as tee2

from improvisers.improviser import fit, improviser
from improvisers.unrolled import unroll
from improvisers.policy import parametric_policy


DELAY = DFA(
    start=1,
    label=lambda s: s, 
    transition=lambda _, c: c,
    inputs={0, 1},
    outputs={0, 1},
)
UNIVERSAL = universal({0, 1})
ALWAYS_1 = DFA(start=1, label=bool, transition=min, inputs={0, 1})


def test_smoke():
    dyn = tee(ALWAYS_1, UNIVERSAL)
    assert len(dyn.states()) == 2
    unrolled = unroll(3, PA.lift(dyn))
    ppolicy = parametric_policy(unrolled)
    policy = ppolicy(2)
    policy.value(unrolled.start)

    for s in unrolled.states():
        if s.time == 0:
            assert policy.value(s) in (0.0, 2.0)
        elif s.val[0]:
            assert policy.value(s, False) < policy.value(s, True)
            assert policy.value(s, True) < policy.value(s)
        else:
            assert policy.value(s, False) == policy.value(s, True)
            assert policy.value(s, True) < policy.value(s)

    assert 0 < policy.psat() < 1
    assert pytest.approx(0.8, rel=1e-3) == fit(ppolicy, 0.8).psat()
    

def test_invalid_states():
    hard = DFA(start=True, label=bool, transition=op.and_, inputs={0, 1})
    soft = DFA(start=True, label=bool, transition=op.and_, inputs={0, 1})
    dyn = tee(soft, hard)
    unrolled = unroll(3, PA.lift(dyn))
    ppolicy = parametric_policy(unrolled)
    
    assert fit(ppolicy, 1).psat() == 1
