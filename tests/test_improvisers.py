import operator as op

import pytest
import probabilistic_automata as PA
from dfa import DFA
from dfa.utils import tee, universal

from improvisers.improviser import fit, improviser
from improvisers.unrolled import unroll
from improvisers.policy import parametric_policy


def test_parametric_policy_smoke():
    hard = universal({0, 1})
    soft = DFA(start=True, label=bool, transition=op.and_, inputs={0, 1})
    dyn = tee(soft, hard)

    assert len(dyn.states()) == 2

    unrolled = unroll(3, PA.lift(dyn))
    ppolicy = parametric_policy(unrolled)
    
    psat, (state_val, action_val) = ppolicy(2)
    assert 0 < float(psat) < 1

    for s in unrolled.states():
        if s.time == 0:
            assert float(state_val(s)) in (0.0, 2.0)
        elif s.val[0]:
            assert float(action_val(s, False)) < float(action_val(s, True))
            assert float(action_val(s, True)) < float(state_val(s))
        else:
            assert float(action_val(s, False)) == float(action_val(s, True))
            assert float(action_val(s, True)) < float(state_val(s))

    psat, (state_val, action_val) = fit(ppolicy, 0.8)
    assert pytest.approx(0.8, rel=1e-3) == float(psat)
    

def test_invalid_states():
    hard = DFA(start=True, label=bool, transition=op.and_, inputs={0, 1})
    soft = DFA(start=True, label=bool, transition=op.and_, inputs={0, 1})
    dyn = tee(soft, hard)
    unrolled = unroll(3, PA.lift(dyn))
    ppolicy = parametric_policy(unrolled)
    
    psat, (state_val, action_val) = fit(ppolicy, 1)

    assert psat == 1
