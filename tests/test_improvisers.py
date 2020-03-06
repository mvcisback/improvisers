import operator as op

import pytest
import probabilistic_automata as PA
from dfa import DFA

from improvisers.improviser import fit, improviser
from improvisers.unrolled import unroll
from improvisers.policy import parametric_policy


def test_parametric_policy_smoke():
    dyn = DFA(start=True, label=bool, transition=op.and_, inputs={True, False})

    unrolled = unroll(3, PA.lift(dyn))
    ppolicy = parametric_policy(unrolled)
    
    psat, (state_val, action_val) = ppolicy(2)
    assert 0 < float(psat) < 1

    for s in unrolled.states():
        if s.time == 0:
            assert float(state_val(s)) in (0.0, 2.0)
        elif s.val:
            assert float(action_val(s, False)) < float(action_val(s, True))
            assert float(action_val(s, True)) < float(state_val(s))
        else:
            assert float(action_val(s, False)) == float(action_val(s, True))
            assert float(action_val(s, True)) < float(state_val(s))

    psat, (state_val, action_val) = fit(ppolicy, 0.8)
    assert pytest.approx(0.8, rel=1e-3) == float(psat)
    
