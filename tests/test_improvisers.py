import probabilistic_automata as PA
from dfa import DFA

from improvisers.unrolled import TimedState
from improvisers.policy import parametric_policy


def test_parametric_policy_smoke():
    composed = PA.lift(DFA(
        start=TimedState(val=True, time=3),
        label=lambda s: s.val,
        transition=lambda s, c: s.evolve(s.val and c),
        inputs={True, False},
    ))
    ppolicy = parametric_policy(composed)
    
    psat, (state_val, action_val) = ppolicy(2)
    assert 0 < float(psat) < 1

    for s in composed.states():
        if s.time == 0:
            assert float(state_val(s)) in (0.0, 2.0)
        elif s.val:
            assert float(action_val(s, False)) < float(action_val(s, True))
            assert float(action_val(s, True)) < float(state_val(s))
        else:
            assert float(action_val(s, False)) == float(action_val(s, True))
            assert float(action_val(s, True)) < float(state_val(s))

