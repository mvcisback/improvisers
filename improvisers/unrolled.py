from typing import Hashable

import attr
import probabilistic_automata as PA


@attr.s(frozen=True, auto_attribs=True)
class TimedState:
    val: Hashable
    time: int

    def evolve(self, val2):
        return TimedState(val=val2, time=max(self.time - 1, 0))


def unroll(horizon, dyn):
    return PA.pdfa(
        start=TimedState(val=dyn.start, time=horizon),
        label=lambda s: dyn.dfa._label(s.val),
        transition=lambda s, c: s.evolve(dyn.dfa._transition(s.val, c)),
        inputs=dyn.inputs,
        outputs=dyn.outputs,
        env_dist=dyn.env_dist,
        env_inputs=dyn.env_inputs,
    )
    
