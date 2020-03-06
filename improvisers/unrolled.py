from typing import Hashable

import attr


@attr.s(frozen=True, auto_attribs=True)
class TimedState:
    val: Hashable
    time: int

    def evolve(self, val2):
        return TimedState(val=val2, time=max(self.time - 1, 0))
