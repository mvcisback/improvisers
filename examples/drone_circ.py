from __future__ import annotations

from itertools import product
from typing import Tuple

import aiger_ptltl as LTL
import aiger_bv as BV
import aiger_discrete as D
import aiger_gridworld as GW
import attr
from blessings import Terminal


TERM = Terminal()


def left_right(dim: int) -> Tuple[int, int]:
    if dim == 3:
        return 2, 2
    return dim // 3 + 1, (2*dim) // 3 + 1


# ----------------- Visualization --------------------- #


def one_index(val):
    assert val != 0
    count = 0
    while True:
        if (val >> count) & 1:
            break
        count += 1
    return count


@attr.s(auto_attribs=True, frozen=True)
class Point:
    x: int
    y: int


@attr.s(auto_attribs=True, frozen=True)
class GridState:
    dim: int
    yxyx: int

    def to_point(self, yx: int) -> Point:
        y = one_index(yx)
        x = one_index(yx >> self.dim)
        return Point(x, y)

    @property
    def p1(self) -> Point:
        yx = self.yxyx & ~(-1 << (2 * self.dim))
        return self.to_point(yx)

    @property
    def p2(self):
        yx = self.yxyx >> (2 * self.dim)
        return self.to_point(yx)

    def _cell(self, p: Point) -> str:
        left, right = left_right(self.dim)
        if p == self.p1 == self.p2:  # Crash
            return '*'
        elif p == self.p1:
            return f'{TERM.white("x")}'
        elif p == self.p2:
            return TERM.red('+')
        elif p.x in (left - 1, right - 1) and \
             p.y in (left - 1, right - 1):
            return '▪'
        return '□'

    def _row(self, row: int) -> Iterator[str]:
        for col in range(self.dim):
            yield self._cell(Point(row, col))

    @property
    def board(self) -> str:
        buff = '\n'

        for row in range(self.dim-1, -1, -1):
            buff += TERM.yellow(f'{row + 1} ')
            buff += ' '.join(self._row(row)) + '\n'
        cols = range(1, self.dim + 1)
        buff += TERM.yellow('  ' + ' '.join(map(str, cols)) + '\n')
        return buff

# ------------------- Drone Dynamics --------------------------------#

def player_dynamics(suffix: str, dim: int, start: Tuple[int, int]):
    gw = GW.gridworld(dim, start=start, compressed_inputs=True)
    # Relabel i/o to be specific to player.
    gw = gw['i', {'a': f'a{suffix}'}]
    gw = gw['o', {'state': f'state{suffix}'}]
    gw = gw['l', {'x': f'x{suffix}', 'y': f'y{suffix}'}]
    return gw


def state_combiner(dim: int):
    size = 2 * dim
    s1, s2 = BV.uatom(size, 'state₁'), BV.uatom(size, 'state₂')
    return s1.concat(s2).with_output('state').aigbv


def drone_dynamics(dim: int):
    _, right = left_right(dim)
    p1_circ = player_dynamics('₁', dim, (1, 1))
    p2_circ = player_dynamics('₂', dim, (right + 1, right + 1))
    return (p1_circ | p2_circ) >> state_combiner(dim)


# ---------------- Specification Monitors ---------------------- #

def dont_crash(dim: int):
    s12 = BV.uatom(4*dim, 'state')
    s1, s2 = s12[:2*dim], s12[2*dim:]

    # Circuit determining crashed predicate.
    crashed = (s1 == s2).with_output('crashed') \
                        .aigbv
    
    # Circuit monitoring crashed predicate never occurs.
    monitor = (~LTL.atom('crashed')).historically().aig
    monitor = BV.aig2aigbv(monitor)  # Need to lift to work on bitvectors.

    return crashed >> monitor


def goals(dim: int, p1: bool):
    s12 = BV.uatom(4 * dim, 'state')
    state = s12[:2*dim] if p1 else s12[2*dim:]

    left, right = left_right(dim)
    for i in range(4):
        x = left if i & 1 else right
        y = left if i & 2 else right
        goal_xy = (1 << x) | (1 << (y + dim))
        yield (state == goal_xy)


def p2_avoids_goals(dim: int):
    s12 = BV.uatom(2 * dim, 'state')
    s2 = s12[dim:]


    # TODO
    raise NotImplementedError


# -------------------------------------------------------------


def main():
    dim = 7

    world = drone_dynamics(dim)

    actions = {'a₁': '→', 'a₂': '↓'}
    state = world(actions)[0]['state']
    print(GridState(dim, state).board)
    g = list(goals(dim, True))
    
    breakpoint()


if __name__ == '__main__':
    main()
