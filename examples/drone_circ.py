from __future__ import annotations

import time
from itertools import product
from functools import reduce
from typing import Tuple

import aiger_ptltl.ptltl as LTL
import aiger_bv as BV
import aiger_discrete as D
import aiger_gridworld as GW
import attr
from blessings import Terminal


TERM = Terminal()
BVExpr = BV.UnsignedBVExpr


def left_right(dim: int) -> Tuple[int, int]:
    if dim == 3:
        return 2, 2
    return dim // 3, (2*dim) // 3


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
        elif p.x in (left, right) and \
             p.y in (left, right):
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
    left, right = left_right(dim)
    p1_circ = player_dynamics('₁', dim, (1, 1))
    p2_circ = player_dynamics('₂', dim, (right + 1, right + 1))
    return (p1_circ | p2_circ) >> state_combiner(dim)


# ---------------- Sensors -------------------- #

def crashed(dim):
    s12 = BV.uatom(4*dim, 'state')
    s1, s2 = s12[:2*dim], s12[2*dim:]
    return (s1 == s2).with_output('crashed').aigbv


def goal_vec(dim: int, player: str):
    s12 = BV.uatom(4 * dim, 'state')
    state = s12[:2*dim] if player == 'p1' else s12[2*dim:]

    left, right = left_right(dim)

    vec = BV.uatom(0, 0)
    for i in range(4):
        x = left if i & 1 else right
        y = left if i & 2 else right
        yx = (1 << x) | (1 << (y + dim))
        vec = vec.concat(state == yx)
    suffix = "₁" if player == "p1" else "₂"
    return vec.with_output(f'goals{suffix}')


def p2_in_goal(dim: int):
    test = goal_vec(dim, "p2") != 0
    return test.with_output('p2_in_goal').aigbv


def p2_in_interior(dim: int):
    s12 = BV.uatom(4 * dim, 'state')
    s2 = s12[2*dim:]

    # Take union over all interior points.
    left, right = left_right(dim)
    test = BV.uatom(1, 0)
    interior = product(range(left + 1, right), range(left + 1, right))
    for x, y in interior:
        yx = (1 << x) | (1 << (y + dim))
        test |= (s2 == yx)
    return test.with_output('p2_in_interior').aigbv


def feature_sensor(dim):
    state = BV.uatom(4*dim, 'state').with_output('state')
    return state.aigbv \
        |  crashed(dim) \
        |  p2_in_goal(dim) \
        |  p2_in_interior(dim) \
        |  goal_vec(dim, 'p1').aigbv


# ---------------- Specifications ---------------------- #

def dont_crash():
    # Circuit monitoring crashed predicate never occurs.
    test = LTL.parse('H ~crashed')
    return BVExpr(test.aigbv)


def visit_all_goals():
    tests = (LTL.parse(f'P goal{i}') for i in range(4))

    # Take conjunction and convert to BitVector expression.
    test = reduce(lambda x, y: x & y, tests)
    return BVExpr(test.aigbv).bundle_inputs('goals₁') \
                             .with_output('visit_all_goals')


def visit_all_goals_once():
    # Having visited a goal in the past (P), implies (->) that before
    # that (Z), the goal was never visited (H~).
    tests = (LTL.parse(f'(P goal{i} -> ZH ~goal{i})') for i in range(4))

    # Take conjunction and convert to BitVector expression.
    test = reduce(lambda x, y: x & y, tests)
    return BVExpr(test.aigbv).bundle_inputs('goals₁') \
                             .with_output('visit_all_goals_once')


def guarantees():
    return (dont_crash() & visit_all_goals()).with_output('guarantees')

# ---------------------- Shield Synth ------------------------ #


# ---------------------- Patrol Policy ----------------------- #

def p2_patrol_policy(dim):
    p2_in_goal = BV.uatom(1, 'p2_in_goal')
    action = BV.uatom(2, 'a₂')
    turn_around = BV.uatom(1, '🗘')

    update = BV.ite(
        p2_in_goal,
        BV.ite(turn_around, action + 2, action + 1),
        action,
    ).with_output('a₂').aigbv

    return update.loopback({
        'input': 'a₂',
        'output': 'a₂',
        'keep_output': True,
        'init': (1, 1),
    })
    

# ------------------------------------------------------------ #


def main():
    dim = 7

    dyn = drone_dynamics(dim)
    sensor = feature_sensor(dim)
    p2_ctrl = p2_patrol_policy(dim)

    dyn2 = (dyn << p2_ctrl) >> sensor
    dyn2 = dyn2.loopback({
        'input': 'p2_in_goal',
        'output': 'p2_in_goal',
        'keep_output': True,
    })

    sim = dyn2.simulator()
    next(sim)

    with TERM.hidden_cursor():
        for i in range(10):
            actions = {'a₁': '→', '🗘': 0}
            output = sim.send(actions)[0]

            state = output['state']
            print(f"{TERM.clear}{GridState(dim, state).board}")
            del output['state']
            print(output)
            time.sleep(0.4)


if __name__ == '__main__':
    main()
