from __future__ import annotations

import time
import random
import re
from itertools import product
from functools import reduce
from typing import Tuple

import aiger
import aiger_ptltl.ptltl as LTL
import aiger_bdd
import aiger_bv as BV
import aiger_discrete as D
import aiger_gridworld as GW
import attr
import funcy as fn
import mdd as MDD
import networkx as nx
from aiger_discrete.mdd import to_mdd
from blessings import Terminal
from dd.cudd import Function  # <---- Make sure CUDD backend available.
from mdd.nx import to_nx

from improvisers.explicit import ExplicitDist
from improvisers.policy import solve
from improvisers.pareto_critic import ParetoCritic
from improvisers.tabular import TabularCritic


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
            buff += TERM.yellow(f'{row} ')
            buff += ' '.join(self._row(row)) + '\n'
        cols = range(self.dim)
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


def battery(n: int=3):
    drain1 = BV.uatom(1, 'drain1')
    drain2 = BV.uatom(1, 'drain2')
    drain = BV.ite(BV.uatom(1, 'choose_drain'), drain1, drain2)
    level = BV.uatom(n, 'lvl')
    update = BV.ite(drain, level >> 1, level).with_output('lvl')
    init = BV.encode_int(n, 1 << n - 1, signed=False)
    return update.aigbv.loopback({
        'input': 'lvl',
        'output': 'lvl',
        'keep_output': True,
        'init': init,
    })


def drone_dynamics(dim: int):
    left, right = left_right(dim)
    p1_circ = player_dynamics('₁', dim, (1, 1))
    p2_circ = player_dynamics('₂', dim, (right + 1, right + 1))

    dyn = (p1_circ | p2_circ) >> state_combiner(dim)
    # No battery
    # return dyn
    # Add battery
    bat = battery(3)
    return dyn | bat


# ---------------- Sensors -------------------- #


def battery_dead(n=3):
    return (BV.uatom(3, 'lvl') == 0).with_output('batteryDead').aigbv


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


def swapped(dim):
    curr = BV.uatom(4*dim, 'state').with_output('state')
    prev = BV.uatom(4*dim, 'prev').with_output('prev')

    s1_curr, s2_curr = curr[:2 * dim], curr[2 * dim:]
    s1_prev, s2_prev = prev[:2 * dim], prev[2 * dim:]
    test = (s1_curr == s2_prev) & (s2_curr == s1_prev)
    test = test.with_output('swapped')
    return (test.aigbv | curr.aigbv).loopback({
        'input': 'prev',
        'output': 'state',
        'keep_output': False,
        'init': 4*dim*(True,)
    })


def feature_sensor(dim):
    state = BV.uatom(4*dim, 'state').with_output('state')

    return state.aigbv \
        |  swapped(dim) \
        |  battery_dead() \
        |  crashed(dim) \
        |  p2_in_goal(dim) \
        |  p2_in_interior(dim) \
        |  goal_vec(dim, 'p1').aigbv \


# ---------------- Specifications ---------------------- #

def dont_crash():
    # Circuit monitoring crashed predicate never occurs.
    test = LTL.parse('H ~((crashed | swapped) | batteryDead)')
    #test = LTL.parse('H ~crashed')
    return BVExpr(test.aigbv)


def visit_all_goals():
    tests = (LTL.parse(f'P goal{i}') for i in range(4))

    # Take conjunction and convert to BitVector expression.
    test = reduce(lambda x, y: x & y, tests)
    test = BVExpr(test.aigbv).bundle_inputs('goals₁') \
                             .with_output('visit_all_goals')
    return test


def visit_all_goals_once():
    # Having visited a goal in the past (P), implies (->) that before
    # that (Z), the goal was never visited (H~).
    tests = (LTL.parse(f'(P goal{i} -> ZH ~goal{i})') for i in range(4))

    # Take conjunction and convert to BitVector expression.
    test = reduce(lambda x, y: x & y, tests)
    return BVExpr(test.aigbv).bundle_inputs('goals₁') \
                             .with_output('visit_all_goals_once')


def guarantees():
    return (dont_crash() & visit_all_goals()).with_output('guarantees').aigbv

# ---------------------- Patrol Policy ----------------------- #

def p2_patrol_policy(dim):
    p2_in_goal = BV.uatom(1, 'p2_in_goal')
    action = BV.uatom(2, 'a₂')

    # ---- 2 player game ----
    #turn_around = BV.uatom(1, '🗘')

    # ---- Stochastic Game (or determinstic below) -----
    turn_around = BV.ite(
        BV.uatom(1, '🗘'),
        BV.uatom(1, '🎲₁'),
        BV.uatom(1, '🎲₂'),
    )

    # ---- MDP case ----
    #dontcare = BV.uatom(1, '🗘') | BV.uatom(1, '🎲₂') | 1
    #turn_around = BV.uatom(1, '🎲₁') & dontcare
    

    update = BV.ite(
        p2_in_goal,
        #BV.ite(turn_around, action + 2, action + 1),  # Stochastic
        BV.ite(turn_around, action + 1, action + 1),  # Deterministic
        action,
    ).with_output('a₂').aigbv

    return update.loopback({
        'input': 'a₂',
        'output': 'a₂',
        'keep_output': True,
        'init': (True, True),
    })


def onehot_gadget(output: str):
    sat = BV.uatom(1, output)
    false, true = BV.uatom(2, 0b01), BV.uatom(2, 0b10)
    expr = BV.ite(sat, true, false) \
             .with_output('sat')
    
    encoder = D.Encoding(
        encode=lambda x: 1 << int(x),
        decode=lambda x: bool((x >> 1) & 1),
    )

    return D.from_aigbv(
        expr.aigbv,
        output_encodings={'sat': encoder},
    )


def const_true(size, name):
    expr = BV.uatom(1, 1)
    expr = BVExpr(expr.aigbv | BV.sink(size, [name]))
    return expr


def monitor2bdd2mdd(monitor, horizon):
    # Convert to BDD.

    pred = monitor.unroll(horizon, only_last_outputs=True) \
                  .aigbv \
                  .cone(f'guarantees##time_{horizon}')

    imap = pred.imap
    order = []
    for t in range(horizon):
        for action in ['a₁', '🗘', '🎲₁', '🎲₂', 'choose_drain', 'drain1', 'drain2']:
            order.extend(imap[f'{action}##time_{t}'])
    order.append(f'sat##time_{horizon}')
    
    bdd, *_ = aiger_bdd.to_bdd(
        pred, 
        levels={n: i for i, n in enumerate(order)}, 
        renamer=lambda _, x: x
    )
    # 1-hot-ify outputs 
    bdd.bdd.add_var('sat[0]')
    bdd.bdd.add_var('sat[1]')
    bdd2 = bdd.implies(bdd.bdd.var('sat[1]')) & (~bdd).implies(bdd.bdd.var('sat[0]'))

    # Create interface.
    imap = monitor.imap
    inputs = []

    for t in range(horizon):
        for name in ['a₁', '🗘', '🎲₁', '🎲₂', 'choose_drain', 'drain1', 'drain2',]:
            size = imap[name].size

            if name in monitor.input_encodings:
                enc = monitor.input_encodings[name]
                encode, decode = enc.encode, enc.decode
            else:
                encode = decode = lambda x: x

            inputs.append(MDD.Variable(
                valid=const_true(size, f'{name}##time_{t}'),
                encode=encode,
                decode=decode,
            ))
    
    output = MDD.Variable(
        valid=const_true(2, 'sat'),
        encode=lambda x: 1 << int(x),
        decode=lambda x: bool((x >> 1) & 1),
    )
    io = MDD.Interface(inputs, output)
    return MDD.DecisionDiagram(io, bdd2)


def _label(name: str, parity: bool=False):
    if name.startswith('sat'):
        return (name == 'sat[1]') ^ parity
    elif name.startswith('a'):
        return 'p1'
    elif name.startswith('🗘'):
        return 'p2'
    elif name.startswith('choose_drain'):
        return 'p2'

    # Is distribution node. Return bias.
    if name.startswith('drain'):
        return 1/100 if name.startswith('drain1') else 1/50

    assert name.startswith('🎲')
    return 0.3 if name.startswith('🎲₁') else 0.7


@attr.s(frozen=True, auto_attribs=True, auto_detect=True, slots=True)
class BState:
    expr: Function
    parity: bool = False 

    @property
    def ref(self) -> int:
        expr = self.expr if self.parity else ~self.expr
        return int(expr)

    def __eq__(self, other):
        return self.ref == other.ref

    def __hash__(self):
        return self.ref

    @property
    def forbidden(self) -> bool:
        return self.expr in (self.expr.bdd.true, self.expr.bdd.false)

    def moves(self):
        for expr in [self.expr.low, self.expr.high]:
            state = BState(expr, expr.negated ^ self.parity)
            if not self.forbidden:
               yield state 

    @property
    def name(self) -> str:
        return self.expr.var

    @property
    def label(self):
        assert not self.forbidden 
        label = _label(self.name, self.parity)
        if not isinstance(label, float):
            return label
        low, high = self.moves()
        return ExplicitDist({low: 1 - label, high: label}) 


@attr.s(frozen=True, auto_attribs=True, auto_detect=True)
class BinaryGameGraph:
    root: Function = attr.ib(converter=BState)

    def moves(self, node):
        return set(node.moves())

    def label(self, node):
        return node.label


def main():
    dim = 5
    horizon = 12

    workspace = drone_dynamics(dim)      # Add dynamics
    workspace >>= feature_sensor(dim)    # Add features
    workspace <<= p2_patrol_policy(dim)  # Constrain p2.
    workspace = workspace.loopback({
        'input': 'p2_in_goal',
        'output': 'p2_in_goal',
        'keep_output': True,
    })

    spec = guarantees()

    monitor = BV.AIGBV.cone(workspace >> spec, 'guarantees')
    # HACK: swap out aig for lazy aig for unrolling.
    # This avoids unnecessary aiger compositions.
    monitor_aigbv = attr.evolve(monitor.circ, aig=monitor.aigbv.aig.lazy_aig)
    monitor = attr.evolve(monitor, circ=monitor_aigbv)
    # End of Hack

    print('building BDD')
    mdd = monitor2bdd2mdd(monitor, horizon)
    print(mdd.bdd.dag_size)
    print('building converting into game graph')

    print('solving game with psat = 0.8')
    game = BinaryGameGraph(mdd.bdd)
    import time
    start = time.time()
    actor = solve(game, psat=0.8, tol=1e-3/horizon)
    print(time.time() - start)


    n_inputs = len(workspace.aig.inputs)
    assert len(game.root.expr.bdd.vars) == horizon * n_inputs + 2

    def sample_path():
        name_parser = re.compile('(.*)##time_(\d*)\[(\d*)]')
        policy = actor.improvise()
        bdd = game.root.expr.bdd
        state = game.root
        obs, path = None, []
        sim = workspace.simulator()
        next(sim)
        codec = workspace.input_encodings

        for t in range(horizon):
            start, end = t * n_inputs, (t + 1) * n_inputs
            circ_input = {}
            for lvl in range(start, end):
                name = bdd.var_at_level(lvl)
                prefix, time, idx = name_parser.match(name).groups()
                key = f"{prefix}[{idx}]" 
                assert time.startswith(str(t))

                dont_care = name != state.name 

                label = _label(name)
                if label != 'p2' and lvl != 0:
                    obs = (state, path)
                    path = []

                if not dont_care and (label == 'p1'):
                    move, _ = policy.send(obs)
                    circ_input[key] = move.expr == state.expr.high
                    state = move
                    continue

                if isinstance(label, float):                # Flip biased coin.
                    bias = label 
                elif label == 'p2':                         # p2 policy. 
                    bias = 1 
                else:
                    bias = 0.5

                decision = random.random() < bias
                moves = list(state.moves())
            
                if dont_care:
                    assert len(moves) == 2
                    circ_input[key] = decision
                    continue

                assert len(moves) in (1, 2)
                if len(moves) == 1:                  # Forced move.
                    circ_input[key] = moves[0] == state.expr.high
                    state = moves[0]
                    continue
                else:
                    circ_input[key] = decision
                    state = moves[int(decision)]

            # Decode input 
            circ_input = workspace.imap.unblast(circ_input)
            for key, val in circ_input.items():
                val = BV.decode_int(val, signed=False)
                if key in codec:
                    val = codec[key].decode(val)
                circ_input[key] = val 
            yield circ_input, sim.send(circ_input)[0]
            

    while True:
        input('run?')

        with TERM.hidden_cursor():
            for i, (actions, output) in enumerate(sample_path()): 
                state = output['state']

                print(f"{TERM.clear}{GridState(dim, state).board}")
                del output['state']
                print(output)
                print(f'time={i}')
                time.sleep(1)


if __name__ == '__main__':
    main()
