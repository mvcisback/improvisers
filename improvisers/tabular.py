"""This module contains the tabular Critic implementation."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Hashable, List, Tuple, Dict, Callable, Optional, Literal
from typing import Union
from functools import lru_cache, partial

import attr
import numpy as np
from scipy.special import logsumexp, softmax
from scipy.optimize import brentq
from sortedcontainers import SortedDict

from improvisers.game_graph import GameGraph, Node
from improvisers.critic import Critic, Distribution, DistLike
from improvisers.explicit import ExplicitDist as Dist


oo = float('inf')
RealFunc = Callable[[float], float]
Feasibile = Optional[float]
SearchCMD = Union[
    Literal['increase'], 
    Literal['decrease'], 
    None,   # Reject
    float,  # Accept
]


# https://stackoverflow.com/questions/29045162/binary-search-of-reversed-sorted-list-in-python
def bisect(arr, val, cmp):
    l = -1
    r = len(arr)
    while r - l > 1:
        e = (l + r) >> 1
        if cmp(arr[e], val): l = e
        else: r = e
    return r


def binary_search(
        f: RealFunc, lo: float, hi: float, eps: float = 1e-2
) -> float:
    tmp = f(lo)
    if tmp == 0:
        return lo
    elif tmp > 0:
        raise ValueError
    tmp = f(hi)
    if tmp == 0:
        return hi
    elif tmp < 0:
        raise ValueError

    while hi - lo > eps:
        mid = lo + (hi - lo) / 2
        tmp = f(mid)
        if tmp == 0:
            return mid        
        lo, hi = (lo, mid) if tmp > 0 else (mid, hi)
    return lo + (hi - lo) / 2


@attr.s(auto_attribs=True, auto_detect=True, frozen=True, slots=True)
class PPoint:
    entropy: float
    psat: float


@attr.s(auto_attribs=True, auto_detect=True, frozen=True, order=False)
class Itvl:
    low: float
    high: float

    def __lt__(self, other: Itvl):
        return self.high < other.low

    @property
    def size(self) -> float:
        return self.high - self.low



def _psat(lsat: float) -> float:
    sat_prob = math.exp(lsat)
    assert sat_prob < 1.2
    return min(sat_prob, 1)  # Clip at 1 due to numerics.


def psat(lsat: Itvl) -> Itvl:
    return Itvl(_psat(lsat.low), _psat(lsat.high))

    
@attr.s(auto_attribs=True, auto_detect=True, frozen=True)
class ParetoCurve:
    entropies: Mapping[float, float] = attr.ib(factory=SortedDict)
    lsats: Mapping[float, Itvl] = attr.ib(factory=SortedDict)

    @staticmethod
    def new(node: Node, critic: Critic) -> ParetoCurve:
        curve = ParetoCurve()
        curve.entropies[0] = critic._entropy(node, 0)
        curve.entropies[oo] = critic._entropy(node, oo)
        curve.lsats[0] = critic._lsat(node, 0)
        curve.lsats[oo] = critic._lsat(node, oo)
        return curve

    def __getitem__(self, key: float) -> PPoint:
        if key in self:
            raise KeyError
        return PPoint(self.entropies[key], self.lsats[key])

    def __contains__(self, key: float) -> bool:
        return (key in self.entropies) and (key in self.psats)

    def entropy_bounds(self, key: float) -> Itvl:
        if key in self.entropies:
            ent = self.entropies[key]
            return Itvl(ent, ent)

        # Compute montonicity bound.
        idx = self.entropies.bisect_left(key)
        size = len(self.entropies)
        assert idx not in (0, size)

        low = self.entropies.values()[idx]
        high = self.entropies.values()[idx - 1]
        return Itvl(low, high)

    def lsat_bounds(self, key: Optional[float] = None, entropy: Optional[float] = None) -> Itvl:
        if key in self.lsats:
            return self.lsats[key]
        if (key is None) == (entropy is None):
            raise ValueError
        elif key is None:
            edge = self.psat_edge(entropy)
            if edge[0] == edge[1]:
                return self.lsats[edge[0]]
            
            raise NotImplementedError
        
        raise NotImplementedError

    def psat_bounds(self, key: Optional[float] = None, entropy: Optional[float] = None) -> Itvl:
        return psat(self.lsat_bounds(key, entropy))

    def psat_edge(self, entropy: float) -> Tuple[float, float]:
        entropies = self.entropies.values()
        coeffs = self.entropies.keys()
        idx = bisect(entropies, entropy, lambda x, y: x > y)
        if entropies[idx] == entropy:  # entropy already present.
            assert entropies[idx] == entropy
            return coeffs[idx], coeffs[idx]
        assert entropies[idx + 1] <= entropy <= entropies[idx]
        return coeffs[idx + 1], coeffs[idx] 

    def next_psat_key(self, entropy: float) -> float:
        key1, key2 = self.psat_edge(entropy)
        if key2 == oo:
            return 2*key1 
        return (key2 - key1) / 2 + key1
        

@attr.s(auto_attribs=True, auto_detect=True, frozen=True)
class TabularCritic:
    game: GameGraph
    val_cache: Dict[(Node, float), float] = attr.ib(factory=dict, eq=False)
    pareto_curves: Dict[Node, ParetoCurve] = attr.ib(factory=dict, eq=False)

    def __hash__(self) -> int:
        # TODO: Remove
        return hash(self.game)

    def curve(self, node: Node) -> ParetoCurve:
        if node not in self.pareto_curves:
            self.pareto_curves[node] = ParetoCurve.new(node, self)
        return self.pareto_curves[node]

    def _moves(self, get_bounds, refine, node: Node, key: float, moves=None) -> List[Node]:
        """Return moves which minimizes the *achievable* entropy."""
        if moves is None:
            moves = list(self.game.moves(node))

        while True:
            # Pruning Phase.
            move2itvl = {}
            for move in moves:
                itvl = get_bounds(move, key)
                move2itvl[move] = itvl

            # TODO: optimize
            for move in moves:
                if move not in move2itvl:
                    continue
                itvl = move2itvl[move]
                if any(itvl2 < itvl for itvl2 in move2itvl.values()):
                    del move2itvl[move]  # Delete dominated move

            moves = list(move2itvl.keys())

            if len(moves) == 1:
                break  # Only one move left.

            elif all(itvl.size == 0 for ivtl in move2itvl.values()):
                break  # Collapsed to multiple singletons.
 
            # Refinement Phase.
            most_uncertain = max(moves, key=lambda m: move2itvl[m].size) 
            refine(most_uncertain, key)  # Collapse interval.

        return moves
    
    @lru_cache(maxsize=None)
    def min_ent_move(self, node: Node, rationality: float) -> Node:
        moves = self._moves(
            get_bounds=lambda m, x: self.curve(m).entropy_bounds(x),
            refine=self.entropy,
            node=node,
            key=rationality,
        )
        if len(moves) == 1:
            return moves[0]

        # Break ties with psat.
        # Note 1: Triggering this is fairly difficult to arrange in
        #   practice, since entropy and values both sensitive to exact
        #   model.
        # Note 2: Unlike in general min psat move case, rationality
        #   need note be updated since entropy is already matched.
        # Note 3: This step cannot be cached since psat will, in general,
        #   depend on the rationality.
        moves = self._moves(
            get_bounds=lambda m, x: self.curve(m).psat_bounds(x),
            refine=self.psat,  # TODO: consider using angle.
            node=node,
            key=rationality,
            moves=moves,
        )
        return moves[0]  # Remaining moves equivalent.

    @lru_cache(maxsize=None)
    def min_psat(self, node: Node, rationality: float) -> Itvl:
        assert self.game.label(node) == 'p2'

        # Compute entropy of planned move.
        planned_move = self.min_ent_move(node, rationality)
        entropy = self.entropy(planned_move, rationality)

        curves = self.curve

        def refine(move: Node, entropy: float) -> float:
            # Halve rationality on corresponding edge.
            rationality2 = curves(move).next_psat_key(entropy)
            self._lsat(move, rationality2) 

        def psat_bounds(move: Node, rationality: float) -> Itvl:
            return curves(move).psat_bounds(entropy=entropy)

        move = self._moves(
            get_bounds=psat_bounds,
            refine=refine,
            node=node,
            key=entropy,
        )[0]
        # Simulate replanning (TODO: expensive!)
        rationality = self.match_entropy(move, entropy)
        return self.lsat(move, rationality)

    def value(self, node: Node, rationality: float) -> float:
        if (node, rationality) not in self.val_cache:
            self.val_cache[node, rationality] = self._value(
                node=node, rationality=rationality
            )
        return self.val_cache[node, rationality]
    
    def _value(self, node: Node, rationality: float) -> float:
        label = self.game.label(node)

        if isinstance(label, bool):              # Terminal node.
            return rationality * label if rationality < oo else float(label)

        moves = list(self.game.moves(node))  # Fix order of moves.

        if label == 'p2':                        # Player 2 case.
            p2_move = self.min_ent_move(node, rationality)
            return self.value(p2_move, rationality)

        values = [self.value(move, rationality) for move in moves]

        if label == 'p1':                        # Player 1 case.
            return logsumexp(values) if rationality < oo else max(values)

        dist = label                             # Environment case.
        probs = [dist.prob(move) for move in moves]
        return np.average(values, weights=probs)

    def lsat(self, node_dist: DistLike, rationality: float) -> Itvl:
        if isinstance(node_dist, Dist):  # Reduce dist to calls over support.
            dist = node_dist
            probs = [dist.prob(n) for n in dist.support()]
            lsats = [self.lsat(n, rationality) for n in dist.support()]
            low = logsumexp([x.low for x in lsats], b=probs)
            high = logsumexp([x.high for x in lsats], b=probs)
            return Itvl(low, high) 

        lsats = self.curve(node_dist).lsats
        if rationality not in lsats:
            lsats[rationality] = self._lsat(node_dist, rationality)
        return lsats[rationality] 

    def _lsat(self, node: Node, rationality: float) -> float:
        label = self.game.label(node)
        if isinstance(label, bool):
            val = 0 if label else -oo
            return Itvl(val, val)
        elif label == 'p2':
            # Plan against optimal deterministic p2 policy.
            return self.min_psat(node, rationality)

        node_dist2 = self.move_dist(node, rationality)
        return self.lsat(node_dist2, rationality)

    def psat(self, node: Node, rationality: float) -> float:
        return psat(self.lsat(node, rationality))
        
    def _rationality(self, node: Node, target: float,
                     match_entropy: bool = False,
                     num_iter: int = 5) -> float:
        """Bracketed search for rationality to match either psat or entropy."""
        assert target >= 0, "Entropy or probabilities must be positive."
        if not match_entropy:  # Matching psat.
            assert target <= 1, "Probabilities are less than 1!"

        if match_entropy:
            stat = self.entropy
        else:
           def stat(node: Node, coeff: float) -> float:
               itvl = self.psat(node, coeff)
               assert itvl.low == itvl.high
               return itvl.high

        def f(coeff: float) -> float:
            return stat(node, coeff) - target  # type: ignore

        # TODO: properly support negative rationality.
        if f(oo) <= 0:
            return oo

        # Doubling trick.
        bot = 0
        for i in range(num_iter):
            try:
                top = 1 << i
                return binary_search(f, bot, top)
            except ValueError:
                bot = top

        return oo  # Effectively infinite.

    @lru_cache(maxsize=None)
    def match_entropy(self, node: Node, target: float) -> float:
        return self._rationality(node, target, match_entropy=True)

    @lru_cache(maxsize=None)
    def match_psat(self, node: Node, target: float) -> float:
        return self._rationality(node, target, match_entropy=False)

    def entropy(self, node_dist: DistLike, rationality: float) -> float:
        if isinstance(node_dist, Dist):  # Reduce dist to calls over support.
            dist = node_dist
            entropy = 0.0
            # Contribution from children. H(A[t+1:T] || S[t+1: T], S[:t]).
            for node in dist.support():
                entropy += dist.prob(node) * self.entropy(node, rationality)
            return entropy

        entropies = self.curve(node_dist).entropies
        if rationality not in entropies:
            entropies[rationality] = self._entropy(node_dist, rationality)
        return entropies[rationality] 

    def _entropy(self, node: Node, rationality: float) -> float:
        label = self.game.label(node)
        if isinstance(label, bool):
            return 0.0  # Terminal node has no entropy.

        node_dist2 = self.move_dist(node, rationality)
        entropy = self.entropy(node_dist2, rationality)

        if label == 'p1':
            entropy += node_dist2.entropy
        return entropy

    def move_dist(self, state: Node, rationality: float) -> Distribution:
        label = self.game.label(state)
        if isinstance(label, bool):
            return Dist({})
        elif label == 'p2':
            p2_move = self.min_ent_move(state, rationality)
            return Dist({p2_move: 1})  # Assume worst case.

        moves = self.game.moves(state)

        if label == 'p1':
            vals = [self.value(move, rationality) for move in moves]

            if rationality < oo:
                probs = softmax(vals)
                return Dist({move: p for move, p in zip(moves, probs)})

            # If rationality = oo, then we pick uniformly from the best move.
            optimal = max(vals)
            support = [a for a, v in zip(moves, vals) if v == optimal]
            return Dist({node: 1 / len(support) for node in support})

        return label  # Environment Case. label *is* the distribution.

    def state_dist(self, move: Node, rationality: float) -> Distribution:
        stack = [(0.0, move, rationality)]
        node2prob = {}
        while stack:
            lprob, node, rationality = stack.pop()
            label = self.game.label(node)

            if isinstance(label, bool) or label == 'p1':
                node2prob[node] = lprob
                continue
            elif label == 'p2':  # Plan against deterministic adversary.
                p2_move = self.min_ent_move(node, rationality)
                stack.append((lprob, p2_move, rationality))
                continue
            else:
                dist = label
                for node2 in dist.support():
                    lprob2 = lprob + math.log(dist.prob(node2))
                    stack.append((lprob2, node2, rationality))
        node2prob = {k: math.exp(v) for k, v in node2prob.items()}
        return Dist(node2prob)

    def feasible(self, node: Node, entropy: float, psat: float) -> Feasible:
        if (self.entropy(node, 0) < entropy) or (self.psat(node, oo).high < psat):
            return None

        if (self.entropy(node, oo) >= entropy):
            return oo  # Already know self.psat(node, oo) >= 0.

        def get_cmd(coeff: float) -> SearchCMD:
            # TODO: introduce tolerance here.
            hfeasible = self.entropy(node, coeff) >= entropy
            pfeasible = self.psat(node, coeff).low >= psat

            if hfeasible and pfeasible:
                return coeff
            elif hfeasible and not pfeasible:
                return "increase"
            elif not hfeasible and pfeasible:
                return "decrease"
            else:
                return None
        
        # Doubling trick for range.
        lo = hi = 0
        for i in range(5):
            cmd = get_cmd(hi)
            if not isinstance(cmd, str):
                return cmd
            elif cmd == 'decrease':
                break
            lo, hi = hi, 1 << i

        while True:
            mid = lo + (hi - lo) / 2
            cmd = get_cmd(mid)
            if not isinstance(cmd, str):
                return cmd
            lo, hi = (lo, mid) if cmd == 'decrease' else (mid, hi)

    @staticmethod
    def from_game_graph(game_graph: GameGraph) -> Critic:
        return TabularCritic(game_graph)


NodeStatFunc = Callable[[TabularCritic, Node, float], float]


__all__ = ['TabularCritic']
