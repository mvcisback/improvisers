"""This module contains the tabular Critic implementation."""
from __future__ import annotations

import math
from typing import Hashable, List, Tuple, Dict, Callable

import attr
import numpy as np
from scipy.special import logsumexp, softmax
from scipy.optimize import brentq

from improvisers.game_graph import GameGraph, Node
from improvisers.critic import Critic, Distribution, DistLike
from improvisers.explicit import ExplicitDist as Dist


oo = float('inf')


CacheKey = Tuple[Node, Hashable, float]


@attr.s(frozen=True, auto_attribs=True)
class Cache:
    data: Dict[Tuple[Node, Hashable], Tuple[float, float]] = attr.ib(
        factory=dict
    )

    def __contains__(self, key: CacheKey) -> bool:
        node, stat_key, rationality = key
        if (node, stat_key) not in self.data:
            return False
        return self.data[node, stat_key][1] == rationality

    def __getitem__(self, key: CacheKey) -> float:
        node, stat_key, _ = key
        if key not in self:
            raise ValueError(f"key: {key} not in cache.")
        return self.data[node, stat_key][0]

    def __setitem__(self, key: CacheKey, val: float) -> None:
        node, stat_key, rationality = key
        self.data[node, stat_key] = (val, rationality)


def cached_stat(func: NodeStatFunc) -> NodeStatFunc:
    """Cache node level statistics. Only once per rationality."""
    def wrap(critic: TabularCritic,
             node_dist: DistLike,
             rationality: float) -> float:
        if isinstance(node_dist, Dist):  # Don't cache distributions.
            return func(critic, node_dist, rationality)
        node = node_dist
        if (node, func, rationality) in critic.cache:
            return critic.cache[node, func, rationality]
        val = func(critic, node, rationality)
        critic.cache[node, func, rationality] = val
        return val
    return wrap


@attr.s(auto_attribs=True, frozen=True)
class TabularCritic:
    game: GameGraph
    cache: Cache = attr.ib(factory=Cache)
    _min_ent_moves: Dict[Node, List[Node]] = attr.ib(factory=dict)

    def min_ent_moves(self, node: Node) -> List[Node]:
        """Return moves which minimizes the *achievable* entropy."""
        if node in self._min_ent_moves:
            return self._min_ent_moves[node]

        moves, worst = [], oo
        for node2 in self.game.moves(node):
            entropy = self.entropy(node2, 0)
            if entropy < worst:
                moves, worst = [node2], entropy
            elif entropy == worst:
                moves.append(node2)
        self._min_ent_moves[node] = moves
        return moves

    def min_ent_move(self, node: Node, rationality: float) -> Node:
        moves = self.min_ent_moves(node)

        # Optimization. If all values are the same, the resulting
        # policy will assign same probability to transitioning to this
        # node. Commonly happens when two subtrees are equivalent.
        val0 = self.value(moves[0], 0)

        other_vals = (self.value(move, 0) for move in moves[1:])
        if all(val == val0 for val in other_vals):
            return moves[0]

        # Break ties with psat.
        # Note 1: Triggering this is fairly difficult to arrange in
        #   practice, since entropy and values both sensitive to exact
        #   model.
        # Note 2: Unlike in general min psat move case, rationality
        #   need note be updated since entropy is already matched.
        # Note 3: This step cannot be cached since psat will, in general,
        #   depend on the rationality.
        return min(moves, key=lambda n: self.psat(n, rationality))

    def min_psat_move(
            self, node: Node, rationality: float) -> Tuple[Node, float]:
        assert self.game.label(node) == 'p2'

        # Compute entropy of planned move.
        planned_move = self.min_ent_move(node, rationality)
        entropy = self.entropy(planned_move, rationality)

        # p1 will increase rationality until target entropy matched.
        def replanned_psat(move: Node) -> float:
            replanned_rationality = rationality
            if rationality < oo:  # Note: can't increase rationality past oo.
                replanned_rationality = self.match_entropy(move, entropy)
            return self.psat(move, max(replanned_rationality, 0))

        # p2 will take the minimum psat of the replanned moves.
        moves = self.game.moves(node)
        p2_move = min(moves, key=replanned_psat)

        if rationality < oo:
            rationality = self.match_entropy(p2_move, entropy)

        return p2_move, rationality

    @cached_stat
    def value(self, node: Node, rationality: float) -> float:
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

    @cached_stat
    def lsat(self, node_dist: DistLike, rationality: float) -> float:
        if isinstance(node_dist, Dist):  # Reduce dist to calls over support.
            dist = node_dist
            probs = [dist.prob(n) for n in dist.support()]
            lsats = [self.lsat(n, rationality) for n in dist.support()]
            return logsumexp(lsats, b=probs)
        node = node_dist

        label = self.game.label(node)
        if isinstance(label, bool):
            return 0 if label else -oo
        elif label == 'p2':
            # Plan against optimal deterministic p2 policy.
            p2_move, rationality = self.min_psat_move(node, rationality)

            return self.lsat(p2_move, rationality)

        node_dist2 = self.move_dist(node, rationality)
        return self.lsat(node_dist2, rationality)

    def psat(self, node: Node, rationality: float) -> float:
        sat_prob = math.exp(self.lsat(node, rationality))
        assert sat_prob < 1.2
        return min(sat_prob, 1)  # Clip at 1 due to numerics.

    def _rationality(self, node: Node, target: float,
                     match_entropy: bool = False,
                     num_iter: int = 100) -> float:
        """Bracketed search for rationality to match either psat or entropy."""
        assert target >= 0, "Entropy or probabilities must be positive."
        if not match_entropy:  # Matching psat.
            assert target <= 1, "Probabilities are less than 1!"

        stat = self.entropy if match_entropy else self.psat

        def f(coeff: float) -> float:
            return stat(node, coeff) - target

        # TODO: properly support negative rationality.
        if f(-100) > 0:
            return -100   # TODO: support -oo.
        elif f(oo) < 0:
            return oo

        top = 1
        for _ in range(num_iter):
            try:
                return brentq(f, -top, top)
            except ValueError:
                top *= 2

        return oo  # Effectively infinite.

    @cached_stat
    def match_entropy(self, node: Node, target: float) -> float:
        return self._rationality(node, target, match_entropy=True)

    @cached_stat
    def match_psat(self, node: Node, target: float) -> float:
        return self._rationality(node, target, match_entropy=False)

    @cached_stat
    def entropy(self, node_dist: DistLike, rationality: float) -> float:
        if isinstance(node_dist, Dist):  # Reduce dist to calls over support.
            dist = node_dist
            entropy = dist.entropy
            # Contribution from children. H(A[t+1:T] || S[t+1: T], S[:t]).
            for node in dist.support():
                entropy += dist.prob(node) * self.entropy(node, rationality)
            return entropy

        node = node_dist
        label = self.game.label(node)
        if isinstance(label, bool):
            return 0.0  # Terminal node has no entropy.

        node_dist2 = self.move_dist(node, rationality)
        return self.entropy(node_dist2, rationality)

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

    @staticmethod
    def from_game_graph(game_graph: GameGraph) -> Critic:
        return TabularCritic(game_graph)


NodeStatFunc = Callable[[TabularCritic, Node, float], float]


__all__ = ['TabularCritic']
