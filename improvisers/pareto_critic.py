"""This module contains the tabular Critic implementation."""
from __future__ import annotations

import math
from functools import partial, lru_cache
from typing import Hashable, List, Sequence, Tuple, Dict, Callable
from typing import no_type_check

import attr
import numpy as np
from scipy.special import logsumexp, softmax
from scipy.optimize import brentq

from improvisers.game_graph import GameGraph, Node
from improvisers.critic import Critic, Distribution, DistLike
from improvisers.explicit import ExplicitDist as Dist
from improvisers.pareto2 import Pareto


oo = float('inf')
Nodes = Sequence[Node]


@attr.s(auto_attribs=True, frozen=True)
class ParetoCritic:
    game: GameGraph
    tol: float = 1e-2

    @lru_cache(maxsize=None)
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

    def entropy(self, node_dist: DistLike, rationality: float) -> float:
        if isinstance(node_dist, Dist):  # Reduce dist to calls over support.
            dist = node_dist
            entropy = 0.0
            # Contribution from children. H(A[t+1:T] || S[t+1: T], S[:t]).
            for node in dist.support():
                entropy += dist.prob(node) * self.entropy(node, rationality)
            return entropy

        node = node_dist
        label = self.game.label(node)
        if isinstance(label, bool):
            return 0.0  # Terminal node has no entropy.

        node_dist2 = self.move_dist(node, rationality)
        entropy = self.entropy(node_dist2, rationality)

        if label == 'p1':
            entropy += node_dist2.entropy
        return entropy

    def psat(self, node: Node, rationality: float,
             lower: bool = True) -> float:
        return math.exp(self.lsat(node, rationality))

    def lsat(self, node_dist: DistLike, rationality: float, 
             lower: bool = True) -> float:
        if isinstance(node_dist, Dist):  # Reduce dist to calls over support.
            dist = node_dist
            probs = [dist.prob(n) for n in dist.support()]

            #pwins = []
            #for n in dist.support():
            #    curve = self.pareto(n)
            #pwins = [
            #    self.pareto(n) for n in dist.support()
            #]
            lsats = [self.lsat(n, rationality) for n in dist.support()]
            return logsumexp(lsats, b=probs)
        node = node_dist

        label = self.game.label(node)
        if isinstance(label, bool):
            return 0 if label else -oo
        elif label == 'p2':
            entropy = self.entropy(node, rationality)
            curve = self.pareto(node)

            if lower:
                psat = curve.lower_win_prob(entropy)
            else:
                psat = curve.upper_win_prob(entropy)

            return math.log(psat)
        elif label == 'p1':
            entropy = self.entropy(node, rationality)

        node_dist2 = self.move_dist(node, rationality)
        return self.lsat(node_dist2, rationality)

    def match_entropy(self, node: Node, target: float) -> float:
        raise NotImplementedError

    def match_psat(self, node: Node, target: float) -> float:
        raise NotImplementedError

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

    def min_ent_move(self, node: Node, rationality: float) -> Node:
        moves = list(self.game.moves(node))

        # Compute min entropy moves.
        entropies = {m: self.entropy(m, rationality) for m in moves}
        target = min(entropies.values())
        moves = [m for m, e in entropies.items() if e == target]
        
        if len(moves) == 1:
            return moves[0]

        # Collapse moves that yield indistinguishable policies.
        values = {self.value(m, rationality): m for m in moves}        
        moves = list(values.values())

        if len(moves) == 1:
            return moves[0]

        move1 = min(moves, key=lambda m: self.pareto(m).lower_win_prob(target))
        move2 = min(moves, key=lambda m: self.pareto(m).upper_win_prob(target))
        if move1 != move2:
            raise ValueError('Need to lower tolerance!')
        return move1

    def min_psat_move(self, node: Node, rationality: float) -> Tuple[Node, float]:  # noqa: E501
        raise NotImplementedError

    @staticmethod
    def from_game_graph(game_graph: GameGraph, tol: float = 1e-2) -> Critic:
        return ParetoCritic(game_graph, tol=tol)

    def psat_min(self, node: Node, rationality: float, lower: bool) -> float:
        move = self.min_ent_move(node, rationality)
        target = self.entropy(move, rationality)
        
        name = 'lower_win_prob' if lower else 'upper_win_prob'

        moves = self.game.moves(node)
        return min(getattr(self.pareto(m), name)(target) for m in moves)  # type: ignore

    @lru_cache(maxsize=None)
    def pareto(self, node: Node) -> Pareto:
        label = self.game.label(node)
        psat = self.psat_min if label == 'p2' else self.psat
        return Pareto.build(
            entropy=lambda x: self.entropy(node, x),
            lower_win_prob=lambda x: psat(node, x, lower=True),  # type: ignore
            upper_win_prob=lambda x: psat(node, x, lower=False), # type: ignore
            tol=self.tol,
        )


NodeStatFunc = Callable[[ParetoCritic, Node, float], float]


__all__ = ['ParetoCritic']
