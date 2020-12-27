"""This module contains the tabular Critic implementation."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Hashable, List, Literal, Mapping, Optional, Tuple, TypeVar, Union, DefaultDict, Dict
from typing import cast, Callable, TypeVar, Iterable

import attr
import numpy as np
from scipy.special import logsumexp, softmax
from scipy.optimize import brentq

from improvisers.game_graph import Node, Action, GameGraph
from improvisers.critic import Critic, Distribution


@attr.s(frozen=True, auto_attribs=True)
class Dist:
    data: Dict[Node, float] = attr.ib(factory=dict)

    def entropy(self) -> float:
        probs = [v for v in self.data.values() if v > 0]
        return -np.log(probs).sum()

    def sample(self, seed: Optional[int] = None) -> Node:
        if seed is not None:
            random.seed(seed)
        return random.choices(*zip(*self.data.items()))[0]  # type: ignore

    def prob(self, node: Node) -> float:
        return self.data[node]

    def support(self) -> Iterable[Node]:
        return self.data.keys()

    def lsat(self, critic: Critic, rationality: float) -> float:
        probs = [self.prob(n) for n in self.support()]
        lsats = [critic.lsat(n, rationality) for n in self.support()]
        return logsumexp(lsats, b=probs)

    def psat(self, critic: Critic, rationality: float) -> float:
        return math.exp(self.lsat(critic, rationality))


@attr.s(frozen=True, auto_attribs=True)
class Cache:
    data: Dict[Tuple[Node, Hashable], Tuple[float, float]] = attr.ib(factory=dict)

    def __contains__(self, key: Tuple[Node, Hashable, float]) -> bool:
        node, stat_key, rationality = key
        if (node, stat_key) not in self.data:
            return False
        return self.data[node, stat_key][1] == rationality

    def __getitem__(self, key: Tuple[Node, Hashable, float]) -> float:
        node, stat_key, _ = key
        if key not in self:
            raise ValueError(f"key: {key} not in cache.")
        return self.data[node, stat_key][0]

    def __setitem__(self, key: Tuple[Node, Hashable, float], val: float) -> None:
        node, stat_key, rationality = key
        self.data[node, stat_key] = (val, rationality)


def cached_stat(func: NodeStatFunc) -> NodeStatFunc:
    def wrapped(critic: TabularCritic, node: Node, rationality: float) -> float:
        if (node, func, rationality) in critic.cache:
            return critic.cache[node, func, rationality]
        val = func(critic, node, rationality)
        critic.cache[node, func, rationality] = val
        return val
    return wrapped


@attr.s(frozen=True, auto_attribs=True)
class TabularCritic:
    game: GameGraph
    cache: Cache = attr.ib(factory=Cache)

    def p2_action(self, node: Node, rationality: float) -> Tuple[Action, float]:
        """Optimal player two action to plan against."""
        assert self.game.label(node) == 'p2'
        actions = list(self.game.actions(node))
        worst_psat = min(self.psat(a.node, rationality) for a in actions)

        def replanned_entropy(action: Action) -> float:
            node = action.node
            rationality = max(self.rationality(node, worst_psat), 0)
            return self.entropy(node, rationality)

        p2_action = min(actions, key=replanned_entropy)
        return p2_action, replanned_entropy(p2_action)

    def action_value(self, action: Action, rationality: float) -> float:
        return self.value(action.node, rationality) + math.log(action.size)

    @cached_stat
    def value(self, node: Node, rationality: float) -> float:
        label = self.game.label(node)
        if isinstance(label, bool):              # Terminal node.
            return rationality * label

        actions = list(self.game.actions(node))  # Fix order of actions.

        if label == 'p2':                        # Player 2 case.
            p2_action, rationality = self.p2_action(node, rationality)
            return self.action_value(p2_action, rationality)

        values = [self.action_value(a, rationality) for a in actions]
        
        if label == 'p1':                        # Player 1 case.
            return logsumexp(values)

        assert label == 'env'                    # Environment case.
        dist = self.action_dist(node, rationality)
        probs = [dist.prob(n) for n in dist.support()]
        return np.average(values, weights=probs)

    @cached_stat
    def lsat(self, node: Node, rationality: float) -> float:
        label = self.game.label(node)
        if isinstance(label, bool):
            return 0 if label else -float('inf')
        elif label == 'p2':
            # Plan against optimal deterministic p2 policy.
            p2_action, rationality = self.p2_action(node, rationality)
            return self.lsat(p2_action.node, rationality)

        dist = self.action_dist(node, rationality)
        return dist.lsat(self, rationality)

    def psat(self, node: Node, rationality: float) -> float:
        return math.exp(self.lsat(node, rationality))

    @cached_stat
    def rationality(self, node: Node, psat: float, top: int = 100) -> float:
        """Bracketed search for rationality to match psat."""
        assert 0 <= psat <= 1

        def f(coeff: float) -> float:
            return self.psat(node, coeff) - psat

        if f(-top) > 0:
            return -top
        elif f(top) < 0:
            return top
        else:
            return brentq(f, -top, top)

    @cached_stat
    def entropy(self, node: Node, rationality: float) -> float:
        label = self.game.label(node)
        entropy = 0.0
        if isinstance(label, bool):
            return entropy  # Terminal node has no entropy.

        dist = self.action_dist(node, rationality)
        entropy += dist.entropy()  # Entropy contribution of this action.

        if label == 'p2':  # Need to update rationality of children.
            _, rationality = self.p2_action(node, rationality)

        # Contribution from children. H(A[t+1:T] || S[t+1: T], S[:t]).
        for node2 in dist.support():
            entropy += dist.prob(node2) * self.entropy(node2, rationality)
        return entropy

    def action_dist(self, state: Node, rationality: float) -> Distribution:
        label = self.game.label(state)
        if isinstance(label, bool):
            return Dist({})
        elif label == 'p2':
            p2_action, _ = self.p2_action(state, rationality)
            return Dist({p2_action.node: 1})  # Assume worst case.

        actions = self.game.actions(state)

        if label == 'env':
            return Dist({a.node: a.prob for a in actions})  # type: ignore
        else:
            assert label == 'p1'
            values = [self.action_value(a, rationality) for a in actions]
            probs = softmax(values)
            return Dist({a.node: p for a, p in zip(actions, probs)})

    def state_dist(self, action: Node, rationality: float) -> Distribution:
        stack = [(0.0, action, rationality)]
        node2prob = {}
        while stack:
            lprob, node, rationality = stack.pop()
            label = self.game.label(node)

            if isinstance(label, bool) or label == 'p1':
                node2prob[node] = lprob
                continue
            elif label == 'p2':  # Plan against deterministic adversary.
                p2_action, rationality = self.p2_action(node, rationality)
                stack.append((lprob, p2_action.node, rationality))
                continue
            else:
                dist = self.action_dist(node, rationality)
                for node2 in dist.support():
                    lprob2 = lprob + math.log(dist.prob(node2))
                    stack.append((lprob2, node2, rationality))

        return Dist(node2prob)


    @staticmethod
    def from_game_graph(game_graph: GameGraph) -> Critic:
        return TabularCritic(game_graph)


NodeStatFunc = Callable[[TabularCritic, Node, float], float]


__all__ = ['TabularCritic']
