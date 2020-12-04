"""This module contains the tabular Critic implementation."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Hashable, List, Literal, Mapping, Optional, Tuple, TypeVar, Union, DefaultDict, Dict
from typing import cast, Callable, TypeVar

import attr
import numpy as np
from scipy.special import logsumexp

from improvisers.game_graph import Node, Action, GameGraph
from improvisers.critic import Critic, Distribution


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


NodeStatFunc = Callable[[TabularCritic, Node, float], float]


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

    @cached_stat
    def value(self, node: Node, rationality: float) -> float:
        label = self.game.label(node)
        if isinstance(label, bool):              # Terminal node.
            return rationality * label

        actions = list(self.game.actions(node))  # Fix order of actions.

        if label == 'p2':                        # Player 2 case.
            def key(action: Action) -> Tuple[float, float]:
                psat = self.psat(action.node, rationality)
                val = self.value(action.node, rationality)
                return psat, val

            p2_action = min(actions, key=key)
            return self.value(p2_action.node, rationality)

        values = np.array([self.value(a.node, rationality) for a in actions])
        values += np.log(np.array([a.size for a in actions]))
        
        if label == 'p1':                        # Player 1 case.
            return logsumexp(values)

        assert label == 'env'                    # Environment case.
        probs = cast(List[float], [a.prob for a in actions])
        return np.average(values, weights=probs)

    @cached_stat
    def psat(self, node: Node, rationality: float) -> float:
        pass

    @cached_stat
    def rationality(self, node: Node, psat: float) -> float:
        pass

    @cached_stat
    def entropy(self, node: Node, rationality: float) -> float:
        pass

    def action_dist(self, state: Node, rationality: float) -> Distribution:
        pass

    def state_dist(self, state: Node, action: Node) -> Distribution:
        pass

    @staticmethod
    def from_game_graph(game_graph: GameGraph) -> Critic:
        pass
