from __future__ import annotations

from bisect import bisect_left
from functools import total_ordering
from typing import Any, Callable, Sequence, Tuple, Union, Mapping, NamedTuple

import attr


oo = float('inf')


@attr.s(auto_attribs=True, frozen=True, auto_detect=True)
class Interval:
    low: float
    high: float

    def __attrs_post_init__(self) -> None:
        assert self.low <= self.high

    def __lt__(self, other: float) -> bool:
        return self.high < other

    def __gt__(self, other: float) -> bool:
        return self.low > other

    def __contains__(self, other: float) -> bool:
        return self.low <= other <= self.high

    def __add__(self, other: Union[float, Interval]) -> Interval:
        if isinstance(other, float):
            other = Interval(other, other)
        return attr.evolve(Interval(
            low=self.low + other.low,
            high=self.high + other.high,
        ))

    def __mul__(self, other: float) -> Interval:
        return attr.evolve(Interval(
            low=self.low * other,
            high=self.high * other,
        ))

    @property
    def size(self) -> float:
        return self.high - self.low


class Point(NamedTuple):
    entropy: float
    rationality: float
    win_prob: float


@attr.s(auto_detect=True, auto_attribs=True, frozen=True)
class Region:
    left: Point   # min rationality point.
    right: Point  # max rationality point.

    def __attrs_post_init__(self) -> None:
        left, right = self.left, self.right
        assert left.rationality < right.rationality
        assert left.win_prob < right.win_prob
        assert left.entropy > right.entropy

    @property
    def size(self) -> float:
        return self.left.win_prob - self.right.win_prob


Points = Sequence[Point]
FindPoint = Callable[[float], Point]


@attr.s(auto_attribs=True, frozen=True)
class Pareto:
    margin: float
    points: Sequence[Point] = attr.ib(converter=sorted)

    def __getitem__(self, entropy: float) -> Point:
        """Look up point by entropy."""
        point = (entropy, -oo, -oo)  # Lift to partially defined point.
        index = bisect_left(self.points, point)
        return self.points[index]

    def rationality(self, entropy: float) -> float:
        """Look up rationality by entropy."""
        return self[entropy].rationality

    def win_prob(self, entropy: float) -> Interval:
        """Look up win_prob by entropy."""
        lower = self[entropy].win_prob
        return Interval(lower, lower + self.margin)

    @staticmethod
    def build(find_point: FindPoint, tol: float, prev_margin: float) -> Pareto:
        queue = [Region(find_point(0), find_point(oo))]
        raise NotImplementedError
