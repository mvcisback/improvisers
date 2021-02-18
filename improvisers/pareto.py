from __future__ import annotations

from bisect import bisect_left
from functools import total_ordering
from typing import Any, Sequence, Tuple, Union

import attr


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


@attr.s(auto_attribs=True, frozen=True, auto_detect=True)
@total_ordering
class Point:
    entropy: float
    rationality: float
    win_prob: Interval

    def __hash__(self) -> int:
        return hash(self.entropy)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Point):
            return False
        entropy = other.entropy if isinstance(other, Point) else other
        return self.entropy == entropy

    def __lt__(self, other: Union[Point, float]) -> bool:
        entropy = other.entropy if isinstance(other, Point) else other
        return self.entropy < entropy


@attr.s(auto_attribs=True, frozen=True)
class ParetoCurve:
    sorted_points: Sequence[Point] = attr.ib(converter=sorted)

    def __getitem__(self, entropy: float) -> Point:
        """Look up point by entropy."""
        index = bisect_left(self.sorted_points, entropy)
        point = self.sorted_points[index]
        assert entropy <= point.entropy
        return point

    def rationality(self, entropy: float) -> float:
        """Look up rationality by entropy."""
        return self[entropy].rationality

    def win_prob(self, entropy: float) -> Interval:
        """Look up win_prob by entropy."""
        return self[entropy].win_prob


@attr.s(auto_attribs=True, frozen=True)
class Region:
    left: Point
    right: Point

    def __contains__(self, entropy: float) -> bool:
        ...
