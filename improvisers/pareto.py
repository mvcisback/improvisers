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
        return self.right.win_prob - self.left.win_prob

    def subdivide(self, middle: Point) -> Tuple[Region, Region]:
        return Region(self.left, middle), Region(middle, self.right)

    @property
    def avg_coeff(self) -> float:
        left, right = self.left.rationality, self.right.rationality
        right = min(right, 2 * left + 3)  # Doubling trick for infinite coeff.
        return left + (right - left) / 2  # Avoid large floats.


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
        upper = min(lower + self.margin, self.max_win_prob)  # Clipped error.
        return Interval(lower, upper)

    @property
    def max_win_prob(self):
        return self.points[0].win_prob

    @property
    def max_entropy(self):
        return self.points[-1].entropy

    @staticmethod
    def build(find_point: FindPoint, tol: float, prev_margin: float) -> Pareto:
        from heapq import heappop as pop, heappush as push

        root = Region(find_point(0), find_point(oo))

        queue = [(-root.size, root)]   # Priority queue in region sizes.
        while queue[0][1].size > tol:  # Get smallest region and test tol.
            _, region = pop(queue)
            kids = region.subdivide(find_point(region.avg_coeff))
            queue.extend([(k.size, k) for k in kids])

        # Convert regions to spanned points and then lift to Pareto approx.
        points = [root.left] + [r.right for _, r in queue]
        return Pareto(points=points, margin=tol + prev_margin)  # type: ignore
