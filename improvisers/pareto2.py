from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Iterable, Tuple, List, Set
from typing import no_type_check

import attr
import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull


oo = float('inf')
RealFunc = Callable[[float], float]
List2d = Tuple[List[float], List[float]]


@attr.s(auto_detect=True, auto_attribs=True, frozen=True)
class Region:
    left: Tuple[float, float]   # (coeff, win_prob)
    right: Tuple[float, float]  # (coeff, win_prob)

    def __attrs_post_init__(self) -> None:
        left, right = self.left, self.right
        assert left[0] < right[0]
        assert left[1] <= right[1]

    @property
    def size(self) -> float:
        return self.right[1] - self.left[1]

    def subdivide(self, middle: Tuple[float, float]) -> Tuple[Region, Region]:
        return Region(self.left, middle), Region(middle, self.right)

    @property
    def avg_coeff(self) -> float:
        left, right = self.left[0], self.right[0]
        right = min(right, 2 * left + 3)  # Doubling trick for infinite coeff.
        return left + (right - left) / 2  # Avoid large floats.


def discretize(func: RealFunc, tol: float) -> List2d:
    from heapq import heappop as pop, heappush as push

    root = Region((0, func(0)), (oo, func(oo)))
    queue = [(-root.size, root)]   # Priority queue in region sizes.
    while queue[0][1].size > tol:  # Get smallest region and test tol.
        _, region = pop(queue)
        mid = region.avg_coeff
        kids = region.subdivide((mid, func(mid)))
        queue.extend([(k.size, k) for k in kids])

    # TODO: convert to np array.
    points = [root.left] + [r.right for _, r in queue]
    return list(zip(*points))  # type: ignore


def interp(x: ArrayLike, y: ArrayLike, kind: str) -> RealFunc:
    func = interp1d(x, y, kind=kind, assume_sorted=True)
    return lambda z: float(func(z))  # Type shenanigans.


@attr.s(auto_attribs=True, frozen=True, auto_detect=True)
class Pareto:
    """Upper and Lower bound on convex pareto front."""
    size: int
    entropy: RealFunc             # Get entropy by rationality.
    lower_win_prob: RealFunc      # Get lower win prob by entropy.
    upper_win_prob: RealFunc      # Get upper win prob by entropy. 
    lower_rationality: RealFunc   # Get lower rationality by entropy.
    upper_rationality: RealFunc   # Get upper rationality by entropy.


    @no_type_check
    def show(self) -> None:
        import plotext as plt

        lo, hi = self.entropy(oo), self.entropy(0)
        entropies = np.linspace(lo, hi, 50)
        pwins_upper = [self.upper_win_prob(e) for e in entropies]
        pwins_lower = [self.lower_win_prob(e) for e in entropies]
        plt.plot(entropies, pwins_upper, fill=True, point_marker=2)
        plt.plot(entropies, pwins_lower, fill=True, point_marker=1)


        plt.xlabel('entropy')
        plt.ylabel('pwin')
        plt.canvas_color('none')
        plt.ticks_color('white')
        plt.axes_color('black')
        plt.show()

    def __contains__(self, entropy: float, pwin: float) -> bool:
        if pwin < self.lower_win_prob(entropy):
            return True
        elif pwin > self.upper_win_prob(entropy):
            return False
        raise ValueError('Need to refine boundary!')

    def rationality(self, entropy: float) -> Tuple[float, float, float]:
        """Look up rationality by entropy.

        Returns:
          Convex combination of rationality coefficents, 

            λ ≜ p·λ₁ + (1 - p)·λ₂,

          represented as a tuple, (λ₁, λ₂, p).
        """
        low_coeff = self.lower_rationality(entropy)
        high_coeff = self.upper_rationality(entropy)

        low_entropy = self.entropy(low_coeff)
        high_entropy = self.entropy(high_coeff)

        mixture = (entropy - low_entropy) / (high_entropy - low_entropy)
        return (low_coeff, high_coeff, mixture)

    @no_type_check
    @staticmethod
    def build(lower_win_prob: RealFunc,
              upper_win_prob: RealFunc,
              entropy: RealFunc, 
              tol: float,) -> Pareto:

        coeffs, probs_lo = map(np.array, discretize(lower_win_prob, tol))
        entropies = np.array([entropy(x) for x in coeffs])

        # Find indicies in convex hull of lower pareto front.
        points = np.array([entropies, probs_lo]).T
        points = np.vstack([points, [-1, -1]])  # Add dummy bottom point.
        dummy_idx = len(points) - 1

        mask = list(set(ConvexHull(points).vertices) - {dummy_idx})
        mask = sorted(mask, key=lambda i: entropies[i])

        # Apply mask and interpolate.
        coeffs = coeffs[mask]
        entropies = entropies[mask]
        probs_lo = probs_lo[mask]
        probs_hi = [upper_win_prob(x) for x in coeffs]

        return Pareto(
            size=len(coeffs),
            entropy=interp(coeffs[::-1], entropies[::-1], 'linear'),
            lower_win_prob=interp(entropies, probs_lo, 'linear'),
            upper_win_prob=interp(entropies, probs_hi, 'linear'),
            lower_rationality=interp(entropies, coeffs, 'previous'),
            upper_rationality=interp(entropies, coeffs, 'next'),
        )

__all__ = ['Pareto']
