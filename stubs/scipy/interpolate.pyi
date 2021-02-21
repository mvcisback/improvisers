from typing import Callable
from numpy.typing import ArrayLike


def interp1d(
        x: ArrayLike,
        y: ArrayLike,
        kind: str = 'linear', 
        assumed_sorted: bool = False,
) -> Callable[[float], float]:
    ...
