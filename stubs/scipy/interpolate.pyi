from typing import Callable
from numpy.typing import ArrayLike


def interp1d(
        x: ArrayLike,
        y: ArrayLike,
        kind: str = 'linear', 
        assume_sorted: bool = False,
        copy: bool = False,
) -> Callable[[float], float]:
    ...
