from typing import Callable, Optional
from numpy.typing import ArrayLike


def interp1d(
        x: ArrayLike,
        y: ArrayLike,
        kind: str = 'linear', 
        assume_sorted: bool = False,
        copy: bool = False,
        fill_value: Optional[str] = None,
) -> Callable[[float], float]:
    ...
