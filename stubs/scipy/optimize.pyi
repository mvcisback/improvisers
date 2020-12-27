from typing import Callable


def brentq(
        f: Callable[[float], float],
        a: float,
        b: float,
) -> float:
    ...
