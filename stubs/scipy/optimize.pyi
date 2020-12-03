from typing import Any, Callable, Protocol, Tuple


def brentq(
        f: Callable[[float], float],
        a: float,
        b: float,
) -> Tuple[float, Any]:
    ...
