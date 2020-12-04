from typing import Iterable, Optional

from numpy import Array


def logsumexp(vals: Iterable[float], b: Optional[Iterable[float]]=None) -> float:
    ...


def softmax(vals: Iterable[float]) -> Array:
    ...
