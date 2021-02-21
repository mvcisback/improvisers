from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike


class ConvexHull:
    def __init__(self, points: ArrayLike) -> None:
        ...

    @property
    def vertices(self) -> Sequence[int]:
        ...
