from typing import Sequence, Union, Iterable, Iterator


class Array:
    def __add__(self, other: Union[Array, float]) -> Array:
        ...

    def __mul__(self, other: Union[Array, float]) -> Array:
        ...

    def sum(self) -> float:
        ...

    def min(self) -> float:
        ...

    def __iter__(self) -> Iterator[float]:
        ...


def array(data: Sequence[float]) -> Array:
    ...


def log(data: Union[Array, float]) -> Array:
    ...


def average(a: Iterable[float], weights: Iterable[float]) -> float:
    ...
