from typing import Hashable, Mapping, Set, Iterable, TypeVar


Node = TypeVar('Node')
AdjacencyMap = Mapping[Node, Set[Node]]


def toposort_flatten(data: AdjacencyMap[Node], sort: bool) -> Iterable[Node]:
    ...
