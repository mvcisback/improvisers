from typing import Hashable, Mapping, Set, Iterable


Node = Hashable
AdjacencyMap = Mapping[Node, Set[Node]]


def toposort_flatten(data: AdjacencyMap) -> Iterable[Node]:
    ...
