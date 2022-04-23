# https://github.com/cexen/procon-cexen/blob/main/py/HeapQueueKV.py
from typing import TypeVar, Generic
import heapq

# >= 3.8

# from typing import Protocol, runtime_checkable
# from abc import abstractmethod
# T = TypeVar("T", bound="SupportsLt")
# @runtime_checkable
# class SupportsLt(Protocol):
#     __slots__ = ()

#     @abstractmethod
#     def __lt__(self: T, other: T) -> bool:
#         pass
# K = TypeVar("K", bound=SupportsLt)


K = TypeVar("K")
V = TypeVar("V")


class HeapQueue(Generic[K, V]):
    """
    q = HeapQueue()
    q.append(p, v)
    v = q.pop()
    """

    def __init__(self):
        from typing import List, Tuple

        self.heap: List[Tuple[K, V]] = []

    def __len__(self) -> int:
        return len(self.heap)

    def append(self, p: K, v: V) -> None:
        heapq.heappush(self.heap, (p, v))

    def appendpop(self, p: K, v: V) -> V:
        return heapq.heappushpop(self.heap, (p, v))[1]

    def popappend(self, p: K, v: V) -> V:
        return heapq.heapreplace(self.heap, (p, v))[1]

    def pop(self) -> V:
        return heapq.heappop(self.heap)[1]

    def front(self) -> V:
        return self.heap[0][1]


HeapQueueInt = HeapQueue[int, V]
