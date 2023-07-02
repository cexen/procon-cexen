# https://github.com/cexen/procon-cexen/blob/main/py/heapqueue.py
from typing import TypeVar, Generic, Iterable, List
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


K_ = TypeVar("K_")


class HeapQueue(Generic[K_]):
    """
    v1.1 @cexen
    >>> q = HeapQueue([3, 4, 4])
    >>> q.append(3)
    >>> q.append(2)
    >>> q.append(5)
    >>> len(q)
    6
    >>> q.pop()
    2
    >>> q.pop()
    3
    >>> q.pop()
    3
    >>> q.front()
    4
    >>> len(q)
    3
    >>> q.pop()
    4
    >>> q.pop()
    4
    >>> bool(q)
    True
    >>> q.pop()
    5
    >>> len(q)
    0
    >>> bool(q)
    False
    """

    def __init__(self, iterable: Iterable[K_] = []):
        self.heap: List[K_] = list(iterable)
        heapq.heapify(self.heap)

    def __len__(self) -> int:
        return len(self.heap)

    def append(self, v: K_) -> None:
        heapq.heappush(self.heap, v)

    def appendpop(self, v: K_) -> K_:
        return heapq.heappushpop(self.heap, v)

    def popappend(self, v: K_) -> K_:
        return heapq.heapreplace(self.heap, v)

    def pop(self) -> K_:
        return heapq.heappop(self.heap)

    def front(self) -> K_:
        return self.heap[0]


HeapQueueInt = HeapQueue[int]


class HeapQueueDiscardable(Generic[K_]):
    """
    v1.0 @cexen
    >>> q = HeapQueueDiscardable([3, 4, 4])
    >>> q.append(5)
    >>> len(q)
    4
    >>> q.discard(4)
    >>> q.discard(3)
    >>> q.pop()
    4
    >>> q.pop()
    5
    """

    def __init__(self, iterable: Iterable[K_] = []):
        self.heap: List[K_] = list(iterable)
        heapq.heapify(self.heap)
        self.disc: List[K_] = []

    def __len__(self) -> int:
        return len(self.heap)

    def _sync(self) -> None:
        while self.disc and self.heap[0] == self.disc[0]:
            heapq.heappop(self.heap)
            heapq.heappop(self.disc)

    def append(self, v: K_) -> None:
        heapq.heappush(self.heap, v)

    def pop(self) -> K_:
        v = heapq.heappop(self.heap)
        self._sync()
        return v

    def discard(self, v: K_) -> None:
        """
        `v` must exist in heap.
        """
        if self.heap[0] == v:
            heapq.heappop(self.heap)
            self._sync()
        else:
            heapq.heappush(self.disc, v)

    def front(self) -> K_:
        return self.heap[0]


HeapQueueDiscardableInt = HeapQueueDiscardable[int]
