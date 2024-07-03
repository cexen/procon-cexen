# https://github.com/cexen/procon-cexen/blob/main/py/swag.py
from collections.abc import Callable, Iterable, Sequence
from functools import reduce
from typing import TypeVar, overload

_T = TypeVar("_T")


class AggregatedQueue(Sequence[_T]):
    """
    v1.0 @cexen.
    Sliding Window Aggregation.
    cf. https://motsu-xe.hatenablog.com/entry/2021/05/13/224016
    cf. https://motsu-xe.hatenablog.com/entry/2019/12/06/192424

    See AggregatedQueueInt() for usage.
    """

    def __init__(
        self,
        f: Callable[[_T, _T], _T],
        e: _T,
        iterable: Iterable[_T] = [],
    ):
        """
        O(len(iterable)).
        Required: f(f(u, v), w) == f(u, f(v, w)).
        Required: f(v, e) == f(e, v) == f(v).
        """
        self.f = f
        self.e = e
        self.qleft = list[_T]()
        self.qright = list(iterable)
        self.aleft = list[_T]()
        self.aright: _T = reduce(f, self.qright, e)

    @property
    def value(self) -> _T:
        """O(1). Returns reduce(q, f, e)."""
        if not self.aleft:
            return self.aright
        return self.f(self.aleft[-1], self.aright)

    def append(self, value: _T) -> None:
        """O(1)."""
        self.qright.append(value)
        self.aright = self.f(self.aright, value)

    def popleft(self) -> _T:
        """O(1) amortized."""
        if not self.qleft and not self.qright:
            return self.e
        if not self.qleft:
            assert not self.aleft
            acc = self.e
            for v in reversed(self.qright):
                acc = self.f(v, acc)
                self.qleft.append(v)
                self.aleft.append(acc)
            self.qright.clear()
            self.aright = self.e
        self.aleft.pop()
        return self.qleft.pop()

    def clear(self) -> None:
        """O(1)."""
        self.qleft.clear()
        self.qright.clear()
        self.aleft.clear()
        self.aright = self.e

    def __len__(self) -> int:
        """O(1)."""
        return len(self.qleft) + len(self.qright)

    @overload
    def __getitem__(self, index: int) -> _T:
        """O(1)."""

    @overload
    def __getitem__(self, index: slice) -> list[_T]:
        """O(len(i))."""

    def __getitem__(self, index: int | slice) -> _T | list[_T]:
        i = range(len(self))[index]
        if isinstance(i, range):
            return [self[ii] for ii in i]
        if i < len(self.qleft):
            return self.qleft[-1 - i]
        return self.qright[i - len(self.qleft)]


class AggregatedQueueInt(AggregatedQueue[int]):
    """
    >>> q = AggregatedQueueInt(f=min, e=10**9, iterable=[3, 1, 4])
    >>> len(q)
    3
    >>> q[:]
    [3, 1, 4]
    >>> q[0]
    3
    >>> q.value  # min(3, 1, 4)
    1
    >>> q.popleft()
    3
    >>> q[:]
    [1, 4]
    >>> q.value
    1
    >>> q.popleft()
    1
    >>> q[:]
    [4]
    >>> q.value
    4
    >>> q.append(1)
    >>> q[:]
    [4, 1]
    >>> q.value
    1
    >>> q.clear()
    >>> q[:]
    []
    >>> q.value
    1000000000
    """

    def __init__(
        self,
        f: Callable[[int, int], int] = min,
        e: int = 10**18,
        iterable: Iterable[int] = [],
    ):
        super().__init__(f, e, iterable)


# --------------------
