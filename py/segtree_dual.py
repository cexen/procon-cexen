# https://github.com/cexen/procon-cexen/blob/main/py/segtree_dual.py
import operator
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Generic, TypeVar, overload

_V = TypeVar("_V")
_X = TypeVar("_X")


class SegtreeDualCommut(Generic[_V, _X]):
    """
    v1.1 @cexen
    Based on: https://hackmd.io/@tatyam-prime/DualSegmentTree
    >>> st = SegtreeDualCommut[int, int]([0, 1, 2, 3, 4, 5], fvx=operator.pow, fxx=operator.mul, ex=1)
    >>> st.operate(2, 3, -1)
    >>> st[-1]
    5
    >>> st[:]
    [0, 1, 2, 9, 16, 5]
    """

    def __init__(
        self,
        iterable: Iterable[_V],
        fvx: Callable[[_V, _X], _V],
        fxx: Callable[[_X, _X], _X],
        ex: _X,
    ):
        """
        fvx must be Commutative: fvx(fvx(v, x), y) == fvx(fvx(v, y), x)
        fvx(fvx(v, x), y) == fvx(v, fxx(x, y))
        """

        treev = list(iterable)
        n = len(treev)
        self.r = range(n)
        self.size: int = 2 ** ((n - 1).bit_length())
        self.treev = treev
        self.treex = [ex] * (self.size - 1)
        self.fvx = fvx
        self.fxx = fxx
        self.ex = ex

    def __len__(self) -> int:
        return len(self.r)

    @overload
    def __getitem__(self, i: int) -> _V:
        """O(log n)."""
        ...

    @overload
    def __getitem__(self, i: slice) -> list[_V]:
        """O(len(i) log n)."""
        ...

    def __getitem__(self, i: int | slice) -> _V | list[_V]:
        if isinstance(i, slice):
            return [self[j] for j in self.r[i]]

        i = self.r[i]
        ans = self.treev[i]
        i = self.size - 1 + i
        while i > 0:
            i = (i - 1) >> 1
            ans = self.fvx(ans, self.treex[i])
        return ans

    def operate(self, x: _X, i: int = 0, j: int | None = None) -> None:
        """O(log n). v = f(v, x) for v in data[i:j]."""
        if j is None:
            j = len(self)
        r = self.r[i:j]
        if not len(r):
            return
        i, j = r[0], 1 + r[-1]
        if i & 1 == 1:
            self.treev[i] = self.fvx(self.treev[i], x)
        if j & 1 == 1:
            self.treev[j - 1] = self.fvx(self.treev[j - 1], x)
        i = self.size - 1 + i
        j = self.size - 1 + j
        i = i >> 1
        j = (j - 1) >> 1
        while j - i > 0:
            if i & 1 == 0:
                self.treex[i] = self.fxx(self.treex[i], x)
            if j & 1 == 0:
                self.treex[j - 1] = self.fxx(self.treex[j - 1], x)
            i = i >> 1
            j = (j - 1) >> 1


class SegtreeDualCommutCompress(SegtreeDualCommut[_V, _X]):
    """@cexen v1.2"""

    def __init__(
        self,
        iterable: Iterable[_V],
        indices: Iterable[int],
        fv: Callable[[_V, _X], _V],
        fx: Callable[[_X, _X], _X],
        ex: _X,
    ):
        super().__init__(iterable, fv, fx, ex)
        self.indices = sorted(indices)
        if len(self.treev) != len(self.indices):
            raise ValueError

    @overload
    def __getitem__(self, i: int) -> _V:
        """O(log n)."""
        ...

    @overload
    def __getitem__(self, i: slice) -> list[_V]:
        """O(len(i) log n)."""
        ...

    def __getitem__(self, i: int | slice) -> _V | list[_V]:
        if isinstance(i, slice):
            return [self[j] for j in self.r[i]]

        from bisect import bisect_right

        i = bisect_right(self.indices, i) - 1
        return super().__getitem__(i)

    def operate(self, x: _X, i: int | None = None, j: int | None = None) -> None:
        """O(log n). v = f(v, x) for v in data[i:j]."""
        from bisect import bisect_left

        if i is None:
            i = 0
        else:
            i = bisect_left(self.indices, i)
        if j is None:
            j = len(self)
        else:
            j = bisect_left(self.indices, j)
        super().operate(x, i, j)


class SegtreeDualCommutInt(SegtreeDualCommut[int, int]):
    """
    >>> seg = SegtreeDualCommutInt([0, 1, 2, 3, 4])
    >>> seg[:]
    [0, 1, 2, 3, 4]
    >>> seg.operate(100, 1, 4)
    >>> seg[:]
    [0, 101, 102, 103, 4]
    """

    def __init__(
        self,
        iterable: Iterable[int],
        fvx: Callable[[int, int], int] = operator.add,
        fxx: Callable[[int, int], int] = operator.add,
        ex: int = 0,
    ):
        super().__init__(iterable, fvx, fxx, ex)


class SegtreeDualAssign(Generic[_V]):
    """
    >>> st = SegtreeDualAssign[int]([0] * 5, ex=0)
    >>> st[2:4] = 7
    >>> st[:]
    [0, 0, 7, 7, 0]
    >>> st[3:5] = 2
    >>> st[:]
    [0, 0, 7, 2, 2]
    """

    def __init__(
        self,
        iterable: Iterable[_V],
        ex: _V,
    ):
        self.st = SegtreeDualCommut[tuple[int, _V], tuple[int, _V]](
            [(0, v) for v in iterable], max, max, (-1, ex)
        )
        self._cnt = 0

    def __len__(self) -> int:
        return len(self.st)

    @overload
    def __getitem__(self, i: int) -> _V:
        """O(log n)."""
        ...

    @overload
    def __getitem__(self, i: slice) -> list[_V]:
        """O(len(i) log n)."""
        ...

    def __getitem__(self, i: int | slice) -> _V | list[_V]:
        if isinstance(i, slice):
            return [v[1] for v in self.st[i]]
        return self.st[i][1]

    @overload
    def __setitem__(self, i: int, v: _V) -> None:
        """O(log n)."""
        ...

    @overload
    def __setitem__(self, i: slice, v: _V) -> None:
        """O(log n)."""
        ...

    def __setitem__(self, i: int | slice, v: _V) -> None:
        if isinstance(i, int):
            i = slice(i, i + 1, 1)
        if i.step not in (1, -1, None):
            raise ValueError(f"{i}")
        if i.step == -1:
            i = slice(self.st.r[i.stop] + 1, self.st.r[i.start] + 1, 1)
        self.operate(v, i.start, i.stop)

    def operate(self, x: _V, i: int = 0, j: int | None = None) -> None:
        """O(log n). data[i:j] = x."""
        self._cnt += 1
        self.st.operate((self._cnt, x), i, j)


class Segtree2DDualCommutCompress(Generic[_V, _X]):
    """
    v1.2 @cexen
    Based on: https://blog.hamayanhamayan.com/entry/2017/12/09/015937
    >>> st = Segtree2DDualCommutCompress[int, int]([(0, 0), (0, 1), (2, 0), (3, 3)], 0, fvx=operator.add, fxx=operator.add, ex=0)
    >>> st.operate(10, 0, 1, 3, 3)
    >>> st[0, 1]
    10
    >>> st[2, 0]
    0
    >>> st[3, 3]
    0
    """

    def __init__(
        self,
        yxs: Iterable[tuple[int, int]],
        v: _V,
        fvx: Callable[[_V, _X], _V],
        fxx: Callable[[_X, _X], _X],
        ex: _X,
    ):
        """
        yxs: all possible (y, x)s on subsequent operate((yl, xl), (yr, xr)) and __getitem__((y, x)).
        Other (y, x)s may cause incorrect answers.
        """
        d = defaultdict[int, list[int]](list)
        for yi, xi in sorted(set(yxs)):
            d[yi].append(xi)
        self.ys = list(d.keys())
        self.xs = list(d.values())
        self.ysize = 1 << (len(self.ys) - 1).bit_length()
        self.fvx = fvx
        self.treev = [
            SegtreeDualCommutCompress[_V, _X]([v] * len(x), x, fvx, fxx, ex)
            for x in self.xs
        ]
        self.treev = self.treev + [
            SegtreeDualCommutCompress[_V, _X]([], [], fvx, fxx, ex)
        ] * (self.ysize - len(self.ys))
        treex: list[SegtreeDualCommutCompress[_X, _X] | None] = [None] * (
            self.ysize - 1
        )
        self.treex: list[SegtreeDualCommutCompress[_X, _X]] = []
        for i in reversed(range(self.ysize - 1)):
            if (i << 1) + 1 >= self.ysize - 1:
                j = (i << 1) + 1 - (self.ysize - 1)
                x = sorted(set(self.treev[j].indices + self.treev[j + 1].indices))
            else:
                st1 = treex[(i << 1) + 1]
                st2 = treex[(i << 1) + 2]
                assert st1 is not None and st2 is not None
                x = sorted(set(st1.indices + st2.indices))
            st = SegtreeDualCommutCompress[_X, _X]([ex] * len(x), x, fxx, fxx, ex)
            treex[i] = st
            self.treex.append(st)
        self.treex.reverse()

    def __len__(self) -> int:
        if not len(self.treex):
            return 0
        return len(self.ys) * len(self.treex[0])

    def __getitem__(self, yx: tuple[int, int]) -> _V:
        """O(log^2 n)."""
        y, x = yx
        i = bisect_right(self.ys, y) - 1
        ans = self.treev[i][x]
        i = self.ysize - 1 + i
        while i > 0:
            i = (i - 1) >> 1
            ans = self.fvx(ans, self.treex[i][x])
        return ans

    def operate(
        self,
        x: _X,
        yl: int | None = None,
        xl: int | None = None,
        yr: int | None = None,
        xr: int | None = None,
    ) -> None:
        """O(log^2 n). v = f(v, x) for v in data[yl:yr, xl:xr]."""
        if yl is None:
            i = 0
        else:
            i = bisect_left(self.ys, yl)
        if yr is None:
            j = len(self.ys)
        else:
            j = bisect_left(self.ys, yr)
        if not i < j:
            return
        if i & 1 == 1:
            self.treev[i].operate(x, xl, xr)
        if j & 1 == 1:
            self.treev[j - 1].operate(x, xl, xr)
        i = self.ysize - 1 + i
        j = self.ysize - 1 + j
        i = i >> 1
        j = (j - 1) >> 1
        while j - i > 0:
            if i & 1 == 0:
                self.treex[i].operate(x, xl, xr)
            if j & 1 == 0:
                self.treex[j - 1].operate(x, xl, xr)
            i = i >> 1
            j = (j - 1) >> 1


class Segtree2DDualCommutCompressInt(Segtree2DDualCommutCompress[int, int]):
    def __init__(
        self,
        yxs: Iterable[tuple[int, int]],
        v: int = 0,
        fvx: Callable[[int, int], int] = operator.add,
        fxx: Callable[[int, int], int] = operator.add,
        ex: int = 0,
    ):
        super().__init__(yxs, v, fvx, fxx, ex)


# --------------------
