# https://github.com/cexen/procon-cexen/blob/main/py/segtree_primal.py
import operator
from bisect import bisect_right
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar, cast, overload

_V = TypeVar("_V")


class SegtreePrimal(Sequence[_V]):
    """
    v1.9 @cexen
    >>> st = SegtreePrimal[int](4, f=operator.add, e=0)
    >>> st[1:] = [4, 9, 16]
    >>> st[0] = -1
    >>> st[-1]
    16
    >>> st[:]
    [-1, 4, 9, 16]
    >>> st.grasp(1, 3)
    13
    >>> st.grasp()
    28
    >>> st.max_right(lambda v: v < 28, l=0)
    3
    >>> st.max_right(lambda v: v <= 28, l=0)
    4
    >>> st.min_left(lambda v: v < 25)
    3
    >>> st.min_left(lambda v: v <= 25)
    2
    """

    def __init__(self, n: int, f: Callable[[_V, _V], _V], e: _V):
        """all(f(v, e) == f(e, v) == v for v in data)"""
        self.r = range(n)
        self.size = 1 << (n - 1).bit_length()
        self.tree = [e] * (2 * self.size - 1)
        self.f = f
        self.e = e

    def __len__(self) -> int:
        return len(self.r)

    def __repr__(self) -> str:
        return repr(self[:])

    def grasp(self, i: int = 0, j: int | None = None) -> _V:
        """O(log n). reduce(f, data[i:j], e)."""
        if j is None:
            j = len(self)
        r_ = self.r[i:j]
        if not len(r_):
            return self.e
        i = self.size - 1 + r_[0]
        j = self.size - 1 + r_[-1] + 1

        vl = list[_V]()
        vr = list[_V]()
        while i < j:
            if i & 1 == 0:
                vl.append(self.tree[i])
            if j & 1 == 0:
                vr.append(self.tree[j - 1])
            i = i >> 1
            j = (j - 1) >> 1
        ans = self.e
        for v in vl:
            ans = self.f(ans, v)
        for v in reversed(vr):
            ans = self.f(ans, v)
        return ans

    @overload
    def __getitem__(self, i: int) -> _V:
        """O(1)."""
        ...

    @overload
    def __getitem__(self, i: slice) -> list[_V]:
        """O(len(i))."""
        ...

    def __getitem__(self, i: int | slice) -> _V | list[_V]:
        r = self.r[i]
        if isinstance(r, int):
            return self.tree[self.size - 1 + r]
        else:
            return [self.tree[self.size - 1 + ri] for ri in r]

    @overload
    def __setitem__(self, i: int, v: _V) -> None:
        """O(log n)."""
        ...

    @overload
    def __setitem__(self, i: slice, v: Iterable[_V]) -> None:
        """O(len(i) log n). Extra elements will be ignored."""
        ...

    def __setitem__(self, i: int | slice, v: _V | Iterable[_V]) -> None:
        if isinstance(i, int):
            j = self.size - 1 + self.r[i]
            self.tree[j] = cast(_V, v)
            j = (j - 1) >> 1
            while j >= 0:
                vl = self.tree[(j << 1) + 1]
                vr = self.tree[(j << 1) + 2]
                self.tree[j] = self.f(vl, vr)
                j = (j - 1) >> 1
        else:
            r = self.r[i]
            assert isinstance(v, Iterable)
            q = deque[int]()
            s = set[int]()
            for ri, vi in zip(r, v):
                j = self.size - 1 + ri
                self.tree[j] = vi
                nj = (j - 1) >> 1
                if nj >= 0 and nj not in s:
                    q.append(nj)
                    s.add(nj)
            while q:
                j = q.popleft()
                vl = self.tree[(j << 1) + 1]
                vr = self.tree[(j << 1) + 2]
                self.tree[j] = self.f(vl, vr)
                nj = (j - 1) >> 1
                if nj >= 0 and nj not in s:
                    q.append(nj)
                    s.add(nj)

    def _rebuild(self) -> None:
        for j in reversed(range(self.size - 1)):
            vl = self.tree[(j << 1) + 1]
            vr = self.tree[(j << 1) + 2]
            self.tree[j] = self.f(vl, vr)

    def setall(self, values: Iterable[_V]) -> None:
        """O(n). Faster than self[:] = values."""
        j = -1
        off = self.size - 1
        for j, v in enumerate(values):
            self.tree[off + j] = v
        assert j + 1 == len(self)
        for j in range(len(self), self.size):
            self.tree[off + j] = self.e
        self._rebuild()

    def max_right(self, g: Callable[[_V], bool], l: int = 0) -> int:
        """
        O(log n). Returns one r s.t. l <= r <= n and g(grasp(l, r)) and not g(grasp(l, r+1)).
        g(e) is not calculated but assumed to be True.
        cf. https://github.com/atcoder/ac-library/blob/master/atcoder/segtree.hpp
        """
        assert 0 <= l <= len(self)
        if l == len(self):
            return len(self)
        r = l + self.size  # 1-indexed, exclusive
        v = self.e
        while True:
            while not r & 1:
                r >>= 1
            nv = self.f(v, self.tree[r - 1])
            if not g(nv):
                while r < self.size:
                    r <<= 1
                    nv = self.f(v, self.tree[r - 1])
                    if g(nv):
                        v = nv
                        r |= 1
                return r - self.size  # 0-indexed, exclusive
            v = nv
            r |= 1
            if (r & -r) == r:
                return len(self)

    def min_left(self, g: Callable[[_V], bool], r: int | None = None):
        """
        O(log n). Returns one l s.t. 0 <= l <= r and g(grasp(l, r)) and not g(grasp(l-1, r)).
        g(e) is not calculated but assumed to be True.
        cf. https://github.com/atcoder/ac-library/blob/master/atcoder/segtree.hpp
        """
        if r is None:
            r = len(self)
        assert 0 <= r <= len(self)
        if r == 0:
            return 0
        l = r + self.size - 1  # 1-indexed, exclusive
        v = self.e
        while True:
            while l & 1 and l > 1:
                l >>= 1
            nv = self.f(self.tree[l - 1], v)
            if not g(nv):
                while l < self.size:
                    l = (l << 1) | 1
                    nv = self.f(self.tree[l - 1], v)
                    if g(nv):
                        v = nv
                        l ^= 1
                return l - self.size + 1  # 0-indexed, inclusive
            if l & (-l) == l:
                return 0
            v = nv
            l -= 1


class SegtreePrimalInt(SegtreePrimal[int]):
    def __init__(self, n: int, f: Callable[[int, int], int] = operator.add, e: int = 0):
        super().__init__(n, f, e)


class Segtree2DPrimal(Sequence[Sequence[_V]]):
    """
    v1.5 @cexen
    """

    Index = int | slice

    def __init__(self, h: int, w: int, f: Callable[[_V, _V], _V], e: _V):
        """all(f(v, e) == f(e, v) == v for v in data)"""
        self.ry = range(h)
        self.rx = range(w)
        self.sizey = 1 << (h - 1).bit_length()
        self.sizex = 1 << (w - 1).bit_length()
        self.tree = [[e] * (2 * self.sizex - 1) for _ in range(2 * self.sizey - 1)]
        self.f = f
        self.e = e

    def __len__(self) -> int:
        return len(self.ry) * len(self.rx)

    def __repr__(self) -> str:
        return repr(self[:, :])

    def grasp(
        self,
        yl: int = 0,
        xl: int = 0,
        yr: int | None = None,
        xr: int | None = None,
    ) -> _V:
        """O(log h log w). reduce(f, data[yl:yr, xl:xr], e)."""
        if yr is None:
            yr = len(self.ry)
        if xr is None:
            xr = len(self.rx)
        ry_ = self.ry[yl:yr]
        rx_ = self.rx[xl:xr]
        if not len(ry_) or not len(rx_):
            return self.e
        yl = self.sizey - 1 + ry_[0]
        yr = self.sizey - 1 + ry_[-1] + 1
        xl = self.sizex - 1 + rx_[0]
        xr = self.sizex - 1 + rx_[-1] + 1

        def graspx(
            tree: list[_V], f: Callable[[_V, _V], _V], e: _V, i: int, j: int
        ) -> _V:
            vl = list[_V]()
            vr = list[_V]()
            while i < j:
                if i & 1 == 0:
                    vl.append(tree[i])
                if j & 1 == 0:
                    vr.append(tree[j - 1])
                i = i >> 1
                j = (j - 1) >> 1
            ans = e
            for v in vl:
                ans = f(ans, v)
            for v in reversed(vr):
                ans = f(ans, v)
            return ans

        vl = list[_V]()
        vr = list[_V]()
        while yl < yr:
            if yl & 1 == 0:
                ansx = graspx(self.tree[yl], self.f, self.e, xl, xr)
                vl.append(ansx)
            if yr & 1 == 0:
                ansx = graspx(self.tree[yr - 1], self.f, self.e, xl, xr)
                vr.append(ansx)
            yl = yl >> 1
            yr = (yr - 1) >> 1
        ans = self.e
        for v in vl:
            ans = self.f(ans, v)
        for v in reversed(vr):
            ans = self.f(ans, v)
        return ans

    @overload
    def __getitem__(self, yx: tuple[int, int]) -> _V:
        """O(1)."""
        ...

    @overload
    def __getitem__(self, y: int) -> list[_V]:
        """O(w)."""
        ...

    @overload
    def __getitem__(self, yx: tuple[int, slice]) -> list[_V]:
        """O(len(x))."""
        ...

    @overload
    def __getitem__(self, yx: tuple[slice, int]) -> list[_V]:
        """O(len(y))."""
        ...

    @overload
    def __getitem__(self, y: slice) -> list[list[_V]]:
        """O(len(y) * w)."""
        ...

    @overload
    def __getitem__(self, yx: tuple[slice, slice]) -> list[list[_V]]:
        """O(len(y) * len(x))."""
        ...

    def __getitem__(  # type: ignore
        self, yx: Index | tuple[Index, Index]
    ) -> _V | list[_V] | list[list[_V]]:
        if not isinstance(yx, tuple):
            yx = (yx, slice(None))
        y, x = yx
        ry = self.ry[y]
        if isinstance(ry, range):
            return [self[yi, x] for yi in ry]  # type: ignore
        rx = self.rx[x]
        if isinstance(rx, range):
            return [self[y, xi] for xi in rx]  # type: ignore
        return self.tree[self.sizey - 1 + ry][self.sizex - 1 + rx]

    @overload
    def __setitem__(self, yx: tuple[int, int], v: _V) -> None:
        """O(log h log w)."""
        ...

    @overload
    def __setitem__(self, y: int, v: Iterable[_V]) -> None:
        """O(w log h log w)."""
        ...

    @overload
    def __setitem__(self, yx: tuple[int, slice], v: Iterable[_V]) -> None:
        """O(len(x) log h log w). Extra elements will be ignored."""
        ...

    @overload
    def __setitem__(self, y: slice, v: Iterable[Iterable[_V]]) -> None:
        """O(len(y) w log h log w). Extra elements will be ignored."""
        ...

    @overload
    def __setitem__(self, y: tuple[slice, slice], v: Iterable[Iterable[_V]]) -> None:
        """O(len(y) len(x) log h log w). Extra elements will be ignored."""
        ...

    def __setitem__(self, yx: Index | tuple[Index, Index], v) -> None:  # type: ignore
        if not isinstance(yx, tuple):
            yx = (yx, slice(None))
        y, x = yx
        if isinstance(y, int):
            ry = range(self.ry[y], self.ry[y] + 1)
            v = [v]
        else:
            ry = self.ry[y]
        if isinstance(x, int):
            rx = range(self.rx[x], self.rx[x] + 1)
            v = [[vi] for vi in v]
        else:
            rx = self.rx[x]

        w = 2 * self.sizex

        q = deque[int]()
        s = set[int]()

        def addnext(q: deque[int], s: set[int], i: int, j: int):
            ni = (i - 1) >> 1
            nj = (j - 1) >> 1
            nk = ni * w + j
            if ni >= 0 and nk not in s:
                q.append(nk)
                s.add(nk)
            nk = i * w + nj
            if nj >= 0 and nk not in s:
                q.append(nk)
                s.add(nk)

        for i, row in zip(ry, v):
            i = self.sizey - 1 + i
            for j, vi in zip(rx, row):
                j = self.sizex - 1 + j
                self.tree[i][j] = vi
                addnext(q, s, i, j)
        while q:
            k = q.popleft()
            i, j = divmod(k, w)
            if i >= self.sizey - 1:
                vl = self.tree[i][(j << 1) + 1]
                vr = self.tree[i][(j << 1) + 2]
                self.tree[i][j] = self.f(vl, vr)
            elif j >= self.sizex - 1:
                vl = self.tree[(i << 1) + 1][j]
                vr = self.tree[(i << 1) + 2][j]
                self.tree[i][j] = self.f(vl, vr)
            else:
                vll = self.tree[(i << 1) + 1][(j << 1) + 1]
                vlr = self.tree[(i << 1) + 1][(j << 1) + 2]
                vrl = self.tree[(i << 1) + 2][(j << 1) + 1]
                vrr = self.tree[(i << 1) + 2][(j << 1) + 2]
                self.tree[i][j] = self.f(self.f(vll, vlr), self.f(vrl, vrr))
            addnext(q, s, i, j)

    def _rebuild(self) -> None:
        for i in reversed(range(self.sizey - 1, 2 * self.sizey - 1)):
            for j in reversed(range(self.sizex - 1)):
                vl = self.tree[i][(j << 1) + 1]
                vr = self.tree[i][(j << 1) + 2]
                self.tree[i][j] = self.f(vl, vr)
        for i in reversed(range(self.sizey - 1)):
            for j in reversed(range(self.sizex - 1, 2 * self.sizex - 1)):
                vl = self.tree[(i << 1) + 1][j]
                vr = self.tree[(i << 1) + 2][j]
                self.tree[i][j] = self.f(vl, vr)
            for j in reversed(range(self.sizex - 1)):
                vll = self.tree[(i << 1) + 1][(j << 1) + 1]
                vlr = self.tree[(i << 1) + 1][(j << 1) + 2]
                vrl = self.tree[(i << 1) + 2][(j << 1) + 1]
                vrr = self.tree[(i << 1) + 2][(j << 1) + 2]
                self.tree[i][j] = self.f(self.f(vll, vlr), self.f(vrl, vrr))

    def setall(self, values: Iterable[Iterable[_V]]) -> None:
        """O(h*w). Faster than self[:, :] = values."""
        i = -1
        offy = self.sizey - 1
        offx = self.sizex - 1
        for i, row in enumerate(values):
            j = -1
            for j, v in enumerate(row):
                self.tree[offy + i][offx + j] = v
            assert j + 1 == len(self.rx)
        assert i + 1 == len(self.ry)
        for i in range(len(self.ry), self.sizey):
            for j in range(self.sizex):
                self.tree[offy + i][offx + j] = self.e
        for i in range(self.sizey):
            for j in range(len(self.rx), self.sizex):
                self.tree[offy + i][offx + j] = self.e
        self._rebuild()


class Segtree2DPrimalInt(Segtree2DPrimal[int]):
    """
    >>> st = Segtree2DPrimalInt(3, 4)
    >>> st[2, 1:] = [4, 9, 16]
    >>> st[:]
    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 4, 9, 16]]
    >>> st[1, 0] = 1
    >>> st[:, :]
    [[0, 0, 0, 0], [1, 0, 0, 0], [0, 4, 9, 16]]
    >>> st.grasp(0, 1, 3, 3)
    13
    """

    def __init__(
        self, h: int, w: int, f: Callable[[int, int], int] = operator.add, e: int = 0
    ):
        super().__init__(h, w, f, e)


class SegtreePrimalCompress(SegtreePrimal[_V]):
    """
    v1.2 @cexen
    """

    def __init__(self, indices: Iterable[int], f: Callable[[_V, _V], _V], e: _V):
        """all(f(v, e) == f(e, v) == v for v in data)"""
        _indices = sorted(set(indices))
        n = len(_indices)
        super().__init__(n, f, e)
        self.indices = _indices
        self.vi = {v: i for i, v in enumerate(_indices)}

    def grasp(self, i: int | None = None, j: int | None = None) -> _V:
        """O(log n). reduce(f, data[i:j], e)."""
        if i is None:
            i = 0
        else:
            i = bisect_right(self.indices, i - 1)
        if j is None:
            j = len(self)
        else:
            j = bisect_right(self.indices, j - 1)
        return super().grasp(i, j)

    @overload
    def __getitem__(self, k: int) -> _V:
        """O(1)."""
        ...

    @overload
    def __getitem__(self, k: slice) -> list[_V]:
        """
        O(len(k) + log n). Requires that k.step == 1.
        Note that len(return value) might not be == len(i).
        """
        ...

    def __getitem__(self, k: int | slice) -> _V | list[_V]:
        i: int | slice | None
        if isinstance(k, int):
            i = self.vi.get(k)
            if i is None:
                return self.e
        elif isinstance(k, slice):
            if k.step not in (None, 1):
                raise ValueError("Requires that k.step == 1")
            l = 0 if k.start is None else self.vi.get(k.start)
            if l is None:
                l = bisect_right(self.indices, k.start - 1)
            r = len(self.vi) if k.stop is None else self.vi.get(k.stop)
            if r is None:
                r = bisect_right(self.indices, k.stop - 1)
            i = slice(l, r)
        else:
            raise TypeError
        return super().__getitem__(i)

    @overload
    def __setitem__(self, k: int, v: _V) -> None:
        """O(log n)."""
        ...

    @overload
    def __setitem__(self, k: slice, v: Iterable[_V]) -> None:
        """O(len(k) + log n). Requires that k.step == 1. Extra elements will be ignored."""
        ...

    def __setitem__(self, k: int | slice, v: _V | Iterable[_V]) -> None:
        i: int | slice
        if isinstance(k, int):
            i = self.vi[k]
        elif isinstance(k, slice):
            if k.step not in (None, 1):
                raise ValueError("Requires that k.step == 1")
            l = 0 if k.start is None else self.vi.get(k.start)
            if l is None:
                l = bisect_right(self.indices, k.start - 1)
            r = len(self.vi) if k.stop is None else self.vi.get(k.stop)
            if r is None:
                r = bisect_right(self.indices, k.stop - 1)
            i = slice(l, r)
        else:
            raise TypeError
        super().__setitem__(i, v)  # type: ignore


class SegtreePrimalCompressInt(SegtreePrimalCompress[int]):
    """
    v1.0 @cexen
    >>> seg = SegtreePrimalCompressInt([3, 1, 4, 1, 5, 9, 2])
    >>> len(seg)
    6
    >>> seg[2] = 10
    >>> seg[4:9] = [100, 1000]
    >>> seg[:]
    [0, 10, 0, 100, 1000, 0]
    >>> seg.grasp(1, 2)
    0
    >>> seg.grasp(1, 3)
    10
    >>> seg.grasp(4)
    1100
    >>> seg.grasp(9)
    0
    >>> seg.grasp(2, 5)
    110
    """

    def __init__(
        self,
        indices: Iterable[int],
        f: Callable[[int, int], int] = operator.add,
        e: int = 0,
    ):
        super().__init__(indices, f, e)


class Segtree2DPrimalCompress(Sequence[Sequence[_V]]):
    """
    v1.3 @cexen
    Based on: https://blog.hamayanhamayan.com/entry/2017/12/09/015937
    """

    Index = int | slice

    def __init__(
        self, yxs: Iterable[tuple[int, int]], f: Callable[[_V, _V], _V], e: _V
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
        self.f = f
        self.e = e
        self.yi = {y: i for i, y in enumerate(self.ys)}

        self.tree = [SegtreePrimalCompress[_V]([], f, e)] * (self.ysize - len(self.ys))
        self.tree += [SegtreePrimalCompress[_V](x, f, e) for x in reversed(self.xs)]  # type: ignore
        n = 2 * self.ysize - 1
        for i in reversed(range(self.ysize - 1)):
            if not 0 <= n - 1 - ((i << 1) + 1) < len(self.tree):
                raise IndexError(n - 1 - ((i << 1) + 1), len(self.tree))
            st1 = self.tree[n - 1 - ((i << 1) + 1)]
            st2 = self.tree[n - 1 - ((i << 1) + 2)]
            x = sorted(set(st1.indices + st2.indices))
            st = SegtreePrimalCompress[_V](x, f, e)
            self.tree.append(st)
        self.tree.reverse()

    def __len__(self) -> int:
        if not len(self.tree):
            return 0
        return len(self.ys) * len(self.tree[0])

    def grasp(
        self,
        yl: int | None = None,
        xl: int | None = None,
        yr: int | None = None,
        xr: int | None = None,
    ) -> _V:
        """O(log h log w). reduce(f, data[yl:yr, xl:xr], e)."""
        if yl is None:
            i = 0
        else:
            i = bisect_right(self.ys, yl - 1)
        if yr is None:
            j = len(self.ys)
        else:
            j = bisect_right(self.ys, yr - 1)
        i += self.ysize - 1
        j += self.ysize - 1

        vl = list[_V]()
        vr = list[_V]()
        while i < j:
            if not i & 1:
                vl.append(self.tree[i].grasp(xl, xr))
            if not j & 1:
                vr.append(self.tree[j - 1].grasp(xl, xr))
            i = i >> 1
            j = (j - 1) >> 1
        ans = self.e
        for v in vl:
            ans = self.f(ans, v)
        for v in reversed(vr):
            ans = self.f(ans, v)
        return ans

    @overload
    def __getitem__(self, yx: tuple[int, int]) -> _V:
        """O(1)."""
        ...

    @overload
    def __getitem__(self, y: int) -> list[_V]:
        """O(w)."""
        ...

    @overload
    def __getitem__(self, yx: tuple[int, slice]) -> list[_V]:
        """O(len(x) + log w)."""
        ...

    @overload
    def __getitem__(self, yx: tuple[slice, int]) -> list[_V]:
        """O(len(y) + log h)."""
        ...

    @overload
    def __getitem__(self, y: slice) -> list[list[_V]]:
        """O(len(y) * w + log h)."""
        ...

    @overload
    def __getitem__(self, yx: tuple[slice, slice]) -> list[list[_V]]:
        """O(len(y) * len(x) + log wh)."""
        ...

    def __getitem__(  # type: ignore
        self, yx: Index | tuple[Index, Index]
    ) -> _V | list[_V] | list[list[_V]]:
        if not isinstance(yx, tuple):
            yx = (yx, slice(None))
        y, x = yx
        if isinstance(y, int):
            i = self.yi.get(y)
            if i is None:
                ans = self.e if isinstance(x, int) else list[_V]()
                return ans
            return self.tree[self.ysize - 1 + i][x]
        elif isinstance(y, slice):
            if y.step not in (None, 1):
                raise ValueError("Requires that y.step == 1")
            l = 0 if y.start is None else self.yi.get(y.start)
            if l is None:
                l = bisect_right(self.ys, y.start - 1)
            r = len(self.yi) if y.stop is None else self.yi.get(y.stop)
            if r is None:
                r = bisect_right(self.ys, y.stop - 1)
            return [self.tree[self.ysize - 1 + i][x] for i in range(l, r)]  # type: ignore
        else:
            raise TypeError

    @overload
    def __setitem__(self, yx: tuple[int, int], v: _V) -> None:
        """O(log h log w)."""
        ...

    @overload
    def __setitem__(self, y: int, v: Iterable[_V]) -> None:
        """O(w log h log w)."""
        ...

    @overload
    def __setitem__(self, yx: tuple[int, slice], v: Iterable[_V]) -> None:
        """O(len(x) log h log w). Extra elements will be ignored."""
        ...

    @overload
    def __setitem__(self, y: slice, v: Iterable[Iterable[_V]]) -> None:
        """O(len(y) w log h log w). Extra elements will be ignored."""
        ...

    @overload
    def __setitem__(self, y: tuple[slice, slice], v: Iterable[Iterable[_V]]) -> None:
        """O(len(y) len(x) log h log w). Extra elements will be ignored."""
        ...

    def __setitem__(self, yx: Index | tuple[Index, Index], v) -> None:  # type: ignore
        if not isinstance(yx, tuple):
            yx = (yx, slice(None))
        y, x = yx
        j: int | None
        if isinstance(y, int):
            j = self.ysize - 1 + self.yi[y]
            self.tree[j][x] = v
            j = (j - 1) >> 1
            while j >= 0:
                if isinstance(x, slice):
                    vls = self.tree[(j << 1) + 1][x]
                    vrs = self.tree[(j << 1) + 2][x]
                    self.tree[j][x] = [self.f(vl, vr) for vl, vr in zip(vls, vrs)]
                else:
                    vl = self.tree[(j << 1) + 1][x]
                    vr = self.tree[(j << 1) + 2][x]
                    self.tree[j][x] = self.f(vl, vr)
                j = (j - 1) >> 1
        elif isinstance(y, slice):
            if y.step not in (None, 1):
                raise ValueError("Requires that y.step == 1")
            i = 0 if y.start is None else self.yi.get(y.start)
            if i is None:
                i = bisect_right(self.ys, y.start - 1)
            j = len(self.yi) if y.stop is None else self.yi.get(y.stop)
            if j is None:
                j = bisect_right(self.ys, y.stop - 1)
            r = range(i, j)
            q = deque[int]()
            s = set[int]()
            for ri, vi in zip(r, v):
                j = self.ysize - 1 + ri
                self.tree[j][x] = vi
                nj = (j - 1) >> 1
                if nj >= 0 and nj not in s:
                    q.append(nj)
                    s.add(nj)
            while q:
                j = q.popleft()
                if isinstance(x, slice):
                    vls = self.tree[(j << 1) + 1][x]
                    vrs = self.tree[(j << 1) + 2][x]
                    self.tree[j][x] = [self.f(vl, vr) for vl, vr in zip(vls, vrs)]
                else:
                    vl = self.tree[(j << 1) + 1][x]
                    vr = self.tree[(j << 1) + 2][x]
                    self.tree[j][x] = self.f(vl, vr)
                nj = (j - 1) >> 1
                if nj >= 0 and nj not in s:
                    q.append(nj)
                    s.add(nj)
        else:
            raise TypeError


class Segtree2DPrimalCompressInt(Segtree2DPrimalCompress[int]):
    """
    v1.0 @cexen
    >>> st = Segtree2DPrimalCompressInt([(0, 0), (0, 1), (2, 0), (3, 3)])
    >>> st[2, 0] = 10
    >>> st.grasp(0, 0, 3, 3)
    10
    >>> st.grasp(0, 0, 2, 3)
    0
    >>> st.grasp(0, 0, 3, 0)
    0
    >>> st.grasp(2, 0, 3, 3)
    10
    >>> st.grasp(2, 1, 3, 3)
    0
    >>> st.grasp(3, 0, 3, 3)
    0
    """

    def __init__(
        self,
        yxs: Iterable[tuple[int, int]],
        f: Callable[[int, int], int] = operator.add,
        e: int = 0,
    ):
        super().__init__(yxs, f, e)


# --------------------


def solve_yosupojudge_point_add_range_sum():
    """
    Library Checker: Point Add Range Sum
    https://judge.yosupo.jp/problem/point_add_range_sum
    """
    N, Q = map(int, input().split())
    A = [int(v) for v in input().split()]
    seg = SegtreePrimalInt(N)
    seg.setall(A)
    ans = list[int]()
    for _ in range(Q):
        q, *args = map(int, input().split())
        if q == 0:
            p, x = args
            seg[p] += x
        elif q == 1:
            l, r = args
            ans.append(seg.grasp(l, r))
        else:
            raise RuntimeError
    for a in ans:
        print(a)


def solve_yosupojudge_point_add_rectangle_sum():
    """
    TLE
    Library Checker: Point Add Rectangle Sum
    https://judge.yosupo.jp/problem/point_add_rectangle_sum
    """
    N, Q = map(int, input().split())
    XYW = list[tuple[int, int, int]]()
    for _ in range(N):
        x, y, w = map(int, input().split())
        XYW.append((x, y, w))
    yxs = set((y, x) for x, y, w in XYW)
    queries = list[tuple[int, list[int]]]()
    for _ in range(Q):
        q, *args = map(int, input().split())
        if q == 0:
            x, y, w = args
            yxs.add((y, x))
        queries.append((q, args))
    seg = Segtree2DPrimalCompressInt(yxs)
    for x, y, w in XYW:
        seg[y, x] += w
    ans = list[int]()
    for q, args in queries:
        if q == 0:
            x, y, w = args
            seg[y, x] += w
        elif q == 1:
            l, d, r, u = args
            ans.append(seg.grasp(d, l, u, r))
        else:
            raise RuntimeError
    for a in ans:
        print(a)


def solve_atcoder_practice2_j():
    """
    AtCoder Library Practice Contest: J - Segment Tree
    https://atcoder.jp/contests/practice2/tasks/practice2_j
    """
    N, Q = map(int, input().split())
    A = [int(v) for v in input().split()]
    seg = SegtreePrimalInt(N, max, -(10**9))
    seg.setall(A)
    anses = list[int]()
    for _ in range(Q):
        t, *args = map(int, input().split())
        if t == 1:
            x, v = args
            seg[x - 1] = v
        elif t == 2:
            l, r = args
            anses.append(seg.grasp(l - 1, r))
        elif t == 3:
            x, v = args
            anses.append(1 + seg.max_right(lambda m: m < v, l=x - 1))
        else:
            raise RuntimeError
    for ans in anses:
        print(ans)


def solve_atcoder_abc287_g():
    """
    AtCoder Beginner Contest 287: G - Balance Update Query
    https://atcoder.jp/contests/abc287/tasks/abc287_g
    """
    N = int(input())
    A = list[int]()
    B = list[int]()
    for _ in range(N):
        a, b = map(int, input().split())
        A.append(a)
        B.append(b)
    S = A.copy()
    Q = int(input())
    qs = list[tuple[int, list[int]]]()
    for _ in range(Q):
        p, *args = map(int, input().split())
        if p == 1:
            x, y = args
            S.append(y)
        qs.append((p, args))
    S = sorted(set(S))
    d = {v: i for i, v in enumerate(S)}
    V = tuple[int, int]

    def f(x: V, y: V) -> V:
        return (x[0] + y[0], x[1] + y[1])

    seg = SegtreePrimal[V](len(d), f, (0, 0))
    for a, b in zip(A, B):
        ia = d[a]
        k, s = seg[ia]
        seg[ia] = (k + b, s + a * b)
    anses = list[int]()
    for p, args in qs:
        if p == 1:
            x, y = args
            a = A[x - 1]
            A[x - 1] = na = y
            b = B[x - 1]
            ia = d[a]
            ina = d[na]
            k, s = seg[ia]
            seg[ia] = (k - b, s - a * b)
            k, s = seg[ina]
            seg[ina] = (k + b, s + na * b)
        elif p == 2:
            x, y = args
            b = B[x - 1]
            B[x - 1] = nb = y
            a = A[x - 1]
            ia = d[a]
            k, s = seg[ia]
            seg[ia] = (k + nb - b, s + a * (nb - b))
        elif p == 3:
            (x,) = args
            l = seg.min_left(lambda v: v[0] < x)
            if l == 0:
                anses.append(-1)
            else:
                k, s = seg.grasp(l, len(seg))
                anses.append((x - k) * S[l - 1] + s)
        else:
            raise RuntimeError
    print(*anses, sep="\n")
