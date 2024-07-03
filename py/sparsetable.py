# https://github.com/cexen/procon-cexen/blob/main/py/sparsetable.py
from collections.abc import Callable, Iterable, Sequence
from itertools import accumulate
from typing import TypeVar, overload

_T = TypeVar("_T")


class SparseTable(Sequence[_T]):
    """
    v1.1 @cexen.
    Works for associative & idenpotent f(x, y).
    cf. https://atcoder.jp/contests/abc282/editorial/5403
    cf. https://ikatakos.com/pot/programming_algorithm/data_structure/sparse_table

    >>> st = SparseTable([3, 1, 4, 1, 5], f=min)
    >>> st[2]
    4
    >>> st[:]
    [3, 1, 4, 1, 5]
    >>> st.grasp()  # min(3, 1, 4, 1, 5)
    1
    >>> st.grasp(3)  # min(1, 5)
    1
    >>> st.grasp(4, 5)  # min(5)
    5
    """

    def __init__(self, data: Iterable[_T], f: Callable[[_T, _T], _T]):
        """
        O(n log n).
        Required: f(f(x, y), z) == f(x, f(y, z)).
        Required: f(x, x) == x.
        """
        self.table = [list(data)]
        self.n = len(self.table[0])
        self.f = f
        for b in range(self.n.bit_length() - 1):
            w = 1 << b
            prow = self.table[b]
            row = [f(prow[i], prow[i + w]) for i in range(self.n + 1 - (w << 1))]
            self.table.append(row)

    def __len__(self):
        return self.n

    @overload
    def __getitem__(self, i: int) -> _T:
        """O(1)."""
        ...

    @overload
    def __getitem__(self, i: slice) -> list[_T]:
        """O(len(i))."""
        ...

    def __getitem__(self, i: int | slice) -> _T | list[_T]:
        table0 = self.table[0]
        if isinstance(i, slice):
            return [table0[j] for j in range(self.n)[i]]
        return table0[i]

    def grasp(self, i: int = 0, j: int | None = None) -> _T:
        """O(1). 0 <= i < j <= n. Take care that i != j."""
        if j is None:
            j = self.n
        # i < j because we don't know identity of f
        assert 0 <= i < j <= self.n
        b = (j - i).bit_length() - 1
        w = 1 << b
        row = self.table[b]
        return self.f(row[i], row[j - w])


class SparseTableDisjoint(Sequence[_T]):
    """
    v1.1 @cexen.
    Works for associative f(x, y).
    cf. https://noshi91.hatenablog.com/entry/2018/05/08/183946

    >>> from operator import xor
    >>> st = SparseTableDisjoint([0b011, 0b001, 0b100, 0b001, 0b101], f=xor)
    >>> st[2]  # 0b100
    4
    >>> st[:]
    [3, 1, 4, 1, 5]
    >>> st.grasp()  # 0b010 == xor(0b011, 0b001, 0b100, 0b001, 0b101)
    2
    >>> st.grasp(3)  # 0b100 == xor(0b001, 0b101)
    4
    >>> st.grasp(4, 5)  # 0b101 == min(0b101)
    5
    """

    def __init__(self, data: Iterable[_T], f: Callable[[_T, _T], _T]):
        """
        O(n log n).
        Required: f(f(x, y), z) == f(x, f(y, z)).
        """
        row0 = list(data)
        self.table = [row0]
        self.n = len(self.table[0])
        self.f = f
        for b in range(1, self.n.bit_length()):
            w = 1 << b
            row = list[_T]()
            for i in range(w, self.n, w << 1):
                seg = accumulate((row0[i - 1 - j] for j in range(w)), f)
                row.extend(reversed(list(seg)))
                seg = accumulate((row0[j] for j in range(i, min(i + w, self.n))), f)
                row.extend(seg)
            self.table.append(row)

    def __len__(self):
        return self.n

    @overload
    def __getitem__(self, i: int) -> _T:
        """O(1)."""
        ...

    @overload
    def __getitem__(self, i: slice) -> list[_T]:
        """O(len(i))."""
        ...

    def __getitem__(self, i: int | slice) -> _T | list[_T]:
        table0 = self.table[0]
        if isinstance(i, slice):
            return [table0[j] for j in range(self.n)[i]]
        return table0[i]

    def grasp(self, i: int = 0, j: int | None = None) -> _T:
        """O(1). 0 <= i < j <= n. Take care that i != j."""
        if j is None:
            j = self.n
        # i < j because we don't know identity of f
        assert 0 <= i < j <= self.n
        j -= 1
        b = (i ^ j).bit_length()
        if b == 0:
            return self.table[0][i]
        row = self.table[b - 1]
        return self.f(row[i], row[j])


class SparseTable2D(Sequence[Sequence[_T]]):
    """
    v1.1 @cexen.
    Works for associative & idenpotent f(x, y).

    >>> st = SparseTable2D([[1, 2], [3, 4]], f=min)
    >>> len(st)
    2
    >>> st.size
    4
    >>> st[1, 1]
    4
    >>> st[:]
    [[1, 2], [3, 4]]
    >>> st.grasp()  # min(1, 2, 3, 4)
    1
    >>> st.grasp(0, 1)  # min(1, 2)
    1
    >>> st.grasp(1, 2, 0, 1)  # min(3)
    3
    """

    def __init__(self, data: Iterable[Iterable[_T]], f: Callable[[_T, _T], _T]):
        """
        O(hw log hw).
        Required: f(f(x, y), z) == f(x, f(y, z)).
        Required: f(x, x) == x.
        """
        self.table = [[[list(row) for row in data]]]
        self.h = len(self.table[0][0])
        self.w = len(self.table[0][0][0])
        self.f = f
        for bi in range(self.h.bit_length() - 1):
            h = 1 << bi
            pdat = self.table[bi][0]
            dat = [
                [f(pdat[i][j], pdat[i + h][j]) for j in range(self.w)]
                for i in range(self.h + 1 - (h << 1))
            ]
            self.table.append([dat])
        for bi in range(self.h.bit_length()):
            h = 1 << bi
            tabi = self.table[bi]
            for bj in range(self.w.bit_length() - 1):
                w = 1 << bj
                pdat = tabi[bj]
                dat = [
                    [
                        f(pdat[i][j], pdat[i][j + w])
                        for j in range(self.w + 1 - (w << 1))
                    ]
                    for i in range(self.h + 1 - h)
                ]
                tabi.append(dat)

    def __len__(self):
        return self.h

    @property
    def size(self) -> int:
        return self.h * self.w

    @overload
    def __getitem__(self, s: tuple[int, int]) -> _T:
        """O(1)."""
        ...

    @overload
    def __getitem__(self, s: int) -> list[_T]:
        """O(w)."""
        ...

    @overload
    def __getitem__(self, s: tuple[int, slice]) -> list[_T]:
        """O(len(j))."""
        ...

    @overload
    def __getitem__(self, s: tuple[slice, int]) -> list[_T]:
        """O(len(i))."""
        ...

    @overload
    def __getitem__(self, s: slice) -> list[list[_T]]:
        """O(len(i) * w)."""
        ...

    @overload
    def __getitem__(self, s: tuple[slice, slice]) -> list[list[_T]]:
        """O(len(i) * len(j))."""
        ...

    def __getitem__(self, s):
        if not isinstance(s, tuple):
            s = (s, slice(None))
        i, j = s
        if isinstance(i, slice):
            return [self[ii, j] for ii in range(self.h)[i]]
        if isinstance(j, slice):
            return [self[i, jj] for jj in range(self.w)[j]]
        return self.table[0][0][i][j]

    def grasp(
        self,
        il: int = 0,
        ir: int | None = None,
        jl: int = 0,
        jr: int | None = None,
    ) -> _T:
        """O(1). 0 <= i < j <= n. Take care that i != j."""
        if ir is None:
            ir = self.h
        if jr is None:
            jr = self.w
        # l < r because we don't know identity of f
        assert 0 <= il < ir <= self.h
        assert 0 <= jl < jr <= self.w
        bi = (ir - il).bit_length() - 1
        bj = (jr - jl).bit_length() - 1
        h = 1 << bi
        w = 1 << bj
        dat = self.table[bi][bj]
        return self.f(
            self.f(dat[il][jl], dat[il][jr - w]),
            self.f(dat[ir - h][jl], dat[ir - h][jr - w]),
        )


# --------------------


def solve_yosupojudge_staticrmq():
    """
    Using SparseTable.
    Library Checker: Static RMQ
    https://judge.yosupo.jp/problem/staticrmq
    """
    N, Q = map(int, input().split())
    A = [int(v) for v in input().split()]
    LR = list[tuple[int, int]]()
    for _ in range(Q):
        l, r = map(int, input().split())
        LR.append((l, r))
    st = SparseTable(A, min)
    for l, r in LR:
        print(st.grasp(l, r))


def solve_yosupojudge_staticrmq_disjoint():
    """
    Using SparseTableDisjoint.
    Library Checker: Static RMQ
    https://judge.yosupo.jp/problem/staticrmq
    """
    N, Q = map(int, input().split())
    A = [int(v) for v in input().split()]
    LR = list[tuple[int, int]]()
    for _ in range(Q):
        l, r = map(int, input().split())
        LR.append((l, r))
    st = SparseTableDisjoint(A, min)
    for l, r in LR:
        print(st.grasp(l, r))


def solve_yosupojudge_static_range_sum():
    """
    Using SparseTableDisjoint.
    Library Checker: Static Range Sum
    https://judge.yosupo.jp/problem/static_range_sum
    """
    from operator import add

    N, Q = map(int, input().split())
    A = [int(v) for v in input().split()]
    LR = list[tuple[int, int]]()
    for _ in range(Q):
        l, r = map(int, input().split())
        LR.append((l, r))
    st = SparseTableDisjoint(A, add)
    for l, r in LR:
        print(st.grasp(l, r))
