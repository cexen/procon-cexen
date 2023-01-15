# https://github.com/cexen/procon-cexen/blob/main/py/sparsetable.py
from typing import TypeVar, Callable, Sequence, Optional, Union, List, overload

T_ = TypeVar("T_")


class SparseTable(Sequence[T_]):
    """
    v1.0 @cexen.
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

    def __init__(self, data: Sequence[T_], f: Callable[[T_, T_], T_]):
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
    def __getitem__(self, i: int) -> T_:
        """O(1)."""
        ...

    @overload
    def __getitem__(self, i: slice) -> List[T_]:
        """O(len(i))."""
        ...

    def __getitem__(self, i: Union[int, slice]) -> Union[T_, List[T_]]:
        table0 = self.table[0]
        if isinstance(i, slice):
            return [table0[j] for j in range(self.n)[i]]
        return table0[i]

    def grasp(self, i: int = 0, j: Optional[int] = None) -> T_:
        """O(1). 0 <= i < j <= n. Take care that i != j."""
        if j is None:
            j = self.n
        # i < j because we don't know identity of f
        assert 0 <= i < j <= self.n
        b = (j - i).bit_length() - 1
        w = 1 << b
        row = self.table[b]
        return self.f(row[i], row[j - w])


from itertools import accumulate


class SparseTableDisjoint(Sequence[T_]):
    """
    v1.0 @cexen.
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

    def __init__(self, data: Sequence[T_], f: Callable[[T_, T_], T_]):
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
            row: List[T_] = []
            for i in range(w, self.n, w << 1):
                seg = accumulate((row0[i - 1 - j] for j in range(w)), f)
                row.extend(reversed(list(seg)))
                seg = accumulate((row0[j] for j in range(i, min(i + w, self.n))), f)
                row.extend(seg)
            self.table.append(row)

    def __len__(self):
        return self.n

    @overload
    def __getitem__(self, i: int) -> T_:
        """O(1)."""
        ...

    @overload
    def __getitem__(self, i: slice) -> List[T_]:
        """O(len(i))."""
        ...

    def __getitem__(self, i: Union[int, slice]) -> Union[T_, List[T_]]:
        table0 = self.table[0]
        if isinstance(i, slice):
            return [table0[j] for j in range(self.n)[i]]
        return table0[i]

    def grasp(self, i: int = 0, j: Optional[int] = None) -> T_:
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


# --------------------


def solve_yosupojudge_staticrmq():
    """
    Using SparseTable.
    Library Checker: Static RMQ
    https://judge.yosupo.jp/problem/staticrmq
    """
    N, Q = map(int, input().split())
    A = [int(v) for v in input().split()]
    LR = []
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
    LR = []
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
    LR = []
    for _ in range(Q):
        l, r = map(int, input().split())
        LR.append((l, r))
    st = SparseTableDisjoint(A, add)
    for l, r in LR:
        print(st.grasp(l, r))
