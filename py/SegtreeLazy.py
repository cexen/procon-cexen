# https://github.com/cexen/procon-cexen/blob/main/py/SegtreeLazy.py
import operator
from typing import TypeVar, Generic

V_ = TypeVar("V_")
X_ = TypeVar("X_")


class SegtreeLazy(Generic[V_, X_]):
    """
    v1.6 @cexen
    Based on: https://algo-logic.info/segment-tree/
    >>> st = SegtreeLazy[int, int]([0, 1, 2, 3, 4, 5], fvv=operator.add, fvxn=lambda v, x, n: v + n * x, fxx=operator.add, ev=0, ex=0)
    >>> st.treev
    [15, 6, 9, 1, 5, 9, 0, 0, 1, 2, 3, 4, 5, 0, 0]
    >>> st.grasp()
    15
    >>> st.grasp(2, 6)
    14
    >>> st[0:2]
    [0, 1]
    >>> st.operate(100, 1, 4)
    >>> st.treev
    [315, 306, 9, 101, 205, 9, 0, 0, 101, 2, 3, 4, 5, 0, 0]
    >>> st.treex
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 100, 0, 0, 0, 0]
    >>> st.grasp()
    315
    >>> st.grasp(2, 6)
    214
    >>> st[0:2]
    [0, 101]
    >>> st.treev
    [315, 306, 9, 101, 205, 9, 0, 0, 101, 2, 3, 4, 5, 0, 0]
    >>> st.treex
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 100, 0, 0, 0, 0]
    >>> st.operate(1000, 0, 3)
    >>> st.treev
    [3315, 3306, 9, 2101, 1205, 9, 0, 0, 101, 1102, 103, 4, 5, 0, 0]
    >>> st.treex
    [0, 0, 0, 0, 0, 0, 0, 1000, 1000, 0, 0, 0, 0, 0, 0]
    >>> st.grasp()
    3315
    >>> st.grasp(2, 6)
    1214
    >>> st[0:2]
    [1000, 1101]
    >>> st[:]
    [1000, 1101, 1102, 103, 4, 5]
    >>> st[3:6] = [5, 4, 103]
    >>> st.grasp()
    3315
    >>> st[:]
    [1000, 1101, 1102, 5, 4, 103]
    """

    from typing import Iterable, Callable, Union, Optional, List, overload

    def __init__(
        self,
        iterable: Iterable[V_],
        fvv: Callable[[V_, V_], V_],
        fvxn: Callable[[V_, X_, int], V_],
        fxx: Callable[[X_, X_], X_],
        ev: V_,
        ex: X_,
    ):

        values = list(iterable)
        n = len(values)
        self.r = range(n)
        self.size: int = 2 ** (n - 1).bit_length()
        self.fvv = fvv
        self.fvxn = fvxn
        self.fxx = fxx
        self.ev = ev
        self.ex = ex
        self.treev = self._build(values)
        self.treex = [ex] * (2 * self.size - 1)
        self.treeb = [False] * (2 * self.size - 1)

    def __len__(self) -> int:
        return len(self.r)

    def _build(self, values: List[V_]) -> List[V_]:
        treev = [self.ev] * (2 * self.size - 1)
        treev[self.size - 1 : self.size - 1 + len(values)] = values
        for i in reversed(range(self.size - 1)):
            treev[i] = self.fvv(treev[(i << 1) + 1], treev[(i << 1) + 2])
        return treev

    def _eval(self, i: int, l: int, r: int) -> None:
        """O(1). treev[k] == reduce(fvv, data[l:r], ev)."""
        if not self.treeb[i]:
            return
        n = r - l
        if n > 1:
            self.treex[(i << 1) + 1] = self.fxx(self.treex[(i << 1) + 1], self.treex[i])
            self.treex[(i << 1) + 2] = self.fxx(self.treex[(i << 1) + 2], self.treex[i])
            self.treeb[(i << 1) + 1] = True
            self.treeb[(i << 1) + 2] = True
        self.treev[i] = self.fvxn(self.treev[i], self.treex[i], n)
        self.treex[i] = self.ex
        self.treeb[i] = False

    @overload
    def __getitem__(self, i: int) -> V_:
        """O(log n)."""
        ...

    @overload
    def __getitem__(self, i: slice) -> List[V_]:
        """O(len(i) log n)."""
        ...

    def __getitem__(self, i: Union[int, slice]) -> Union[V_, List[V_]]:
        if isinstance(i, slice):
            return [self[j] for j in self.r[i]]
        return self.grasp(self.r[i], self.r[i] + 1)

    @overload
    def __setitem__(self, i: int, v: V_) -> None:
        """O(log n)."""
        ...

    @overload
    def __setitem__(self, i: slice, v: Iterable[V_]) -> None:
        """O(len(i) log n)."""
        ...

    def __setitem__(self, i: Union[int, slice], v: Union[V_, Iterable[V_]]) -> None:
        from typing import Iterable, Tuple, List

        v_: List[V_]
        if isinstance(i, int):
            r_ = self.r[i : i + 1]
            v_ = [v]  # type: ignore
        elif isinstance(i, slice):
            r_ = self.r[i]
            assert isinstance(v, Iterable)
            v_ = list(v)
        else:
            raise TypeError
        if not len(r_):
            return
        i, j = min(r_), 1 + max(r_)
        assert len(r_) == len(v_)

        nq: List[int] = []
        q: List[Tuple[int, int, int]] = []
        q.append((0, 0, self.size))
        while len(q):
            k, l, r = q.pop()  # treev[k] == reduce(fvv, data[l:r], ev)
            self._eval(k, l, r)
            if not l < j or not i < r:
                continue
            if r - l > 1:
                nq.append(k)
                q.append(((k << 1) + 1, l, (l + r) // 2))
                q.append(((k << 1) + 2, (l + r) // 2, r))
        nq.reverse()
        for ri, vi in zip(r_, v_):
            self.treev[self.size - 1 + ri] = vi
        for k in nq:
            self.treev[k] = self.fvv(self.treev[(k << 1) + 1], self.treev[(k << 1) + 2])

    def grasp(self, i: int = 0, j: Optional[int] = None) -> V_:
        """O(log n). reduce(fvv, data[i:j], ev)."""
        from typing import Tuple, Deque

        if j is None:
            j = len(self)
        r_ = self.r[i:j]
        if not len(r_):
            return self.ev
        i, j = r_[0], 1 + r_[-1]

        q = Deque[Tuple[int, int, int]]()
        q.append((0, 0, self.size))
        ans = self.ev
        while len(q):
            k, l, r = q.pop()  # treev[k] == reduce(fvv, data[l:r], ev)
            if not l < j or not i < r:
                continue
            self._eval(k, l, r)
            if i <= l and r <= j:
                ans = self.fvv(self.treev[k], ans)
            else:
                q.append(((k << 1) + 1, l, (l + r) // 2))
                q.append(((k << 1) + 2, (l + r) // 2, r))
        return ans

    def operate(self, x: X_, i: int = 0, j: Optional[int] = None) -> None:
        """O(log n). v = f(v, x) for v in data[i:j]."""
        from typing import Deque, Tuple

        if j is None:
            j = len(self)
        r_ = self.r[i:j]
        if not len(r_):
            return
        i, j = r_[0], 1 + r_[-1]

        q = Deque[Tuple[int, int, int]]()
        q.append((0, 0, self.size))
        while len(q):
            k, l, r = q.pop()  # treev[k] == reduce(fvv, data[l:r], ev)
            if k < 0:  # postorder
                self.treev[~k] = self.fvv(
                    self.treev[(~k << 1) + 1], self.treev[(~k << 1) + 2]
                )
                continue
            self._eval(k, l, r)
            if not l < j or not i < r:
                continue
            if i <= l and r <= j:
                self.treex[k] = x
                self.treeb[k] = True
                self._eval(k, l, r)
            else:
                q.append((~k, l, r))  # postorder
                q.append(((k << 1) + 1, l, (l + r) // 2))
                q.append(((k << 1) + 2, (l + r) // 2, r))


class SegtreeLazyInt(SegtreeLazy[int, int]):

    from typing import Iterable, Callable

    def __init__(
        self,
        iterable: Iterable[int],
        fvv: Callable[[int, int], int] = operator.add,
        fvxn: Callable[[int, int, int], int] = lambda v, x, n: v + n * x,
        fxx: Callable[[int, int], int] = operator.add,
        ev: int = 0,
        ex: int = 0,
    ):
        super().__init__(iterable, fvv, fvxn, fxx, ev, ex)

    @classmethod
    def max_assign(cls, iterable: Iterable[int]):
        """
        >>> st = SegtreeLazyInt.max_assign([0, 1, 2, 3, 4])
        >>> st.grasp()
        4
        >>> st.operate(8, 1, 4)
        >>> st.grasp(0, 1)
        0
        >>> st.grasp(4, 5)
        4
        >>> st.grasp()
        8
        >>> st[:]
        [0, 8, 8, 8, 4]

        Memo: ex might cause bug??
        fxx=lambda x, y: x if y == ex else y
        cf. https://atcoder.jp/contests/abl/editorial/1204
        I haven't seen any bugs with fxx=lambda x, y: y
        """
        return cls(iterable, fvv=max, fvxn=lambda v, x, n: x, fxx=lambda x, y: y)

    @classmethod
    def min_assign(cls, iterable: Iterable[int], ev: int = 10**9):
        return cls(iterable, fvv=min, fvxn=lambda v, x, n: x, fxx=lambda x, y: y, ev=ev)

    @classmethod
    def min_add(cls, iterable: Iterable[int], ev: int = 10**9):
        return cls(
            iterable, fvv=min, fvxn=lambda v, x, n: v + x, fxx=lambda x, y: x + y, ev=ev
        )

    @classmethod
    def sum_assign(cls, iterable: Iterable[int]):
        """
        >>> st = SegtreeLazyInt.sum_assign([0, 1, 2, 3, 4])
        >>> st.grasp()
        10
        >>> st.operate(8, 1, 4)
        >>> st.grasp(0, 1)
        0
        >>> st.grasp(4, 5)
        4
        >>> st.grasp()
        28
        >>> st[:]
        [0, 8, 8, 8, 4]
        """
        return cls(
            iterable, fvv=operator.add, fvxn=lambda v, x, n: n * x, fxx=lambda x, y: y
        )


def solve_yosupojudge_setitem_grasp():
    from typing import Tuple, List

    MOD = 998244353
    N, Q = map(int, input().split())
    AB: List[Tuple[int, int]] = []
    for _ in range(N):
        a, b = map(int, input().split())
        AB.append((a, b))

    def fvv(vl: Tuple[int, int], vr: Tuple[int, int]) -> Tuple[int, int]:
        a, b = vl
        c, d = vr
        return a * c % MOD, (b * c + d) % MOD

    # does not use operate (x)
    seg = SegtreeLazy[Tuple[int, int], int](
        AB, fvv=fvv, fvxn=lambda v, x, n: v, fxx=operator.add, ev=(1, 0), ex=0
    )
    ans = []
    for _ in range(Q):
        m, *args = map(int, input().split())
        if m == 0:
            p, c, d = args
            seg[p] = (c, d)
        elif m == 1:
            l, r, x = args
            a, b = seg.grasp(l, r)
            ans.append((a * x + b) % MOD)
        else:
            raise RuntimeError
    for a in ans:
        print(a)
