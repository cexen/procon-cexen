# https://github.com/cexen/procon-cexen/blob/main/py/SegtreeLazy.py
import operator
from typing import TypeVar, Generic, Iterable, List

V_ = TypeVar("V_")
X_ = TypeVar("X_")


class SegtreeLazy(Generic[V_, X_]):
    """
    v1.8 @cexen
    Based on: https://algo-logic.info/segment-tree/
    >>> st = SegtreeLazy[int, int]([0, 1, 2, 3, 4, 5], fvv=operator.add, fvxn=lambda v, x, n: v + n * x, fxx=operator.add, ev=0, ex=0)
    >>> st.grasp()
    15
    >>> st.grasp(2, 6)
    14
    >>> st[0:2]
    [0, 1]
    >>> st.operate(100, 1, 4)
    >>> st.grasp()
    315
    >>> st.grasp(2, 6)
    214
    >>> st[0:2]
    [0, 101]
    >>> st.operate(1000, 0, 3)
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
        self.n = n = len(values)
        self.r = range(n)
        self.size: int = 1 << (n - 1).bit_length()
        self.fvv = fvv
        self.fvxn = fvxn
        self.fxx = fxx
        self.ev = ev
        self.ex = ex
        self.treev = self._build(values)
        self.treex = [ex] * (self.size << 1)
        self.treeb = [0] * (self.size << 1)

    def __len__(self) -> int:
        return self.n

    def _build(self, values: List[V_]) -> List[V_]:
        treev = [self.ev] * (self.size << 1)
        treev[self.size : self.size + len(values)] = values
        for i in reversed(range(1, self.size)):
            treev[i] = self.fvv(treev[i << 1], treev[(i << 1) + 1])
        return treev

    def _eval(self, i: int, l: int, r: int) -> None:
        """O(1). treev[i] == reduce(fvv, data[l:l+r], ev)."""
        if self.treeb[i] == 0:
            return
        if r - l > 1:
            self.treex[i << 1] = self.fxx(self.treex[i << 1], self.treex[i])
            self.treex[(i << 1) + 1] = self.fxx(self.treex[(i << 1) + 1], self.treex[i])
            self.treeb[i << 1] = 1
            self.treeb[(i << 1) + 1] = 1
        n = min(r, self.n) - min(l, self.n)
        self.treev[i] = self.fvxn(self.treev[i], self.treex[i], n)
        self.treex[i] = self.ex
        self.treeb[i] = 0

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
        q = [1]
        while q:
            k = q.pop()  # treev[k] == reduce(fvv, data[l:r], ev)
            b = k.bit_length() - 1
            w = self.size >> b
            l = w * (k ^ (1 << b))
            r = l + w
            self._eval(k, l, r)
            if not l < j or not i < r:
                continue
            if w > 1:
                nq.append(k)
                q.append(k << 1)
                q.append((k << 1) + 1)
        for ri, vi in zip(r_, v_):
            self.treev[self.size + ri] = vi
        for k in reversed(nq):
            self.treev[k] = self.fvv(self.treev[k << 1], self.treev[(k << 1) + 1])

    def grasp(self, i: int = 0, j: Optional[int] = None) -> V_:
        """O(log n). reduce(fvv, data[i:j], ev)."""
        if j is None:
            j = len(self)
        r_ = self.r[i:j]
        if not len(r_):
            return self.ev
        i, j = r_[0], 1 + r_[-1]

        q = [1]
        ans = self.ev
        while q:
            k = q.pop()  # treev[k] == reduce(fvv, data[l:r], ev)
            b = k.bit_length() - 1
            w = self.size >> b
            l = w * (k ^ (1 << b))
            r = l + w
            self._eval(k, l, r)
            if i <= l and r <= j:
                ans = self.fvv(self.treev[k], ans)
            elif w > 1:
                if i < (l + r) // 2:
                    q.append(k << 1)
                if (l + r) // 2 < j:
                    q.append((k << 1) + 1)
        return ans

    def operate(self, x: X_, i: int = 0, j: Optional[int] = None) -> None:
        """O(log n). v = f(v, x) for v in data[i:j]."""
        if j is None:
            j = len(self)
        r_ = self.r[i:j]
        if not len(r_):
            return
        i, j = r_[0], 1 + r_[-1]

        nq: List[int] = []
        q = [1]
        while q:
            k = q.pop()  # treev[k] == reduce(fvv, data[l:r], ev)
            b = k.bit_length() - 1
            w = self.size >> b
            l = w * (k ^ (1 << b))
            r = l + w
            self._eval(k, l, r)
            if not l < j or not i < r:
                continue
            if i <= l and r <= j:
                self.treex[k] = x
                self.treeb[k] = 1
                self._eval(k, l, r)
            elif w > 1:
                nq.append(k)
                q.append(k << 1)
                q.append((k << 1) + 1)
        for k in reversed(nq):
            self.treev[k] = self.fvv(self.treev[k << 1], self.treev[(k << 1) + 1])


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
    def max_add(cls, iterable: Iterable[int], ev: int = -(10**9)):
        return cls(
            iterable, fvv=max, fvxn=lambda v, x, n: v + x, fxx=lambda x, y: x + y, ev=ev
        )

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


############
# template #
############
# V = int
# X = int
# def fvv(u: V, v: V) -> V:
#     raise NotImplementedError
# def fvxn(v: V, x: X, n: int) -> V:
#     raise NotImplementedError
# def fxx(x: X, y: X) -> X:
#     raise NotImplementedError
# ev = 0
# ex = 0
# seg = SegtreeLazy[V, X]([], fvv=fvv, fvxn=fvxn, fxx=fxx, ev=ev, ex=ex)


# --------------------


def solve_yosupojudge_setitem_grasp():
    """
    Point Set Range Composite
    https://judge.yosupo.jp/problem/point_set_range_composite
    """
    from typing import Tuple, List

    MOD = 998244353
    N, Q = map(int, input().split())
    AB: List[Tuple[int, int]] = []
    for _ in range(N):
        a, b = map(int, input().split())
        AB.append((a, b))

    V = Tuple[int, int]
    X = int

    def fvv(u: V, v: V) -> V:
        a, b = u
        c, d = v
        return a * c % MOD, (b * c + d) % MOD

    def fxx(x: X, y: X) -> X:
        return (x + y) % MOD

    seg = SegtreeLazy[V, X](
        AB, fvv=fvv, fvxn=lambda v, x, n: v, fxx=fxx, ev=(1, 0), ex=0
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


def solve_yosupojudge_getitem_setitem_grasp():
    """
    Point Add Range Sum
    https://judge.yosupo.jp/problem/point_add_range_sum
    """
    N, Q = map(int, input().split())
    A = [int(v) for v in input().split()]
    seg = SegtreeLazyInt(A)
    ans = []
    for _ in range(Q):
        m, *args = map(int, input().split())
        if m == 0:
            p, x = args
            seg[p] += x
        elif m == 1:
            l, r = args
            ans.append(seg.grasp(l, r))
    for a in ans:
        print(a)


def solve_yosupojudge_operate_grasp():
    """
    TLE
    Range Affine Range Sum
    https://judge.yosupo.jp/problem/range_affine_range_sum
    """
    from typing import Tuple

    MOD = 998244353
    N, Q = map(int, input().split())
    A = [int(v) for v in input().split()]

    V = int
    X = Tuple[int, int]

    def fvv(u: V, v: V) -> V:
        return (u + v) % MOD

    def fvxn(v: V, x: X, n: int) -> V:
        a, b = x
        return (a * v + b * n) % MOD

    def fxx(x: X, y: X) -> X:
        a, b = x
        c, d = y
        return a * c % MOD, (b * c + d) % MOD

    seg = SegtreeLazy[V, X](A, fvv=fvv, fvxn=fvxn, fxx=fxx, ev=0, ex=(1, 0))
    ans = []
    for _ in range(Q):
        m, *args = map(int, input().split())
        if m == 0:
            l, r, b, c = args
            seg.operate((b, c), l, r)
        elif m == 1:
            l, r = args
            ans.append(seg.grasp(l, r))
        else:
            raise RuntimeError
    for a in ans:
        print(a)
