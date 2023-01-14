# https://github.com/cexen/procon-cexen/blob/main/py/bit.py
import operator
from typing import TypeVar, Generic

T = TypeVar("T")


class BIT(Generic[T]):
    """v1.3 @cexen"""

    from typing import Callable, Optional

    def __init__(
        self, n: int, f: Callable[[T, T], T], e: T, increasing: Optional[bool] = None
    ):
        """
        increasing: Required at bisect.
        True if grasp(i) <= grasp(i + 1).
        False if grasp(i) >= grasp(i + 1).
        """
        self.size = n
        self.tree = [e] * (n + 1)
        self.f = f
        self.e = e
        self.increasing = increasing

    def __len__(self) -> int:
        return self.size

    def clear(self) -> None:
        """O(n)."""
        self.tree[:] = [self.e] * len(self.tree)

    def grasp(self, i: Optional[int] = None) -> T:
        """O(log n). reduce(f, data[:i], e)."""
        if i is None:
            i = self.size
        i = min(i, self.size)
        s = self.e
        while i > 0:
            s = self.f(s, self.tree[i])
            i -= i & -i
        return s

    def operate(self, i: int, v: T) -> None:
        """O(log n). bit[i] = f(bit[i], v)."""
        i += 1  # to 1-indexed
        while i <= self.size:
            self.tree[i] = self.f(self.tree[i], v)
            i += i & -i

    def bisect_left(self, v: T) -> int:
        return self._bisect_any(v, left=True)

    def bisect_right(self, v: T) -> int:
        return self._bisect_any(v, left=False)

    def _bisect_any(self, v: T, left: bool = True) -> int:
        if self.increasing is None:
            raise RuntimeError("Specify increasing.")
        incr = self.increasing  # type: ignore
        i = 0  # 0-indexed
        u = self.e
        for s in reversed(range(self.size.bit_length())):
            k = i + (1 << s)  # 1-indexed
            if not k <= self.size:
                continue
            w = self.f(u, self.tree[k])
            if left and incr and not w < v:  # type: ignore
                continue
            if not left and incr and not w <= v:  # type: ignore
                continue
            if left and not incr and not v < w:  # type: ignore
                continue
            if not left and not incr and not v <= w:  # type: ignore
                continue
            i = k  # 0-indexed
            u = w
        return i  # 0-indexed


class BITInt(BIT[int]):
    """
    >>> b = BITInt(5, increasing=True)
    >>> b.operate(0, 10)
    >>> b.operate(1, 10)
    >>> b.operate(3, 10)
    >>> b.grasp(1)
    10
    >>> b.grasp(2)
    20
    >>> b.grasp(3)
    20
    >>> b.grasp(4)
    30
    >>> b.grasp(5)
    30
    >>> b.grasp()
    30
    >>> b.bisect_left(10)
    0
    >>> b.bisect_left(11)
    1
    >>> b.bisect_left(20)
    1
    >>> b.bisect_left(21)
    3
    >>> b.bisect_left(30)
    3
    >>> b.bisect_left(31)
    5
    >>> b.bisect_right(29)
    3
    >>> b.bisect_right(30)
    5

    >>> b = BITInt(3, f=min, e=10**9, increasing=False)
    >>> b.bisect_left(0), b.bisect_right(0)
    (3, 3)
    >>> b.operate(1, 5)
    >>> b.operate(2, 2)
    >>> b.bisect_left(6), b.bisect_right(6)
    (1, 1)
    >>> b.bisect_left(5), b.bisect_right(5)
    (1, 2)
    >>> b.bisect_left(4), b.bisect_right(4)
    (2, 2)
    """

    from typing import Callable, Optional

    def __init__(
        self,
        n: int,
        f: Callable[[int, int], int] = operator.add,
        e: int = 0,
        increasing: Optional[bool] = None,
    ):
        super().__init__(n, f, e, increasing)


class BITFloat(BIT[float]):
    from typing import Callable, Optional

    def __init__(
        self,
        n: int,
        f: Callable[[float, float], float] = operator.add,
        e: float = 0.0,
        increasing: Optional[bool] = None,
    ):
        super().__init__(n, f, e, increasing)


class DisjointSetUnion:
    """For find_manhattan_mst."""

    def __init__(self, n: int):
        self.roots = [-1] * n

    def root(self, i: int) -> int:
        p = self.roots[i]
        if p < 0:
            return i
        p = self.roots[i] = self.root(p)
        return p

    def connected(self, i: int, j: int) -> bool:
        return self.root(i) == self.root(j)

    def connect(self, i: int, j: int) -> bool:
        i = self.root(i)
        j = self.root(j)
        if i == j:
            return False
        if not -self.roots[i] >= -self.roots[j]:
            i, j = j, i
        if self.roots[i] == self.roots[j]:
            self.roots[i] -= 1
        self.roots[j] = i
        return True


from typing import Tuple, Sequence, List


def find_manhattan_mst(
    xs: Sequence[int], ys: Sequence[int]
) -> Tuple[int, List[int], List[int]]:
    """
    v1.0 @cexen.
    Returns (dist, us, vs)
     s.t. dist = sum(dist(u, v) for u, v in zip(us, vs))
     and (u, v) is an edge of manhattan MST.
    cf. https://knuu.github.io/manhattan_mst.html
    cf. This code uses the region 4 (max x+y s.t. y>=x && y<=0).
    """
    assert len(xs) == len(ys)
    n = len(xs)
    idxs = list(range(n))

    def f(i: int, j: int) -> int:
        if i < 0:
            return j
        if j < 0:
            return i
        if sgnx * (xs[i] - xs[j]) + sgny * (ys[i] - ys[j]) >= 0:
            return i
        return j

    bit = BITInt(n, f=f, e=-1)
    ds0 = []
    us0 = []
    vs0 = []
    for _ in range(2):
        for sgny in (1, -1):
            dyi = {y: i for i, y in enumerate(sorted(sgny * y for y in ys))}
            for sgnx in (1, -1):
                idxs.sort(key=lambda i: sgnx * xs[i] - sgny * ys[i])
                bit.clear()
                for i in idxs:
                    x0 = sgnx * xs[i]
                    y0 = sgny * ys[i]
                    iy0 = dyi[y0]
                    j = bit.grasp(iy0 + 1)
                    bit.operate(iy0, i)
                    if j != -1:
                        x = sgnx * xs[j]
                        y = sgny * ys[j]
                        d = (x0 + y0) - (x + y)
                        ds0.append(d)
                        us0.append(i)
                        vs0.append(j)
        xs, ys = ys, xs
    dist = 0
    us = []
    vs = []
    idxs = list(range(len(us0)))
    idxs.sort(key=lambda i: ds0[i])
    dsu = DisjointSetUnion(n)
    for i in idxs:
        d = ds0[i]
        u = us0[i]
        v = vs0[i]
        if dsu.connect(u, v):
            dist += d
            us.append(u)
            vs.append(v)
    assert len(us) == len(vs) == n - 1
    return dist, us, vs


# --------------------


def solve_yosupojudge_static_range_sum():
    """
    Library Checker: Static Range Sum
    https://judge.yosupo.jp/problem/static_range_sum
    """
    N, Q = map(int, input().split())
    A = [int(v) for v in input().split()]
    LR = []
    for _ in range(Q):
        l, r = map(int, input().split())
        LR.append((l, r))
    bit = BITInt(N)
    for i, a in enumerate(A):
        bit.operate(i, a)
    for l, r in LR:
        print(bit.grasp(r) - bit.grasp(l))


def solve_yosupojudge_unionfind():
    """
    Library Checker: Unionfind
    https://judge.yosupo.jp/problem/unionfind
    """
    N, Q = map(int, input().split())
    dst = DisjointSetUnion(N)
    ans = []
    for _ in range(Q):
        t, u, v = map(int, input().split())
        if t == 0:
            dst.connect(u, v)
        elif t == 1:
            ans.append(dst.connected(u, v))
        else:
            raise RuntimeError
    for a in ans:
        print(int(a))


def solve_yosupojudge_manhattanmst():
    """
    TLE.
    Library Checker: Manhattan MST
    https://judge.yosupo.jp/problem/manhattanmst
    """
    N = int(input())
    xs = []
    ys = []
    for _ in range(N):
        x, y = map(int, input().split())
        xs.append(x)
        ys.append(y)
    dist, us, vs = find_manhattan_mst(xs, ys)
    print(dist)
    for u, v in zip(us, vs):
        print(u, v)
