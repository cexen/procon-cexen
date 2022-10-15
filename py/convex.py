# https://github.com/cexen/procon-cexen/blob/main/py/convex.py
from typing import Tuple, Sequence, List


def pick_convex_hull_lower(xys: Sequence[Tuple[int, int]]) -> List[int]:
    """
    O(len(xys)).
    Returns indices of xys
    s.t. [xys[i] for i in indices]
    constitutes the lower convex hull of xys.
    xys MUST BE SORTED (ORDER BY x, y).
    Linearly dependent points are eliminated.
    """
    if len(xys) <= 2:
        return list(range(len(xys)))
    q = [0, 1]
    for i in range(2, len(xys)):
        while len(q) >= 2:
            x0, y0 = xys[q[-2]]
            x1, y1 = xys[q[-1]]
            x2, y2 = xys[i]
            dx0 = x1 - x0
            dy0 = y1 - y0
            dx1 = x2 - x1
            dy1 = y2 - y1
            assert dx0 * dx1 >= 0, "unsorted!"
            if dx0 == dx1 == 0:
                assert dy0 * dy1 >= 0, "unsorted!"
            # == 0: linearly dependent
            if dx0 * dy1 - dy0 * dx1 > 0:
                break
            q.pop()
        q.append(i)
    return q


def pick_convex_hull_upper(xys: Sequence[Tuple[int, int]]) -> List[int]:
    return pick_convex_hull_lower([(x, -y) for x, y in xys])


class ConvexHullTrick_AIncrXIncr:
    """
    @cexen v1.0
    Add `a` incrementally. Query `x` incrementally.
    cf. https://satanic0258.hatenablog.com/entry/2016/08/16/181331
    >>> cht = ConvexHullTrick_AIncrXIncr(greater=True)
    >>> cht.add(-1, 3)
    >>> cht.add(0, 1)
    >>> cht.add(1, -2)
    >>> cht.query(-10)
    13
    >>> cht.query(0)
    3
    >>> cht.query(1)
    2
    >>> cht.query(2)
    1
    >>> cht.query(3)
    1
    >>> cht.query(4)
    2
    >>> cht.query(5)
    3
    >>> cht.query(15)
    13
    """

    def __init__(self, inf: int = 10**18, greater: bool = True):
        """
        greater=True: find maximum.
        greater=False: find minimum.
        """
        from typing import List

        self.inf = inf
        self.greater = greater
        self.sgn = 1 if greater else -1
        self.a: List[int] = []
        self.b: List[int] = []
        self.k = 0

    def add(self, a: int, b: int):
        """a must be added incrementally."""
        while len(self.a) >= 2:
            l = -2
            c = -1
            al = self.a[l]
            ac = self.a[c]
            bl = self.b[l]
            bc = self.b[c]
            assert al <= ac <= a
            if self.sgn * (a - ac) * (b - bl) > self.sgn * (a - al) * (b - bc):
                break
            self.a.pop()
            self.b.pop()
        self.a.append(a)
        self.b.append(b)

    def query(self, x: int) -> int:
        """x must be added incrementally."""
        v = -self.inf
        for j in range(self.k, len(self.a)):
            nv = self.a[j] * x + self.b[j]
            if self.sgn * v > self.sgn * nv:
                break
            v = nv
            self.k = j
        return v


# --------------------


def solve_dp_z():
    """
    Frog 3 (dp_z)
    https://atcoder.jp/contests/dp/tasks/dp_z
    """
    N, C = map(int, input().split())
    H = [int(v) for v in input().split()]
    inf = 10**18
    cht = ConvexHullTrick_AIncrXIncr(greater=True, inf=inf)
    dp = [inf] * N
    dp[0] = 0
    cht.add(2 * H[0], -0 - H[0] ** 2)
    for i in range(1, N):
        h = H[i]
        dp[i] = -cht.query(h) + h**2 + C
        cht.add(2 * h, -dp[i] - h**2)
    print(dp[-1])
