# https://github.com/cexen/procon-cexen/blob/main/py/mincostflow.py
class MinCostFlow:
    """
    @cexen v0.1
    Primal-Dual.
    cf. 『プログラミングコンテストチャレンジブック第2版』, p.203
    >>> mcf = MinCostFlow(4)
    >>> mcf.connect(0, 1, 5, 1)
    >>> mcf.connect(0, 2, 3, 10)
    >>> mcf.connect(1, 3, 10**9, 100)
    >>> mcf.connect(2, 3, 10**9, 1000)
    >>> mcf.flow(0, 3)
    ([5, 8], [505, 3535])
    """

    from typing import Tuple, List, Optional

    def __init__(self, n: int):
        from typing import List

        self.n = n
        self.m = 0
        self.adjs: List[List[int]] = [[] for _ in range(n)]
        self.frs: List[int] = []
        self.tos: List[int] = []
        self.caps: List[int] = []
        self.rems: List[int] = []
        self.costs: List[int] = []
        self.fws: List[int] = []
        self.revs: List[int] = []
        self.pots = [0] * self.n  # potential
        self.f = self.c = 0

    def connect(self, i: int, j: int, cap: int, cost: int) -> None:
        """i -> j; directed."""
        assert 0 <= i < self.n
        assert 0 <= j < self.n
        k = self.m
        self.m += 2

        self.adjs[i].append(k)
        self.frs.append(i)
        self.tos.append(j)
        self.caps.append(cap)
        self.rems.append(cap)
        self.costs.append(cost)
        self.fws.append(1)
        self.revs.append(k + 1)

        self.adjs[j].append(k + 1)
        self.frs.append(j)
        self.tos.append(i)
        self.caps.append(cap)
        self.rems.append(0)
        self.costs.append(-cost)
        self.fws.append(0)
        self.revs.append(k)

    def flow(
        self, s: int, t: int, limit: Optional[int] = None
    ) -> Tuple[List[int], List[int]]:
        """O(f m log n) where f = max(flows). f <= limit."""
        from heapq import heappop, heappush
        from typing import Tuple, List

        assert 0 <= s < self.n
        assert 0 <= t < self.n

        pots = self.pots
        flows: List[int] = []
        costs: List[int] = []
        while True:
            dists = [-1] * self.n
            dists[s] = 0  # this requires s != t
            preves = [-1] * self.n
            q: List[Tuple[int, int]] = [(0, s)]
            while q:
                d, i = heappop(q)
                if 0 <= dists[i] < d:
                    continue
                for k in self.adjs[i]:
                    if self.rems[k] == 0:
                        continue
                    j = self.tos[k]
                    nd = d + self.costs[k] + pots[i] - pots[j]
                    if 0 <= dists[j] <= nd:
                        continue
                    dists[j] = nd
                    preves[j] = k
                    heappush(q, (nd, j))
            if preves[t] == -1:
                break
            for i in range(self.n):
                pots[i] += dists[i]
            edges = []
            i = t
            while i != s:
                k = preves[i]
                edges.append(k)
                i = self.frs[k]
            f = min(self.rems[k] for k in edges)
            if limit is not None:
                f = min(f, limit - self.f)
            for k in edges:
                self.rems[k] -= f
                self.rems[self.revs[k]] += f
            self.f += f
            self.c += f * sum(self.costs[k] for k in edges)
            flows.append(self.f)
            costs.append(self.c)
            if limit is not None and self.f == limit:
                break
        return flows, costs


# --------------------


def solve_atcoder_practice2_e():
    """
    E - MinCostFlow
    https://atcoder.jp/contests/practice2/tasks/practice2_e
    """
    N, K = map(int, input().split())
    A = [[int(v) for v in input().split()] for _ in range(N)]
    amax = max(max(a) for a in A)
    S = 0
    T = 1
    rows = [2 + i for i in range(N)]
    cols = [2 + N + i for i in range(N)]
    mcf = MinCostFlow(2 + 2 * N)
    for i in rows:
        mcf.connect(S, i, K, 0)
    for j in cols:
        mcf.connect(j, T, K, 0)
    for i in range(N):
        for j in range(N):
            mcf.connect(rows[i], cols[j], 1, amax - A[i][j])
    ans = 0
    table = [[0] * N for _ in range(N)]
    for f in range(1, 1 + N * N):
        flows, costs = mcf.flow(S, T, limit=f)
        if len(flows) == 0:
            break
        assert len(flows) == len(costs) == 1
        flow = flows[0]
        cost = costs[0]
        nans = amax * flow - cost
        if ans < nans:
            ans = nans
            for k in range(mcf.m):
                if not mcf.fws[k] or mcf.frs[k] == S or mcf.tos[k] == T:
                    continue
                i = mcf.frs[k] - 2
                j = mcf.tos[k] - 2 - N
                table[i][j] = mcf.rems[k] ^ 1
    print(ans)
    for row in table:
        print("".join("X" if v == 1 else "." for v in row))


def solve_atcoder_abc247_g():
    """
    G - Dream Team
    https://atcoder.jp/contests/abc247/tasks/abc247_g
    """
    N = int(input())
    A = []
    B = []
    C = []
    for _ in range(N):
        a, b, c = map(int, input().split())
        A.append(a - 1)
        B.append(b - 1)
        C.append(c)
    amax = 1 + max(A)
    bmax = 1 + max(B)
    cmax = max(C)
    S = 0
    T = 1
    univs = [2 + i for i in range(amax)]
    fields = [2 + amax + i for i in range(bmax)]
    mcf = MinCostFlow(2 + amax + bmax)
    for i in univs:
        mcf.connect(S, i, 1, 0)
    for i in fields:
        mcf.connect(i, T, 1, 0)
    for a, b, c in zip(A, B, C):
        mcf.connect(univs[a], fields[b], 1, cmax - c)
    flows, costs = mcf.flow(S, T)
    flows = [0] + flows
    costs = [0] + costs
    print(flows[-1])
    for i in range(len(flows) - 1):
        fl, fr = flows[i : i + 2]
        cl, cr = costs[i : i + 2]
        for f in range(fl + 1, fr + 1):
            c = (cl * (fr - f) + cr * (f - fl)) // (fr - fl)
            ans = f * cmax - c
            print(ans)
