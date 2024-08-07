# https://github.com/cexen/procon-cexen/blob/main/py/maxflow.py
from collections import deque


class MaxFlow:
    """
    @cexen v0.3
    Dinic.
    cf. [Dinic 法とその時間計算量 - みさわめも](https://misawa.github.io/others/flow/dinic_time_complexity.html)
    >>> mf = MaxFlow(4)
    >>> mf.connect(0, 1, 10)
    >>> mf.connect(0, 2, 3)
    >>> mf.connect(1, 3, 10**9)
    >>> mf.connect(2, 3, 10**9)
    >>> mf.flow(0, 3)
    13
    """

    def __init__(self, n: int):
        self.n = n
        self.m = 0
        self.adjs = [list[int]() for _ in range(n)]
        self.frs = list[int]()
        self.tos = list[int]()
        self.caps = list[int]()
        self.rems = list[int]()
        self.fws = list[int]()
        self.revs = list[int]()

    def connect(self, i: int, j: int, cap: int) -> None:
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
        self.fws.append(1)
        self.revs.append(k + 1)

        self.adjs[j].append(k + 1)
        self.frs.append(j)
        self.tos.append(i)
        self.caps.append(cap)
        self.rems.append(0)
        self.fws.append(0)
        self.revs.append(k)

    def flow(self, s: int, t: int) -> int:
        """O(n**2 m)."""
        assert 0 <= s < self.n
        assert 0 <= t < self.n

        ans = 0
        while True:
            # dual step
            dists = [-1] * self.n
            dists[s] = 0
            q = deque([s])
            while q:
                i = q.popleft()
                if i == t:
                    break
                d = dists[i]
                for k in self.adjs[i]:
                    if self.rems[k] == 0:
                        continue
                    j = self.tos[k]
                    if dists[j] != -1:
                        continue
                    dists[j] = d + 1
                    q.append(j)
            else:
                break

            # primal step
            iadjs = [0] * self.n
            iadjs[t] = len(self.adjs[t])
            while True:
                vs = [s]
                while vs:
                    i = vs[-1]
                    adj = self.adjs[i]
                    if iadjs[i] < len(adj):
                        k = adj[iadjs[i]]
                        assert i == self.frs[k]
                        j = self.tos[k]
                        if self.rems[k] == 0 or dists[i] + 1 != dists[j]:
                            iadjs[i] += 1
                            continue
                        vs.append(j)
                        continue
                    j = vs.pop()
                    if not vs:  # Note: including s == t
                        continue  # go to else break
                    if j == t:
                        break
                    iadjs[vs[-1]] += 1
                    continue
                else:
                    break  # end primal step
                flow = min(self.rems[self.adjs[v][iadjs[v]]] for v in vs)
                ans += flow
                for v in vs:
                    k = self.adjs[v][iadjs[v]]
                    self.rems[k] -= flow
                    self.rems[self.revs[k]] += flow
                    if self.rems[k] == 0:
                        iadjs[v] += 1
        return ans


class MaxBiMatchDinic:
    """
    @cexen v0.1
    Maximum bipartite matching based on Dinic
    >>> mbm = MaxBiMatch(3, 4)
    >>> mbm.connect(0, 0)
    >>> mbm.connect(0, 2)
    >>> mbm.connect(0, 3)
    >>> mbm.connect(1, 0)
    >>> mbm.connect(2, 2)
    >>> mbm.match()
    [(0, 3), (1, 0), (2, 2)]
    """

    _S = 0
    _T = 1

    def __init__(self, nl: int, nr: int):
        self.nl = nl
        self.nr = nr
        self.mf = MaxFlow(2 + nl + nr)
        for i in range(self.nl):
            self.mf.connect(self._S, 2 + i, 1)
        for j in range(self.nr):
            self.mf.connect(2 + self.nl + j, self._T, 1)

    def connect(self, i: int, j: int) -> None:
        """i <-> j."""
        assert 0 <= i < self.nl
        assert 0 <= j < self.nr
        self.mf.connect(2 + i, 2 + self.nl + j, 1)

    def match(self) -> list[tuple[int, int]]:
        """O(sqrt(nl + nr) (nl + nr + |E|))."""
        flow = self.mf.flow(self._S, self._T)
        matches = list[tuple[int, int]]()
        for k in range(self.mf.m):
            if (
                self.mf.fws[k] == 0
                or self.mf.frs[k] == self._S
                or self.mf.tos[k] == self._T
                or self.mf.rems[k] > 0
            ):
                continue
            matches.append((self.mf.frs[k] - 2, self.mf.tos[k] - 2 - self.nl))
        assert flow == len(matches)
        return matches


# --------------------
