# https://github.com/cexen/procon-cexen/blob/main/py/MaxFlowFF.py
class MaxFlowFF:
    """
    @cexen v0.2
    Ford-Fulkerson
    """

    def __init__(self, n: int):
        from typing import List

        self.n = n
        self.m = 0
        self.adjs: List[List[int]] = [[] for _ in range(n)]
        self.frs: List[int] = []
        self.tos: List[int] = []
        self.caps: List[int] = []
        self.rems: List[int] = []
        self.fws: List[bool] = []
        self.revs: List[int] = []

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
        self.fws.append(True)
        self.revs.append(k + 1)

        self.adjs[j].append(k + 1)
        self.frs.append(j)
        self.tos.append(i)
        self.caps.append(cap)
        self.rems.append(0)
        self.fws.append(False)
        self.revs.append(k)

    def flow(self, s: int, t: int) -> int:
        """O(n max(f))."""
        ans = 0
        while True:
            prevs = [-1] * self.n
            prevs[s] = self.m  # None
            q = [s]
            while q:
                i = q.pop()
                if i == t:
                    break
                for k in self.adjs[i]:
                    if self.rems[k] == 0:
                        continue
                    j = self.tos[k]
                    if prevs[j] != -1:
                        continue
                    prevs[j] = k
                    q.append(j)
            else:
                break
            edges = []
            i = t
            while i != s:
                k = prevs[i]
                edges.append(k)
                assert self.tos[k] == i
                i = self.frs[k]
            flow = min(self.rems[k] for k in edges)
            ans += flow
            for k in edges:
                self.rems[k] -= flow
                self.rems[self.revs[k]] += flow
        return ans


class MaxBiMatchFF:
    """
    @cexen v0.1
    Maximum bipartite matching based on Ford-Fulkerson
    """

    from typing import Tuple, List

    _S = 0
    _T = 1

    def __init__(self, nl: int, nr: int):
        self.nl = nl
        self.nr = nr
        self.mf = MaxFlowFF(2 + nl + nr)
        for i in range(self.nl):
            self.mf.connect(self._S, 2 + i, 1)
        for j in range(self.nr):
            self.mf.connect(2 + self.nl + j, self._T, 1)

    def connect(self, i: int, j: int) -> None:
        """i <-> j."""
        assert 0 <= i < self.nl
        assert 0 <= j < self.nr
        self.mf.connect(2 + i, 2 + self.nl + j, 1)

    def match(self) -> List[Tuple[int, int]]:
        """O((nl + nr + |E|) max(nl, nr))."""
        from typing import Tuple, List

        flow = self.mf.flow(self._S, self._T)
        matches: List[Tuple[int, int]] = []
        for k in range(self.mf.m):
            if (
                not self.mf.fws[k]
                or self.mf.frs[k] == self._S
                or self.mf.tos[k] == self._T
                or self.mf.rems[k] > 0
            ):
                continue
            matches.append((self.mf.frs[k] - 2, self.mf.tos[k] - 2 - self.nl))
        assert flow == len(matches)
        return matches
