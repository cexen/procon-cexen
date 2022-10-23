# https://github.com/cexen/procon-cexen/blob/main/py/maxbimatch.py
from typing import Tuple, Deque, List


class MaxBiMatch:
    """
    v1.0 @cexen.
    Maximum Bipartite Matching by snuke's modified Kuhn (BFS).
    cf.
    https://snuke.hatenablog.com/entry/2019/05/07/013609
    https://cp-algorithms.com/graph/kuhn_maximum_bipartite_matching.html
    """

    def __init__(self, m: int, n: int):
        self.m = m
        self.n = n
        self.adjs: List[List[int]] = [[] for _ in range(m)]

    def connect(self, i: int, j: int) -> None:
        """O(1)."""
        assert 0 <= i < self.m
        assert 0 <= j < self.n
        self.adjs[i].append(j)

    def solve(self) -> Tuple[int, List[int], List[int]]:
        """Not sure but O((m+n)sqrt(v)) where v = sum(len(adj)) ?"""
        num_pairs = 0
        pair_of_l = [-1] * self.m
        pair_of_r = [-1] * self.n
        will_continue = 1
        while will_continue:
            will_continue = 0
            roots = [-1] * self.m
            prevs = [-1] * self.m
            q = Deque[int]()
            for i in range(self.m):
                if pair_of_l[i] == -1:
                    roots[i] = i
                    q.append(i)
            while q:
                i = q.popleft()
                if pair_of_l[roots[i]] != -1:
                    continue
                for j in self.adjs[i]:
                    ni = pair_of_r[j]
                    if ni == -1:
                        num_pairs += 1
                        will_continue = 1
                        while j != -1:
                            pair_of_r[j] = i
                            pair_of_l[i], j = j, pair_of_l[i]
                            i = prevs[i]
                        break
                    if prevs[ni] != -1:
                        continue
                    roots[ni] = roots[i]
                    prevs[ni] = i
                    q.append(ni)
        return num_pairs, pair_of_l, pair_of_r


# --------------------


def solve_typical90_by():
    """
    077 - Planes on a 2D Plane
    https://atcoder.jp/contests/typical90/tasks/typical90_by
    """
    N, T = map(int, input().split())
    A: List[Tuple[int, int]] = []
    B: List[Tuple[int, int]] = []
    for _ in range(N):
        x, y = map(int, input().split())
        A.append((x, y))
    for _ in range(N):
        x, y = map(int, input().split())
        B.append((x, y))
    dxs = [1, 1, 0, -1, -1, -1, 0, 1]
    dys = [0, 1, 1, 1, 0, -1, -1, -1]
    d = {(x, y): i for i, (x, y) in enumerate(B)}
    mbm = MaxBiMatch(N, N)
    for i, (x, y) in enumerate(A):
        for dx, dy in zip(dxs, dys):
            nx = x + dx * T
            ny = y + dy * T
            j = d.get((nx, ny), -1)
            if j != -1:
                mbm.connect(i, j)
    num_pairs, pair_of_l, pair_of_r = mbm.solve()
    if num_pairs < N:
        print("No")
        exit()
    print("Yes")
    D = [-1] * N
    dxys = [(dx, dy) for dx, dy in zip(dxs, dys)]
    for i, j in enumerate(pair_of_l):
        x0, y0 = A[i]
        x1, y1 = B[j]
        dx = (x1 - x0) // T
        dy = (y1 - y0) // T
        D[i] = dxys.index((dx, dy))
    print(*(d + 1 for d in D))
