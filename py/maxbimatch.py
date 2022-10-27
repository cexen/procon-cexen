# https://github.com/cexen/procon-cexen/blob/main/py/maxbimatch.py
from typing import Tuple, Deque, List


class MaxBiMatch:
    """
    v1.1 @cexen.
    Maximum Bipartite Matching by snuke's modified Kuhn (multi-source BFS).
    cf.
    https://snuke.hatenablog.com/entry/2019/05/07/013609
    https://cp-algorithms.com/graph/kuhn_maximum_bipartite_matching.html
    """

    def __init__(self, m: int, n: int):
        self.m = m
        self.n = n
        self.adjs: List[List[int]] = [[] for _ in range(m)]
        self.revs: List[List[int]] = [[] for _ in range(n)]

    def connect(self, i: int, j: int) -> None:
        """O(1)."""
        assert 0 <= i < self.m
        assert 0 <= j < self.n
        self.adjs[i].append(j)
        self.revs[j].append(i)

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

    def find_min_vertex_cover(
        self, pair_of_l: List[int], pair_of_r: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        O(m+n+v) where v = sum(len(adj)).
        cf. https://qiita.com/drken/items/7f98315b56c95a6181a4
        """
        visited_l = [0] * self.m
        visited_r = [0] * self.n
        q: List[int] = []
        for i in range(self.m):
            if pair_of_l[i] == -1:
                q.append(i)
                visited_l[i] = 1
        while q:
            i = q.pop()
            for j in self.adjs[i]:
                if j == pair_of_l[i]:
                    continue
                if visited_r[j]:
                    continue
                visited_r[j] = 1
                ni = pair_of_r[j]
                if ni != -1:
                    if visited_l[ni]:
                        continue
                    visited_l[ni] = 1
                    q.append(ni)
        ls = [i for i in range(self.m) if not visited_l[i]]
        rs = [i for i in range(self.n) if visited_r[i]]
        return ls, rs

    def find_max_independent_set(
        self, pair_of_l: List[int], pair_of_r: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        O(m+n+v) where v = sum(len(adj)).
        Returns complement set of find_min_vertex_cover().
        cf. https://qiita.com/drken/items/7f98315b56c95a6181a4
        """
        cls, crs = self.find_min_vertex_cover(pair_of_l, pair_of_r)
        cls.append(self.m)
        crs.append(self.n)
        ls = [*range(cls[0])]
        for i in range(len(cls) - 1):
            ls.extend(range(cls[i] + 1, cls[i + 1]))
        rs = [*range(crs[0])]
        for i in range(len(crs) - 1):
            rs.extend(range(crs[i] + 1, crs[i + 1]))
        return ls, rs

    def find_min_edge_cover(
        self, pair_of_l: List[int], pair_of_r: List[int]
    ) -> List[Tuple[int, int]]:
        """
        O(m+n).
        cf. https://qiita.com/drken/items/7f98315b56c95a6181a4
        """
        edges: List[Tuple[int, int]] = []
        for i, j in enumerate(pair_of_l):
            if not self.adjs[i]:
                continue
            if j == -1:
                j = self.adjs[i][0]
            edges.append((i, j))
        for j, i in enumerate(pair_of_r):
            if not self.revs[j]:
                continue
            if i == -1:
                i = self.revs[j][0]
                edges.append((i, j))
        return edges


# --------------------


def solve_yosupojudge_matching():
    """
    Matching on Bipartite Graph
    https://judge.yosupo.jp/problem/bipartitematching
    """
    L, R, M = map(int, input().split())
    mbm = MaxBiMatch(L, R)
    for _ in range(M):
        a, b = map(int, input().split())
        mbm.connect(a, b)
    num_pairs, pair_of_l, pair_of_r = mbm.solve()
    print(num_pairs)
    for i, j in enumerate(pair_of_l):
        if j != -1:
            print(i, j)


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


def solve_abc274_g():
    """
    G - Security Camera 3
    https://atcoder.jp/contests/abc274/tasks/abc274_g
    """
    H, W = map(int, input().split())
    S = [input() + "#" for _ in range(H)]
    S.append("#" * (W + 1))
    R = [[-1] * W for _ in range(H)]
    C = [[-1] * W for _ in range(H)]
    m = 0
    for i in range(H):
        for j in range(W):
            if S[i][j] == ".":
                R[i][j] = m
                if S[i][j + 1] == "#":
                    m += 1
    n = 0
    for j in range(W):
        for i in range(H):
            if S[i][j] == ".":
                C[i][j] = n
                if S[i + 1][j] == "#":
                    n += 1
    mbm = MaxBiMatch(m, n)
    for i in range(H):
        for j in range(W):
            if S[i][j] == ".":
                mbm.connect(R[i][j], C[i][j])
    num_pairs, pair_of_l, pair_of_r = mbm.solve()
    print(num_pairs)
