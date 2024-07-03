# https://github.com/cexen/procon-cexen/blob/main/py/tree_simple.py
from collections import deque


class Tree:
    """v1.1 @cexen"""

    def __init__(self, n: int):
        self.n = n
        self.adjs = [list[int]() for _ in range(n)]
        self.root: int | None = None
        self.maxdepth: int | None = None
        self.depths: list[int] | None = None
        self.parents: list[list[int]] | None = None
        self.logmaxdepth: int | None = None

    def connect(self, i: int, j: int) -> None:
        """O(1). i -> j. DIRECTED."""
        if not 0 <= i < self.n:
            raise ValueError
        if not 0 <= j < self.n:
            raise ValueError
        self.adjs[i].append(j)

    def locate_all_from_root(self, root: int = 0, save: bool = True) -> tuple:
        """
        Prerequisite: none
        """
        depths = [-1] * self.n
        parents = [[-1] * self.n]  # [k of 2^k][i]
        q = deque[tuple[int, int, int]]()
        q.append((root, 0, -1))
        i, d = root, 0
        while len(q):
            i, d, parent = q.popleft()  # BFS
            depths[i] = d
            parents[0][i] = parent
            for j in self.adjs[i]:
                if parent == j:
                    continue
                q.append((j, d + 1, i))
        maxdepth = d
        one_of_deepest = i
        if save:
            self.root = root
            self.maxdepth = maxdepth
            self.depths = depths
            self.parents = parents
        return maxdepth, depths, parents, one_of_deepest

    def double(self) -> None:
        """
        Prerequisite: locate_all_from_root()
        """
        assert self.parents is not None, "Prerequisite: locate_all_from_root()"
        assert self.maxdepth is not None, "Prerequisite: locate_all_from_root()"
        assert len(self.parents) == 1, "double() called twice?"
        logmaxdepth = 1
        while (1 << logmaxdepth) <= self.maxdepth:
            logmaxdepth += 1
        self.logmaxdepth = logmaxdepth
        for k in range(self.logmaxdepth - 1):
            self.parents.append([-1] * self.n)
            for i in range(self.n):
                self.parents[k + 1][i] = self.parents[k][self.parents[k][i]]

    def find_lca(self, u: int, v: int) -> int:
        """
        Lowest Common Ancestor
        Prerequisite: double()
        """
        assert self.logmaxdepth is not None, "Prerequisite: double()"
        assert self.depths is not None, "Prerequisite: double()"
        assert self.parents is not None, "Prerequisite: double()"
        if self.depths[u] < self.depths[v]:
            u, v = v, u  # ensures depths[u] >= depths[v]
        for k in range(self.logmaxdepth):
            if 1 & ((self.depths[u] - self.depths[v]) >> k):
                u = self.parents[k][u]
        if u == v:
            return u
        for k in reversed(range(self.logmaxdepth)):
            if self.parents[k][u] != self.parents[k][v]:
                u = self.parents[k][u]
                v = self.parents[k][v]
        return self.parents[0][u]

    def find_dist(self, u: int, v: int) -> int:
        """
        Prerequisite: double()
        """
        assert self.depths is not None, "Prerequisite: locate_all_from_root()"
        return self.depths[u] + self.depths[v] - 2 * self.depths[self.find_lca(u, v)]

    def find_diameter(self, root: int = 0, save: bool = True) -> tuple:
        """
        MUST BE UNDIRECTED.
        Prerequisite: none
        """
        _, _, _, one_end = self.locate_all_from_root(root, save)
        diameter, _, _, other_end = self.locate_all_from_root(one_end, save=False)
        return diameter, one_end, other_end


# --------------------
