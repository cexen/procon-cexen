# https://github.com/cexen/procon-cexen/blob/main/py/Graph.py
from typing import TypeVar, Generic, Sequence


C = TypeVar("C")

# >= 3.8

# C = TypeVar("C", bound="Cost")

# from typing import Protocol, runtime_checkable
# from abc import abstractmethod
# @runtime_checkable
# class Cost(Protocol):
#     @abstractmethod
#     def __add__(self: C, other: C) -> C:
#         ...

#     @abstractmethod
#     def __sub__(self: C, other: C) -> C:
#         ...

#     @abstractmethod
#     def __lt__(self: C, other: C) -> bool:
#         ...


class Graph(Generic[C], Sequence[int]):
    """v1.14 @cexen"""

    from typing import Union, Tuple, List, FrozenSet, overload

    def __init__(self, n: int, _0: C):
        from typing import List, Tuple, FrozenSet, Optional

        self._n = n
        self._0 = _0  # will be used for initial cost values
        self._adjs: List[List[Tuple[int, C, int]]] = [[] for _ in range(n)]
        self._revs: List[List[Tuple[int, C, int]]] = [[] for _ in range(n)]
        self._idxs = list(range(n))
        self._is_sorted = True
        self._sccs: Optional[List[FrozenSet[int]]] = None
        self._locs_in_sccs: Optional[List[int]] = None

    def __len__(self) -> int:
        return self._n

    @overload
    def __getitem__(self, i: int) -> int:
        ...

    @overload
    def __getitem__(self, i: slice) -> List[int]:
        ...

    def __getitem__(self, i: Union[int, slice]) -> Union[int, List[int]]:
        return self._idxs[i]

    def _require_nonempty(self):
        if not self._n > 0:
            raise RuntimeError("n must be > 0.")

    def connect(self, i: int, j: int, cost: C, edgeidx: int = -1) -> None:
        """
        O(1). i -> j. DIRECTED.
        Specify edgeidx if you need it when rerooting.
        """
        if not 0 <= i < self._n:
            raise ValueError
        if not 0 <= j < self._n:
            raise ValueError
        self._adjs[i].append((j, cost, edgeidx))
        self._revs[j].append((i, cost, edgeidx))
        self._is_sorted = False
        self._sccs = None
        self._locs_in_sccs = None

    def sort(self) -> None:
        """
        O(n + m). m: num of connections.
        If not sorted, sorts.
        Sort nodes topologically.
        """
        from typing import List

        idxs: List[int] = []
        visited = [False] * self._n
        # reversing just for aesthetic goodness of orders
        for istart in reversed(range(self._n)):
            if visited[istart]:
                continue
            q: List[int] = []
            q.append(istart)
            # postorder
            while q:
                i = q.pop()
                if i < 0:
                    idxs.append(~i)
                    continue
                if visited[i]:
                    continue
                visited[i] = True
                q.append(~i)
                for j, *_ in self._adjs[i]:
                    if not visited[j]:
                        # Note: it is incorrect to do here visited[j] = True
                        q.append(j)
        assert len(idxs) == self._n
        self._idxs = idxs[::-1]
        self._is_sorted = True

    def find_sccs(
        self,
        require_sccadjs: bool = False,
    ) -> Tuple[
        List[FrozenSet[int]],
        List[List[Tuple[int, int]]],
        List[List[Tuple[int, int]]],
    ]:
        """
        O(n + m). m: num of connections.
        Return: A topologically sorted list of SCCs.
        """
        from typing import Tuple, List, FrozenSet

        if not self._is_sorted:
            self.sort()

        sccs: List[FrozenSet[int]] = []
        locs_in_sccs = [-1] * self._n
        sccadjs: List[List[Tuple[int, int]]] = [[] for _ in range(self._n)]
        sccrevs: List[List[Tuple[int, int]]] = [[] for _ in range(self._n)]
        for idx in self._idxs:
            if locs_in_sccs[idx] != -1:
                continue
            scc: List[int] = []
            q: List[int] = []
            si = len(sccs)
            locs_in_sccs[idx] = si
            q.append(idx)
            while q:
                i = q.pop()
                scc.append(i)
                for j, _, e in self._revs[i]:
                    if locs_in_sccs[j] != -1:
                        if require_sccadjs:
                            sj = locs_in_sccs[j]
                            sccadjs[sj].append((si, e))
                            sccrevs[si].append((sj, e))
                        continue
                    locs_in_sccs[j] = si
                    q.append(j)
            sccs.append(frozenset(scc))

        self._sccs = sccs
        self._locs_in_sccs = locs_in_sccs
        return sccs[:], sccadjs[: len(sccs)], sccrevs[: len(sccs)]

    def is_dag(self) -> bool:
        """
        O(m). m: num of connections.
        Prerequisites: find_sccs().
        """
        if self._sccs is None:
            raise RuntimeError("Do find_sccs() first.")
        if len(self._sccs) < self._n:
            return False
        if any(i == j for i, adj in enumerate(self._adjs) for j, _, _ in adj):
            return False  # has self loop
        return True

    def are_equivalent(self, i: int, j: int) -> bool:
        """
        O(1). Prerequisites: find_sccs().
        Returns: Whether i and j are in the same scc.
        """
        if not 0 <= i < self._n:
            raise ValueError
        if not 0 <= j < self._n:
            raise ValueError
        if self._locs_in_sccs is None:
            raise RuntimeError("Do find_sccs() first.")
        return self._locs_in_sccs[i] == self._locs_in_sccs[j]


class GraphInt(Graph[int]):
    def __init__(self, n: int):
        super().__init__(n, _0=0)

    def connect(self, i: int, j: int, cost: int = 1, edgeidx: int = -1) -> None:
        super().connect(i, j, cost, edgeidx)


class Tree(Graph[C]):
    """v1.5 @cexen"""

    from typing import Optional, List, Tuple

    def __init__(self, n: int, _0: C, root: Optional[int] = None):
        from typing import Optional, List

        super().__init__(n, _0)

        if root is not None and not 0 <= root < n:
            raise ValueError
        self._root = root
        self._maxdepth: Optional[int] = None
        self._depths: Optional[List[int]] = None
        self._weighted_depths: Optional[List[C]] = None
        self._parents: Optional[List[List[int]]] = None
        self._bitlen_maxdepth: Optional[int] = None

    def find_root(self) -> None:
        """O(n). If directional and not sorted, sorts."""
        self._require_nonempty()
        # Just set to 0 if it seems to be a bidirectional tree
        if self._adjs[0] == self._revs[0]:
            self._root = 0
            return
        self.sort()
        self._root = self._idxs[0]

    def locate_all_from_root(
        self, root: Optional[int] = None, save: bool = True
    ) -> Tuple[int, List[int], List[C], List[List[int]]]:
        """
        O(n). Prerequisite: none.
        Returns: (maxdepth, depths, parents, one_of_deepest).
        """
        from collections import deque
        from typing import Deque, Tuple

        self._require_nonempty()
        if root is None:
            if self._root is None:
                raise ValueError("Specify root or do find_root() first.")
            root = self._root
        if not 0 <= root < self._n:
            raise ValueError

        depths = [-1] * self._n
        weighted_depths = [self._0] * self._n
        parents = [[-1] * self._n]  # [k of 2^k][i]
        q: Deque[Tuple[int, int, C, int]] = deque()
        q.append((root, 0, self._0, -1))
        while len(q):
            i, d, wd, parent = q.popleft()  # BFS
            depths[i] = d
            weighted_depths[i] = wd
            parents[0][i] = parent
            for j, c, _ in self._adjs[i]:
                if parent == j:
                    continue
                q.append((j, d + 1, wd + c, i))  # type: ignore
        maxdepth = d  # type: ignore
        if save:
            self._root = root
            self._maxdepth = maxdepth
            self._depths = depths
            self._weighted_depths = weighted_depths
            self._parents = parents
        return maxdepth, depths, weighted_depths, parents

    def double(self) -> None:
        """
        O(n log n). Prerequisite: locate_all_from_root()
        """
        assert self._maxdepth is not None, "Prerequisite: locate_all_from_root()"
        assert self._parents is not None, "Prerequisite: locate_all_from_root()"
        assert len(self._parents) == 1, "double() called twice?"
        self._bitlen_maxdepth = self._maxdepth.bit_length()
        for _ in range(self._bitlen_maxdepth - 1):
            self._parents.append(
                [self._parents[-1][self._parents[-1][i]] for i in range(self._n)]
            )

    def find_lca(self, u: int, v: int) -> int:
        """
        O(log n). LCA (Lowest Common Ancestor).
        Prerequisite: double()
        """
        assert self._bitlen_maxdepth is not None, "Prerequisite: double()"
        assert self._depths is not None, "Prerequisite: double()"
        assert self._parents is not None, "Prerequisite: double()"
        if self._depths[u] < self._depths[v]:
            u, v = v, u  # ensures depths[u] >= depths[v]
        for k in range(self._bitlen_maxdepth):
            if 1 & (self._depths[u] - self._depths[v]) >> k:
                u = self._parents[k][u]
        if u == v:
            return u
        for k in reversed(range(self._bitlen_maxdepth)):
            if self._parents[k][u] != self._parents[k][v]:
                u = self._parents[k][u]
                v = self._parents[k][v]
        return self._parents[0][u]

    def find_dist(self, u: int, v: int) -> C:
        """
        O(log n). Prerequisite: double()
        """
        assert self._weighted_depths is not None, "Prerequisite: locate_all_from_root()"
        lca = self.find_lca(u, v)
        return (
            self._weighted_depths[u]  # type: ignore
            + self._weighted_depths[v]
            - self._weighted_depths[lca]
            - self._weighted_depths[lca]
        )

    def find_diameter(self, root: int = 0, save: bool = True) -> Tuple[C, int, int]:
        """
        MUST BE BIDIRECTIONAL. DOES NOT ACCOUNT FOR COSTS.
        Prerequisite: none
        """
        _, _, weighted_depths, _ = self.locate_all_from_root(root, save)
        maxwd = max(weighted_depths)  # type: ignore
        one_end = weighted_depths.index(maxwd)  # type: ignore
        _, _, weighted_depths, _ = self.locate_all_from_root(one_end, save=False)
        diameter = max(weighted_depths)  # type: ignore
        other_end = weighted_depths.index(diameter)  # type: ignore
        return diameter, one_end, other_end  # type: ignore


class TreeInt(Tree[int]):
    from typing import Optional

    def __init__(self, n: int, root: Optional[int] = None):
        super().__init__(n, _0=0, root=root)

    def connect(self, i: int, j: int, cost: int = 1, edgeidx: int = -1) -> None:
        super().connect(i, j, cost, edgeidx)
