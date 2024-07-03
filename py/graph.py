# https://github.com/cexen/procon-cexen/blob/main/py/graph.py
from abc import abstractmethod
from collections import deque
from collections.abc import Sequence
from typing import Generic, Protocol, TypeVar, overload, runtime_checkable

C_ = TypeVar("C_", bound="Cost")


@runtime_checkable
class Cost(Protocol):
    @abstractmethod
    def __add__(self: C_, value: C_, /) -> C_: ...

    @abstractmethod
    def __sub__(self: C_, value: C_, /) -> C_: ...

    @abstractmethod
    def __lt__(self: C_, value: C_, /) -> bool: ...


class Graph(Generic[C_], Sequence[int]):
    """v1.14 @cexen"""

    def __init__(self, n: int, _0: C_):
        self._n = n
        self._0 = _0  # will be used for initial cost values
        self._adjs = [list[tuple[int, C_, int]]() for _ in range(n)]
        self._revs = [list[tuple[int, C_, int]]() for _ in range(n)]
        self._idxs = list(range(n))
        self._is_sorted = True
        self._sccs: list[frozenset[int]] | None = None
        self._locs_in_sccs: list[int] | None = None

    def __len__(self) -> int:
        return self._n

    @overload
    def __getitem__(self, i: int) -> int: ...

    @overload
    def __getitem__(self, i: slice) -> list[int]: ...

    def __getitem__(self, i: int | slice) -> int | list[int]:
        return self._idxs[i]

    def _require_nonempty(self) -> None:
        if not self._n > 0:
            raise RuntimeError("n must be > 0.")

    def connect(self, i: int, j: int, cost: C_, edgeidx: int = -1) -> None:
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
        idxs = list[int]()
        visited = [False] * self._n
        # reversing just for aesthetic goodness of orders
        for istart in reversed(range(self._n)):
            if visited[istart]:
                continue
            q = list[int]()
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
    ) -> tuple[
        list[frozenset[int]],
        list[list[tuple[int, int]]],
        list[list[tuple[int, int]]],
    ]:
        """
        O(n + m). m: num of connections.
        Return: A topologically sorted list of SCCs.
        """
        if not self._is_sorted:
            self.sort()

        sccs = list[frozenset[int]]()
        locs_in_sccs = [-1] * self._n
        sccadjs = [list[tuple[int, int]]() for _ in range(self._n)]
        sccrevs = [list[tuple[int, int]]() for _ in range(self._n)]
        for idx in self._idxs:
            if locs_in_sccs[idx] != -1:
                continue
            scc = list[int]()
            q = list[int]()
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
    """
    >>> g = GraphInt(4)
    >>> g.connect(1, 0)
    >>> g.connect(1, 2)
    >>> g.connect(2, 1)
    >>> g.connect(3, 2)
    >>> g.sort()
    >>> g.find_sccs()
    ([frozenset({3}), frozenset({1, 2}), frozenset({0})], [[], [], []], [[], [], []])
    """

    def __init__(self, n: int):
        super().__init__(n, _0=0)

    def connect(self, i: int, j: int, cost: int = 1, edgeidx: int = -1) -> None:
        super().connect(i, j, cost, edgeidx)


class Tree(Graph[C_]):
    """v1.5 @cexen"""

    def __init__(self, n: int, _0: C_, root: int | None = None):
        super().__init__(n, _0)

        if root is not None and not 0 <= root < n:
            raise ValueError
        self._root = root
        self._maxdepth: int | None = None
        self._depths: list[int] | None = None
        self._weighted_depths: list[C_] | None = None
        self._parents: list[list[int]] | None = None
        self._bitlen_maxdepth: int | None = None

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
        self, root: int | None = None, save: bool = True
    ) -> tuple[int, list[int], list[C_], list[list[int]]]:
        """
        O(n). Prerequisite: none.
        Returns: (maxdepth, depths, weighted_depths, parents).
        """
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
        q = deque[tuple[int, int, C_, int]]([(root, 0, self._0, -1)])
        d = 0
        while q:
            i, d, wd, parent = q.popleft()  # BFS
            depths[i] = d
            weighted_depths[i] = wd
            parents[0][i] = parent
            for j, c, _ in self._adjs[i]:
                if parent == j:
                    continue
                q.append((j, d + 1, wd + c, i))
        maxdepth = d
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

    def find_dist(self, u: int, v: int) -> C_:
        """
        O(log n). Prerequisite: double()
        """
        assert self._weighted_depths is not None, "Prerequisite: locate_all_from_root()"
        lca = self.find_lca(u, v)
        return (
            self._weighted_depths[u]
            + self._weighted_depths[v]
            - self._weighted_depths[lca]
            - self._weighted_depths[lca]
        )

    def find_diameter(self, root: int = 0, save: bool = True) -> tuple[C_, int, int]:
        """
        MUST BE BIDIRECTIONAL. DOES NOT ACCOUNT FOR COSTS.
        Prerequisite: none
        """
        _, _, weighted_depths, _ = self.locate_all_from_root(root, save)
        maxwd = max(weighted_depths)
        one_end = weighted_depths.index(maxwd)
        _, _, weighted_depths, _ = self.locate_all_from_root(one_end, save=False)
        diameter = max(weighted_depths)
        other_end = weighted_depths.index(diameter)
        return diameter, one_end, other_end


class TreeInt(Tree[int]):
    """
    >>> tree = TreeInt(4)
    >>> tree.connect(0, 1)
    >>> tree.connect(1, 0)
    >>> tree.connect(0, 2)
    >>> tree.connect(2, 0)
    >>> tree.connect(2, 3)
    >>> tree.connect(3, 2)
    >>> tree.locate_all_from_root(root=0)
    (2, [0, 1, 1, 2], [0, 1, 1, 2], [[-1, 0, 0, 2]])
    >>> tree.double()
    >>> tree.find_lca(2, 3)
    2
    >>> tree.find_dist(1, 3)
    3
    >>> tree.find_diameter(root=0)  # max(dist(i, j)) == 3 == dist(3, 1)
    (3, 3, 1)
    """

    def __init__(self, n: int, root: int | None = None):
        super().__init__(n, _0=0, root=root)

    def connect(self, i: int, j: int, cost: int = 1, edgeidx: int = -1) -> None:
        super().connect(i, j, cost, edgeidx)


# --------------------
