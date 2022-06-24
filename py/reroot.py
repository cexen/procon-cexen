# https://github.com/cexen/procon-cexen/blob/main/py/tree.py
from typing import TypeVar

E_ = TypeVar("E_")
V_ = TypeVar("V_")


class TreeReroot:
    """
    v1.5 cexen.
    """

    from typing import Callable, Optional, Tuple, List

    def __init__(self, n: int):
        from typing import Tuple, List

        self.n = n
        self.root = -1
        self.idxs = [-1] * n
        self.adjs: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
        self.parents: List[Tuple[int, int]] = [(-1, -1)] * n

    def connect(self, u: int, v: int, e: int = -1) -> None:
        """
        Specify e (edge index) if you need it when add_e.
        """

        assert 0 <= u < self.n
        assert 0 <= v < self.n
        self.adjs[u].append((v, e))
        self.adjs[v].append((u, e))

    def sort(self, root: int = 0) -> None:
        assert 0 <= root < self.n
        if root == self.root:
            return
        self.root = root
        self.idxs.clear()
        q = [root]
        visited = [False] * self.n
        visited[root] = True
        while q:
            i = q.pop()
            self.idxs.append(i)
            for adj in self.adjs[i]:
                j, e = adj
                if visited[j]:
                    self.parents[i] = adj
                    continue
                visited[j] = True
                q.append(j)

    def collect_for_root(
        self,
        add_e: Callable[[V_, int, int], E_],
        merge: Callable[[E_, E_], E_],
        add_v: Callable[[E_, int, int], V_],
        ee: Callable[[], E_],
        root: int = 0,
    ) -> Tuple[List[V_], List[List[Optional[E_]]]]:
        from typing import Optional, List

        self.sort(root)
        ans0: List[V_] = [None] * self.n  # type: ignore
        edgeses: List[List[Optional[E_]]] = [[] for _ in range(self.n)]
        for i in reversed(self.idxs):
            parent = self.parents[i]
            edges = edgeses[i]
            accum_l = ee()
            for adj in self.adjs[i]:
                j, e = adj
                edge = None if adj == parent else add_e(ans0[j], j, e)
                edges.append(edge)
                if edge is not None:
                    accum_l = merge(accum_l, edge)
            ans0[i] = add_v(accum_l, i, parent[0])
        return ans0, edgeses

    def collect_for_all(
        self,
        add_e: Callable[[V_, int, int], E_],
        merge: Callable[[E_, E_], E_],
        add_v: Callable[[E_, int, int], V_],
        ee: Callable[[], E_],
        root: int = 0,
    ) -> List[V_]:
        from typing import List

        _, edgeses = self.collect_for_root(add_e, merge, add_v, ee, root)
        ans: List[V_] = [None] * self.n  # type: ignore
        q = [(root, ee())]
        while q:
            i, edge_parent = q.pop()
            adjsi = self.adjs[i]
            parent = self.parents[i]
            edges = edgeses[i]
            accums_l = [ee()]
            for k, edge in enumerate(edges):
                if edge is None:
                    edge = edge_parent
                    edges[k] = edge_parent
                accums_l.append(merge(accums_l[-1], edge))
            accums_l.pop()  # not used
            accum_r = ee()
            for adj, accum_l, edge in zip(
                reversed(adjsi), reversed(accums_l), reversed(edges)
            ):
                assert edge is not None
                if adj != parent:
                    j, e = adj
                    # myself as a child of j
                    myself = add_v(merge(accum_l, accum_r), i, j)
                    q.append((j, add_e(myself, i, e)))
                accum_r = merge(edge, accum_r)
            ans[i] = add_v(accum_r, i, i)
        return ans


# E = int
# V = int
# def add_e(v: V, i: int, e: int) -> E:
#     raise NotImplementedError
# def merge(l: E, r: E) -> E:
#     raise NotImplementedError
# def add_v(e: E, i: int, parent: int) -> V:
#     raise NotImplementedError
# def ee() -> E:
#     raise NotImplementedError


# --------------------


def solve_dp_v():
    """
    https://atcoder.jp/contests/dp/tasks/dp_v
    """
    N, M = map(int, input().split())
    tree = TreeReroot(N)
    for _ in range(N - 1):
        x, y = map(int, input().split())
        tree.connect(x - 1, y - 1)
    E = int
    V = int

    def add_e(v: V, i: int, e: int) -> E:
        return (v + 1) % M

    def merge(l: E, r: E) -> E:
        return l * r % M

    def add_v(e: E, i: int, parent: int) -> V:
        return e

    def ee() -> E:
        return 1

    for ans in tree.collect_for_all(add_e, merge, add_v, ee):
        print(ans)


def solve_abc220_f():
    """
    https://atcoder.jp/contests/abc220/tasks/abc220_f
    """
    N = int(input())
    tree = TreeReroot(N)
    for _ in range(N - 1):
        u, v = map(int, input().split())
        tree.connect(u - 1, v - 1)

    from typing import Tuple

    E = Tuple[int, int]
    V = Tuple[int, int]

    def add_e(v: V, i: int, e: int) -> E:
        n, d = v
        return n, d + n

    def merge(l: E, r: E) -> E:
        nl, dl = l
        nr, dr = r
        return nl + nr, dl + dr

    def add_v(e: E, i: int, parent: int) -> V:
        n, d = e
        return n + 1, d

    def ee() -> E:
        return 0, 0

    for _, ans in tree.collect_for_all(add_e, merge, add_v, ee):
        print(ans)


def solve_abc222_f():
    """
    https://atcoder.jp/contests/abc222/tasks/abc222_f
    """
    N = int(input())
    tree = TreeReroot(N)
    C = [-1] * (N - 1)
    for i in range(N - 1):
        a, b, c = map(int, input().split())
        tree.connect(a - 1, b - 1, i)
        C[i] = c
    D = [int(v) for v in input().split()]

    E = int
    V = int

    def add_e(v: V, i: int, e: int) -> E:
        return v + C[e]

    def merge(l: E, r: E) -> E:
        return max(l, r)

    def add_v(e: E, i: int, parent: int) -> V:
        return max(e, D[i]) if i != parent else e

    def ee() -> E:
        return 0

    for ans in tree.collect_for_all(add_e, merge, add_v, ee):
        print(ans)
