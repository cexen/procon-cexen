# https://github.com/cexen/procon-cexen/blob/main/py/unionfind.py
class UnionFind:
    """
    v1.1 @cexen
    >>> uf = UnionFind(5)
    >>> uf.unite(2, 3)
    >>> uf.connected(0, 2)
    False
    >>> uf.connected(3, 2)
    True
    >>> uf.roots()
    [0, 1, 2, 4]
    >>> uf.root(3)
    2
    >>> uf.groupsize(3)
    2
    """

    def __init__(self, n: int):
        self.size = n
        self.data = [-1] * n
        self.sizes = [1] * n
        # self.weights = [0] * n

    def __len__(self) -> int:
        return self.size

    def root(self, i: int) -> int:
        """O(α(n)). 0 <= i < n."""
        data = self.data
        p = data[i]
        if p < 0:
            return i
        data[i] = p = self.root(p)
        return p

    def connected(self, i: int, j: int) -> bool:
        return self.root(i) == self.root(j)

    def groupsize(self, i: int) -> int:
        return self.sizes[self.root(i)]

    def roots(self):
        """O(n)."""
        return [i for i, v in enumerate(self.data) if v < 0]

    def unite(self, i: int, j: int) -> None:
        i = self.root(i)
        j = self.root(j)
        if i == j:
            return
        data = self.data
        sizes = self.sizes
        # weights = self.weights
        if (-data[i]) < (-data[j]):
            data[i] = j
            sizes[j] += sizes[i]
            # weights[j] += weights[i]
        elif (-data[i]) > (-data[j]):
            data[j] = i
            sizes[i] += sizes[j]
            # weights[i] += weights[j]
        else:
            data[i] += -1
            data[j] = i
            sizes[i] += sizes[j]
            # weights[i] += weights[j]


# --------------------


def solve_yosupojudge_unionfind():
    """
    Library Checker: Unionfind
    https://judge.yosupo.jp/problem/unionfind
    """
    N, Q = map(int, input().split())
    dst = UnionFind(N)
    ans = []
    for _ in range(Q):
        t, u, v = map(int, input().split())
        if t == 0:
            dst.unite(u, v)
        elif t == 1:
            ans.append(dst.connected(u, v))
        else:
            raise RuntimeError
    for a in ans:
        print(int(a))
