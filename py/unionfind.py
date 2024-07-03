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


from typing import Generic, Protocol, TypeVar

Self = TypeVar("Self")


class SupportsAddSubNeg(Protocol):
    def __add__(self: Self, other: Self, /) -> Self: ...

    def __sub__(self: Self, other: Self, /) -> Self: ...

    def __neg__(self: Self) -> Self: ...


_T = TypeVar("_T", bound=SupportsAddSubNeg)


class UnionFindWithPotential(Generic[_T]):
    """
    v1.0 @cexen.
    See UnionFindWithPotentialInt for usage.
    cf. https://qiita.com/drken/items/cce6fc5c579051e64fab
    """

    def __init__(self, n: int, _0: _T):
        self._data = [~1] * n
        self._sizes = [1] * n
        self._pots = [_0] * n

    def __len__(self) -> int:
        return len(self._data)

    def size(self, i: int) -> int:
        assert 0 <= i < len(self)
        return self._sizes[self.root(i)]

    def root(self, i: int) -> int:
        assert 0 <= i < len(self)
        if self._data[i] < 0:
            return i
        r = self._data[i]
        self._data[i] = self.root(r)
        self._pots[i] += self._pots[r]
        return self._data[i]

    def united(self, i: int, j: int) -> bool:
        return self.root(i) == self.root(j)

    def diff(self, i: int, j: int) -> _T:
        """
        YOU MUST ENSURE THAT self.united(i, j).
        """
        assert self.united(i, j)  # potentials are refreshed here
        return self._pots[j] - self._pots[i]

    def unite(self, i: int, j: int, diff: _T) -> bool:
        """
        YOU MUST ENSURE THAT not self.united(i, j) or self.diff(i, j) == diff.
        Returns if newly united.
        """
        ri = self.root(i)
        rj = self.root(j)
        if ri == rj:
            assert self.diff(i, j) == diff
            return False
        diff += self.diff(ri, i) + self.diff(j, rj)
        del i, j
        if ~self._data[ri] < ~self._data[rj]:
            ri, rj = rj, ri
            diff = -diff
        elif ~self._data[ri] == ~self._data[rj]:
            self._data[ri] = ~(~self._data[ri] + 1)
        self._data[rj] = ri
        self._sizes[ri] += self._sizes[rj]
        self._pots[rj] += diff
        return True


class UnionFindWithPotentialInt(UnionFindWithPotential[int]):
    """
    >>> uf = UnionFindWithPotentialInt(5)
    >>> uf.unite(0, 1, -5)
    True
    >>> uf.unite(1, 2, 3)
    True
    >>> uf.diff(0, 2)
    -2
    >>> uf.unite(0, 2, -2)
    False
    >>> uf.unite(2, 0, 2)
    False
    >>> uf.unite(0, 2, 3)
    Traceback (most recent call last):
        ...
    AssertionError
    >>> uf.diff(3, 4)
    Traceback (most recent call last):
        ...
    AssertionError
    """

    def __init__(self, n: int):
        return super().__init__(n, 0)


# --------------------


def solve_yosupojudge_unionfind():
    """
    Library Checker: Unionfind
    https://judge.yosupo.jp/problem/unionfind
    """
    N, Q = map(int, input().split())
    dst = UnionFind(N)
    ans = list[int]()
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


def solve_atcoder_abc328_f():
    """
    AtCoder: abc328_f
    https://atcoder.jp/contests/abc328/tasks/abc328_f
    """
    N, Q = map(int, input().split())
    ans = list[int]()
    uf = UnionFindWithPotentialInt(N)
    for i in range(Q):
        a, b, d = map(int, input().split())
        a -= 1
        b -= 1
        if uf.united(a, b) and uf.diff(a, b) != d:
            continue
        uf.unite(a, b, d)
        ans.append(i)
    print(*(i + 1 for i in ans))
