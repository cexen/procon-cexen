# https://github.com/cexen/procon-cexen/blob/main/py/BIT.py
import operator
from typing import TypeVar, Generic

T = TypeVar("T")


class BIT(Generic[T]):
    """v1.2 @cexen"""

    from typing import Callable, Optional

    def __init__(
        self, n: int, f: Callable[[T, T], T], e: T, increasing: Optional[bool] = None
    ):
        """
        increasing: Required at bisect.
        True if grasp(i) <= grasp(i + 1).
        False if grasp(i) >= grasp(i + 1).
        """
        self.size = n
        self.tree = [e] * (n + 1)
        self.f = f
        self.e = e
        self.increasing = increasing

    def __len__(self) -> int:
        return self.size

    def grasp(self, i: Optional[int] = None) -> T:
        """O(log n). reduce(f, data[:i], e)."""
        if i is None:
            i = self.size
        i = min(i, self.size)
        s = self.e
        while i > 0:
            s = self.f(s, self.tree[i])
            i -= i & -i
        return s

    def operate(self, i: int, v: T) -> None:
        """O(log n). bit[i] = f(bit[i], v)."""
        i += 1  # to 1-indexed
        while i <= self.size:
            self.tree[i] = self.f(self.tree[i], v)
            i += i & -i

    def bisect_left(self, v: T) -> int:
        return self._bisect_any(v, left=True)

    def bisect_right(self, v: T) -> int:
        return self._bisect_any(v, left=False)

    def _bisect_any(self, v: T, left: bool = True) -> int:
        if self.increasing is None:
            raise RuntimeError("Specify increasing.")
        incr = self.increasing  # type: ignore
        i = 0  # 0-indexed
        u = self.e
        for s in reversed(range(self.size.bit_length())):
            k = i + (1 << s)  # 1-indexed
            if not k <= self.size:
                continue
            w = self.f(u, self.tree[k])
            if left and incr and not w < v:  # type: ignore
                continue
            if not left and incr and not w <= v:  # type: ignore
                continue
            if left and not incr and not v < w:  # type: ignore
                continue
            if not left and not incr and not v <= w:  # type: ignore
                continue
            i = k  # 0-indexed
            u = w
        return i  # 0-indexed


class BITInt(BIT[int]):
    """
    >>> b = BITInt(5, increasing=True)
    >>> b.operate(0, 10)
    >>> b.operate(1, 10)
    >>> b.operate(3, 10)
    >>> b.grasp(1)
    10
    >>> b.grasp(2)
    20
    >>> b.grasp(3)
    20
    >>> b.grasp(4)
    30
    >>> b.grasp(5)
    30
    >>> b.grasp()
    30
    >>> b.bisect_left(10)
    0
    >>> b.bisect_left(11)
    1
    >>> b.bisect_left(20)
    1
    >>> b.bisect_left(21)
    3
    >>> b.bisect_left(30)
    3
    >>> b.bisect_left(31)
    5
    >>> b.bisect_right(29)
    3
    >>> b.bisect_right(30)
    5

    >>> b = BITInt(3, f=min, e=10**9, increasing=False)
    >>> b.bisect_left(0), b.bisect_right(0)
    (3, 3)
    >>> b.operate(1, 5)
    >>> b.operate(2, 2)
    >>> b.bisect_left(6), b.bisect_right(6)
    (1, 1)
    >>> b.bisect_left(5), b.bisect_right(5)
    (1, 2)
    >>> b.bisect_left(4), b.bisect_right(4)
    (2, 2)
    """

    from typing import Callable, Optional

    def __init__(
        self,
        n: int,
        f: Callable[[int, int], int] = operator.add,
        e: int = 0,
        increasing: Optional[bool] = None,
    ):
        super().__init__(n, f, e, increasing)


class BITFloat(BIT[float]):
    from typing import Callable, Optional

    def __init__(
        self,
        n: int,
        f: Callable[[float, float], float] = operator.add,
        e: float = 0.0,
        increasing: Optional[bool] = None,
    ):
        super().__init__(n, f, e, increasing)
