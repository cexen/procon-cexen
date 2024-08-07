# https://github.com/cexen/procon-cexen/blob/main/py/avl.py
import sys

sys.setrecursionlimit(10**9)


from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterable, Mapping, MutableMapping, MutableSet
from typing import Generic, Protocol, TypeVar, overload, runtime_checkable

_C = TypeVar("_C", bound="Comparable")


@runtime_checkable
class Comparable(Protocol):
    @abstractmethod
    def __lt__(self: _C, value: _C, /) -> bool: ...

    @abstractmethod
    def __gt__(self: _C, value: _C, /) -> bool: ...


_K = TypeVar("_K", bound=Comparable)
_V = TypeVar("_V")


class AVLNodeBase(ABC, Generic[_K, _V]):
    __slots__ = ()

    height: int
    size: int
    _l: "AVLNodeBase[_K, _V]"
    _r: "AVLNodeBase[_K, _V]"

    @property
    def bias(self) -> int:
        return self.l.height - self.r.height

    @property
    def l(self):
        return self._l

    @l.setter
    def l(self, val: "AVLNodeBase[_K, _V]"):
        self._l_setter(val)

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, val: "AVLNodeBase[_K, _V]"):
        self._r_setter(val)

    def __bool__(self) -> bool:
        """O(1)."""
        return self.height > 0

    def __len__(self) -> int:
        """O(1)."""
        return self.size

    @abstractmethod
    def __contains__(self, key):
        """O(log n)."""
        ...

    def __iter__(self):
        """O(n)."""
        return iter(self.keys())

    @abstractmethod
    def __getitem__(self, key: _K) -> _V:
        """O(log n)."""
        ...

    @abstractmethod
    def setitem(self, key: _K, value: _V) -> "AVLNodeBase[_K, _V]":
        """O(log n)."""
        ...

    @abstractmethod
    def insertitem(self, key: _K, value: _V) -> tuple["AVLNodeBase[_K, _V]", bool]:
        """O(log n). Insert an item only if the key does not exist. Returns whether inserted."""
        ...

    @abstractmethod
    def pop(self, key: _K) -> tuple["AVLNodeBase[_K, _V]", _V]:
        """O(log n). Delete and return an item by key."""
        ...

    @abstractmethod
    def delitem(self, key: _K) -> "AVLNodeBase[_K, _V]":
        """O(log n)."""
        ...

    @abstractmethod
    def popmin(self) -> tuple["AVLNodeBase[_K, _V]", _K, _V]:
        """O(log n). Delete and return min item."""
        ...

    @abstractmethod
    def popmax(self) -> tuple["AVLNodeBase[_K, _V]", _K, _V]:
        """O(log n). Delete and return max item."""
        ...

    @abstractmethod
    def min(self) -> tuple[_K, _V]:
        """O(log n)."""
        ...

    @abstractmethod
    def max(self) -> tuple[_K, _V]:
        """O(log n)."""
        ...

    @abstractmethod
    def bisect_left(self, key: _K) -> tuple[_K, _V] | None:
        """O(log n)."""

    @abstractmethod
    def bisect_right(self, key: _K) -> tuple[_K, _V] | None:
        """O(log n)."""

    @abstractmethod
    def _l_setter(self, val: "AVLNodeBase[_K, _V]"): ...

    @abstractmethod
    def _r_setter(self, val: "AVLNodeBase[_K, _V]"): ...

    @abstractmethod
    def clear(self) -> "AVLNodeBase[_K, _V]": ...

    @abstractmethod
    def keys(self) -> set[_K]:
        """O(n)."""
        ...

    @abstractmethod
    # https://github.com/python/typeshed/issues/4435
    def values(self) -> list[_V]:
        """O(n)."""
        ...

    @abstractmethod
    def items(self) -> set[tuple[_K, _V]]:
        """O(n)."""
        ...

    @abstractmethod
    def rotate_l(self) -> "AVLNodeBase[_K, _V]": ...

    @abstractmethod
    def rotate_r(self) -> "AVLNodeBase[_K, _V]": ...

    @abstractmethod
    def rotate_lr(self) -> "AVLNodeBase[_K, _V]": ...

    @abstractmethod
    def rotate_rl(self) -> "AVLNodeBase[_K, _V]": ...

    @abstractmethod
    def dump(self, indent: str = "", head: str = "") -> str: ...


class AVLNodeEmptyError(RuntimeError):
    pass


class AVLNodeEmpty(AVLNodeBase[_K, _V]):
    __slots__ = ("_l", "_r")

    height = 0
    size = 0

    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
            cls.__init__(cls._instance)
        return cls._instance

    def __init__(self):
        self._l = self
        self._r = self

    def __str__(self) -> str:
        return "{}"

    def __repr__(self) -> str:
        return f"AVLNodeEmpty()"

    def __contains__(self, _) -> bool:
        return False

    def __getitem__(self, _):
        raise KeyError

    def setitem(self, key: _K, value: _V) -> "AVLNode[_K, _V]":
        newroot = AVLNode(key, value)
        return newroot

    def insertitem(self, key: _K, value: _V):
        return self.setitem(key, value), True

    def pop(self, _):
        raise KeyError

    def delitem(self, _):
        raise KeyError

    def popmax(self):
        raise KeyError

    def popmin(self):
        raise KeyError

    def min(self):
        raise KeyError

    def max(self):
        raise KeyError

    def bisect_left(self, _):
        return None

    def bisect_right(self, _):
        return None

    def _l_setter(self, _):
        raise AVLNodeEmptyError("You should never want to set l of empty node")

    def _r_setter(self, _):
        raise AVLNodeEmptyError("You should never want to set r of empty node")

    def clear(self):
        return self

    def keys(self):
        return set()

    def values(self):
        return []

    def items(self):
        return set()

    def rotate_l(self):
        raise AVLNodeEmptyError("You should never want to rotate empty node")

    def rotate_r(self):
        raise AVLNodeEmptyError("You should never want to rotate empty node")

    def rotate_lr(self):
        raise AVLNodeEmptyError("You should never want to rotate empty node")

    def rotate_rl(self):
        raise AVLNodeEmptyError("You should never want to rotate empty node")

    def dump(self, indent: str = "", head: str = "") -> str:
        graph = f"{indent}{head}.\n"
        return graph


class AVLNode(AVLNodeBase[_K, _V]):
    __slots__ = ("key", "value", "height", "size", "_l", "_r")

    is_empty = False

    def __init__(self, key: _K, value: _V):
        self.key = key
        self.value = value
        self.height = 1
        self.size = 1
        self._l = AVLNodeEmpty[_K, _V]()
        self._r = AVLNodeEmpty[_K, _V]()

    def __str__(self) -> str:
        return f"{{{self.key}: {self.value}}}"

    def __repr__(self) -> str:
        return f"AVLNode(key={self.key}, value={self.value})"

    def __contains__(self, key) -> bool:
        if key < self.key:
            return key in self.l
        if key > self.key:
            return key in self.r
        return key == self.key

    def __getitem__(self, key) -> _V:
        if key < self.key:  # type: ignore
            return self.l[key]  # type: ignore
        if key > self.key:  # type: ignore
            return self.r[key]  # type: ignore
        if key == self.key:
            return self.value
        raise KeyError

    def setitem(self, key: _K, value: _V) -> "AVLNode[_K, _V]":
        if key < self.key:  # type: ignore
            self.l = self.l.setitem(key, value)
            return self._balance()
        elif key > self.key:  # type: ignore
            self.r = self.r.setitem(key, value)
            return self._balance()
        elif key == self.key:
            self.value = value
            return self
        else:
            raise KeyError

    def insertitem(self, key: _K, value: _V) -> tuple["AVLNode[_K, _V]", bool]:
        if key == self.key:
            return self, False
        return self.setitem(key, value), True

    def pop(self, key: _K) -> tuple[AVLNodeBase[_K, _V], _V]:
        if key < self.key:  # type: ignore
            self.l, value = self.l.pop(key)
            return self._balance(), value
        elif key > self.key:  # type: ignore
            self.r, value = self.r.pop(key)
            return self._balance(), value
        elif key == self.key:
            value = self.value
            if self.l.height == 0:
                return self.r, value
            self.l, mkey, mvalue = self.l.popmax()
            self.key = mkey
            self.value = mvalue
            return self._balance(), value
        else:
            raise KeyError

    def delitem(self, key: _K) -> AVLNodeBase[_K, _V]:
        return self.pop(key)[0]

    def popmin(self) -> tuple[AVLNodeBase[_K, _V], _K, _V]:
        if self.l.height:
            self.l, key, value = self.l.popmin()
            return self._balance(), key, value
        return self.r, self.key, self.value

    def popmax(self) -> tuple[AVLNodeBase[_K, _V], _K, _V]:
        if self.r.height:
            self.r, key, value = self.r.popmax()
            return self._balance(), key, value
        return self.l, self.key, self.value

    def max(self) -> tuple[_K, _V]:
        if self.r.height:
            return self.r.max()
        return self.key, self.value

    def min(self) -> tuple[_K, _V]:
        if self.l.height:
            return self.l.min()
        return self.key, self.value

    def bisect_left(self, key: _K) -> tuple[_K, _V] | None:
        if key < self.key or key == self.key:
            if self.l.height:
                ret = self.l.bisect_left(key)
                if ret is not None:
                    return ret
            return self.key, self.value
        elif key > self.key:
            return self.r.bisect_left(key)
        elif key == self.key:
            return self.key, self.value
        else:
            raise ValueError

    def bisect_right(self, key: _K) -> tuple[_K, _V] | None:
        if key < self.key:
            if self.l.height:
                ret = self.l.bisect_right(key)
                if ret is not None:
                    return ret
            return self.key, self.value
        elif key > self.key or key == self.key:
            return self.r.bisect_right(key)
        else:
            raise ValueError

    def _l_setter(self, value: AVLNodeBase[_K, _V]) -> None:
        self._l = value
        self._refresh_property()

    def _r_setter(self, value: AVLNodeBase[_K, _V]) -> None:
        self._r = value
        self._refresh_property()

    def _refresh_property(self):
        self.height = 1 + max(self.l.height, self.r.height)
        self.size = 1 + self.l.size + self.r.size

    def clear(self):
        return AVLNodeEmpty()

    def keys(self) -> set[_K]:
        return self.l.keys() | set([self.key]) | self.r.keys()

    def values(self) -> list[_V]:
        return self.l.values() + [self.value] + self.r.values()

    def items(self) -> set[tuple[_K, _V]]:
        return self.l.items() | set([(self.key, self.value)]) | self.r.items()

    def _balance(self) -> "AVLNode[_K, _V]":
        if self.bias > 1:
            return self.rotate_r() if self.l.bias >= 0 else self.rotate_lr()  # type: ignore
        if self.bias < -1:
            return self.rotate_l() if self.r.bias <= 0 else self.rotate_rl()  # type: ignore
        return self

    def rotate_l(self):
        newroot = self.r
        self.r, newroot.l = newroot.l, self
        return newroot

    def rotate_r(self):
        newroot = self.l
        self.l, newroot.r = newroot.r, self
        return newroot

    def rotate_lr(self):
        self.l = self.l.rotate_l()
        return self.rotate_r()

    def rotate_rl(self):
        self.r = self.r.rotate_r()
        return self.rotate_l()

    def dump(self, indent: str = "", head: str = "") -> str:
        graph = (
            self.r.dump(f"{indent}    ", "")
            + f"{indent}{head}{self.key}={self.value}({self.height})\n"
            + self.l.dump(f"{indent}    ", "")
        )
        return graph


class AVLMap(MutableMapping[_K, _V]):
    """
    v1.1 @cexen
    Based on: http://wwwa.pikara.ne.jp/okojisan/avl-tree/
    >>> am = AVLMap[int, str]()
    >>> am[-1] = "Z"
    >>> am[0] = "a"
    >>> am[0]
    'a'
    >>> am[-1]
    'Z'
    >>> len(am)
    2
    >>> del am[0]
    >>> len(am)
    1
    >>> am.popmin()
    (-1, 'Z')
    >>> len(am)
    0
    >>> test_avlmap()
    """

    @overload
    def __init__(self, iterable: Mapping[_K, _V] | None = None): ...

    @overload
    def __init__(self, iterable: Iterable[tuple[_K, _V]] | None = None): ...

    def __init__(self, iterable=None):
        self._root: AVLNodeBase[_K, _V] = AVLNodeEmpty[_K, _V]()
        if iterable is not None:
            if isinstance(iterable, Mapping):
                iterable = iterable.items()
            for key, value in iterable:
                self[key] = value

    def __bool__(self) -> bool:
        return bool(self._root)

    def __len__(self) -> int:
        return len(self._root)

    def __iter__(self):
        return iter(self._root)

    def __str__(self):
        return str(dict(self.items()))

    def __repr__(self):
        return f"AVLMap({str(self)})"

    def __contains__(self, key) -> bool:
        return key in self._root

    def __getitem__(self, key: _K) -> _V:
        return self._root[key]

    def __setitem__(self, key: _K, value: _V) -> None:
        self._root = self._root.setitem(key, value)

    def __delitem__(self, key: _K) -> None:
        self._root = self._root.delitem(key)

    def clear(self) -> None:
        self._root = self._root.clear()

    def insert(self, key: _K, value: _V) -> bool:
        """Insert an item only if the key does not exist. Returns whether inserted."""
        self._root, success = self._root.insertitem(key, value)
        return success

    def pop(self, key: _K, default=None) -> _V:
        """
        >>> am = AVLMap[int, str]({0: "a", 3: "d", 4: "e"})
        >>> am.pop(3)
        'd'
        """
        try:
            self._root, value = self._root.pop(key)
        except KeyError:
            if default is not None:
                return default
            else:
                raise KeyError
        return value

    def popmin(self) -> tuple[_K, _V]:
        """
        >>> am = AVLMap[int, str]({0: "a", 3: "d", 4: "e"})
        >>> am.popmin()
        (0, 'a')
        >>> am.popmin()
        (3, 'd')
        """
        self._root, key, value = self._root.popmin()
        return key, value

    def popmax(self) -> tuple[_K, _V]:
        """
        >>> am = AVLMap[int, str]({0: "a", 3: "d", 4: "e"})
        >>> am.popmax()
        (4, 'e')
        >>> am.popmax()
        (3, 'd')
        """
        self._root, key, value = self._root.popmax()
        return key, value

    def min(self) -> tuple[_K, _V]:
        """
        >>> am = AVLMap[int, str]({0: "a", 3: "d", 4: "e"})
        >>> am.min()
        (0, 'a')
        """
        return self._root.min()

    def max(self) -> tuple[_K, _V]:
        """
        >>> am = AVLMap[int, str]({0: "a", 3: "d", 4: "e"})
        >>> am.max()
        (4, 'e')
        """
        return self._root.max()

    def bisect_left(self, key: _K) -> tuple[_K, _V] | None:
        """
        >>> am = AVLMap[int, str]({0: "a", 3: "d", 4: "e"})
        >>> am.bisect_left(-1)
        (0, 'a')
        >>> am.bisect_left(0)
        (0, 'a')
        >>> am.bisect_left(1)
        (3, 'd')
        >>> am.bisect_left(4)
        (4, 'e')
        >>> am.bisect_left(5)
        """
        return self._root.bisect_left(key)

    def bisect_right(self, key: _K) -> tuple[_K, _V] | None:
        """
        >>> am = AVLMap[int, str]({0: "a", 3: "d", 4: "e"})
        >>> am.bisect_right(-1)
        (0, 'a')
        >>> am.bisect_right(0)
        (3, 'd')
        >>> am.bisect_right(1)
        (3, 'd')
        >>> am.bisect_right(4)
        >>> am.bisect_right(5)
        """
        return self._root.bisect_right(key)

    def keys(self) -> set[_K]:
        return self._root.keys()

    def values(self) -> list[_V]:
        return self._root.values()

    def items(self) -> set[tuple[_K, _V]]:
        return self._root.items()

    def dump(self) -> str:
        return self._root.dump()


class AVLDefaultMap(AVLMap[_K, _V]):
    """
    v1.1 @cexen
    Based on: http://wwwa.pikara.ne.jp/okojisan/avl-tree/
    >>> am = AVLDefaultMap[int, str](default_factory=str)
    >>> am[-1] = "Z"
    >>> am[0]
    ''
    >>> am[-1]
    'Z'
    >>> len(am)
    2
    >>> del am[0]
    >>> len(am)
    1
    >>> am.popmin()
    (-1, 'Z')
    >>> len(am)
    0
    """

    @overload
    def __init__(
        self,
        iterable: Mapping[_K, _V] | None = None,
        default_factory: Callable[[], _V] | None = None,
    ): ...

    @overload
    def __init__(
        self,
        iterable: Iterable[tuple[_K, _V]] | None = None,
        default_factory: Callable[[], _V] | None = None,
    ): ...

    def __init__(self, iterable=None, default_factory: Callable[[], _V] | None = None):
        super().__init__(iterable)
        self.default_factory = default_factory

    def __missing__(self, key: _K) -> _V:
        if self.default_factory is None:
            raise KeyError(key)
        value = self.default_factory()
        self[key] = value
        return value

    def __getitem__(self, key: _K) -> _V:
        try:
            return self._root[key]
        except KeyError:
            return self.__missing__(key)


class AVLset(MutableSet[_K]):
    """
    v1.1 @cexen
    >>> s = AVLset[int]([3, 1, 4])
    >>> s.discard(1)
    >>> s.remove(3)
    >>> s.pop()
    4
    >>> len(s)
    0
    >>> from typing import Union
    >>> s = AVLset[Union[int, float]]()
    >>> s.insert(1)
    True
    >>> s.insert(1.0)
    False
    >>> s.pop()
    1
    >>> test_avlset()
    """

    def __init__(self, iterable: Iterable[_K] | None = None):
        self._am = AVLMap[_K, _K]()
        if iterable is not None:
            for k in iterable:
                self._am.insert(k, k)

    def __bool__(self) -> bool:
        return bool(self._am)

    def __len__(self) -> int:
        return len(self._am)

    def __iter__(self):
        return iter(self._am)

    def __str__(self):
        return str(set(self))

    def __repr__(self):
        return f"AVLset({str(self)})"

    def __contains__(self, key) -> bool:
        return key in self._am

    def clear(self) -> None:
        self._am.clear()

    def add(self, key: _K) -> None:
        self.insert(key)

    def insert(self, key: _K) -> bool:
        return self._am.insert(key, key)

    def remove(self, key: _K) -> None:
        del self._am[key]

    def discard(self, key: _K) -> None:
        try:
            self.remove(key)
        except KeyError:
            pass

    def pop(self) -> _K:
        return self.popmin()

    def popmin(self) -> _K:
        return self._am.popmin()[0]

    def popmax(self) -> _K:
        return self._am.popmax()[0]

    def min(self) -> _K:
        return self._am.min()[0]

    def max(self) -> _K:
        return self._am.max()[0]

    def bisect_left(self, key: _K) -> _K | None:
        ret = self._am.bisect_left(key)
        return None if ret is None else ret[0]

    def bisect_right(self, key: _K) -> _K | None:
        ret = self._am.bisect_right(key)
        return None if ret is None else ret[0]

    def dump(self) -> str:
        return self._am.dump()


class AVLMultiset(MutableSet[_K]):
    """
    v1.1 @cexen
    >>> s = AVLMultiset[int]([3, 1, 1, 4])
    >>> s.discard(1)
    >>> s.remove(3)
    >>> s.popmax()
    4
    >>> len(s)
    1
    >>> s.pop()
    1
    >>> len(s)
    0
    >>> s = AVLMultiset[int | float]()
    >>> s.insert(1)
    True
    >>> s.insert(1.0)
    False
    >>> len(s)
    1
    >>> s.add(1.0)
    >>> len(s)
    2
    >>> s.max()
    1.0
    >>> s.min()
    1
    >>> test_avlmultiset()
    """

    def __init__(self, iterable: Iterable[_K] | None = None):
        self._am = AVLDefaultMap[_K, deque[_K]](default_factory=deque)
        self.size = 0
        if iterable is not None:
            for k in iterable:
                self._am[k].append(k)
                self.size += 1

    def __bool__(self) -> bool:
        return bool(self._am)

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        return iter(sum((list(q) for q in self._am.values()), []))

    def __str__(self):
        return str(list(self))

    def __repr__(self):
        return f"AVLMultiset({str(self)})"

    def __contains__(self, key) -> bool:
        return key in self._am

    def clear(self) -> None:
        self._am.clear()
        self.size = 0

    def add(self, key: _K) -> None:
        self._am[key].append(key)
        self.size += 1

    def insert(self, key: _K) -> bool:
        success = self._am.insert(key, deque([key]))
        if success:
            self.size += 1
        return success

    def _popqleft(self, key: _K, q: deque[_K]) -> _K:
        value = q.popleft()
        if not len(q):
            del self._am[key]
        self.size -= 1
        return value

    def _popq(self, key: _K, q: deque[_K]) -> _K:
        value = q.pop()
        if not len(q):
            del self._am[key]
        self.size -= 1
        return value

    def remove(self, key: _K) -> None:
        self._popqleft(key, self._am[key])

    def discard(self, key: _K) -> None:
        try:
            self.remove(key)
        except KeyError:
            pass

    def pop(self) -> _K:
        return self.popmin()

    def popmin(self) -> _K:
        return self._popqleft(*self._am.min())

    def popmax(self) -> _K:
        return self._popq(*self._am.max())

    def min(self) -> _K:
        return self._am.min()[1][0]

    def max(self) -> _K:
        return self._am.max()[1][-1]

    def bisect_left(self, key: _K) -> _K | None:
        ret = self._am.bisect_left(key)
        return None if ret is None else ret[1][0]

    def bisect_right(self, key: _K) -> _K | None:
        ret = self._am.bisect_right(key)
        return None if ret is None else ret[1][0]

    def dump(self) -> str:
        return self._am.dump()


# --------------------


def test_avlmap_balanced(s: AVLMap):
    q = deque[AVLNodeBase]()
    q.append(s._root)
    while len(q):
        node = q.pop()
        if node.height == 0:
            continue
        assert -1 <= node.bias <= 1
        q.append(node.l)
        q.append(node.r)


def test_avlmap(num: int = 10000):
    import heapq
    from random import randint

    values = [randint(-1000, 1000) for _ in range(num)]
    q = list[int]()
    qs = set[int]()
    s = AVLMap[int, int]()
    while len(values):
        if len(q) and randint(0, 1):
            k, v = s.popmin()
            ans = heapq.heappop(q)
            qs.remove(ans)
            assert ans == k == v
        else:
            v = values.pop()
            if v not in qs:
                heapq.heappush(q, v)
                qs.add(v)
            s[v] = v
    test_avlmap_balanced(s)


def test_avlset(num: int = 10000):
    import heapq
    from random import randint

    values = [randint(-1000, 1000) for _ in range(num)]
    q = list[int]()
    qs = set[int]()
    s = AVLset[int]()
    while len(values):
        if len(q) and randint(0, 1):
            v = s.popmin()
            ans = heapq.heappop(q)
            qs.remove(ans)
            assert ans == v
        else:
            v = values.pop()
            if v not in qs:
                heapq.heappush(q, v)
                qs.add(v)
            s.add(v)


def test_avlmultiset(num: int = 10000):
    import heapq
    from random import randint

    values = [randint(-1000, 1000) for _ in range(num)]
    q = list[int]()
    s = AVLMultiset[int]()
    while len(values):
        if len(q) and randint(0, 1):
            assert heapq.heappop(q) == s.pop()
        else:
            v = values.pop()
            heapq.heappush(q, v)
            s.add(v)
