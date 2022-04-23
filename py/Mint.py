# https://github.com/cexen/procon-cexen/blob/main/py/Mint.py
from numbers import Number


class Mint(Number):
    """
    v1.3 @cexen
    >>> Mint(3) * Mint(7)
    21
    >>> Mint.factorial(10)
    3628800
    >>> Mint.comb(10, 2)
    45
    """

    MOD = 1_000_000_007  # Must be a prime
    CACHE_FACTORIALS = [1, 1]
    __slots__ = ("v",)

    from typing import SupportsInt, Union

    # https://stackoverflow.com/questions/39505379/whats-the-type-hint-for-can-be-converted-to-mean
    AcceptableToInt = Union[SupportsInt, str, bytes, bytearray]

    def __init__(self, v: AcceptableToInt):
        self.v = int(v) % self.MOD

    @property
    def inv(self):
        return self ** (self.MOD - 2)

    @classmethod
    def factorial(cls, v: AcceptableToInt):
        for i in range(len(cls.CACHE_FACTORIALS), int(v) + 1):
            cls.CACHE_FACTORIALS.append(cls.CACHE_FACTORIALS[-1] * i % cls.MOD)
        return cls(cls.CACHE_FACTORIALS[int(v)])

    @classmethod
    def perm(cls, n: int, r: int):
        if n < r or r < 0:
            return cls(0)
        return cls.factorial(n) // cls.factorial(n - r)

    @classmethod
    def comb(cls, n: int, r: int):
        if n < r or r < 0:
            return cls(0)
        return cls.perm(n, r) // cls.factorial(r)

    def __str__(self) -> str:
        return str(self.v)

    __repr__ = __str__

    def __int__(self) -> int:
        return self.v

    def __hash__(self) -> int:
        return hash(self.v)

    def __eq__(self, w: object) -> bool:
        return isinstance(w, self.__class__.AcceptableToInt) and self.v == int(w)

    def __neg__(self):
        return self.__class__(-self.v)

    def __pos__(self):
        return self.__class__(self.v)

    def __abs__(self):
        return self.__class__(self.v)

    def __add__(self, w: AcceptableToInt):
        return self.__class__(self.v + int(w))

    __radd__ = __add__

    def __sub__(self, w: AcceptableToInt):
        return self.__class__(self.v - int(w))

    def __rsub__(self, u: AcceptableToInt):
        return self.__class__(int(u) - self.v)

    def __mul__(self, w: AcceptableToInt):
        return self.__class__(self.v * int(w))

    __rmul__ = __mul__

    def __floordiv__(self, w: AcceptableToInt):
        return self * self.__class__(w).inv

    def __rfloordiv__(self, u: AcceptableToInt):
        return u * self.inv

    def __pow__(self, w: AcceptableToInt):
        return self.__class__(pow(self.v, int(w), self.MOD))

    def __rpow__(self, u: AcceptableToInt):
        return self.__class__(u) ** self
