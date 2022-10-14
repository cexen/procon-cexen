# https://github.com/cexen/procon-cexen/blob/main/py/frac.py
from numbers import Number


class Frac(Number):
    """
    v1.6 @cexen
    >>> Frac(2, 3) == Frac(4, 6)
    True
    >>> Frac(2, 3) <= Frac(4, 6)
    True
    >>> Frac(2, 3) >= Frac(4, 6)
    True
    >>> Frac(2, 3) == Frac(4, 6)
    True
    >>> Frac(6, 1) == 6
    False
    >>> Frac(10000, 1) < Frac(1, 0)
    True
    >>> Frac(1, 0) == Frac(2, 0)
    True
    """

    __slots__ = (
        "n",
        "m",
    )

    def __init__(self, numerator: int = 0, denominator: int = 1):
        self.n = numerator
        self.m = denominator
        if denominator == 0:
            self.n = 1

    def __str__(self) -> str:
        return f"{self.n}/{self.m}"

    __repr__ = __str__

    def __hash__(self) -> int:
        return self.n * self.m  # Note: hash((self.n, self.m)) is slow

    def __eq__(self, w: object) -> bool:
        return isinstance(w, self.__class__) and self.n * w.m - self.m * w.n == 0

    def __le__(self, w: object) -> bool:
        return isinstance(w, self.__class__) and self.n * w.m - self.m * w.n <= 0

    def __lt__(self, w: object) -> bool:
        return isinstance(w, self.__class__) and self.n * w.m - self.m * w.n < 0
