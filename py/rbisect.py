# https://github.com/cexen/procon-cexen/blob/main/py/rbisect.py
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from typing import TypeVar

_T = TypeVar("_T")


class ReversedSequence(Sequence[_T]):
    def __init__(self, seq: Sequence[_T]):
        self.seq = seq

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, i: int) -> _T:
        n = len(self.seq)
        if not 0 <= i < n:
            raise IndexError
        return self.seq[n - i - 1]


def rbisect_left(a, x, lo: int = 0, hi: int | None = None):
    """all(e > x for e in a[:i]) and all(x >= e for e in a[i:])"""
    n = len(a)
    if hi is None:
        hi = n
    rseq = ReversedSequence(a)
    i = bisect_right(rseq, x, lo=(n - hi), hi=(n - lo))
    return n - i


def rbisect_right(a, x, lo: int = 0, hi: int | None = None):
    """all(e >= x for e in a[:i]) and all(x > e for e in a[i:])"""
    n = len(a)
    if hi is None:
        hi = n
    rseq = ReversedSequence(a)
    i = bisect_left(rseq, x, lo=(n - hi), hi=(n - lo))
    return n - i


# --------------------
