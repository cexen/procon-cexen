# https://github.com/cexen/procon-cexen/blob/main/py/lazy_sequence.py
from collections.abc import Callable, Sequence
from typing import TypeVar, overload

_T = TypeVar("_T")
Self = TypeVar("Self", bound="LazySequence")


class LazySequence(Sequence[_T]):
    def __init__(self, f: Callable[[int], _T], n: int | range):
        self.f = f
        if isinstance(n, int):
            self.r = range(n)
        elif isinstance(n, range):
            self.r = n
        else:
            raise TypeError

    def __len__(self):
        return len(self.r)

    @overload
    def __getitem__(self, i: int) -> _T: ...

    @overload
    def __getitem__(self: Self, i: slice) -> Self: ...

    def __getitem__(self: Self, i: int | slice) -> _T | Self:
        r = self.r[i]
        if isinstance(r, int):
            return self.f(r)
        else:
            return self.__class__(self.f, r)


# --------------------
