# https://github.com/cexen/procon-cexen/blob/main/py/lazy_sequence.py
from collections.abc import Sequence


class LazySequence(Sequence):
    def __init__(self, f, n):
        self.f = f
        if isinstance(n, int):
            self.r = range(n)
        elif isinstance(n, range):
            self.r = n
        else:
            raise TypeError

    def __len__(self):
        return len(self.r)

    def __getitem__(self, i):
        r = self.r[i]
        if isinstance(r, int):
            return self.f(r)
        else:
            return self.__class__(self.f, r)
