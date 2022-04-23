# https://github.com/cexen/procon-cexen/blob/main/py/BitSimple.py
class Bit:
    def __init__(self, n: int):
        self.size = n
        self.tree = [0] * (n + 1)

    def __len__(self):
        return self.size

    def sum(self, i: int) -> int:
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

    def add(self, i: int, v: int) -> None:
        while i <= self.size:
            self.tree[i] += v
            i += i & -i
