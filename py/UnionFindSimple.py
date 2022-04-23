# https://github.com/cexen/procon-cexen/blob/main/py/UnionFindSimple.py
class UnionFind:
    def __init__(self, n: int):
        self.size = n
        self.data = [-1 for _ in range(n)]
        self.sizes = [1 for _ in range(n)]

    def __len__(self):
        return self.size

    def root(self, i: int):
        """0 <= i < n"""
        if self.data[i] < 0:
            return i
        self.data[i] = self.root(self.data[i])
        return self.data[i]

    def connected(self, i: int, j: int):
        return self.root(i) == self.root(j)

    def groupsize(self, i: int):
        return self.sizes[self.root(i)]

    def roots(self):
        """O(n)"""
        return [i for i, v in enumerate(self.data) if v < 0]

    def unite(self, i: int, j: int):
        i = self.root(i)
        j = self.root(j)
        if i == j:
            return
        if (-self.data[i]) < (-self.data[j]):
            self.data[i] = j
            self.sizes[j] += self.sizes[i]
        elif (-self.data[i]) > (-self.data[j]):
            self.data[j] = i
            self.sizes[i] += self.sizes[j]
        else:
            self.data[i] += -1
            self.data[j] = i
            self.sizes[i] += self.sizes[j]
