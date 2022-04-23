# https://github.com/cexen/procon-cexen/blob/main/py/UnionFind.py
class UnionFind:
    def __init__(self, n: int):
        self.size = n
        self.data = [-1] * n
        self.sizes = [1] * n
        self.weights = [0] * n

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
            self.weights[j] += self.weights[i]
        elif (-self.data[i]) > (-self.data[j]):
            self.data[j] = i
            self.sizes[i] += self.sizes[j]
            self.weights[i] += self.weights[j]
        else:
            self.data[i] += -1
            self.data[j] = i
            self.sizes[i] += self.sizes[j]
            self.weights[i] += self.weights[j]
