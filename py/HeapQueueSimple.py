# https://github.com/cexen/procon-cexen/blob/main/py/HeapQueueSimple.py
from heapq import heappop, heappush, heappushpop, heapreplace


class HeapQueue:
    """
    q = HeapQueue()
    q.append(p, v)
    v = q.pop()
    """

    def __init__(self, reverse=False):
        self.reverse = reverse
        self.heap = []

    def __len__(self):
        return len(self.heap)

    def append(self, p, v):
        heappush(self.heap, (-p if self.reverse else p, v))

    def appendpop(self, p, v):
        return heappushpop(self.heap, (-p if self.reverse else p, v))[1]

    def popappend(self, p, v):
        return heapreplace(self.heap, (-p if self.reverse else p, v))[1]

    def pop(self):
        return heappop(self.heap)[1]

    def front(self):
        return self.heap[0][1]
