# https://github.com/cexen/procon-cexen/blob/main/py/util_lin.py
from typing import Sequence


def matmul(
    a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], mod: int
) -> list[list[int]]:
    h = len(a)
    if h == 0:
        return []
    l = len(a[0])
    assert l == len(b)
    if l == 0:
        return [[] for _ in range(h)]
    w = len(b[0])
    bT = [[0] * l for _ in range(w)]
    for i in range(l):
        bi = b[i]
        for j in range(w):
            bT[j][i] = bi[j]
    c = [[0] * w for _ in range(h)]
    for i in range(h):
        ai = a[i]
        ci = c[i]
        for j in range(w):
            bTj = bT[j]
            v = 0
            for k in range(l):
                v = (v + ai[k] * bTj[k]) % mod
            ci[j] = v
    return c


# --------------------
