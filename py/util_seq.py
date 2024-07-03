# https://github.com/cexen/procon-cexen/blob/main/py/util_seq.py
from typing import Sequence, TypeVar

_T = TypeVar("_T")


def rleify(s: Sequence[_T]) -> list[tuple[_T, int]]:
    """
    O(len(s)). Run Length Encoding.
    >>> rleify("assassin")
    [('a', 1), ('s', 2), ('a', 1), ('s', 2), ('i', 1), ('n', 1)]
    """
    if not len(s):
        return []
    i = 0
    rle: list[tuple[_T, int]] = []
    for j in range(len(s) + 1):
        if j == len(s) or s[i] != s[j]:
            rle.append((s[i], j - i))
            i = j
    return rle


from typing import Any, Sequence


def calc_common_prefix_lengths_z(seq: Sequence[Any]) -> list[int]:
    """
    O(len(seq)). Z-algorithm.
    Returns dp s.t. len(dp) == len(seq).
    dp[i] := common prefix length of seq[:] and seq[i:].
    cf. https://sen-comp.hatenablog.com/entry/2020/01/16/174230
    >>> calc_common_prefix_lengths_z("abcaabc")
    [7, 0, 0, 1, 3, 0, 0]
    """
    dp = [-1] * len(seq)
    dp[0] = len(seq)
    i = j = 0
    for ni in range(1, len(seq)):
        j = max(ni, j)
        if ni - i < ni:
            nj = ni + dp[ni - i]
            if nj < j:
                dp[ni] = nj - ni
                continue
        nj = j
        while nj < len(seq):
            if seq[nj - ni] != seq[nj]:
                break
            nj += 1
        dp[ni] = nj - ni
        i = ni
        j = nj
    return dp


# --------------------
