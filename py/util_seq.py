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


from typing import Any, MutableSequence


def next_permutation(seq: MutableSequence[Any]) -> bool:
    """
    Modify `seq` to the next distinct permutation in lexicographic order.
    Returns False if `seq` is the last permutation.
    O(len(seq)).
    cf. https://atcoder.jp/contests/abc363/editorial/10481
    Note: To enumerate all permutations, `seq` must be sorted first.
    >>> seq = [1, 1, 3, 3]
    >>> seq.sort()
    >>> while True:
    ...     print(seq)
    ...     if not next_permutation(seq):
    ...         break
    [1, 1, 3, 3]
    [1, 3, 1, 3]
    [1, 3, 3, 1]
    [3, 1, 1, 3]
    [3, 1, 3, 1]
    [3, 3, 1, 1]
    """
    n = len(seq)
    for i in range(n - 1)[::-1]:
        if seq[i] < seq[i + 1]:
            break
    else:
        return False
    for j in range(i + 1, n)[::-1]:
        if seq[i] < seq[j]:
            break
    else:
        assert False  # unreachable
    seq[i], seq[j] = seq[j], seq[i]
    for k in range((n - 1 - i) // 2):
        seq[i + 1 + k], seq[n - 1 - k] = seq[n - 1 - k], seq[i + 1 + k]
    return True


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


def test_atcoder_abc363_c():
    """
    Atcoder: abc363_c - Avoid K Palindrome 2
    https://atcoder.jp/contests/abc363/tasks/abc363_c
    """
    from math import factorial

    N, K = map(int, input().split())
    S = list(input())
    if len(set(S)) == 10:
        # miscellaneous special handling to avoid TLE
        print(factorial(len(S)))
        exit()
    S.sort()
    ans = 0
    while True:
        if not any(
            all(S[i + j] == S[i + K - 1 - j] for j in range(K // 2))
            for i in range(N - K + 1)
        ):
            ans += 1
        if not next_permutation(S):
            break
    print(ans)
