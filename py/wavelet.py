# https://github.com/cexen/procon-cexen/blob/main/py/wavelet.py
def popcount(n: int) -> int:
    """
    Returns the number of 1 bits.
    cf. https://nixeneko.hatenablog.com/entry/2018/03/04/000000
    >>> popcount(0b11110001)
    5
    """
    assert n >= 0
    s = 0
    while n:
        # Note: 63 bit is much faster than 64 bit on PyPy3 7.3.0
        c = n & 0x7FFFFFFFFFFFFFFF  # c: uint64_t
        c = (c & 0x5555555555555555) + ((c >> 1) & 0x5555555555555555)
        c = (c & 0x3333333333333333) + ((c >> 2) & 0x3333333333333333)
        c = (c & 0x0F0F0F0F0F0F0F0F) + ((c >> 4) & 0x0F0F0F0F0F0F0F0F)
        c = (c & 0x00FF00FF00FF00FF) + ((c >> 8) & 0x00FF00FF00FF00FF)
        c = (c & 0x0000FFFF0000FFFF) + ((c >> 16) & 0x0000FFFF0000FFFF)
        c = (c & 0x00000000FFFFFFFF) + ((c >> 32) & 0x00000000FFFFFFFF)
        s += c
        n >>= 63
    return s


from typing import Sequence


class SuccinctBV(Sequence[bool]):
    """
    v1.3 @cexen.
    Succinct Bit Vector.
    cf. https://miti-7.hatenablog.com/entry/2018/04/15/155638

    >>> bv = SuccinctBV([True, False, False, False, True] + [False] * 10)
    >>> len(bv)
    15
    >>> bv[0]
    True
    >>> bv[1]
    False
    >>> bv[4:1:-1]
    [True, False, False]
    >>> bv.rank1(0), bv.rank0(0)
    (0, 0)
    >>> bv.rank1(1), bv.rank0(1)
    (1, 0)
    >>> bv.rank1(4), bv.rank1(5), bv.rank1(6)
    (1, 2, 2)
    >>> bv.rank0(4), bv.rank0(5), bv.rank0(6)
    (3, 3, 4)
    >>> bv.rank1(14), bv.rank1(15)
    (2, 2)
    >>> bv.rank0(14), bv.rank0(15)
    (12, 13)
    >>> assert all(bv.rank1(i) == bv.rank(True, i) for i in range(15))
    >>> assert all(bv.rank0(i) == bv.rank(False, i) for i in range(15))
    >>> [bv.select1(k) for k in range(15)]
    [0, 4, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
    >>> [bv.select0(k) for k in range(15)]
    [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15]
    >>> assert all(bv.select1(i) == bv.select(True, i) for i in range(15))
    >>> assert all(bv.select0(i) == bv.select(False, i) for i in range(15))
    >>> bv2 = SuccinctBV([False] * 5000 + [True] + [False] * 5000)
    >>> len(bv2)
    10001
    >>> bv2.rank1(5000), bv2.rank1(5001), bv2.rank1(5002)
    (0, 1, 1)
    >>> bv2.rank0(5000), bv2.rank0(5001), bv2.rank0(5002)
    (5000, 5000, 5001)
    >>> bv2.select1(0), bv2.select1(1), bv2.select1(10**9)
    (5000, 10001, 10001)
    >>> bv2.select0(4999), bv2.select0(5000), bv2.select0(10**9)
    (4999, 5001, 10001)
    """

    # Note: len(P) is O(sqrt(n)).
    P = [0, 1, 1, 2, 1, 2, 2, 3]

    from typing import Iterable, Union, Tuple, List, overload

    def __init__(self, bits: Iterable[bool]):
        """O(len(bits))."""
        bits_ = list(bits)
        self.n = len(bits_)
        bitlen = max(1, self.n.bit_length())
        self.s = (bitlen + 1) // 2
        self.l = 2 * bitlen * self.s
        assert bitlen <= 126
        assert (
            self.s <= 63
        )  # not required but desirable for faster bit operations on PyPy3 7.3.0
        self.ll = ll = [0] * (1 + (self.n + self.l - 1) // self.l)
        self.ls = ls = [0] * (1 + (self.n + self.s - 1) // self.s)
        self.lb = lb = [0] * (1 + (self.n + self.s - 1) // self.s)
        for i, v in enumerate(bits_):
            d, m = divmod(i, self.s)
            b = int(v)
            ll[1 + i // self.l] += b
            ls[1 + d] += b
            lb[d] |= b << m
        for i in range(len(ll) - 1):
            ll[i + 1] += ll[i]
        for i in range(len(ls) - 1):
            if (i + 1) % (self.l // self.s) == 0:
                ls[i + 1] = 0
            else:
                ls[i + 1] += ls[i]
        self.P.extend(popcount(i) for i in range(len(self.P), 1 << self.s))

    def __len__(self) -> int:
        """O(1)."""
        return self.n

    @overload
    def __getitem__(self, i: int) -> bool:
        """O(1)."""
        ...

    @overload
    def __getitem__(self, i: slice) -> List[bool]:
        """O(len(i))."""
        ...

    def __getitem__(self, i: Union[int, slice]) -> Union[bool, List[bool]]:
        if isinstance(i, slice):
            return [self[j] for j in range(len(self))[i]]
        if not 0 <= i < self.n:
            raise IndexError
        d, m = divmod(i, self.s)
        return bool((self.lb[d] >> m) & 1)

    def rank1(self, j: int) -> int:
        """O(1). Returns self[:j].count(True)."""
        assert 0 <= j <= self.n
        x = j // self.l
        d, m = divmod(j, self.s)
        mask = (1 << m) - 1
        return self.ll[x] + self.ls[d] + self.P[self.lb[d] & mask]

    def rank0(self, j: int) -> int:
        """O(1). Returns self[:j].count(False)."""
        return j - self.rank1(j)

    def rankall(self, j: int) -> Tuple[int, int]:
        """O(1). Returns (self[:j].count(False), self[:j].count(True))."""
        k = self.rank1(j)
        return j - k, k

    def rank(self, v: bool, j: int) -> int:
        """O(1). Returns self[:j].count(v)."""
        return self.rank1(j) if v else self.rank0(j)

    def select(self, v: bool, k: int) -> int:
        """
        O(log n).
        Returns argwhere(self[:], v)[k].
        Returns n if not found.
        """
        assert 0 <= k
        xl = 0
        xr = len(self.ll) - 1
        while xl < xr:
            m = (xl + xr + 1) // 2
            n = self.ll[m] if v else min(self.l * m, self.n) - self.ll[m]
            if k < n:
                xr = m - 1
            else:
                xl = m
        if xl == len(self.ll) - 1:
            return self.n  # not found
        k -= self.ll[xl] if v else min(self.l * xl, self.n) - self.ll[xl]

        w = self.l // self.s
        yl = yl0 = xl * w
        yr = min((xl + 1) * w, len(self.ls)) - 1
        while yl < yr:
            m = (yl + yr + 1) // 2
            n = self.ls[m] if v else min(self.s * m, self.n) - self.s * yl0 - self.ls[m]
            if k < n:
                yr = m - 1
            else:
                yl = m
        k -= self.ls[yl] if v else min(self.s * yl, self.n) - self.s * yl0 - self.ls[yl]

        b = self.lb[yl]
        nv = int(not v)
        for i in range(self.s):  # linear search
            k -= ((b >> i) & 1) ^ nv
            if k < 0:
                break
        else:
            raise RuntimeError(f"Unreachable")
        return self.s * yl + i

    def select1(self, k: int) -> int:
        """
        O(log n).
        Returns argwhere(self[:], True)[k].
        Returns n if not found.
        """
        return self.select(True, k)

    def select0(self, k: int) -> int:
        """
        O(log n).
        Returns argwhere(self[:], False)[k].
        Returns n if not found.
        """
        return self.select(False, k)


class WaveletMatrix(Sequence[int]):
    """
    v1.5 @cexen.
    cf.
    https://miti-7.hatenablog.com/entry/2018/04/28/152259
    https://miti-7.hatenablog.com/entry/2019/02/01/152131
    https://www.slideshare.net/pfi/ss-15916040
    https://ei1333.github.io/library/structure/wavelet/wavelet-matrix.cpp.html

    >>> wm = WaveletMatrix([3, 1, 4, 1, 5, 9, 2])
    >>> len(wm)
    7
    >>> wm[2]
    4
    >>> wm[:]
    [3, 1, 4, 1, 5, 9, 2]
    >>> [wm.rank(1, j) for j in range(1 + len(wm))]
    [0, 0, 1, 1, 2, 2, 2, 2]
    >>> [wm.rank(7, j) for j in range(1 + len(wm))]
    [0, 0, 0, 0, 0, 0, 0, 0]
    >>> [wm.select(1, k) for k in range(len(wm))]
    [1, 3, 7, 7, 7, 7, 7]
    >>> [wm.select(7, k) for k in range(len(wm))]
    [7, 7, 7, 7, 7, 7, 7]
    >>> [wm.quantile(0, len(wm), k) for k in range(len(wm))]
    [1, 1, 2, 3, 4, 5, 9]
    >>> [wm.quantile(2, 5, k) for k in range(3)]
    [1, 4, 5]
    >>> [wm.topk(1, 5, k) for k in range(5)]
    [[], [(1, 2)], [(1, 2), (4, 1)], [(1, 2), (4, 1), (5, 1)], [(1, 2), (4, 1), (5, 1)]]
    >>> [wm.sum(1, k) for k in range(1, 6)]
    [0, 1, 5, 6, 11]
    >>> [wm.rangefreq(0, len(wm), 2, k) for k in range(2, 12)]
    [0, 1, 2, 3, 4, 4, 4, 4, 5, 5]
    >>> [wm.rangelist(1, 5, k, 5) for k in range(6)]
    [[(1, 2), (4, 1)], [(1, 2), (4, 1)], [(4, 1)], [(4, 1)], [(4, 1)], []]
    >>> [wm.rangemaxk(1, 5, k) for k in range(5)]
    [[], [(5, 1)], [(5, 1), (4, 1)], [(5, 1), (4, 1), (1, 2)], [(5, 1), (4, 1), (1, 2)]]
    >>> [wm.rangemink(1, 5, k) for k in range(5)]
    [[], [(1, 2)], [(1, 2), (4, 1)], [(1, 2), (4, 1), (5, 1)], [(1, 2), (4, 1), (5, 1)]]
    >>> [wm.prevvalue(1, 5, k, 5) for k in range(6)]
    [(4, 1), (4, 1), (4, 1), (4, 1), (4, 1), None]
    >>> [wm.nextvalue(1, 5, k, 5) for k in range(6)]
    [(1, 2), (1, 2), (4, 1), (4, 1), (4, 1), None]
    >>> wm.intersect(0, 5, 2, 6)
    [(1, 2, 1), (4, 1, 1), (5, 1, 1)]
    """

    from typing import Iterable, Optional, List, Tuple, Union, overload

    def __init__(self, iterable: Iterable[int], bitlen: Optional[int] = None):
        """
        O(n * bitlen) where n == len(iterable).
        If bitlen omitted: bitlen = 1 + max(iterable).bit_length().
        """
        from typing import List, Dict

        tab = list(iterable)
        assert min(tab, default=0) >= 0
        if bitlen is None:
            # take one more bit to ensure `max(tab) < (1 << bitlen) - 1`
            bitlen = 1 + max(tab, default=0).bit_length()
        matrix: List[SuccinctBV] = []
        num0s: List[int] = []
        for ib in reversed(range(bitlen)):
            bits = [1 == (v >> ib) & 1 for v in tab]
            zeros: List[int] = []
            ones: List[int] = []
            for v, b in zip(tab, bits):
                if b:
                    ones.append(v)
                else:
                    zeros.append(v)
            tab[: len(zeros)] = zeros
            tab[len(zeros) :] = ones
            matrix.append(SuccinctBV(bits))
            num0s.append(len(zeros))
        d_idxs: Dict[int, int] = {}
        for i in reversed(range(len(tab))):
            d_idxs[tab[i]] = i

        self.bitlen = bitlen
        self.matrix = matrix
        self.num0s = num0s
        self.d_idxs = d_idxs

    def __len__(self) -> int:
        """O(1)."""
        return len(self.matrix[0])

    @overload
    def __getitem__(self, i: int) -> int:
        """O(bitlen)."""
        ...

    @overload
    def __getitem__(self, i: slice) -> List[int]:
        """O(bitlen * len(i))."""
        ...

    def __getitem__(self, i: Union[int, slice]) -> Union[int, List[int]]:
        if isinstance(i, slice):
            return [self[k] for k in range(len(self))[i]]
        if not 0 <= i < len(self):
            raise IndexError
        ans = 0
        for ib in range(self.bitlen):
            nib = self.bitlen - 1 - ib
            bv, num0 = self.matrix[ib], self.num0s[ib]
            b = bv[i]
            ans += b << nib
            offset = num0 if b else 0
            i = offset + bv.rank(b, i)
        return ans

    def rank(self, v: int, j: int) -> int:
        """O(bitlen). Returns self[:j].count(v)."""
        assert 0 <= j <= len(self)
        if v not in self.d_idxs:
            return 0
        for ib in range(self.bitlen):
            nib = self.bitlen - 1 - ib
            bv, num0 = self.matrix[ib], self.num0s[ib]
            b = bool((v >> nib) & 1)
            offset = num0 if b else 0
            j = offset + bv.rank(b, j)
        return j - self.d_idxs[v]

    def select(self, v: int, k: int) -> int:
        """
        O(bitlen * log n).
        Returns argwhere(self[:], v)[k].
        Returns len(self) if not found.
        """
        if v not in self.d_idxs:
            return len(self)
        k = self.d_idxs[v] + k
        for ib in reversed(range(self.bitlen)):
            nib = self.bitlen - 1 - ib
            bv, num0 = self.matrix[ib], self.num0s[ib]
            b = bool((v >> nib) & 1)
            offset = num0 if b else 0
            k = bv.select(b, k - offset)
        return k

    def quantile(self, i: int, j: int, k: int) -> int:
        """O(bitlen). Returns sorted(self[i:j])[k]."""
        assert 0 <= i < len(self)
        assert 0 <= j <= len(self)
        assert 0 <= k < j - i
        ans = 0
        for ib in range(self.bitlen):
            nib = self.bitlen - 1 - ib
            bv, num0 = self.matrix[ib], self.num0s[ib]
            l0, l1 = bv.rankall(i)
            r0, r1 = bv.rankall(j)
            b = r0 - l0 <= k
            ans += b << nib
            if b:
                i = num0 + l1
                j = num0 + r1
                k -= r0 - l0
            else:
                i = l0
                j = r0
        assert 0 <= k < j - i
        return ans

    def topk(self, i: int, j: int, k: Optional[int] = None) -> List[Tuple[int, int]]:
        """
        O(bitlen + m log m) where m = len(set(self[i:j])).
        Returns Counter(self[i:j]).most_common(k).
        (value, count) LIMIT k ORDER BY count DESC, value ASC.
        Returns all values if k is None.
        Note that 0 <= len(keys) <= k; not necessarily == k if not found.
        """
        from heapq import heappop, heappush
        from typing import Tuple, List

        assert 0 <= i <= j <= len(self)
        if k is None:
            k = j - i
        assert 0 <= k

        if k == 0:
            return []
        q = [(-(j - i), 0, 0, i, j)]
        ans: List[Tuple[int, int]] = []
        while q:
            _, v, ib, i, j = heappop(q)
            if ib == self.bitlen:
                if i < j:
                    ans.append((v, j - i))
                    if len(ans) == k:
                        break
                continue
            nib = self.bitlen - 1 - ib
            bv, num0 = self.matrix[ib], self.num0s[ib]
            l0, l1 = bv.rankall(i)
            r0, r1 = bv.rankall(j)
            if l0 < r0:
                heappush(q, (-(r0 - l0), v, ib + 1, l0, r0))
            if l1 < r1:
                heappush(q, (-(r1 - l1), v + (1 << nib), ib + 1, num0 + l1, num0 + r1))
        return ans

    def sum(self, i: int, j: int) -> int:
        """
        O(bitlen + m log m) where m = len(set(self[i:j])).
        Returns sum(self[i:j]).
        """
        return sum(v * c for v, c in self.topk(i, j)) if i < j else 0

    def rangefreq(self, i: int, j: int, x: int = 0, y: Optional[int] = None) -> int:
        """O(bitlen). Returns len([v for v in self[i:j] if x <= v < y])."""
        assert 0 <= i <= j <= len(self)  # accept i == len(self) for len(self) == 0
        if y is None:
            y = (1 << self.bitlen) - 1
        assert 0 <= x <= y
        if x > 0:
            return self.rangefreq(i, j, 0, y) - self.rangefreq(i, j, 0, x)
        assert x == 0
        ans = 0
        for ib in range(self.bitlen):
            nib = self.bitlen - 1 - ib
            bv, num0 = self.matrix[ib], self.num0s[ib]
            l0, l1 = bv.rankall(i)
            r0, r1 = bv.rankall(j)
            b = bool((y >> nib) & 1)
            if b:
                ans += r0 - l0
                i = num0 + l1
                j = num0 + r1
            else:
                i = l0
                j = r0
        return ans

    def rangelist(
        self,
        i: int,
        j: int,
        x: int = 0,
        y: Optional[int] = None,
        k: Optional[int] = None,
        reverse: bool = False,
    ) -> List[Tuple[int, int]]:
        """
        O(bitlen + k).
        Returns sorted(Counter([v for v in self[i:j] if x <= v < y]).items(), reverse=reverse)[:k].
        (value, count) WHERE x <= value < y LIMIT k ORDER BY value {DESC if reverse else ASC}.
        """
        from typing import Tuple, List

        assert 0 <= i <= j <= len(self)
        if y is None:
            y = (1 << self.bitlen) - 1
        assert 0 <= x <= y
        if k is None:
            k = j - i

        if k == 0:
            return []
        TypeQ = List[Tuple[int, int, int, int, bool, bool]]
        q: TypeQ = [(0, 0, i, j, False, False)]
        ans: List[Tuple[int, int]] = []
        while q:
            v, ib, i, j, gtx, lty = q.pop()
            if ib == self.bitlen:
                if i < j and lty:
                    assert x <= v < y
                    ans.append((v, j - i))
                    if len(ans) == k:
                        break
                continue
            nib = self.bitlen - 1 - ib
            bv, num0 = self.matrix[ib], self.num0s[ib]
            l0, l1 = bv.rankall(i)
            r0, r1 = bv.rankall(j)
            x_is_0 = (x >> nib) & 1 == 0
            y_is_1 = (y >> nib) & 1 == 1
            nq: TypeQ = []
            if l1 < r1 and (lty or y_is_1):
                l1 += num0
                r1 += num0
                nq.append((v + 1 << nib, ib + 1, l1, r1, gtx or x_is_0, lty))
            if l0 < r0 and (gtx or x_is_0):
                nq.append((v, ib + 1, l0, r0, gtx, lty or y_is_1))
            if reverse:
                nq.reverse()
            q.extend(nq)
        return ans

    def rangemaxk(
        self, i: int, j: int, k: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """
        O(bitlen + k).
        Returns [(v, self.count(v)) for v in sorted(self[i:j], reversed=True)[:k]].
        Returns all if k is None.
        """
        return self.rangelist(i, j, k=k, reverse=True)

    def rangemink(
        self, i: int, j: int, k: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """
        O(bitlen + k).
        Returns [(v, self.count(v)) for v in sorted(self[i:j])[:k]].
        Returns all if k is None.
        """
        return self.rangelist(i, j, k=k)

    def prevvalue(self, i: int, j: int, x: int, y: int) -> Optional[Tuple[int, int]]:
        """
        O(bitlen).
        Returns (v, self[i:j].count(v)) where v = max(v for v in self[i:j] if x <= v < y).
        Returns None if not found.
        """
        ans = self.rangelist(i, j, x, y, k=1, reverse=True)
        return ans[0] if len(ans) else None

    def nextvalue(self, i: int, j: int, x: int, y: int) -> Optional[Tuple[int, int]]:
        """
        O(bitlen).
        Returns (v, self[i:j].count(v)) where v = min(v for v in self[i:j] if x <= v < y).
        Returns None if not found.
        """
        ans = self.rangelist(i, j, x, y, k=1)
        return ans[0] if len(ans) else None

    def intersect(
        self, i1: int, j1: int, i2: int, j2: int
    ) -> List[Tuple[int, int, int]]:
        """
        O(bitlen + j1-i1 + j2-i2).
        Returns [(v, l1.count(v), l2.count(v)) for v in sorted(set(l1) & set(l2))].
        where l1, l2 = self[i1:j1], self[i2:j2].
        (value, count1, count2) ORDER BY value ASC.
        """
        from typing import Tuple, List

        assert 0 <= i1 <= j1 <= len(self)
        assert 0 <= i2 <= j2 <= len(self)

        TypeQ = List[Tuple[int, int, int, int, int, int]]
        q: TypeQ = [(0, 0, i1, j1, i2, j2)]
        ans: List[Tuple[int, int, int]] = []
        while q:
            v, ib, i1, j1, i2, j2 = q.pop()
            if ib == self.bitlen:
                if i1 < j1 and i2 < j2:
                    ans.append((v, j1 - i1, j2 - i2))
                continue
            nib = self.bitlen - 1 - ib
            bv, num0 = self.matrix[ib], self.num0s[ib]
            l0_1, l1_1 = bv.rankall(i1)
            r0_1, r1_1 = bv.rankall(j1)
            l0_2, l1_2 = bv.rankall(i2)
            r0_2, r1_2 = bv.rankall(j2)
            if l1_1 < r1_1 and l1_2 < r1_2:
                l1_1 += num0
                r1_1 += num0
                l1_2 += num0
                r1_2 += num0
                q.append((v + (1 << nib), ib + 1, l1_1, r1_1, l1_2, r1_2))
            if l0_1 < r0_1 and l0_2 < r0_2:
                q.append((v, ib + 1, l0_1, r0_1, l0_2, r0_2))
        return ans


def solve_yosupojudge_rank():
    # https://judge.yosupo.jp/problem/static_range_frequency
    N, Q = map(int, input().split())
    A = [int(v) for v in input().split()]
    ans = []
    wav = WaveletMatrix(A)
    for _ in range(Q):
        l, r, x = map(int, input().split())
        ans.append(wav.rank(x, r) - wav.rank(x, l))
    for a in ans:
        print(a)


def solve_yosupojudge_rangefreq():
    # https://judge.yosupo.jp/problem/static_range_frequency
    N, Q = map(int, input().split())
    A = [int(v) for v in input().split()]
    ans = []
    wav = WaveletMatrix(A)
    for _ in range(Q):
        l, r, x = map(int, input().split())
        ans.append(wav.rangefreq(l, r, x, x + 1))
    for a in ans:
        print(a)


def solve_yosupojudge_quantile():
    # https://judge.yosupo.jp/problem/range_kth_smallest
    N, Q = map(int, input().split())
    A = [int(v) for v in input().split()]
    ans = []
    wav = WaveletMatrix(A)
    for _ in range(Q):
        l, r, k = map(int, input().split())
        ans.append(wav.quantile(l, r, k))
    for a in ans:
        print(a)
