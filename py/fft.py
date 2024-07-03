# https://github.com/cexen/procon-cexen/blob/main/py/fft.py
from collections.abc import MutableSequence, Sequence
from math import gcd


class FFTNaive:
    """
    @cexen v1.8
    cf.)
    [競プロのための高速フーリエ変換](https://www.creativ.xyz/fast-fourier-transform/)
    [NTT(数論変換)のやさしい解説 - Senの競技プログラミング備忘録](https://sen-comp.hatenablog.com/entry/2021/02/06/180310)
    [【競プロer向け】FFT を習得しよう！ | 東京工業大学デジタル創作同好会traP](https://trap.jp/post/1386/)

    >>> fft = FFTNaive()
    >>> a = [0, 1, 2]
    >>> fft.fft(a)
    >>> fft.ifft(a)
    >>> a
    [0, 1, 2, 0]
    >>> b = [1, 0, 0]
    >>> fft.conv(a, b)
    [0, 1, 2, 0, 0, 0]
    >>> fft.inv([1, -1, -1], 7)  # fibonacci, 1/(1-x-x^2)
    [1, 1, 2, 3, 5, 8, 13]
    >>> fft.div_at_n([1], [1, -1, -1], 6)  # fibonacci, 1/(1-x-x^2)
    13
    >>> fft.div_at_n([0, 1], [1, -1, -1], 6)  # fibonacci, x/(1-x-x^2)
    8
    """

    def __init__(self, mod: int = 998244353, g: int = 3):
        """mod must be a prime. requires gcd(mod, g) == 1."""
        roots, iroots = self._gen_roots(mod, g, *self._calc_am(mod))
        self.mod = mod
        self._roots = roots
        self._iroots = iroots

    @staticmethod
    def _calc_am(mod: int) -> tuple[int, int]:
        """
        returns (a, m) s.t. mod == a * (2 ** m) + 1.
        >>> FFTNaive._calc_am(998244353)
        (119, 23)
        >>> FFTNaive._calc_am(1000000007)
        (500000003, 1)
        >>> FFTNaive._calc_am(167772161)
        (5, 25)
        >>> FFTNaive._calc_am(469762049)
        (7, 26)
        >>> FFTNaive._calc_am(1224736769)
        (73, 24)
        """
        a = mod - 1
        m = 0
        while a & 1 == 0:
            a >>= 1
            m += 1
        return a, m

    @staticmethod
    def _gen_roots(mod: int, g: int, a: int, m: int) -> tuple[list[int], list[int]]:
        """mod must be a prime."""
        assert a * 2**m + 1 == mod
        if gcd(mod, g) != 1:
            raise ValueError("gcd(mod, g) != 1")

        ig = pow(g, -1, mod)
        roots = [pow(g, (mod - 1) >> (i + 1), mod) for i in range(m)]
        iroots = [pow(ig, (mod - 1) >> (i + 1), mod) for i in range(m)]
        return roots, iroots

    @staticmethod
    def _invert_indices(arr: MutableSequence[int], d: int) -> None:
        """e.g.) 6 (110) <-> 3 (011)"""
        assert len(arr) == 1 << d
        for i in range(len(arr)):
            j = 0
            for k in range(d):
                j |= ((i >> k) & 1) << (d - 1 - k)
            if i < j:
                arr[i], arr[j] = arr[j], arr[i]

    @staticmethod
    def _butterfly(
        arr: MutableSequence[int], d: int, mod: int, roots: Sequence[int]
    ) -> None:
        n = len(arr)
        for i in range(d):
            b = 1 << i  # block size // 2
            for k in range(0, n, b << 1):
                w = 1
                for j in range(b):
                    l = j + k
                    r = l + b
                    arr[r] *= w
                    arr[l], arr[r] = (arr[l] + arr[r]) % mod, (arr[l] - arr[r]) % mod
                    w = w * roots[i] % mod
            # assert w**2 % mod == 1, w

    @classmethod
    def _fft(cls, arr: MutableSequence[int], mod: int, roots: Sequence[int]) -> int:
        """O(n log n). mod must be a prime. returns n s.t. n == 2 ** ceil(log2(len(arr)))."""
        orglen = len(arr)
        d = (orglen - 1).bit_length()  # ceil(log2(orglen))
        n = 1 << d
        arr.extend([0] * (n - orglen))  # extend len to 2 ** d
        cls._invert_indices(arr, d)
        if d > len(roots):
            raise ValueError(f"arr is too large for this mod ({n} > 2 ** {len(roots)})")
        cls._butterfly(arr, d, mod, roots)
        return n

    def fft(self, arr: MutableSequence[int]) -> None:
        """O(n log n). in-place. len(arr) will be extended to 2 ** k."""
        self._fft(arr, self.mod, self._roots)

    def ifft(self, arr: MutableSequence[int]) -> None:
        """O(n log n). in-place. len(arr) will be extended to 2 ** k."""
        n = self._fft(arr, self.mod, self._iroots)
        ninv = pow(n, -1, self.mod)
        for i in range(len(arr)):
            arr[i] = arr[i] * ninv % self.mod

    def conv_cyclic(self, a: Sequence[int], b: Sequence[int], n: int) -> list[int]:
        """O(n log n). returns c s.t. len(c) == n and ck == sum(ai * bj if (i + j) % n == k)"""
        if not n >= max(len(a), len(b)):
            raise ValueError("n must be >= max(len(a), len(b))")
        if n != n & (-n):
            raise ValueError("n must be 2 ** k")
        fa = [0] * n
        fb = [0] * n
        fa[: len(a)] = a
        fb[: len(b)] = b
        self.fft(fa)
        self.fft(fb)
        fc = [u * v for u, v in zip(fa, fb)]
        self.ifft(fc)
        return fc

    def conv(self, a: Sequence[int], b: Sequence[int]) -> list[int]:
        """O(n log n). returns c s.t. len(c) == n == len(a) + len(b) - 1 and ck == sum(ai * bj if i + j == k)"""
        n = len(a) + len(b) - 1
        c = self.conv_cyclic(a, b, 1 << (n - 1).bit_length())
        return c[:n]

    def inv(self, a: Sequence[int], n: int) -> list[int]:
        """
        O(n log n). returns b' s.t. b[:n] == b' and a*b == [1, 0, 0, ...].
        cf.
        https://nyaannyaan.github.io/library/fps/formal-power-series.hpp.html
        https://nyaannyaan.github.io/library/fps/ntt-friendly-fps.hpp
        https://paper.dropbox.com/doc/fps-EoHXQDZxfduAB8wD1PMBW
        """
        assert len(a) > 0 and n >= 0
        assert a[0] != 0  # non-singular
        b = [0] * n
        b[0] = pow(a[0], -1, self.mod)
        for i in range((n - 1).bit_length()):
            d = 1 << i
            f = [0] * (d << 1)
            g = [0] * (d << 1)
            k = min(len(a), d << 1)
            f[:k] = a[:k]
            g[:d] = b[:d]
            self.fft(f)
            self.fft(g)
            for j in range(d << 1):
                f[j] = f[j] * g[j] % self.mod
            self.ifft(f)
            f[:d] = [0] * d
            self.fft(f)
            for j in range(d << 1):
                f[j] = f[j] * g[j] % self.mod
            self.ifft(f)
            for j in range(d, min(n, d << 1)):
                b[j] = -f[j] % self.mod
        return b

    def div_at_n(self, a: Sequence[int], b: Sequence[int], n: int) -> int:
        """
        O(k log k log n) where k = len(a) + len(b). returns (a/b)[n].
        Bostan-Mori.
        cf. https://qiita.com/ryuhe1/items/da5acbcce4ac1911f47a
        cf. https://nyaannyaan.github.io/library/fps/kitamasa.hpp.html
        """
        while n > 0:
            nb = [(-1) ** i * v for i, v in enumerate(b)]
            p = self.conv(a, nb)
            q = self.conv(b, nb)
            a = p[n % 2 :: 2]
            b = q[::2]
            n = n // 2
        return a[0] * pow(b[0], -1, self.mod) % self.mod


class FFT:
    """
    NotImplemented yet
    cf: [任意modでの畳み込み演算をO(n log(n))で - math314のブログ](https://math314.hateblo.jp/entry/2015/05/07/014908)
    """

    ...


# --------------------


def solve_atcoder_practice2_f():
    """
    AtCoder ACL Practice Contest F - Convolution
    https://atcoder.jp/contests/practice2/tasks/practice2_f
    """
    MOD = 998244353
    fft = FFTNaive(MOD)
    N, M = map(int, input().split())
    A = [int(v) for v in input().split()]
    B = [int(v) for v in input().split()]
    C = fft.conv(A, B)
    print(*C)


def solve_yosupojudge_inv():
    """
    Library Checker: Inv of Formal Power Series
    https://judge.yosupo.jp/problem/inv_of_formal_power_series
    """
    MOD = 998244353
    fft = FFTNaive(MOD)
    N = int(input())
    A = [int(v) for v in input().split()]
    B = fft.inv(A, N)
    print(*B)


# solve_atcoder_practice2_f()
# solve_yosupojudge_inv()
