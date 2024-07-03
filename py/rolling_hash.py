# https://github.com/cexen/procon-cexen/blob/main/py/rolling_hash.py
from collections.abc import Iterable
from math import gcd
from random import randrange


class RollingHash:
    """
    mod: 61 bit Mersenne prime.
    v1.3 @cexen.
    e.g.) hash([s0, s1, s2, s3]) == (s0 * b**3 + s1 * b**2 + s2 * b**1 + s3) % mod.
    cf. https://qiita.com/keymoon/items/11fac5627672a6d6a9f6

    You may generate a random base:
    >>> rh = RollingHash(base=RollingHash.generate_base(minval=26))

    >>> rh = RollingHash(base=37)
    >>> rh.calc([1, 2, 3])  # [0, 1, 1*37+2, 1*(37**2)+2*37+3]
    [0, 1, 39, 1446]
    >>> rh.slice([0, 1, 39, 1446], 0, 3)  # [1, 2, 3]
    1446
    >>> rh.slice([0, 1, 39, 1446], 0, 2)  # [1, 2]
    39
    >>> rh.slice([0, 1, 39, 1446], 0, 1)  # [1]
    1
    >>> rh.slice([0, 1, 39, 1446], 0, 0)  # []
    0
    >>> rh.slice([0, 1, 39, 1446], 2, 3)  # [3]
    3
    >>> rh.lshift(1, 1)  # [1, 0]
    37
    >>> rh.lshift(39, 2)  # [1, 2, 0, 0]
    53391
    """

    mod = (1 << 61) - 1

    def __init__(self, base: int):
        self.base = base
        self.bs = [1, base]

    @classmethod
    def generate_base(cls, minval: int) -> int:
        """
        Generates a random base (> minval).
        cf. https://trap.jp/post/1036/
        """
        mod = (1 << 61) - 1
        assert minval < mod // 2
        r = 37
        while True:
            k = randrange(1, mod - 1)
            if gcd(k, mod - 1) != 1:
                continue
            base = cls.pow(r, k)
            if base > minval:
                break
        return base

    def calc(self, iter: Iterable[int]) -> list[int]:
        """
        O(len(iter)).
        Returns hashes of [iter[:0], iter[:1], ..., iter[:len(iter)]].
        Required for non-collision: min(iter) > 0.
        e.g.) hash == iter[0] * base**2 + iter[1] * base**1 + iter[2] * base**0.
        """
        hashes = [0]
        for v in iter:
            assert 0 < v < self.base
            hashes.append(self.per_mod(self.mul(hashes[-1], self.base) + v))
        self._cache_bs(len(hashes))
        return hashes

    def slice(self, hashes: list[int], i: int = 0, j: int | None = None) -> int:
        """Returns hash of iter[i:j]."""
        if j is None:
            j = len(hashes) - 1
        assert 0 <= i < len(hashes)
        assert 0 <= j < len(hashes)
        assert i <= j
        self._cache_bs(j - i)
        return self.per_mod(hashes[j] - self.mul(hashes[i], self.bs[j - i]))

    def concat(self, hashl: int, lenr: int, hashr: int) -> int:
        """Returns hash of (iterl + iterr). Requires lenr := len(iterr)."""
        assert lenr >= 0
        # return self.per_mod(self.lshift(hashl, lenr) + hashr)
        self._cache_bs(lenr)
        return self.per_mod(self.mul(hashl, self.bs[lenr]) + hashr)

    def lshift(self, hash: int, b: int) -> int:
        """Returns hash of (iter + [0] * b)."""
        assert b >= 0
        self._cache_bs(b)
        return self.mul_mod(hash, self.bs[b])

    def _cache_bs(self, n: int):
        """O(Δn). Ensure len(self.bs) - 1 >= n."""
        base = self.base
        bs = self.bs
        for _ in range(n + 1 - len(bs)):
            bs.append(self.mul_mod(bs[-1], base))

    @classmethod
    def mul_mod(cls, a: int, b: int) -> int:
        return cls.per_mod(cls.mul(a, b))

    @classmethod
    def pow(cls, base: int, exp: int) -> int:
        """
        Note: maybe `builtins.pow(base, exp, mod)` exceeds 63 bits internally?
        (not confirmed)
        """
        ans = 1
        for i in reversed(range(exp.bit_length())):
            ans = cls.mul_mod(ans, ans)
            if exp & (1 << i):
                ans = cls.mul_mod(ans, base)
        return ans

    @staticmethod
    def mul(a: int, b: int) -> int:
        """
        Recommended for speed: max(a, b) <= mod.
        Note: ans can be larger than mod (ans < 4 * mod).
        """
        au = a >> 31
        bu = b >> 31
        ad = a & ((1 << 31) - 1)
        bd = b & ((1 << 31) - 1)
        mid = ad * bu + au * bd
        midu = mid >> 30
        midd = mid & ((1 << 30) - 1)
        ans = ((au * bu) << 1) + midu + (midd << 31) + ad * bd
        return ans

    @staticmethod
    def per_mod(x: int) -> int:
        """
        Returns x % mod.
        Recommended for speed: x.bit_length() <= 63.
        Required for validity: abs(x) <= (mod<<61) + mod.
        Works ok with x < 0.
        """
        mod = (1 << 61) - 1
        xu = x >> 61
        xd = x & mod
        ans = xu + xd
        if ans < 0:
            ans += mod
        elif ans >= mod:
            ans -= mod
        return ans


# --------------------
