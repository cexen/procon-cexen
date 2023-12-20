# https://github.com/cexen/procon-cexen/blob/main/py/mod.py
class Mod:
    """
    v1.3 @cexen
    Expected to be faster than Mint.
    >>> mod = Mod(998244353)
    >>> mod.fact(10)  # factorial (auto-cached)
    3628800
    >>> mod.ifact(2)  # inverse of factorial (auto-cached)
    499122177

    You may make cache manually (O(nmax)):
    >>> mod = Mod(998244353, nmax=10000)

    >>> mod.comb(4, 2)  # combination; [x**2](1+x)**4
    6
    >>> mod.comb(0, 0)
    1
    >>> mod.comb(-3, 2)  # [x**2](1+x)**(-3)
    6
    >>> mod.comb(-3, 3) - mod.val  # [x**3](1+x)**(-3)
    -10
    >>> mod.div(12, 3)  # mod must be a prime here
    4
    """

    KNOWN_PRIMES = {998244353, 1000000007, 1000000009}

    def __init__(
        self, mod: int = 998244353, nmax: int = 1, is_prime: bool | None = None
    ):
        """O(nmax)."""
        self.mod = mod
        self.facts = [1, 1]
        self.ifacts = [1, 1]
        self.is_prime = is_prime
        if is_prime is None and mod in self.KNOWN_PRIMES:
            self.is_prime = True
        self._cache_ifacts(nmax)

    @property
    def val(self) -> int:
        return self.mod

    def _cache_facts(self, n: int) -> None:
        """O(Δn)."""
        if n < len(self.facts):
            return
        mod = self.mod
        facts = self.facts
        i0 = len(facts)
        facts.extend([1] * (n + 1 - i0))
        for i in range(i0, n + 1):
            facts[i] = facts[i - 1] * i % mod

    def _cache_ifacts(self, n: int) -> None:
        """O(Δn + log n)."""
        if n < len(self.ifacts):
            return
        self._cache_facts(n)
        mod = self.mod
        ifacts = self.ifacts
        i0 = len(ifacts)
        ifacts.extend([1] * (n + 1 - i0))
        ifacts[n] = self.inv(self.facts[n])
        for i in reversed(range(i0, n)):
            ifacts[i] = ifacts[i + 1] * (i + 1) % mod

    def fact(self, v: int) -> int:
        self._cache_facts(v)
        return self.facts[v]

    def ifact(self, v: int) -> int:
        self._cache_ifacts(v)
        return self.ifacts[v]

    def perm(self, n: int, r: int) -> int:
        if n < r or r < 0:
            return 0
        return self.fact(n) * self.ifact(n - r) % self.mod

    def comb(self, n: int, r: int) -> int:
        if 0 <= n < r or r < 0:
            return 0
        if n < 0:
            return (-1) ** (r & 1) * self.homo(-n, r) % self.mod
        return self.perm(n, r) * self.ifact(r) % self.mod

    def homo(self, n: int, r: int) -> int:
        return self.comb(n + r - 1, r)

    def inv(self, v: int) -> int:
        """O(log v). mod must be a prime!"""
        assert self.is_prime
        return pow(v, -1, self.mod)

    def add(self, a: int, b: int) -> int:
        return (a + b) % self.mod

    def sub(self, a: int, b: int) -> int:
        return (a - b) % self.mod

    def mul(self, a: int, b: int) -> int:
        return (a * b) % self.mod

    def pow(self, a: int, b: int) -> int:
        return pow(a, b, self.mod)

    def div(self, a: int, b: int) -> int:
        return a * self.inv(b) % self.mod
