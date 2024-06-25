# https://github.com/cexen/procon-cexen/blob/main/py/util_num.py
def popcount63(n: int) -> int:
    """
    Returns the number of 1 bits. Faster than popcount(n).
    Required: 0 <= n < (1 << 63).
    cf. https://nixeneko.hatenablog.com/entry/2018/03/04/000000
    >>> popcount(0b11110001)
    5
    """
    # Note: 63 bit is much faster than 64 bit on PyPy3 7.3.0
    assert 0 <= n <= 0x7FFFFFFFFFFFFFFF  # this tells PyPy to use int64_t
    c = (n & 0x5555555555555555) + ((n >> 1) & 0x5555555555555555)
    c = (c & 0x3333333333333333) + ((c >> 2) & 0x3333333333333333)
    c = (c & 0x0F0F0F0F0F0F0F0F) + ((c >> 4) & 0x0F0F0F0F0F0F0F0F)
    c = (c & 0x00FF00FF00FF00FF) + ((c >> 8) & 0x00FF00FF00FF00FF)
    c = (c & 0x0000FFFF0000FFFF) + ((c >> 16) & 0x0000FFFF0000FFFF)
    c = (c & 0x00000000FFFFFFFF) + ((c >> 32) & 0x00000000FFFFFFFF)
    return c


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


def popcount2(n: int) -> int:
    """
    Returns the number of 1 bits. Good for sparse bits.
    cf. https://nixeneko.hatenablog.com/entry/2018/03/04/000000
    >>> popcount2(0b11110001)
    5
    """
    s = 0
    while n:
        s += 1
        n -= n & (-n)
    return s


def generate_primes(n: int) -> list[int]:
    """
    O(sqrt(n)). Returns list of primes <= n.
    cf. https://strangerxxx.hateblo.jp/entry/20210514/1620925766
    cf. https://strangerxxx.hateblo.jp/entry/20230227/1677491214
    >>> generate_primes(11)
    [2, 3, 5, 7, 11]
    """
    if n <= 1:
        return []
    m = (n + 1) // 2
    isprime = [1] * m
    isprime[0] = 0
    for i in range(1, m):
        k = 2 * i + 1
        if k * k > n:
            break
        if not isprime[i]:
            continue
        for j in range(i, m):
            ni = 2 * i * j + i + j
            # assert 2 * ni + 1 == k * (2 * j + 1)
            if ni >= m:
                break
            isprime[ni] = 0
    p = [2]
    p.extend(2 * i + 1 for i in range(m) if isprime[i])
    return p


from collections import Counter
from collections.abc import Iterable


def fact_in_primes(n: int, primes: Iterable[int]) -> Counter[int]:
    """
    primes must be sorted.
    >>> fact_in_primes(60, generate_primes(60))
    Counter({2: 2, 3: 1, 5: 1})
    """
    # assert 1 not in primes
    c = Counter[int]()
    for p in primes:
        if n < p:
            break
        while n % p == 0:
            n //= p
            c[p] += 1
    assert n == 1
    return c


from math import sqrt


def fact(n: int) -> list[int]:
    """
    O(sqrt(n)). Returned prime list will be sorted.
    >>> fact(60)
    [2, 2, 3, 5]
    """
    primes = list[int]()
    for p in range(2, 2 + int(sqrt(n))):
        while n % p == 0:
            n = n // p
            primes.append(p)
    if n > 1:
        primes.append(n)
    return primes


def list_divisors(n: int) -> list[int]:
    """
    O(sqrt(n)).
    cf. https://algo-logic.info/divisor/
    >>> list_divisors(60)
    [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]
    """
    assert n > 0
    divs1 = list[int]()
    divs2 = list[int]()
    for d in range(1, n + 1):
        if d * d > n:
            break
        if not n % d:
            divs1.append(d)
            if d * d != n:
                divs2.append(n // d)
    divs1.extend(reversed(divs2))
    return divs1


def list_factors_eratosthenes(n: int) -> list[list[int]]:
    """
    O(n log log n). Returns factors of [0, n].
    Factors will be sorted.
    Note that len(factorses[0]) == len(factorses[1]) == 0.
    >>> list_factors_eratosthenes(6)
    [[], [], [2], [3], [2], [5], [2, 3]]
    """
    factorses = [list[int]() for _ in range(n + 1)]
    for i in range(2, n + 1):
        if len(factorses[i]) > 0:
            continue
        for j in range(1, 1 + n // i):
            factorses[i * j].append(i)
    return factorses


from collections import Counter


def factorize_eratosthenes(n: int) -> list[Counter[int]]:
    """
    O(n log n). Returns counters of factors of [0, n].
    Note that len(factorses[0]) == len(factorses[1]) == 0.
    >>> factorize_eratosthenes(4)
    [Counter(), Counter(), Counter({2: 1}), Counter({3: 1}), Counter({2: 2})]
    """
    factorses = [Counter[int]() for _ in range(n + 1)]
    for i in range(2, n + 1):
        if len(factorses[i]) > 0:
            continue
        for j in range(1, 1 + n // i):
            m = j
            cnt = 1
            while m % i == 0:
                m //= i
                cnt += 1
            factorses[i * j][i] = cnt
    return factorses


def list_divisors_eratosthenes(n: int) -> list[list[int]]:
    """
    O(n log n). Returns divisors of of [0, n].
    Divisors will be sorted.
    Note that divisorses[0] == [1].
    >>> list_divisors_eratosthenes(6)
    [[1], [1], [1, 2], [1, 3], [1, 2, 4], [1, 5], [1, 2, 3, 6]]
    """
    divisorses = [[1] for _ in range(n + 1)]
    for i in range(2, n + 1):
        for j in range(1, 1 + n // i):
            divisorses[i * j].append(i)
    return divisorses


from collections import Counter


def list_divisors_from_counter(factors: Counter[int]) -> list[int]:
    """
    >>> list_divisors_from_counter(Counter({2: 2, 3: 1, 5: 1}))
    [1, 5, 3, 15, 2, 10, 6, 30, 4, 20, 12, 60]
    """
    q = [1]
    for k, v in factors.items():
        nq = []
        for m in q:
            nm = m
            for _ in range(v + 1):
                nq.append(nm)
                nm *= k
        q = nq
    return q


from collections.abc import Iterable


def list_divisors_from_list(x: int, factors: Iterable[int]) -> list[int]:
    """
    >>> list_divisors_from_list(60, generate_primes(60))
    [1, 5, 3, 15, 2, 10, 6, 30, 4, 20, 12, 60]
    """
    q = [1]
    for p in factors:
        nq = []
        cnt = 0
        while x % p == 0:
            x //= p
            cnt += 1
        for m in q:
            nm = m
            for _ in range(cnt + 1):
                nq.append(nm)
                nm *= p
        q = nq
    return q


from collections import Counter
from collections.abc import Iterable


def lcmmod(*x: int, mod: int) -> int:
    """
    Returns lcm(*x) % mod.
    O(sum(sqrt(xi))).
    cf. https://blog.hamayanhamayan.com/entry/2017/05/21/001646
    """
    c = Counter[int]()
    for xi in x:
        cc = Counter(fact(xi))
        for k, v in cc.items():
            c[k] = max(c[k], v)
    ans = 1
    for k, v in c.items():
        ans = ans * pow(k, v, mod) % mod
    return ans


def floorlog2(n: int) -> int:
    if not n > 0:
        raise ValueError
    ans = 0
    while 1 << (ans + 1) <= n:
        ans += 1
    return ans


def ndigit2(n: int) -> int:
    return 1 + floorlog2(n)


def ceillog2(n: int) -> int:
    if not n > 0:
        raise ValueError
    ans = 0
    while 1 << ans < n:
        ans += 1
    return ans


def floorlog(n: int, b: int = 10) -> int:
    if not n > 0:
        raise ValueError
    ans = 0
    while b ** (ans + 1) <= n:
        ans += 1
    return ans


def ndigit(n: int, b: int = 10) -> int:
    return 1 + floorlog(n, b)


def ceillog(n: int, b: int = 10) -> int:
    if not n > 0:
        raise ValueError
    ans = 0
    while b**ans < n:
        ans += 1
    return ans


from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T")


def pow_iter(base: _T, exp: int, e: _T, f: Callable[[_T, _T], _T]):
    ans = e
    for i in reversed(range(exp.bit_length())):
        ans = f(ans, ans)
        if exp & (1 << i):
            ans = f(ans, base)
    return ans


def euclid(a: int, b: int):
    """
    Returns (p, q, gcd(a, b)) s.t. ap + bq = gcd(a, b).
    O(log min(a, b)).
    >>> euclid(3, 0)
    (1, 0, 3)
    >>> euclid(0, 3)
    (0, 1, 3)
    >>> euclid(4, 6)
    (-1, 1, 2)
    """
    ks = list[int]()
    while b != 0:
        ks.append(a // b)
        a, b = b, a % b
    p, q = 1, 0
    for k in reversed(ks):
        p, q = q, p - k * q
    return p, q, a


def inv(a: int, mod: int) -> int:
    """
    Returns ia s.t. ia * a = 1 under mod.
    Returns 0 if nonexistent (gcd(a, mod) != 1).
    O(log min(a, mod)).
    On PyPy 3.10: inv(a, mod) is faster than pow(a, -1, mod).
    On CPython 3.11: inv(a, mod) is slower than pow(a, -1, mod).
    >>> def inv2(a: int, mod: int) -> int:
    ...     try:
    ...         return pow(a, -1, mod)
    ...     except ValueError:
    ...         return 0
    >>> for a in range(-100, 100):
    ...     for mod in range(1, 20):
    ...         assert inv(a, mod) == inv2(a, mod)
    """
    ia, _, d = euclid(a, mod)  # ia * a + _ * mod = d
    return ia % mod if d == 1 else 0


from math import ceil, gcd, sqrt


def discrete_log(a: int, b: int, m: int, m_is_prime: bool) -> int:
    """
    O(m**0.5). Baby-step giant-step.
    v1.1 @cexen.
    Returns min x s.t. 0<=x<m and a**x=b mod m.
    Returns -1 if nonexistent.
    Works ok with gcd(a, m) != 1.
    Note that 0**0 == 1.
    cf. https://qiita.com/suisen_cp/items/d597c8ec576ae32ee2d7
    """
    assert m >= 1
    if m == 1:
        return 0
    a %= m
    b %= m
    if a == 0:
        if b == 1:
            return 0
        elif b == 0:
            return 1
        else:
            return -1

    # reduce m to ensure gcd(a, m) == 1
    d = m.bit_length() - 1
    ax = 1
    for x in range(d):
        if ax == b:
            return x
        ax = ax * a % m
    if ax == 0:
        return d if b == 0 else -1
    g = gcd(ax, m)
    if b % g != 0:
        return -1
    m //= g
    if m_is_prime:
        iv = pow(ax, m - 2, m)
    else:
        iv = inv(ax, m)
    assert iv != 0, "unreachable"
    b = b * iv % m

    # baby-step, giant-step
    # assert gcd(a, m) == 1, "unreachable"
    k = ceil(sqrt(m))
    s = dict[int, int]()
    ax = 1
    for x in range(k):
        s.setdefault(ax, x)
        ax = ax * a % m
    if m_is_prime:
        iak = pow(ax, m - 2, m)
    else:
        iak = inv(ax, m)
    assert iak != 0, "unreachable"
    iakx = 1
    for x in range(k + 1):
        r = s.get(iakx * b % m, -1)
        if r != -1:
            return d + x * k + r
        iakx = iakx * iak % m
    return -1


from collections.abc import Sequence


def crt_naive(rs: Sequence[int], ms: Sequence[int]) -> tuple[int | None, int | None]:
    """
    Returns (r, lcm(ms)) of x = r (mod lcm(ms)) s.t. x = ri (mod mi).
    Returns (None, None) if there is no solution.
    0 <= r < lcm(ms) or r is None.
    O(len(ms) sum(log mi)).
    cf. https://qiita.com/drken/items/ae02240cd1f8edfc86fd
    >>> crt_naive([2, 3], [3, 5])
    (8, 15)
    >>> crt_naive([0, 2], [4, 6])
    (8, 12)
    >>> crt_naive([0, 1], [4, 6])
    (None, None)
    """
    assert len(rs) == len(ms)
    r, m = 0, 1
    for ri, mi in zip(rs, ms):
        p, _, d = euclid(m, mi)  # Note that p m/d + _ mi/d = 1
        if (ri - r) % d != 0:
            return (None, None)
        k = (ri - r) // d * p  # km = (ri-r)pm/d = ri-r (mod mi/d)
        l = k % (mi // d)
        r += l * m  # r+lm = r (mod m), r+lm = r+km = ri (mod mi/d)
        m *= mi // d
    return r % m, m


from collections.abc import Sequence
from math import gcd


def crt_garner(rs: Sequence[int], ms: Sequence[int], mod: int) -> int | None:
    """
    Returns r % mod of x = r (mod lcm(ms)) s.t. x = ri (mod mi).
    Returns None if there is no solution.
    0 <= r < min(mod, lcm(ms)) or r is None.
    O(n**2 + n (sum(log mi)+log mod)) where n = len(rs).
    cf.
    https://qiita.com/drken/items/ae02240cd1f8edfc86fd
    https://math314.hateblo.jp/entry/2015/05/07/014908
    >>> crt_garner([2, 3], [3, 5], 1000000007)
    8
    >>> crt_garner([2, 3], [3, 5], 5)
    3
    >>> crt_garner([0, 2], [4, 6], 1000000009)
    8
    >>> crt_garner([0, 2], [4, 6], 4)
    0
    >>> crt_garner([0, 1], [4, 6], 10)  # returns None
    >>> crt_garner([], [], 3)  # lcm([]) == 1
    0
    """
    # reduce the problem so that ms are prime to each other
    # O(n (sum(log mi)+log mod)).
    # cf. https://qiita.com/drken/items/ae02240cd1f8edfc86fd
    ms = list(ms)
    assert len(rs) == len(ms)
    n = len(rs)
    for i in range(n - 1):
        for j in range(i + 1, n):
            g = gcd(ms[i], ms[j])
            if (rs[i] - rs[j]) % g:
                return None
            ms[i] //= g
            ms[j] //= g
            gi = gcd(g, ms[i])
            gj = g // gi
            # O(log log g).
            # sample: g=1000, gi=1, gj=0.
            while (g := gcd(gi, gj)) != 1:
                gi *= g
                gj //= g
            ms[i] *= gi
            ms[j] *= gj

    # garner
    ms.append(mod)  # m_n = mod
    assert len(ms) == n + 1
    # find: x = x0 + x1m0 + x2m0m1 + x3m0m1m2 + ... + x_{n-1}m0m1...m_{n-2}
    # sx = [0%m0, x0%m1, (x0+x1m0)%m2, ..., (x0+...+x_{n-1}m0m1...m_{n-2})%m_n]
    # mm = [1%m0, m0%m1, m0m1%m2, m0m1m2%m3, ..., m0m1m2...m_{n-1}%m_n]
    sx = [0] * (n + 1)
    mm = [1] * (n + 1)
    for i in range(n):
        xi = (rs[i] - sx[i]) * inv(mm[i], ms[i]) % ms[i]
        for j in range(i + 1, n + 1):
            sx[j] = (sx[j] + xi * mm[j]) % ms[j]
            mm[j] = mm[j] * ms[i] % ms[j]
    return sx[-1]


# --------------------


def solve_yosupojudge_counting_primes():
    """
    RE or MLE for max_*.
    Library Checker: Counting Primes
    https://judge.yosupo.jp/problem/counting_primes
    """
    N = int(input())
    primes = generate_primes(N)
    print(len(primes))


def solve_yosupojudge_enumerate_primes():
    """
    RE or MLE for 499477801_00, 499999993_00, and max_*.
    Library Checker: Enumerate Primes
    https://judge.yosupo.jp/problem/enumerate_primes
    """
    N, A, B = map(int, input().split())
    primes = generate_primes(N)
    ans = primes[B::A]
    print(len(primes), len(ans))
    print(*ans)


def solve_yosupojudge_discrete_logarithm_mod():
    """
    Library Checker: Discrete Logarithm
    https://judge.yosupo.jp/problem/discrete_logarithm_mod
    """
    T = int(input())
    anses = []
    for _ in range(T):
        x, y, m = map(int, input().split())
        ans = discrete_log(x, y, m, m_is_prime=False)
        anses.append(ans)
    for ans in anses:
        print(ans)


def solve_yukicoder_no_187():
    """
    yukicoder: No.187 中華風 (Hard)
    https://yukicoder.me/problems/no/187
    """
    MOD = 1000000007
    N = int(input())
    X = list[int]()
    Y = list[int]()
    for _ in range(N):
        x, y = map(int, input().split())
        X.append(x)
        Y.append(y)
    if all(x == 0 for x in X):
        ans = lcmmod(*Y, mod=MOD)
    else:
        ans = crt_garner(X, Y, MOD)
    print(ans if ans is not None else -1)
