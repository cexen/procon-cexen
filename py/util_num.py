# https://github.com/cexen/procon-cexen/blob/main/py/util_num.py
def popcount63(n: int) -> int:
    """
    Returns the number of 1 bits. Faster than popcount().
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


def generate_primes(n: int):
    """
    Returns list of primes <= n.
    >>> generate_primes(11)
    [2, 3, 5, 7, 11]
    """
    from math import sqrt

    isprime = [True] * n
    isprime[0] = False
    for i in range(2, 1 + round(sqrt(n))):
        if not isprime[i - 1]:
            continue
        for j in range(2, 1 + n // i):
            isprime[i * j - 1] = False
    p = [i + 1 for i, v in enumerate(isprime) if v]
    return p


def fact_in_primes(n: int, primes: list):
    """
    primes must be sorted.
    >>> fact_in_primes(60, generate_primes(60))
    Counter({2: 2, 3: 1, 5: 1})
    """
    # assert 1 not in primes
    from typing import Counter

    c = Counter[int]()
    for p in primes:
        if n < p:
            break
        while n % p == 0:
            n //= p
            c[p] += 1
    assert n == 1
    return c


def fact(n: int):
    """
    Returned prime list will be sorted.
    >>> fact(60)
    [2, 2, 3, 5]
    """
    from math import sqrt
    from typing import List

    primes: List[int] = []
    for p in range(2, 2 + int(sqrt(n))):
        while n % p == 0:
            n = n // p
            primes.append(p)
    if n > 1:
        primes.append(n)
    return primes


def list_factors_eratosthenes(n: int):
    """
    O(n log log n). Returns factors of [0, n].
    Factors will be sorted.
    Note that len(factorses[0]) == len(factorses[1]) == 0.
    >>> list_factors_eratosthenes(6)
    [[], [], [2], [3], [2], [5], [2, 3]]
    """
    from typing import List

    factorses: List[List[int]] = [[] for _ in range(n + 1)]
    for i in range(2, n + 1):
        if len(factorses[i]) > 0:
            continue
        for j in range(1, 1 + n // i):
            factorses[i * j].append(i)
    return factorses


def factorize_eratosthenes(n: int):
    """
    O(n log n). Returns counters of factors of [0, n].
    Note that len(factorses[0]) == len(factorses[1]) == 0.
    >>> factorize_eratosthenes(4)
    [Counter(), Counter(), Counter({2: 1}), Counter({3: 1}), Counter({2: 2})]
    """
    from typing import Counter

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


def list_divisors_eratosthenes(n: int):
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


from typing import Counter


def list_divisors_from_counter(factors: Counter[int]):
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


from typing import List


def list_divisors_from_list(x: int, factors: List[int]):
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


def pow_iter(base, exp: int):
    ans = 1
    for i in reversed(range(exp.bit_length())):
        ans = ans * ans
        if exp & (1 << i):
            ans *= base
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
    from typing import List

    ks: List[int] = []
    while b != 0:
        ks.append(a // b)
        a, b = b, a % b
    p, q = 1, 0
    for k in reversed(ks):
        p, q = q, p - k * q
    return p, q, a


def inv(a: int, mod: int):
    ia, _, d = euclid(a, mod)  # ia * a + _ * mod = d
    return ia % mod if d == 1 else None


from typing import List, Tuple, Optional


def crt_naive(rs: List[int], ms: List[int]) -> Tuple[int, Optional[int]]:
    """
    Returns (r, lcm(mi)) of x = r (mod lcm(mi)) s.t. x = ri (mod mi).
    May return (0, None).
    O(len(ms) sum(log mi)).
    cf. https://qiita.com/drken/items/ae02240cd1f8edfc86fd
    >>> crt_naive([2, 3], [3, 5])
    (8, 15)
    >>> crt_naive([0, 2], [4, 6])
    (8, 12)
    >>> crt_naive([0, 1], [4, 6])
    (0, None)
    """
    assert len(rs) == len(ms)
    r, m = 0, 1
    for ri, mi in zip(rs, ms):
        p, _, d = euclid(m, mi)  # Note that p m/d + _ mi/d = 1
        if (ri - r) % d != 0:
            return (0, None)
        k = (ri - r) // d * p  # km = (ri-r)pm/d = ri-r (mod mi/d)
        l = k % (mi // d)
        r += l * m  # r+lm = r (mod m), r+lm = r+km = ri (mod mi/d)
        m *= mi // d
    return r % m, m


def crt_garner(rs: List[int], ms: List[int]) -> Tuple[int, Optional[int]]:
    raise NotImplementedError
