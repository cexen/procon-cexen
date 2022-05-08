// https://github.com/cexen/procon-cexen/blob/main/cpp/mod.cpp
#include <cassert>
#include <vector>
#include <atcoder/modint>

using namespace std;

/**
 * @brief v1.2 cexen.
 * T must be an atcoder::modint.
 * Advanced mod operations with caching of factorials.
 */
template <class T>
struct mod
{
private:
    vector<T> _facts;
    vector<T> _ifacts;

public:
    explicit mod<T>(const size_t n = 2) : _facts(2, 1), _ifacts(2, 1)
    {
        assert(atcoder::internal::is_modint<T>());
        _cache_ifacts(n);
    }
    int32_t val() {
        return T::mod();
    }
    /**
     * O(Δn).
     */
    void _cache_facts(const size_t n)
    {
        if (n < _facts.size())
            return;
        size_t i0 = _facts.size();
        _facts.resize(n + 1);
        for (size_t i = i0; i < n + 1; i++)
        {
            _facts.at(i) = _facts.at(i - 1) * i;
        }
    }
    /**
     * O(Δn + log n).
     */
    void _cache_ifacts(const size_t n)
    {
        if (n < _ifacts.size())
            return;
        _cache_facts(n);
        size_t i0 = _ifacts.size();
        _ifacts.resize(n + 1);
        _ifacts.at(n) = _facts.at(n).inv();
        for (size_t i = n - 1; i >= i0; i--)
        {
            _ifacts.at(i) = _ifacts.at(i + 1) * (i + 1);
        }
    }
    T fact(const size_t v)
    {
        _cache_facts(v);
        return _facts.at(v);
    }
    T ifact(const size_t v)
    {
        _cache_ifacts(v);
        return _ifacts.at(v);
    }
    T perm(const int32_t n, const int32_t r)
    {
        if (n < r || r < 0)
            return 0;
        return fact(n) * ifact(n - r);
    }
    T comb(const int32_t n, const int32_t r)
    {
        if ((0 <= n && n < r) || r < 0)
            return 0;
        if (n < 0)
            return ((r & 1) ? -1 : 1) * homo(-n, r);
        return perm(n, r) * ifact(r);
    }
    T homo(const int32_t n, const int32_t r)
    {
        return comb(n + r - 1, r);
    }
};

// --------------------

void test()
{
    using namespace atcoder;
    using mint = modint998244353;
    mod<mint> mod;
    // factorial (auto-cached)
    assert(3628800 == mod.fact(10).val());
    // inverse of factorial (auto-cached)
    assert(499122177 == mod.ifact(2).val());
    // You may make cache manually (O(n))
    mod._cache_ifacts(10000);
    // combination; [x**2](1+x)**4
    assert(6 == mod.comb(4, 2));
    // [1](1+x)**0
    assert(1 == mod.comb(0, 0));
    // [x**2](1+x)**(-3)
    assert(6 == mod.comb(-3, 2));
    // [x**3](1+x)**(-3)
    assert(-10 == mod.comb(-3, 3) - mod.val());
}

int main()
{
    test();
}