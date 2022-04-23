#include <algorithm>
#include <array>
#include <chrono>
#include <deque>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/hana/functional/fix.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm_ext.hpp>
#include <atcoder/all>

using namespace std;
// using namespace boost::adaptors;
// using namespace boost::algorithm;
// using namespace boost::range;
using namespace atcoder;
using boost::make_iterator_range;
using boost::adaptors::indexed;
using boost::adaptors::map_keys;
using boost::adaptors::reversed;
using boost::hana::fix;
using boost::range::count;
using boost::range::lower_bound;
using boost::range::max_element;
using boost::range::min_element;
using boost::range::reverse;
using boost::range::sort;
using boost::range::upper_bound;

using u32 = uint32_t;
using u64 = uint64_t;
using i32 = int32_t;
using i64 = int64_t;
using f32 = float;
using f64 = double;
using usize = size_t;
[[maybe_unused]] constexpr i32 inf = INT_MAX / 2;
[[maybe_unused]] constexpr i32 inf32 = INT_MAX / 2;
[[maybe_unused]] constexpr i64 inf64 = 1LL << 60;
#define WHOLE(x) (x).begin(), (x).end()
#define rep(i, n) for (int i = 0; i < (int)(n); i++)
#define rep3(i, a, b) for (int i = (int)(a); i < (int)(b); i++)
// #define _rep2(i, n) for (int i = 0; i < (int)(n); i++)
// #define _rep3(i, a, b) for (int i = (int)(a); i < (int)(b); i++)
// #define _overload3(_1, _2, _3, name, ...) name
// #define rep(...) _overload3(__VA_ARGS__, _rep3, _rep2)(__VA_ARGS__)
#define rrep(i, n) for (int i = (int)(n)-1; i >= 0; i--)
template <class T>
bool chmax(T &a, const T &b)
{
    return (a < b) ? (a = b, true) : false;
}
template <class T>
bool chmin(T &a, const T &b)
{
    return (a > b) ? (a = b, true) : false;
}
template <class T>
void dedup(vector<T> &v)
{
    v.erase(unique(v.begin(), v.end()), v.end());
}
constexpr i64 powi(const i64 base, const u64 exp) noexcept
{
    i64 ans = 1, x = base, y = exp;
    while (y > 0)
        (y & 1) && (ans *= x), (x *= x), (y >>= 1);
    return ans;
}
constexpr u64 powi(const i64 base, const u64 exp, const u32 mod) noexcept
{
    u64 ans = 1, x = (base % mod + mod) % mod, y = exp;
    while (y > 0)
        (y & 1) && (ans = ans * x % mod), (x = x * x % mod), (y >>= 1);
    return ans;
}
constexpr u32 bit_length(i64 x) noexcept
{
    x = (x < 0) ? -x : x;
    u32 ans = 0;
    for (; x != 0; ++ans)
        x >>= 1;
    return ans;
}
template <class K, class V, class F>
constexpr boost::iterator_range<typename vector<K>::iterator> equal_range(vector<K> &vec, V &value, F &&key)
{
    return make_iterator_range(
        lower_bound(vec, value, [&key](const K &vec, const V &value)
                    { return key(vec) < value; }),
        upper_bound(vec, value, [&key](const V &value, const K &vec)
                    { return value < key(vec); }));
}

// --------------------

int main()
{
    // ios_base::sync_with_stdio(false);
    // cin.tie(nullptr);
}
