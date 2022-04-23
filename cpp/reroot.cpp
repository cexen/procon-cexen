#include <cassert>
#include <iostream>
#include <vector>
#include <boost/hana/functional/fix.hpp>

using namespace std;
using boost::hana::fix;

/**
 * @brief v1.2 @cexen
 */
struct tree
{
public:
    const size_t n;

private:
    vector<vector<tuple<size_t, size_t>>> _adjs;
    size_t _root = -1;
    vector<size_t> _idxs;
    vector<tuple<size_t, size_t>> _parents;
    vector<vector<tuple<size_t, size_t>>> _childrens;

public:
    tree() : tree(0) {}
    explicit tree(size_t n) : n(n), _adjs(n), _parents(n), _childrens(n) {}

    /**
     * Specify e (edge index) if you need it when add_e.
     */
    void connect(const size_t u, const size_t v, const size_t e = -1)
    {
        assert(0 <= u && u < n);
        assert(0 <= v && v < n);
        _adjs.at(u).emplace_back(v, e);
        _adjs.at(v).emplace_back(u, e);
    }
    const vector<vector<tuple<size_t, size_t>>> &adjs() const noexcept
    {
        return _adjs;
    }
    tuple<
        const vector<size_t> &,
        const vector<tuple<size_t, size_t>> &,
        const vector<vector<tuple<size_t, size_t>>> &>
    sort(size_t root = 0)
    {
        assert(0 <= root && root < n);
        if (_root != root)
        {
            _root = root;
            const auto visit = fix(
                [this](const auto &&self, const size_t i, const size_t parent, const size_t parent_e) -> void
                {
                    _parents.at(i) = {parent, parent_e};
                    vector<tuple<size_t, size_t>> children;
                    for (const auto &[j, e] : _adjs.at(i))
                    {
                        if (j == parent)
                            continue;
                        children.emplace_back(j, e);
                        self(j, i, e);
                    }
                    _childrens.at(i) = children;
                    _idxs.push_back(i);
                });
            visit(_root, _root, -1);
            reverse(_idxs.begin(), _idxs.end());
        }
        return {_idxs, _parents, _childrens};
    }
};

template <
    class E, class V,
    E (*add_e)(const V &, const size_t, const size_t),
    E (*merge)(const E &, const E &),
    V (*add_v)(const E &, const size_t, const size_t),
    E (*ee)()>
vector<V> collect_for_root(tree tree, const size_t root = 0)
{
    return collect_for_root<E, V>(
        tree,
        [](const V &v, const size_t i, const size_t e)
        { return add_e(v, i, e); },
        [](const E &l, const E &r)
        { return merge(l, r); },
        [](const E &e, const size_t i, const size_t parent)
        { return add_v(e, i, parent); },
        []()
        { return ee(); },
        root);
};
template <
    class E, class V,
    class FEV,
    class FEEE,
    class FVE,
    class FE>
vector<V> collect_for_root(tree tree, FEV add_e, FEEE merge, FVE add_v, FE ee, const size_t root = 0)
{
    auto [idxs, parents, childrens] = tree.sort(root);
    vector<V> ans0(tree.n);
    for (auto i = idxs.rbegin(); i != idxs.rend(); i++)
    {
        auto accum_l = ee();
        for (const auto &[j, e] : childrens.at(*i))
        {
            accum_l = merge(accum_l, add_e(ans0.at(j), j, e));
        }
        ans0.at(*i) = add_v(accum_l, *i, get<0>(parents.at(*i)));
    }
    return ans0;
};

template <
    class E, class V,
    E (*add_e)(const V &, const size_t, const size_t),
    E (*merge)(const E &, const E &),
    V (*add_v)(const E &, const size_t, const size_t),
    E (*ee)()>
vector<V> collect_for_all(tree tree, const size_t root = 0)
{
    return collect_for_all<E, V>(
        tree,
        [](const V &v, const size_t i, const size_t e)
        { return add_e(v, i, e); },
        [](const E &l, const E &r)
        { return merge(l, r); },
        [](const E &e, const size_t i, const size_t parent)
        { return add_v(e, i, parent); },
        []()
        { return ee(); },
        root);
}
template <
    class E, class V,
    class FEV,
    class FEEE,
    class FVE,
    class FE>
vector<V> collect_for_all(tree tree, FEV add_e, FEEE merge, FVE add_v, FE ee, const size_t root = 0)
{
    vector<V> ans0 = collect_for_root<E, V>(tree, add_e, merge, add_v, ee, root);
    vector<V> ans(tree.n);
    const auto visit = fix(
        [&tree, &ans0, &ans, &add_e, &merge, &add_v, &ee](const auto &&self, const size_t i, const V &ans_parent) -> void
        {
            auto [idxs, parents, childrens] = tree.sort();
            const auto &adj = tree.adjs().at(i);
            const auto &parent = parents.at(i);
            vector<E> edges(adj.size());
            for (size_t k = 0; k < adj.size(); k++)
            {
                const auto &adji = adj.at(k);
                const auto &[j, e] = adji;
                edges.at(k) = add_e(adji == parent ? ans_parent : ans0.at(j), j, e);
            }
            vector<E> accums_l(1 + edges.size(), ee());
            for (size_t k = 0; k < edges.size(); k++)
            {
                accums_l.at(k + 1) = merge(accums_l.at(k), edges.at(k));
            }
            auto accum_r = ee();
            for (size_t k = adj.size(); k-- > 0;)
            {
                const auto &adji = adj.at(k);
                const auto &[j, e] = adji;
                if (adji != parent)
                {
                    self(j, add_v(merge(accums_l.at(k), accum_r), i, j));
                }
                accum_r = merge(edges.at(k), accum_r);
            }
            ans.at(i) = add_v(accum_r, i, i);
        });
    visit(root, V());
    return ans;
};

// --------------------

/**
 * @brief https://atcoder.jp/contests/abc220/tasks/abc220_f
 */
void solve_abc220_f()
{
    size_t N;
    cin >> N;
    tree tree(N);
    for (size_t i = 0; i < N - 1; i++)
    {
        size_t u, v;
        cin >> u >> v;
        tree.connect(u - 1, v - 1);
    }
    using E = tuple<int32_t, int64_t>;
    using V = tuple<int32_t, int64_t>;
    constexpr auto add_e = [](const V &v, [[maybe_unused]] const size_t i, [[maybe_unused]] const size_t e) -> E
    {
        const auto &[s, d] = v;
        return {s, d + s};
    };
    constexpr auto merge = [](const E &l, const E &r) -> E
    {
        const auto &[ls, ld] = l;
        const auto &[rs, rd] = r;
        return {ls + rs, ld + rd};
    };
    constexpr auto add_v = [](const E &e, [[maybe_unused]] const size_t i, [[maybe_unused]] const size_t parent) -> V
    {
        const auto &[s, d] = e;
        return {s + 1, d};
    };
    constexpr auto ee = []() -> E
    { return {0, 0}; };

    const auto ans = collect_for_all<E, V, add_e, merge, add_v, ee>(tree);
    for (size_t i = 0; i < N; i++)
        cout << get<1>(ans.at(i)) << endl;
}

/**
 * @brief https://atcoder.jp/contests/abc222/tasks/abc222_f
 */
void solve_abc222_f()
{
    size_t N;
    cin >> N;
    tree tree(N);
    vector<int64_t> C(N - 1);
    for (size_t i = 0; i < N - 1; i++)
    {
        size_t a, b;
        int64_t c;
        cin >> a >> b >> c;
        tree.connect(a - 1, b - 1, i);
        C.at(i) = c;
    }
    vector<int64_t> D(N);
    for (size_t i = 0; i < N; i++)
    {
        int64_t d;
        cin >> d;
        D.at(i) = d;
    }
    using E = int64_t;
    using V = int64_t;
    auto add_e = [&C](const V &v, [[maybe_unused]] const size_t i, [[maybe_unused]] const size_t e) -> E
    {
        return v + C.at(e);
    };
    auto merge = [](const E &l, const E &r) -> E
    {
        return max(l, r);
    };
    auto add_v = [&D](const E &e, [[maybe_unused]] const size_t i, [[maybe_unused]] const size_t parent) -> V
    {
        return i == parent ? e : max(e, D.at(i));
    };
    constexpr auto e = []() -> E
    { return 0; };

    const auto ans = collect_for_all<E, V>(tree, add_e, merge, add_v, e);
    for (size_t i = 0; i < N; i++)
        cout << ans.at(i) << endl;
}

int main()
{
    // solve_abc220_f();
    solve_abc222_f();
}