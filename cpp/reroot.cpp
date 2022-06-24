#include <cassert>
#include <iostream>
#include <vector>

using namespace std;

/**
 * @brief v1.5 cexen.
 */
struct tree
{
public:
    const size_t n;

private:
    vector<vector<pair<size_t, size_t>>> _adjs;
    size_t _root = -1;
    vector<size_t> _idxs;
    vector<pair<size_t, size_t>> _parents;

public:
    tree() : tree(0) {}
    explicit tree(const size_t n) : n(n), _adjs(n), _parents(n) {}

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
    const vector<vector<pair<size_t, size_t>>> &adjs() const noexcept
    {
        return _adjs;
    }
    void sort(size_t root = 0)
    {
        assert(0 <= root && root < n);
        if (_root == root)
            return;
        _root = root;
        _idxs.clear();
        vector<size_t> q{root};
        basic_string<bool> visited(n, false);
        visited.at(root) = true;
        while (!q.empty())
        {
            const auto i = move(q.back());
            q.pop_back();
            _idxs.push_back(i);
            for (const auto &adj : _adjs.at(i))
            {
                const auto &[j, e] = adj;
                if (visited.at(j))
                {
                    _parents.at(i) = adj;
                    continue;
                }
                visited.at(j) = true;
                q.push_back(j);
            }
        }
    }
    template <
        class E, class V,
        E (*add_e)(const V &, const size_t, const size_t),
        E (*merge)(const E &, const E &),
        V (*add_v)(const E &, const size_t, const size_t),
        E (*ee)()>
    vector<V> collect_for_root(const size_t root = 0)
    {
        return collect_for_root<E, V>(
            [](const V &v, const size_t i, const size_t e) constexpr { return add_e(v, i, e); },
            [](const E &l, const E &r) constexpr { return merge(l, r); },
            [](const E &e, const size_t i, const size_t parent) constexpr { return add_v(e, i, parent); },
            []() constexpr { return ee(); },
            root);
    }
    template <
        class E, class V,
        class FEV,
        class FEEE,
        class FVE,
        class FE>
    pair<vector<V>, vector<vector<E>>> collect_for_root(const FEV add_e, const FEEE merge, const FVE add_v, const FE ee, const size_t root = 0)
    {
        sort(root);
        vector<V> ans0(n);
        vector<vector<E>> edgeses(n);
        for (auto i = _idxs.rbegin(); i != _idxs.rend(); i++)
        {
            const auto &parent = _parents.at(*i);
            vector<E> &edges = edgeses.at(*i);
            auto accum_l = ee();
            for (const auto &adj : _adjs.at(*i))
            {
                const auto &[j, e] = adj;
                if (adj == parent)
                {
                    edges.emplace_back(ee());
                }
                else
                {
                    const auto edge = add_e(ans0.at(j), j, e);
                    edges.emplace_back(edge);
                    accum_l = merge(accum_l, edge);
                }
            }
            ans0.at(*i) = add_v(accum_l, *i, parent.first);
        }
        return {ans0, edgeses};
    }
    template <
        class E, class V,
        E (*add_e)(const V &, const size_t, const size_t),
        E (*merge)(const E &, const E &),
        V (*add_v)(const E &, const size_t, const size_t),
        E (*ee)()>
    vector<V> collect_for_all(const size_t root = 0)
    {
        return collect_for_all<E, V>(
            [](const V &v, const size_t i, const size_t e) constexpr { return add_e(v, i, e); },
            [](const E &l, const E &r) constexpr { return merge(l, r); },
            [](const E &e, const size_t i, const size_t parent) constexpr { return add_v(e, i, parent); },
            []() constexpr { return ee(); },
            root);
    }
    template <
        class E, class V,
        class FEV,
        class FEEE,
        class FVE,
        class FE>
    vector<V> collect_for_all(const FEV add_e, const FEEE merge, const FVE add_v, const FE ee, const size_t root = 0)
    {
        vector<vector<E>> edgeses = collect_for_root<E, V>(add_e, merge, add_v, ee, root).second;
        vector<V> ans(n);
        vector<pair<size_t, E>> q{{root, ee()}};
        while (!q.empty())
        {
            const auto [i, edge_parent] = move(q.back());
            q.pop_back();
            const auto &adjsi = _adjs.at(i);
            const auto &parent = _parents.at(i);
            vector<E> &edges = edgeses.at(i);
            vector<E> accums_l{ee()};
            assert(adjsi.size() == edges.size());
            for (size_t k = 0; k < edges.size(); ++k)
            {
                if (adjsi.at(k) == parent)
                {
                    edges.at(k) = edge_parent;
                }
                const E &edge = edges.at(k);
                accums_l.push_back(merge(accums_l.back(), edge));
            }
            accums_l.pop_back(); // not used
            E accum_r = ee();
            assert(adjsi.size() == accums_l.size());
            assert(accums_l.size() == edges.size());
            for (size_t k = edges.size(); k-- > 0;)
            {
                const auto &adj = adjsi.at(k);
                const auto &accum_l = accums_l.at(k);
                const auto &edge = edges.at(k);
                if (adj != parent)
                {
                    const auto &[j, e] = adj;
                    // myself as a child of j
                    const auto myself = add_v(merge(accum_l, accum_r), i, j);
                    q.emplace_back(j, add_e(myself, i, e));
                }
                accum_r = merge(edge, accum_r);
            }
            ans.at(i) = add_v(accum_r, i, i);
        }
        return ans;
    }
};

// using E = int32_t;
// using V = int32_t;
// auto add_e = [](const V &v, [[maybe_unused]] const size_t i, [[maybe_unused]] const size_t e) -> E
// { throw logic_error("not_implemented"); };
// auto merge = [](const E &l, const E &r) -> E
// { throw logic_error("not_implemented"); };
// auto add_v = [](const E &e, [[maybe_unused]] const size_t i, [[maybe_unused]] const size_t parent) -> V
// { throw logic_error("not_implemented"); };
// constexpr auto e = []() -> E
// { throw logic_error("not_implemented"); };

// --------------------

/**
 * @brief https://atcoder.jp/contests/dp/tasks/dp_v
 */
void solve_dp_v()
{
    size_t N, M;
    cin >> N >> M;
    tree tree(N);
    for (size_t i = 0; i < N - 1; i++)
    {
        size_t x, y;
        cin >> x >> y;
        tree.connect(x - 1, y - 1);
    }
    using E = int64_t;
    using V = int64_t;
    const auto add_e = [&M](const V &v, [[maybe_unused]] const size_t i, [[maybe_unused]] const size_t e) -> E
    {
        return (v + 1) % M;
    };
    const auto merge = [&M](const E &l, const E &r) -> E
    {
        return l * r % M;
    };
    constexpr auto add_v = [](const E &e, [[maybe_unused]] const size_t i, [[maybe_unused]] const size_t parent) -> V
    {
        return e;
    };
    constexpr auto ee = []() -> E
    {
        return 1;
    };
    const auto ans = tree.collect_for_all<E, V>(add_e, merge, add_v, ee);
    for (size_t i = 0; i < N; i++)
        cout << ans.at(i) << endl;
}

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
    using E = pair<int32_t, int64_t>;
    using V = pair<int32_t, int64_t>;
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

    const auto ans = tree.collect_for_all<E, V, add_e, merge, add_v, ee>();
    for (size_t i = 0; i < N; i++)
        cout << ans.at(i).second << endl;
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

    const auto ans = tree.collect_for_all<E, V>(add_e, merge, add_v, e);
    for (size_t i = 0; i < N; i++)
        cout << ans.at(i) << endl;
}

int main()
{
    // solve_dp_v();
    // solve_abc220_f();
    // solve_abc222_f();
}
