// https://github.com/cexen/procon-cexen/blob/main/cpp/numutils.cpp
#include <cassert>
#include <map>
#include <iostream>
#include <vector>
#include <boost/hana/functional/fix.hpp>

using namespace std;
using boost::hana::fix;

/**
 * O(n log log n). Returns factors of [1, n].
 * Factors will be sorted.
 * Note that factorses[0].size() == factorses[1].size() == 0.
 */
vector<vector<int32_t>> list_factors_eratosthenes(int32_t n)
{
    vector<vector<int32_t>> factorses(n + 1);
    for (int32_t i = 2; i < n + 1; i++)
    {
        if (factorses.at(i).size() > 0)
            continue;
        for (int32_t j = 1; j < 1 + n / i; j++)
        {
            factorses.at(i * j).push_back(i);
        }
    }
    return factorses;
}

/**
 * O(n log^2 n). Returns counters of factors of [1, n].
 * Note that factorses[0].size() == factorses[1].size() == 0.
 */
vector<map<int32_t, int32_t>> factorize_eratosthenes(int32_t n)
{
    vector<map<int32_t, int32_t>> factorses(n + 1);
    for (int32_t i = 2; i < n + 1; i++)
    {
        if (factorses.at(i).size() > 0)
            continue;
        for (int32_t j = 1; j < 1 + n / i; j++)
        {
            auto m = j;
            int32_t cnt = 1;
            while (m % i == 0)
            {
                m /= i;
                cnt++;
            }
            factorses.at(i * j).insert_or_assign(i, cnt);
        }
    }
    return factorses;
}

/**
 * O(n log n). Returns divisors of of [1, n].
 * Divisors will be sorted.
 * Note that divisorses[0] == [1].
 */
vector<vector<int32_t>> list_divisors_eratosthenes(int32_t n)
{
    vector<vector<int32_t>> divisorses(n + 1, vector<int32_t>(1, 1));
    for (int32_t i = 2; i < n + 1; i++)
    {
        for (int32_t j = 1; j < 1 + n / i; j++)
        {
            divisorses.at(i * j).push_back(i);
        }
    }
    return divisorses;
}

vector<int64_t> list_divisors(const map<int32_t, int32_t> &factors)
{
    vector<int64_t> q{1};
    for (const auto &[p, num] : factors)
    {
        vector<int64_t> nq;
        for (const auto &m : q)
        {
            int64_t nm = m;
            for (int32_t i = 0; i <= num; i++)
            {
                nq.push_back(nm);
                nm *= p;
            }
        }
        q = move(nq);
    }
    return q;
}

vector<int64_t> list_divisors(int32_t x, const vector<int32_t> &factors)
{
    vector<int64_t> q(1, 1);
    for (const auto &p : factors)
    {
        vector<int64_t> nq;
        int32_t cnt = 0;
        while (x % p == 0)
        {
            x /= p;
            cnt++;
        }
        for (const auto &m : q)
        {
            int64_t nm = m;
            for (int32_t i = 0; i <= cnt; i++)
            {
                nq.push_back(nm);
                nm *= p;
            }
        }
        q = move(nq);
    }
    return q;
}

// --------------------

int main()
{
}