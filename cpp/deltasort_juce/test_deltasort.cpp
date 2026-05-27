// SPDX-License-Identifier: MIT
// Standalone correctness + fuzz test for the C++ DeltaSort port.
// Compile: clang++ -O2 -std=c++17 test_deltasort.cpp -o test_ds && ./test_ds

#include "deltasort.hpp"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

static int g_failures = 0;
static int g_tests    = 0;

static void check(bool cond, const char* msg) {
    g_tests++;
    if (!cond) {
        g_failures++;
        std::cerr << "FAIL: " << msg << "\n";
    }
}

template <class T>
static bool is_sorted_vec(const std::vector<T>& v) {
    return std::is_sorted(v.begin(), v.end());
}

static void test_paper_example() {
    std::vector<int> arr = {1, 3, 5, 7, 9};
    arr[1] = 8;
    arr[3] = 2;
    std::vector<std::size_t> dirty = {1, 3};
    deltasort::delta_sort_by(arr, dirty, std::less<int>{});
    check(arr == std::vector<int>({1, 2, 5, 8, 9}), "paper example");
}

static void test_movement_cancellation() {
    std::vector<int> arr = {1, 8, 5, 2, 9};
    std::vector<std::size_t> dirty = {1, 3};
    deltasort::delta_sort_by(arr, dirty, std::less<int>{});
    check(arr == std::vector<int>({1, 2, 5, 8, 9}), "movement cancellation");
}

static void test_all_dirty_reverse() {
    std::vector<int> arr = {5, 4, 3, 2, 1};
    std::vector<std::size_t> dirty = {0, 1, 2, 3, 4};
    deltasort::delta_sort_by(arr, dirty, std::less<int>{});
    check(arr == std::vector<int>({1, 2, 3, 4, 5}), "fully reverse-dirty");
}

static void test_empty_dirty() {
    std::vector<int> arr = {1, 2, 3, 4, 5};
    std::vector<std::size_t> dirty;
    deltasort::delta_sort_by(arr, dirty, std::less<int>{});
    check(arr == std::vector<int>({1, 2, 3, 4, 5}), "empty dirty");
}

static void test_move_only() {
    // unique_ptr<int> is move-only.
    std::vector<std::unique_ptr<int>> arr;
    for (int v : {1, 3, 5, 7, 9}) arr.emplace_back(std::make_unique<int>(v));
    *arr[1] = 8;
    *arr[3] = 2;
    std::vector<std::size_t> dirty = {1, 3};
    deltasort::delta_sort_by(arr, dirty,
        [](const std::unique_ptr<int>& a, const std::unique_ptr<int>& b) { return *a < *b; });
    std::vector<int> got;
    for (auto& p : arr) got.push_back(*p);
    check(got == std::vector<int>({1, 2, 5, 8, 9}), "move-only T");
}

static void fuzz(std::mt19937_64& rng, int trial) {
    int n = std::uniform_int_distribution<int>(2, 200)(rng);
    int k = std::uniform_int_distribution<int>(1, n)(rng);
    std::vector<int> arr(n);
    for (auto& x : arr) x = std::uniform_int_distribution<int>(-1000, 1000)(rng);
    std::sort(arr.begin(), arr.end());

    // Choose k unique dirty indices and mutate
    std::vector<std::size_t> all(n);
    for (int i = 0; i < n; ++i) all[i] = i;
    std::shuffle(all.begin(), all.end(), rng);
    std::vector<std::size_t> dirty(all.begin(), all.begin() + k);
    for (auto i : dirty) arr[i] = std::uniform_int_distribution<int>(-1500, 1500)(rng);

    std::vector<int> ref = arr;
    std::sort(ref.begin(), ref.end());

    auto dirty2 = dirty;  // delta_sort mutates dirty
    deltasort::delta_sort_by(arr, dirty2, std::less<int>{});

    if (arr != ref) {
        std::cerr << "FUZZ FAIL trial=" << trial << " n=" << n << " k=" << k << "\n";
        g_failures++;
    }
    g_tests++;
}

int main() {
    test_paper_example();
    test_movement_cancellation();
    test_all_dirty_reverse();
    test_empty_dirty();
    test_move_only();

    std::mt19937_64 rng(0x5EED12345ULL);
    for (int t = 0; t < 5000; ++t) fuzz(rng, t);

    std::printf("%d/%d tests passed (%d failures)\n", g_tests - g_failures, g_tests, g_failures);
    return g_failures == 0 ? 0 : 1;
}
