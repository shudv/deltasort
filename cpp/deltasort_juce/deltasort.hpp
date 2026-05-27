// SPDX-License-Identifier: MIT
//
// DeltaSort — C++ port of the Rust reference (rust/src/lib.rs).
//
// Same algorithm: O(n + k log n) compares, O(n + k sqrt(k)) moves in expectation
// when the dirty index set is known. Header-only, templated, no allocations
// outside Phase 1's scratch buffer of length k.
//
// Public API:
//   delta_sort_by(it_begin, it_end, dirty_begin, dirty_end, cmp)
//
// The cmp argument follows the C++ Compare concept: returns true iff a < b
// (strict weak ordering). This matches std::sort / std::stable_sort.

#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <vector>
#include <utility>

namespace deltasort {

namespace detail {

template <class It, class Cmp>
inline bool needs_left(It arr, std::size_t i, const Cmp& less) {
    // arr[i] should move LEFT iff arr[i-1] > arr[i]  i.e. less(arr[i], arr[i-1]).
    if (i == 0) return false;
    return less(arr[i], arr[i - 1]);
}

// Rotate so that arr[from] ends up at arr[to], shifting intermediate elements
// by one position.
template <class It>
inline void move_element(It arr, std::size_t from, std::size_t to) {
    if (from == to) return;
    if (from < to) {
        std::rotate(arr + from, arr + from + 1, arr + to + 1);
    } else {
        std::rotate(arr + to, arr + from, arr + from + 1);
    }
}

// Binary-search for position in (i, right_bound] where arr[i] should go,
// then rotate.  Returns the new position.
template <class It, class Cmp>
inline std::size_t fix_right(It arr, std::size_t i, std::size_t right_bound, const Cmp& less) {
    std::size_t lo = i + 1, hi = right_bound + 1;
    while (lo < hi) {
        std::size_t mid = lo + ((hi - lo) >> 1);
        // arr[mid] <= arr[i]  iff  !less(arr[i], arr[mid])
        if (!less(arr[i], arr[mid])) lo = mid + 1;
        else                          hi = mid;
    }
    std::size_t target = lo - 1;
    move_element(arr, i, target);
    return target;
}

// Binary-search for position in [left_bound, i) where arr[i] should go,
// then rotate.  Returns the new position.
template <class It, class Cmp>
inline std::size_t fix_left(It arr, std::size_t i, std::size_t left_bound, const Cmp& less) {
    std::size_t lo = left_bound, hi = i;
    while (lo < hi) {
        std::size_t mid = lo + ((hi - lo) >> 1);
        if (less(arr[i], arr[mid])) hi = mid;
        else                         lo = mid + 1;
    }
    move_element(arr, i, lo);
    return lo;
}

} // namespace detail

// Sort `arr` (previously sorted, now has changes at `dirty` indices) in place,
// using `less` as the strict-weak-ordering comparator.
//
// `dirty` is sorted in place. Duplicate indices are not supported (debug check
// in caller).
template <class T, class Cmp>
void delta_sort_by(T* arr, std::size_t n, std::size_t* dirty, std::size_t k, const Cmp& less) {
    if (k == 0) return;

    std::sort(dirty, dirty + k);

    // Phase 1 — extract dirty values, sort, write back. O(k log k) compares,
    // O(k) extra memory. We use a small scratch vector so the move-only T case
    // works without requiring T to be default-constructible.
    std::vector<T> values;
    values.reserve(k);
    for (std::size_t d = 0; d < k; ++d) {
        values.emplace_back(std::move(arr[dirty[d]]));
    }
    std::sort(values.begin(), values.end(), less);
    for (std::size_t d = 0; d < k; ++d) {
        arr[dirty[d]] = std::move(values[d]);
    }

    // Phase 2 — fix ordering violations left-to-right.
    std::size_t left_bound = 0;
    bool        have_seg   = false;
    std::size_t seg_start  = 0;        // index INTO dirty[] where current RIGHT run began

    for (std::size_t d = 0; d < k; ++d) {
        std::size_t i = dirty[d];
        bool left = detail::needs_left(arr, i, less);

        if (left) {
            // Flush pending RIGHT segment in REVERSE order.
            if (have_seg) {
                std::size_t right_bound = i - 1;
                for (std::size_t rd = d; rd-- > seg_start;) {
                    std::size_t idx = dirty[rd];
                    if (idx + 1 < n && less(arr[idx + 1], arr[idx])) {
                        right_bound = detail::fix_right(arr, idx, right_bound, less) - 1;
                    }
                }
                have_seg = false;
            }
            left_bound = detail::fix_left(arr, i, left_bound, less) + 1;
        } else {
            if (!have_seg) {
                have_seg  = true;
                seg_start = d;
                left_bound = i;
            }
        }
    }

    if (have_seg) {
        std::size_t right_bound = n - 1;
        for (std::size_t rd = k; rd-- > seg_start;) {
            std::size_t idx = dirty[rd];
            if (idx + 1 < n && less(arr[idx + 1], arr[idx])) {
                right_bound = detail::fix_right(arr, idx, right_bound, less) - 1;
            }
        }
    }
}

// Convenience: vector overload.
template <class T, class Cmp>
inline void delta_sort_by(std::vector<T>& arr, std::vector<std::size_t>& dirty, const Cmp& less) {
    delta_sort_by(arr.data(), arr.size(), dirty.data(), dirty.size(), less);
}

} // namespace deltasort
