/// Mini-benchmark comparing O(k)-space ESM variants.
///
/// Run with: cargo run --bin benchmark --release -- --esm-compare
use crate::data::User;

// =============================================================================
// Variant 1: Backwards Merge — O(k) space
// =============================================================================
// 1. Extract dirty to buffer, compact clean to left: O(n) pass
// 2. Sort dirty buffer: O(k log k)
// 3. Merge backwards from right end: O(n) pass
//
// Clean elements sit in arr[0..n-k], dirty sorted in buffer[0..k].
// We merge right-to-left into arr[n-1..0], always picking the larger tail.
// This is safe because the merge cursor is always >= the clean read cursor.

pub fn esm_backwards_merge(
    arr: &mut [User],
    dirty_indices: &mut [usize],
    cmp: fn(&User, &User) -> std::cmp::Ordering,
) {
    if dirty_indices.is_empty() {
        return;
    }

    let n = arr.len();
    let k = dirty_indices.len();

    // Sort dirty indices for linear scanning
    dirty_indices.sort_unstable();

    // Extract dirty elements into buffer, compact clean to the left
    let mut dirty_buf: Vec<User> = Vec::with_capacity(k);
    let mut write = 0;
    let mut di = 0;
    for read in 0..n {
        if di < k && dirty_indices[di] == read {
            dirty_buf.push(std::mem::take(&mut arr[read]));
            di += 1;
        } else {
            if write != read {
                arr.swap(write, read);
            }
            write += 1;
        }
    }
    // arr[0..n-k] = sorted clean elements, dirty_buf = unsorted dirty

    dirty_buf.sort_by(cmp);

    // Merge backwards: pick larger of clean tail / dirty tail, write at arr[n-1..0]
    let mut ci = (n - k) as isize - 1; // clean read cursor
    let mut di = k as isize - 1; // dirty read cursor
    let mut wi = n - 1; // write cursor

    loop {
        let take_clean = if ci >= 0 && di >= 0 {
            cmp(&arr[ci as usize], &dirty_buf[di as usize]) != std::cmp::Ordering::Less
        } else {
            ci >= 0
        };

        if take_clean {
            // Move clean element to write position (may be same position)
            if wi != ci as usize {
                arr.swap(wi, ci as usize);
            }
            ci -= 1;
        } else if di >= 0 {
            arr[wi] = std::mem::take(&mut dirty_buf[di as usize]);
            di -= 1;
        } else {
            break;
        }

        if wi == 0 {
            break;
        }
        wi -= 1;
    }
}

// =============================================================================
// Variant 5: Binary Search ESM — O(k log n) comparisons, O(k) space
// =============================================================================
// Same as backwards merge but replaces the element-by-element merge comparison
// with binary search to find the insertion point for each dirty value in the
// clean portion, then bulk-shifts clean elements with copy_within.
//
// Comparisons: O(k log n) instead of O(n)
// Moves: O(n) — same as standard ESM
// Space: O(k) — same dirty buffer, no output buffer

pub fn esm_binary_search(
    arr: &mut [User],
    dirty_indices: &mut [usize],
    cmp: fn(&User, &User) -> std::cmp::Ordering,
) {
    if dirty_indices.is_empty() {
        return;
    }

    let n = arr.len();
    let k = dirty_indices.len();

    dirty_indices.sort_unstable();

    // Extract dirty to buffer, compact clean left — O(n) pass
    let mut dirty_buf: Vec<User> = Vec::with_capacity(k);
    let mut write = 0;
    let mut di = 0;
    for read in 0..n {
        if di < k && dirty_indices[di] == read {
            dirty_buf.push(std::mem::take(&mut arr[read]));
            di += 1;
        } else {
            if write != read {
                arr.swap(write, read);
            }
            write += 1;
        }
    }
    let clean_len = n - k;

    // Sort dirty buffer — O(k log k)
    dirty_buf.sort_by(&cmp);

    // Merge right-to-left using binary search — O(k log n) comparisons
    // arr[0..clean_len] = sorted clean, dirty_buf[0..k] = sorted dirty
    // The gap between ci and wi always equals the number of remaining dirty elements.
    // arr[ci..wi] is "consumed" space — safe to overwrite.
    let mut ci = clean_len; // one past the last unplaced clean element
    let mut wi = n; // one past the last write position

    for dj in (0..k).rev() {
        // Binary search: find split point in clean[0..ci]
        let split =
            arr[..ci].partition_point(|x| cmp(x, &dirty_buf[dj]) != std::cmp::Ordering::Greater);

        // Shift clean[split..ci] right into arr[wi-num_clean..wi].
        // Source and dest don't overlap (gap = remaining dirty elements between them).
        let num_clean = ci - split;
        if num_clean > 0 {
            for i in (0..num_clean).rev() {
                arr.swap(split + i, wi - num_clean + i);
            }
            wi -= num_clean;
        }
        ci = split;

        // Place dirty value
        wi -= 1;
        arr[wi] = std::mem::take(&mut dirty_buf[dj]);
    }

    // Remaining clean elements are already in arr[0..ci] == arr[0..wi], no move needed
    debug_assert_eq!(ci, wi);
}

// =============================================================================
// Variant 6: In-place merge ESM — O(k) auxiliary space
// =============================================================================
// 1. Stable partition: move dirty values to end, preserve clean order: O(n log k)
// 2. Stable sort dirty region: O(k log k) — uses O(k) internally (driftsort)
// 3. In-place merge (merge-without-buffer / SymMerge-style): O(n log k)
//
// Total: O(n log k) time, O(k) auxiliary space.
// The in-place merge eliminates the O(k) value buffer used by standard ESM,
// but stable sorting (step 2) internally allocates O(k) anyway, so total
// auxiliary space remains O(k). This variant is strictly slower than standard
// ESM due to the O(log k) factors in the partition and merge steps.
// Included to empirically validate that in-place merging is not competitive.

pub fn esm_inplace(
    arr: &mut [User],
    dirty_indices: &mut [usize],
    cmp: fn(&User, &User) -> std::cmp::Ordering,
) {
    if dirty_indices.is_empty() {
        return;
    }

    let n = arr.len();
    let k = dirty_indices.len();

    dirty_indices.sort_unstable();

    // Step 1: Stable partition — move dirty values to arr[n-k..n], clean to arr[0..n-k]
    // Uses recursive block rotation: O(n log k) time, O(log k) stack.
    partition_dirty_to_end(arr, dirty_indices, 0, k, 0, n);

    // Step 2: Stable sort dirty region — O(k log k) time, O(k) space internally
    // Stability is needed: equal-keyed elements must retain their relative order.
    // Rust's stable sort (driftsort) allocates O(k) internally, so total auxiliary
    // space is O(k) — same as standard ESM despite the in-place merge.
    arr[n - k..n].sort_by(&cmp);

    // Step 3: In-place merge — O(n log k) time, O(log n) stack
    inplace_merge(arr, 0, n - k, n, cmp);
}

/// Stably partition dirty values to the end of arr[range_start..range_end).
///
/// After this call:
///   arr[range_start .. range_start + clean_count] = clean values in original order
///   arr[range_start + clean_count .. range_end]   = dirty values (order may change)
///
/// dirty[d_lo..d_hi] must be the sorted dirty positions within [range_start..range_end).
/// Uses divide-and-conquer: split dirty indices at midpoint, recurse on left and right
/// non-overlapping sub-ranges, then rotate the middle section to merge.
/// Time: O(n log k) — each recursion level does O(n) rotation work, depth O(log k).
fn partition_dirty_to_end(
    arr: &mut [User],
    dirty: &[usize],
    d_lo: usize,
    d_hi: usize,
    range_start: usize,
    range_end: usize,
) {
    let dc = d_hi - d_lo;
    if dc == 0 || dc == range_end - range_start {
        return;
    }
    if dc == 1 {
        let pos = dirty[d_lo];
        if pos + 1 < range_end {
            arr[pos..range_end].rotate_left(1);
        }
        return;
    }

    let d_mid = d_lo + dc / 2;
    let split = dirty[d_mid];

    // Recurse on non-overlapping sub-ranges
    partition_dirty_to_end(arr, dirty, d_lo, d_mid, range_start, split);
    partition_dirty_to_end(arr, dirty, d_mid, d_hi, split, range_end);

    // After recursion: [left_clean | left_dirty | right_clean | right_dirty]
    // Rotate middle [left_dirty | right_clean] → [right_clean | left_dirty]
    let left_dirty = d_mid - d_lo;
    let right_dirty = d_hi - d_mid;
    let right_clean = (range_end - split) - right_dirty;

    if left_dirty > 0 && right_clean > 0 {
        let left_clean = (split - range_start) - left_dirty;
        let mid_start = range_start + left_clean;
        let mid_end = mid_start + left_dirty + right_clean;
        arr[mid_start..mid_end].rotate_left(left_dirty);
    }
}

/// In-place merge of arr[first..middle] and arr[middle..last].
///
/// Uses the merge-without-buffer algorithm (as in GCC libstdc++ / Go sort):
/// pick median of larger half, binary search in smaller half, rotate, recurse.
/// Time: O(n log(min(m, n))), Stack: O(log(m+n)).
fn inplace_merge(
    arr: &mut [User],
    first: usize,
    middle: usize,
    last: usize,
    cmp: fn(&User, &User) -> std::cmp::Ordering,
) {
    let len1 = middle - first;
    let len2 = last - middle;
    if len1 == 0 || len2 == 0 {
        return;
    }
    if len1 + len2 == 2 {
        if cmp(&arr[first], &arr[middle]) == std::cmp::Ordering::Greater {
            arr.swap(first, middle);
        }
        return;
    }

    let (cut1, cut2) = if len1 >= len2 {
        let c1 = first + len1 / 2;
        // lower_bound: first position in right half where element >= arr[c1]
        let c2 = {
            let s: &[User] = &*arr;
            s[middle..last].partition_point(|x| cmp(x, &s[c1]) == std::cmp::Ordering::Less)
        } + middle;
        (c1, c2)
    } else {
        let c2 = middle + len2 / 2;
        // upper_bound: first position in left half where element > arr[c2]
        let c1 = {
            let s: &[User] = &*arr;
            s[first..middle].partition_point(|x| cmp(x, &s[c2]) != std::cmp::Ordering::Greater)
        } + first;
        (c1, c2)
    };

    arr[cut1..cut2].rotate_left(middle - cut1);
    let new_mid = cut1 + (cut2 - middle);

    inplace_merge(arr, first, cut1, new_mid, cmp);
    inplace_merge(arr, new_mid, cut2, last, cmp);
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    type SortFn = fn(&mut [User], &mut [usize], fn(&User, &User) -> std::cmp::Ordering);

    fn user_cmp(a: &User, b: &User) -> std::cmp::Ordering {
        a.age.cmp(&b.age)
    }

    fn make_user(age: u32) -> User {
        User {
            name: format!("user_{}", age),
            age,
            country: "US".to_string(),
        }
    }

    fn is_sorted(arr: &[User]) -> bool {
        arr.windows(2).all(|w| w[0].age <= w[1].age)
    }

    fn test_variant(sort_fn: SortFn) {
        let mut rng = rand::thread_rng();

        // Empty
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        sort_fn(&mut arr, &mut [], user_cmp);
        assert!(is_sorted(&arr));

        // Single dirty
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[5] = make_user(2);
        sort_fn(&mut arr, &mut [5], user_cmp);
        assert!(is_sorted(&arr));

        // All dirty
        let mut arr = vec![make_user(9), make_user(1), make_user(5), make_user(3)];
        sort_fn(&mut arr, &mut (0..4).collect::<Vec<usize>>(), user_cmp);
        assert!(is_sorted(&arr));

        // Randomized stress test
        for _ in 0..200 {
            let n = rng.gen_range(5..=500);
            let k = rng.gen_range(1..=n);
            let mut arr: Vec<User> = (0..n as u32).map(make_user).collect();
            let mut pool: Vec<usize> = (0..n).collect();
            for i in 0..k {
                let j = rng.gen_range(i..n);
                pool.swap(i, j);
            }
            let mut dirty: Vec<usize> = pool[..k].to_vec();
            for &idx in &dirty {
                arr[idx] = make_user(rng.gen_range(0..n as u32 * 2));
            }

            let mut expected = arr.clone();
            expected.sort_by(user_cmp);

            sort_fn(&mut arr, &mut dirty, user_cmp);
            assert!(
                is_sorted(&arr),
                "Not sorted: n={}, k={}, dirty={:?}",
                n,
                k,
                dirty
            );
            assert_eq!(arr.len(), expected.len());
        }
    }

    #[test]
    fn test_backwards_merge() {
        test_variant(esm_backwards_merge);
    }

    #[test]
    fn test_binary_search() {
        test_variant(esm_binary_search);
    }

    #[test]
    fn test_inplace() {
        test_variant(esm_inplace);
    }
}
