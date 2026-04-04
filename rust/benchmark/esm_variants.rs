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
    arr: &mut Vec<User>,
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
// Variant 2: Forward merge with O(n) boolean mask
// =============================================================================
// Note: despite the name, this uses O(n) space for the is_dirty mask.
// Kept for comparison.

pub fn esm_original(
    arr: &mut Vec<User>,
    dirty_indices: &mut [usize],
    cmp: fn(&User, &User) -> std::cmp::Ordering,
) {
    crate::extract_sort_merge::extract_sort_merge(arr, dirty_indices, cmp);
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
    arr: &mut Vec<User>,
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
    dirty_buf.sort_by(|a, b| cmp(a, b));

    // Merge right-to-left using binary search — O(k log n) comparisons
    // arr[0..clean_len] = sorted clean, dirty_buf[0..k] = sorted dirty
    // The gap between ci and wi always equals the number of remaining dirty elements.
    // arr[ci..wi] is "consumed" space — safe to overwrite.
    let mut ci = clean_len; // one past the last unplaced clean element
    let mut wi = n;         // one past the last write position

    for dj in (0..k).rev() {
        // Binary search: find split point in clean[0..ci]
        let split = arr[..ci]
            .partition_point(|x| cmp(x, &dirty_buf[dj]) != std::cmp::Ordering::Greater);

        // Shift clean[split..ci] right into arr[wi-num_clean..wi]
        // The destination region arr[ci..wi] is consumed garbage — non-overlapping.
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
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

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

    fn test_variant(
        sort_fn: fn(&mut Vec<User>, &mut [usize], fn(&User, &User) -> std::cmp::Ordering),
    ) {
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
    fn test_original() {
        test_variant(esm_original);
    }

    #[test]
    fn test_binary_search() {
        test_variant(esm_binary_search);
    }
}
