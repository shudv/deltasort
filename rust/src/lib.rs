//! DeltaSort: Incremental sorting of arrays with known updates.
//!
//! When a small number of values in a sorted array change, DeltaSort
//! restores sorted order more efficiently than a full re-sort by exploiting
//! knowledge of which indices changed.
//!
//! # Algorithm
//!
//! DeltaSort achieves O(n√k) time with O(1) auxiliary space (the
//! `updated_indices` slice is input, not auxiliary). Phase 1 sorts the
//! dirty values in-place at their scattered positions using heapsort via
//! indirection. Phase 2 scans left-to-right and fixes violations using
//! binary-search + rotation, reusing the indices array as a stack.
//!
//! # Example
//!
//! ```
//! use deltasort::delta_sort;
//! use std::collections::HashSet;
//!
//! let mut arr = vec![1, 3, 5, 7, 9];
//! // Modify indices 1 and 3
//! arr[1] = 8;
//! arr[3] = 2;
//!
//! let updated: HashSet<usize> = [1, 3].into_iter().collect();
//! delta_sort(&mut arr, &updated);
//!
//! assert_eq!(arr, vec![1, 2, 5, 8, 9]);
//! ```

pub mod parallel;

use std::collections::HashSet;

/// Directions for updated indices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    /// Element must move left
    Left,
    /// Element may move right (or is stable)
    Right,
}

/// Sorts an array that was previously sorted but has had some values updated.
///
/// This is a convenience wrapper around [`delta_sort_by`] for types that implement `Ord`.
///
/// # Arguments
///
/// * `arr` - A mutable slice that was previously sorted but has had some values changed
/// * `updated_indices` - Set of indices where values were updated
///
/// # Panics
///
/// Panics if any index in `updated_indices` is out of bounds for `arr`.
///
/// # Example
///
/// ```
/// use deltasort::delta_sort;
/// use std::collections::HashSet;
///
/// let mut arr = vec![1, 3, 5, 7, 9];
/// arr[1] = 8;
/// arr[3] = 2;
///
/// let updated: HashSet<usize> = [1, 3].into_iter().collect();
/// delta_sort(&mut arr, &updated);
///
/// assert_eq!(arr, vec![1, 2, 5, 8, 9]);
/// ```
pub fn delta_sort<T>(arr: &mut [T], updated_indices: &HashSet<usize>)
where
    T: Ord,
{
    delta_sort_by(arr, updated_indices, T::cmp)
}

/// Sorts an array that was previously sorted but has had some values updated,
/// using a custom comparator.
///
/// This is a convenience wrapper around [`delta_sort_by_inplace`] that accepts
/// a `HashSet`. The `HashSet` is converted to a `Vec` internally.
///
/// # Arguments
///
/// * `arr` - A mutable slice that was previously sorted but has had some values changed
/// * `updated_indices` - Set of indices where values were updated
/// * `cmp` - Comparison function returning `Ordering`
///
/// # Panics
///
/// Panics if any index in `updated_indices` is out of bounds for `arr`.
///
/// # Example
///
/// ```
/// use deltasort::delta_sort_by;
/// use std::collections::HashSet;
///
/// let mut arr = vec![1, 3, 5, 7, 9];
/// arr[1] = 8;
/// arr[3] = 2;
///
/// let updated: HashSet<usize> = [1, 3].into_iter().collect();
/// delta_sort_by(&mut arr, &updated, |a, b| a.cmp(b));
///
/// assert_eq!(arr, vec![1, 2, 5, 8, 9]);
/// ```
pub fn delta_sort_by<T, F>(arr: &mut [T], updated_indices: &HashSet<usize>, cmp: F)
where
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    if updated_indices.is_empty() {
        return;
    }
    let mut indices: Vec<usize> = updated_indices.iter().copied().collect();
    delta_sort_by_inplace(arr, &mut indices, cmp);
}

/// Sorts an array that was previously sorted but has had some values updated.
///
/// O(1) auxiliary space variant. The `updated_indices` slice is treated as
/// input (not auxiliary space) and is used as working storage: it will be
/// sorted in-place and partially overwritten during Phase 2.
///
/// # Arguments
///
/// * `arr` - A mutable slice that was previously sorted but has had some values changed
/// * `updated_indices` - Mutable slice of indices where values were updated.
///   Will be sorted in-place and used as scratch space.
/// * `cmp` - Comparison function returning `Ordering`
///
/// # Panics
///
/// Panics if any index in `updated_indices` is out of bounds for `arr`.
pub fn delta_sort_by_inplace<T, F>(arr: &mut [T], updated_indices: &mut [usize], cmp: F)
where
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    let k = updated_indices.len();
    if k == 0 {
        return;
    }

    // Sort the indices in-place
    updated_indices.sort_unstable();

    // Phase 1: Sort dirty values in-place at their scattered positions
    // using heapsort with indirection. O(k log k) time, O(1) aux space.
    heapsort_scattered(arr, updated_indices, &cmp);

    // Phase 2: Scan updated indices left to right, fixing violations.
    // Reuse updated_indices as the pending-RIGHT stack.
    // Invariant: stack_top <= read_pos, so stack writes never overwrite unread data.
    let mut stack_top: usize = 0;
    let mut left_bound: usize = 0;

    for read_pos in 0..k {
        let i = updated_indices[read_pos];
        let direction = get_direction(arr, i, &cmp);

        match direction {
            Direction::Left => {
                // Fix all pending RIGHT directions before fixing LEFT
                let mut right_bound = i - 1;
                while stack_top > 0 {
                    stack_top -= 1;
                    let idx = updated_indices[stack_top];
                    if idx < arr.len() - 1
                        && cmp(&arr[idx], &arr[idx + 1]) == std::cmp::Ordering::Greater
                    {
                        right_bound = fix_right(arr, idx, right_bound, &cmp) - 1;
                    }
                }

                left_bound = fix_left(arr, i, left_bound, &cmp) + 1;
            }
            Direction::Right => {
                if stack_top == 0 {
                    left_bound = i;
                }
                // Safe: stack_top <= read_pos
                updated_indices[stack_top] = i;
                stack_top += 1;
            }
        }
    }

    // Final flush: fix any remaining pending RIGHTs (replaces sentinel logic)
    if stack_top > 0 {
        let mut right_bound = arr.len() - 1;
        while stack_top > 0 {
            stack_top -= 1;
            let idx = updated_indices[stack_top];
            if idx < arr.len() - 1
                && cmp(&arr[idx], &arr[idx + 1]) == std::cmp::Ordering::Greater
            {
                right_bound = fix_right(arr, idx, right_bound, &cmp) - 1;
            }
        }
    }
}

/// Heapsort values at scattered positions via indirection.
///
/// Sorts `arr[indices[0]], arr[indices[1]], ..., arr[indices[k-1]]` in
/// ascending order (according to `cmp`) without touching other elements.
/// O(k log k) time, O(1) auxiliary space.
fn heapsort_scattered<T, F>(arr: &mut [T], indices: &[usize], cmp: &F)
where
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    let k = indices.len();
    if k <= 1 {
        return;
    }

    // Build max-heap
    for i in (0..k / 2).rev() {
        sift_down_scattered(arr, indices, i, k, cmp);
    }

    // Extract max repeatedly
    for end in (1..k).rev() {
        arr.swap(indices[0], indices[end]);
        sift_down_scattered(arr, indices, 0, end, cmp);
    }
}

/// Sift-down for heapsort on scattered positions.
fn sift_down_scattered<T, F>(
    arr: &mut [T],
    indices: &[usize],
    mut root: usize,
    end: usize,
    cmp: &F,
) where
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    loop {
        let mut child = 2 * root + 1;
        if child >= end {
            break;
        }
        if child + 1 < end
            && cmp(&arr[indices[child]], &arr[indices[child + 1]]) == std::cmp::Ordering::Less
        {
            child += 1;
        }
        if cmp(&arr[indices[root]], &arr[indices[child]]) != std::cmp::Ordering::Less {
            break;
        }
        arr.swap(indices[root], indices[child]);
        root = child;
    }
}

/// Determines the direction at updated index i.
///
/// This should only be called for an updated index.
#[inline]
fn get_direction<T, F>(arr: &[T], i: usize, cmp: &F) -> Direction
where
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    if i > 0 && cmp(&arr[i - 1], &arr[i]) == std::cmp::Ordering::Greater {
        Direction::Left
    } else {
        Direction::Right
    }
}

/// Fixes a RIGHT direction at i by moving it to the correct position
/// between i and right_bound.
///
/// Returns the new index of the moved element.
fn fix_right<T, F>(arr: &mut [T], i: usize, right_bound: usize, cmp: &F) -> usize
where
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    // Binary search for target position in (i, right_bound]
    let mut lo = i + 1;
    let mut hi = right_bound + 1;

    while lo < hi {
        let mid = lo + ((hi - lo) >> 1);
        if cmp(&arr[mid], &arr[i]) != std::cmp::Ordering::Greater {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    let target = lo - 1;
    move_element(arr, i, target);
    target
}

/// Fixes a LEFT direction at i by moving it to the correct position
/// between left_bound and i.
///
/// Returns the target position.
fn fix_left<T, F>(arr: &mut [T], i: usize, left_bound: usize, cmp: &F) -> usize
where
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    // Binary search for target position in [left_bound, i)
    // Using half-open interval avoids underflow when target is at position 0
    let mut lo = left_bound;
    let mut hi = i;

    while lo < hi {
        let mid = lo + ((hi - lo) >> 1);
        if cmp(&arr[i], &arr[mid]) == std::cmp::Ordering::Less {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    move_element(arr, i, lo);
    lo
}

/// Moves an element from index `from` to index `to`, shifting intermediate values.
#[inline]
fn move_element<T>(arr: &mut [T], from: usize, to: usize) {
    if from == to {
        return;
    }

    if from < to {
        // Moving right: rotate left
        arr[from..=to].rotate_left(1);
    } else {
        // Moving left: rotate right
        arr[to..=from].rotate_right(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    /// Test scales (powers of 10)
    const SCALES: [u32; 4] = [1, 2, 3, 4];
    /// Delta volumes as percentages
    const DELTA_VOLUMES: [usize; 7] = [0, 1, 5, 10, 20, 50, 80];
    /// Iterations per configuration
    const ITERATIONS: usize = 10;

    #[test]
    fn test_empty_updated_indices() {
        let mut arr = vec![1, 2, 3, 2, 1];
        let updated: HashSet<usize> = HashSet::new();
        delta_sort(&mut arr, &updated);
        // Should be unchanged (no-op)
        assert_eq!(arr, vec![1, 2, 3, 2, 1]);
    }

    #[test]
    fn test_single_element() {
        let mut arr = vec![42];
        let updated: HashSet<usize> = [0].into_iter().collect();
        delta_sort(&mut arr, &updated);
        assert_eq!(arr, vec![42]);
    }

    #[test]
    fn test_already_sorted() {
        let mut arr = vec![1, 2, 3, 4, 5];
        let updated: HashSet<usize> = [1, 3].into_iter().collect();
        delta_sort(&mut arr, &updated);
        assert_eq!(arr, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_movement_cancellation() {
        // Example from paper: values cross but pre-sorting cancels movement
        let mut arr = vec![1, 8, 5, 2, 9];
        let updated: HashSet<usize> = [1, 3].into_iter().collect();
        delta_sort(&mut arr, &updated);
        assert_eq!(arr, vec![1, 2, 5, 8, 9]);
    }

    #[test]
    fn test_all_left_moves() {
        let mut arr = vec![5, 4, 3, 2, 1];
        let updated: HashSet<usize> = [0, 1, 2, 3, 4].into_iter().collect();
        delta_sort(&mut arr, &updated);
        assert_eq!(arr, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_all_right_moves() {
        // Change to reverse order
        let mut arr = vec![1, 5, 4, 3, 2];
        let updated: HashSet<usize> = [1, 2, 3, 4].into_iter().collect();
        delta_sort(&mut arr, &updated);
        assert_eq!(arr, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_custom_comparator() {
        // Sort in descending order
        let mut arr = vec![1, 5, 3, 7, 2];
        let updated: HashSet<usize> = [0, 1, 2, 3, 4].into_iter().collect();
        delta_sort_by(&mut arr, &updated, |a, b| b.cmp(a)); // Reverse comparator
        assert_eq!(arr, vec![7, 5, 3, 2, 1]);
    }

    #[test]
    fn test_with_duplicates() {
        let mut arr = vec![1, 3, 3, 5, 2];
        let updated: HashSet<usize> = [4].into_iter().collect();
        delta_sort(&mut arr, &updated);
        assert_eq!(arr, vec![1, 2, 3, 3, 5]);
    }

    #[test]
    fn test_randomized_correctness() {
        let mut rng = rand::thread_rng();

        for &scale in &SCALES {
            let size = 10_usize.pow(scale);

            for &delta_volume in &DELTA_VOLUMES {
                for _ in 0..ITERATIONS {
                    let delta_count = (delta_volume * size / 100).max(1);

                    // Create sorted array
                    let mut arr: Vec<i32> = (0..size as i32).collect();
                    let mut updated_indices = HashSet::new();

                    // Randomly modify delta_count values
                    for _ in 0..delta_count {
                        let idx = rng.gen_range(0..size);
                        arr[idx] = rng.gen_range(0..size as i32);
                        updated_indices.insert(idx);
                    }

                    // Create expected result via native sort
                    let mut expected = arr.clone();
                    expected.sort();

                    // Sort with DeltaSort
                    delta_sort(&mut arr, &updated_indices);

                    assert_eq!(
                        arr, expected,
                        "Failed at scale={}, delta_volume={}",
                        scale, delta_volume
                    );
                }
            }
        }
    }

    #[test]
    fn test_struct_sorting() {
        #[derive(Clone, Debug, PartialEq, Eq)]
        struct User {
            name: String,
            age: u32,
        }

        let mut users = vec![
            User {
                name: "Alice".into(),
                age: 30,
            },
            User {
                name: "Bob".into(),
                age: 25,
            },
            User {
                name: "Charlie".into(),
                age: 35,
            },
        ];

        // Modify Bob's age
        users[1].age = 40;

        let updated: HashSet<usize> = [1].into_iter().collect();
        delta_sort_by(&mut users, &updated, |a, b| a.age.cmp(&b.age));

        assert_eq!(users[0].name, "Alice");
        assert_eq!(users[1].name, "Charlie");
        assert_eq!(users[2].name, "Bob");
    }
}
