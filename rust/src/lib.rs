//! DeltaSort: Efficient incremental repair of sorted arrays.
//!
//! When a small number of values in a sorted array change, DeltaSort
//! restores sorted order more efficiently than a full re-sort by exploiting
//! knowledge of which indices changed.
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
    T: Clone + Ord,
{
    delta_sort_by(arr, updated_indices, T::cmp)
}

/// Sorts an array that was previously sorted but has had some values updated,
/// using a custom comparator.
///
/// This function efficiently restores sorted order by only moving the values
/// that need to be repositioned, rather than performing a full sort.
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
    T: Clone,
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    if updated_indices.is_empty() {
        return;
    }

    // Phase 1: Extract and sort dirty values, write back in index order
    let mut dirty: Vec<usize> = updated_indices.iter().copied().collect();
    dirty.sort_unstable();

    let mut values: Vec<T> = dirty.iter().map(|&i| arr[i].clone()).collect();
    values.sort_by(&cmp);

    for (i, &idx) in dirty.iter().enumerate() {
        arr[idx] = values[i].clone();
    }

    // Add sentinel to trigger final flush
    dirty.push(arr.len());

    // Phase 2: Scan updated indices left to right

    // Stack for pending RIGHT directions
    let mut pending_right_directions: Vec<usize> = Vec::with_capacity(dirty.len());

    // Left boundary for fixing LEFT violations
    let mut left_bound = 0;

    for &i in &dirty {
        // Determine direction (sentinel is treated as LEFT to trigger final flush)
        let direction = if i == arr.len() {
            Direction::Left
        } else {
            get_direction(arr, i, &cmp)
        };

        match direction {
            Direction::Left => {
                // Fix all pending RIGHT directions before fixing LEFT
                let mut right_bound = i.saturating_sub(1);
                while let Some(idx) = pending_right_directions.pop() {
                    // Fix RIGHT direction at idx if needed
                    if idx < arr.len() - 1
                        && cmp(&arr[idx], &arr[idx + 1]) == std::cmp::Ordering::Greater
                    {
                        right_bound = fix_right(arr, idx, right_bound, &cmp).saturating_sub(1);
                    }
                }

                // Fix actual (non-sentinel) LEFT directions
                if i < arr.len() {
                    left_bound = fix_left(arr, i, left_bound, &cmp) + 1;
                }
            }
            Direction::Right => {
                pending_right_directions.push(i);
            }
        }
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
    // Binary search for target position on the right
    let mut lo = i + 1;
    let mut hi = right_bound as isize;

    while lo as isize <= hi {
        let mid = lo + ((hi as usize - lo) >> 1);
        let c = cmp(&arr[mid], &arr[i]);

        if c != std::cmp::Ordering::Greater {
            lo = mid + 1;
        } else {
            hi = mid as isize - 1;
        }
    }

    let target = hi as usize;
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
    // Binary search for target position on the left
    let mut lo = left_bound;
    let mut hi = i.saturating_sub(1) as isize;

    while lo as isize <= hi {
        let mid = lo + ((hi as usize - lo) >> 1);
        let c = cmp(&arr[i], &arr[mid]);

        if c == std::cmp::Ordering::Less {
            hi = mid as isize - 1;
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
