//! DeltaSort: Incremental sorting of arrays with known updates.
//!
//! When a small number of values in a sorted array change, DeltaSort
//! restores sorted order more efficiently than a full re-sort by exploiting
//! knowledge of which indices changed.
//!
//! # Example
//!
//! ```
//! use deltasort::delta_sort;
//!
//! let mut arr = vec![1, 3, 5, 7, 9];
//! // Modify indices 1 and 3
//! arr[1] = 8;
//! arr[3] = 2;
//!
//! let mut dirty = vec![1, 3];
//! delta_sort(&mut arr, &mut dirty);
//!
//! assert_eq!(arr, vec![1, 2, 5, 8, 9]);
//! ```

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
///
/// let mut arr = vec![1, 3, 5, 7, 9];
/// arr[1] = 8;
/// arr[3] = 2;
///
/// let mut dirty = vec![1, 3];
/// delta_sort(&mut arr, &mut dirty);
///
/// assert_eq!(arr, vec![1, 2, 5, 8, 9]);
/// ```
pub fn delta_sort<T>(arr: &mut [T], dirty: &mut [usize])
where
    T: Default + Ord,
{
    delta_sort_by(arr, dirty, T::cmp)
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
/// * `dirty` - Mutable slice of indices where values were updated (will be sorted internally)
/// * `cmp` - Comparison function returning `Ordering`
///
/// # Panics
///
/// Panics if any index in `dirty` is out of bounds for `arr`.
///
/// # Example
///
/// ```
/// use deltasort::delta_sort_by;
///
/// let mut arr = vec![1, 3, 5, 7, 9];
/// arr[1] = 8;
/// arr[3] = 2;
///
/// let mut dirty = vec![1, 3];
/// delta_sort_by(&mut arr, &mut dirty, |a, b| a.cmp(b));
///
/// assert_eq!(arr, vec![1, 2, 5, 8, 9]);
/// ```
pub fn delta_sort_by<T, F>(arr: &mut [T], dirty: &mut [usize], cmp: F)
where
    T: Default,
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    if dirty.is_empty() {
        return;
    }

    // Sort dirty indices — O(k log k), O(1) aux
    dirty.sort_unstable();
    debug_assert!(
        dirty.windows(2).all(|w| w[0] < w[1]),
        "dirty indices must be distinct"
    );
    let k = dirty.len();

    // Phase 1: Extract dirty values, sort, write back — O(k log k) time, O(k) space
    let mut values: Vec<T> = dirty.iter().map(|&i| std::mem::take(&mut arr[i])).collect();
    values.sort_unstable_by(&cmp);
    for (&idx, val) in dirty.iter().zip(values) {
        arr[idx] = val;
    }

    // Phase 2: Fix ordering violations left to right — O(n√k) expected, O(1) aux
    // Instead of a pending_right stack, track the start of the current RIGHT segment
    // as an index into the dirty array, then iterate backwards when flushing.

    let mut left_bound = 0;
    let mut right_seg_start: Option<usize> = None; // index into dirty[]

    for d in 0..k {
        let i = dirty[d];
        let direction = get_direction(arr, i, &cmp);

        match direction {
            Direction::Left => {
                // Flush pending RIGHT segment in reverse order
                if let Some(start) = right_seg_start {
                    let mut right_bound = i - 1;
                    for rd in (start..d).rev() {
                        let idx = dirty[rd];
                        if idx < arr.len() - 1
                            && cmp(&arr[idx], &arr[idx + 1]) == std::cmp::Ordering::Greater
                        {
                            right_bound = fix_right(arr, idx, right_bound, &cmp) - 1;
                        }
                    }
                    right_seg_start = None;
                }

                // Fix LEFT violation
                left_bound = fix_left(arr, i, left_bound, &cmp) + 1;
            }
            Direction::Right => {
                if right_seg_start.is_none() {
                    right_seg_start = Some(d);
                    left_bound = i;
                }
            }
        }
    }

    // Flush trailing RIGHT segment
    if let Some(start) = right_seg_start {
        let mut right_bound = arr.len() - 1;
        for rd in (start..k).rev() {
            let idx = dirty[rd];
            if idx < arr.len() - 1 && cmp(&arr[idx], &arr[idx + 1]) == std::cmp::Ordering::Greater {
                right_bound = fix_right(arr, idx, right_bound, &cmp) - 1;
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
        arr[from..=to].rotate_left(1);
    } else {
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
        delta_sort(&mut arr, &mut []);
        // Should be unchanged (no-op)
        assert_eq!(arr, vec![1, 2, 3, 2, 1]);
    }

    #[test]
    fn test_single_element() {
        let mut arr = vec![42];
        delta_sort(&mut arr, &mut [0]);
        assert_eq!(arr, vec![42]);
    }

    #[test]
    fn test_already_sorted() {
        let mut arr = vec![1, 2, 3, 4, 5];
        delta_sort(&mut arr, &mut [1, 3]);
        assert_eq!(arr, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_movement_cancellation() {
        // Example from paper: values cross but pre-sorting cancels movement
        let mut arr = vec![1, 8, 5, 2, 9];
        delta_sort(&mut arr, &mut [1, 3]);
        assert_eq!(arr, vec![1, 2, 5, 8, 9]);
    }

    #[test]
    fn test_all_left_moves() {
        let mut arr = vec![5, 4, 3, 2, 1];
        delta_sort(&mut arr, &mut [0, 1, 2, 3, 4]);
        assert_eq!(arr, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_all_right_moves() {
        // Change to reverse order
        let mut arr = vec![1, 5, 4, 3, 2];
        delta_sort(&mut arr, &mut [1, 2, 3, 4]);
        assert_eq!(arr, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_custom_comparator() {
        // Sort in descending order
        let mut arr = vec![1, 5, 3, 7, 2];
        delta_sort_by(&mut arr, &mut [0, 1, 2, 3, 4], |a, b| b.cmp(a)); // Reverse comparator
        assert_eq!(arr, vec![7, 5, 3, 2, 1]);
    }

    #[test]
    fn test_with_duplicates() {
        let mut arr = vec![1, 3, 3, 5, 2];
        delta_sort(&mut arr, &mut [4]);
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

                    // Generate distinct dirty indices
                    let mut indices: Vec<usize> = (0..size).collect();
                    for i in 0..delta_count {
                        let j = rng.gen_range(i..size);
                        indices.swap(i, j);
                    }
                    let mut dirty: Vec<usize> = indices[..delta_count].to_vec();

                    // Randomly modify dirty values
                    for &idx in &dirty {
                        arr[idx] = rng.gen_range(0..size as i32);
                    }

                    // Create expected result via native sort
                    let mut expected = arr.clone();
                    expected.sort();

                    // Sort with DeltaSort
                    delta_sort(&mut arr, &mut dirty);

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
        #[derive(Clone, Debug, Default, PartialEq, Eq)]
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

        delta_sort_by(&mut users, &mut [1], |a, b| a.age.cmp(&b.age));

        assert_eq!(users[0].name, "Alice");
        assert_eq!(users[1].name, "Charlie");
        assert_eq!(users[2].name, "Bob");
    }
}
