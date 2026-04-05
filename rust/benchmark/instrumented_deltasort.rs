//! Instrumented DeltaSort for segment and movement analysis
//!
//! This is a copy of the DeltaSort algorithm that tracks:
//! - Number of segments formed
//! - Total data movement (number of element writes, excluding internal sort)

/// Result of instrumented DeltaSort execution
#[derive(Debug, Clone)]
pub struct InstrumentedResult {
    /// Number of segments that were formed
    pub segments: usize,
    /// Total number of element moves (writes) during Phase 2
    pub movement: usize,
}

/// Directions for updated indices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Left,
    Right,
}

/// Instrumented version of delta_sort_by that tracks segments and movement.
///
/// Takes unsorted dirty indices; sorts them internally.
pub fn delta_sort_instrumented<T, F>(
    arr: &mut [T],
    dirty: &mut [usize],
    cmp: F,
) -> InstrumentedResult
where
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    if dirty.is_empty() {
        return InstrumentedResult {
            segments: 0,
            movement: 0,
        };
    }

    // Instrumentation counters
    let mut segment_count: usize = 1;
    let mut total_movement: usize = 0;

    // Sort dirty indices — O(k log k)
    dirty.sort_unstable();
    debug_assert!(
        dirty.windows(2).all(|w| w[0] < w[1]),
        "dirty indices must be distinct"
    );
    let k = dirty.len();

    // Phase 1: Sort dirty values in-place using indirect heapsort — O(k log k), O(1) aux
    if k > 1 {
        for i in (0..k / 2).rev() {
            sift_down_indirect(arr, &dirty, i, k, &cmp);
        }
        for end in (1..k).rev() {
            arr.swap(dirty[0], dirty[end]);
            sift_down_indirect(arr, &dirty, 0, end, &cmp);
        }
    }

    // Phase 2: Fix ordering violations left to right — O(1) aux
    let mut left_bound = 0;
    let mut right_seg_start: Option<usize> = None;
    let mut prev_direction: Option<Direction> = None;

    for d in 0..k {
        let i = dirty[d];
        let direction = get_direction(arr, i, &cmp);

        // Count L→R transitions (segment boundaries)
        if let Some(prev) = prev_direction {
            if prev == Direction::Left && direction == Direction::Right {
                segment_count += 1;
            }
        }
        prev_direction = Some(direction);

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
                            let target = fix_right(arr, idx, right_bound, &cmp);
                            total_movement += target - idx;
                            right_bound = target - 1;
                        }
                    }
                    right_seg_start = None;
                }

                // Fix LEFT violation
                let target = fix_left(arr, i, left_bound, &cmp);
                total_movement += i - target;
                left_bound = target + 1;
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
                let target = fix_right(arr, idx, right_bound, &cmp);
                total_movement += target - idx;
                right_bound = target - 1;
            }
        }
    }

    InstrumentedResult {
        segments: segment_count,
        movement: total_movement,
    }
}

/// Sift-down for indirect heapsort
#[inline]
fn sift_down_indirect<T, F>(arr: &mut [T], indices: &[usize], start: usize, end: usize, cmp: &F)
where
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    let mut root = start;
    loop {
        let mut max = root;
        let left = 2 * root + 1;
        let right = left + 1;
        if left < end && cmp(&arr[indices[left]], &arr[indices[max]]) == std::cmp::Ordering::Greater
        {
            max = left;
        }
        if right < end
            && cmp(&arr[indices[right]], &arr[indices[max]]) == std::cmp::Ordering::Greater
        {
            max = right;
        }
        if max == root {
            break;
        }
        arr.swap(indices[root], indices[max]);
        root = max;
    }
}

/// Determines the direction at updated index i.
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

/// Fixes a RIGHT direction at i by moving it to the correct position.
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

/// Fixes a LEFT direction at i by moving it to the correct position.
/// Returns the target position.
fn fix_left<T, F>(arr: &mut [T], i: usize, left_bound: usize, cmp: &F) -> usize
where
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    // Binary search for target position in [left_bound, i)
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

    #[test]
    fn test_empty_indices() {
        let mut arr = vec![1, 2, 3, 4, 5];
        let mut updated: Vec<usize> = vec![];
        let result = delta_sort_instrumented(&mut arr, &mut updated, i32::cmp);
        assert_eq!(arr, vec![1, 2, 3, 4, 5]);
        assert_eq!(result.segments, 0);
        assert_eq!(result.movement, 0);
    }

    #[test]
    fn test_update_preserves_order() {
        // Start sorted, update value but it stays in correct position
        let mut arr = vec![1, 2, 3, 4, 5];
        arr[2] = 3; // Update to same value - still sorted
        let mut updated: Vec<usize> = vec![2];
        let result = delta_sort_instrumented(&mut arr, &mut updated, i32::cmp);
        assert_eq!(arr, vec![1, 2, 3, 4, 5]);
        assert_eq!(result.movement, 0); // No movement needed
    }

    #[test]
    fn test_single_left_movement() {
        // Start with sorted array [1, 3, 5, 7, 9]
        // Update index 3 from 7 to 2 -> needs to move left
        let mut arr = vec![1, 3, 5, 7, 9];
        arr[3] = 2; // Now: [1, 3, 5, 2, 9]
        let mut updated: Vec<usize> = vec![3];
        let result = delta_sort_instrumented(&mut arr, &mut updated, i32::cmp);
        assert_eq!(arr, vec![1, 2, 3, 5, 9]);
        assert_eq!(result.segments, 1); // One L segment
        assert_eq!(result.movement, 2); // Moves from index 3 to index 1
    }

    #[test]
    fn test_single_right_movement() {
        // Start with sorted array [1, 3, 5, 7, 9]
        // Update index 1 from 3 to 8 -> needs to move right
        let mut arr = vec![1, 3, 5, 7, 9];
        arr[1] = 8; // Now: [1, 8, 5, 7, 9]
        let mut updated: Vec<usize> = vec![1];
        let result = delta_sort_instrumented(&mut arr, &mut updated, i32::cmp);
        assert_eq!(arr, vec![1, 5, 7, 8, 9]);
        assert_eq!(result.segments, 1); // One R segment
        assert_eq!(result.movement, 2); // 8 moves from index 1 to index 3
    }

    #[test]
    fn test_two_updates_forming_one_segment() {
        // Start with sorted array [1, 3, 5, 7, 9]
        // Update index 1: 3 -> 8 (R violation)
        // Update index 3: 7 -> 2 (L violation)
        let mut arr = vec![1, 3, 5, 7, 9];
        arr[1] = 8; // R violation
        arr[3] = 2; // L violation
                    // Now: [1, 8, 5, 2, 9]
        let mut updated: Vec<usize> = vec![1, 3];
        let result = delta_sort_instrumented(&mut arr, &mut updated, i32::cmp);
        assert_eq!(arr, vec![1, 2, 5, 8, 9]);
        assert_eq!(result.segments, 1); // One R*L* segment
        assert_eq!(result.movement, 0); // No movement
    }

    #[test]
    fn test_leading_multiple_intermediate_trailing_segment() {
        // Start with sorted array [1, 3, 5, 7, 9]
        // Update index 1: 3 -> 8 (R violation)
        // Update index 3: 7 -> 2 (L violation)
        let mut arr = vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21];
        arr[1] = 0;
        arr[2] = 0;
        arr[3] = 6;
        arr[5] = 10;
        arr[7] = 11;
        arr[8] = 22;
        arr[9] = 23;
        // Now: [1, 0, 0, 6, 9, 10, 13, 14, 22, 23, 21]
        let mut updated: Vec<usize> = vec![1, 2, 3, 5, 7, 8, 9];
        let result = delta_sort_instrumented(&mut arr, &mut updated, i32::cmp);
        assert_eq!(arr, vec![0, 0, 1, 6, 9, 10, 11, 13, 21, 22, 23]);
        assert_eq!(result.segments, 3);
        assert_eq!(result.movement, 5);
    }

    #[test]
    fn test_correctness_matches_native_sort() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for _ in 0..10 {
            let n = 100;
            let k = 10;
            let mut arr: Vec<i32> = (0..n).collect();

            // Generate distinct dirty indices
            let mut pool: Vec<usize> = (0..n as usize).collect();
            for i in 0..k {
                let j = rng.gen_range(i..n as usize);
                pool.swap(i, j);
            }
            let mut updated: Vec<usize> = pool[..k].to_vec();

            for &idx in &updated {
                arr[idx] = rng.gen_range(0..n);
            }

            let mut expected = arr.clone();
            expected.sort();

            delta_sort_instrumented(&mut arr, &mut updated, i32::cmp);
            assert_eq!(arr, expected);
        }
    }
}
