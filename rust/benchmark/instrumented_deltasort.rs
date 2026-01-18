//! Instrumented DeltaSort for segment and movement analysis
//!
//! This is a copy of the DeltaSort algorithm that tracks:
//! - Number of segments formed
//! - Total data movement (number of element writes, excluding internal sort)

use std::collections::HashSet;

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
/// Returns (segment_count, total_movement) where:
/// - segment_count: Number of segments formed during Phase 2
/// - total_movement: Number of element positions shifted (excluding Phase 1 sort)
pub fn delta_sort_instrumented<T, F>(
    arr: &mut [T],
    updated_indices: &HashSet<usize>,
    cmp: F,
) -> InstrumentedResult
where
    T: Clone,
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    if updated_indices.is_empty() {
        return InstrumentedResult {
            segments: 0,
            movement: 0,
        };
    }

    // Instrumentation counters
    // Start with 1 segment (there's always at least one if k > 0)
    // Then count L→R transitions which create additional segment boundaries
    let mut segment_count: usize = 1;
    let mut total_movement: usize = 0;

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
    let mut pending_right: Vec<usize> = Vec::with_capacity(dirty.len());

    // Left boundary for fixing LEFT violations
    let mut left_bound = 0;

    // Track previous direction for segment counting (L→R transitions)
    let mut prev_direction: Option<Direction> = None;

    for &i in &dirty {
        // Determine direction (sentinel is treated as LEFT to trigger final flush)
        let direction = if i == arr.len() {
            Direction::Left
        } else {
            get_direction(arr, i, &cmp)
        };

        // Count L→R transitions (segment boundaries), skip sentinel
        if i < arr.len() {
            if let Some(prev) = prev_direction {
                if prev == Direction::Left && direction == Direction::Right {
                    segment_count += 1;
                }
            }
            prev_direction = Some(direction);
        }

        match direction {
            Direction::Left => {
                // Fix all pending RIGHT directions before fixing LEFT
                let mut right_bound = i - 1;
                while let Some(idx) = pending_right.pop() {
                    // Fix RIGHT direction at idx if needed
                    if idx < arr.len() - 1
                        && cmp(&arr[idx], &arr[idx + 1]) == std::cmp::Ordering::Greater
                    {
                        let target = fix_right(arr, idx, right_bound, &cmp);
                        // Movement = distance moved
                        total_movement += target - idx;
                        right_bound = target - 1;
                    }
                }

                // Fix actual (non-sentinel) LEFT directions
                if i < arr.len() {
                    let target = fix_left(arr, i, left_bound, &cmp);
                    // Movement = distance moved
                    total_movement += i - target;
                    left_bound = target + 1;
                }
            }
            Direction::Right => {
                if pending_right.is_empty() {
                    left_bound = i;
                }

                pending_right.push(i);
            }
        }
    }

    InstrumentedResult {
        segments: segment_count,
        movement: total_movement,
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
        let updated: HashSet<usize> = HashSet::new();
        let result = delta_sort_instrumented(&mut arr, &updated, i32::cmp);
        assert_eq!(arr, vec![1, 2, 3, 4, 5]);
        assert_eq!(result.segments, 0);
        assert_eq!(result.movement, 0);
    }

    #[test]
    fn test_update_preserves_order() {
        // Start sorted, update value but it stays in correct position
        let mut arr = vec![1, 2, 3, 4, 5];
        arr[2] = 3; // Update to same value - still sorted
        let updated: HashSet<usize> = [2].into_iter().collect();
        let result = delta_sort_instrumented(&mut arr, &updated, i32::cmp);
        assert_eq!(arr, vec![1, 2, 3, 4, 5]);
        assert_eq!(result.movement, 0); // No movement needed
    }

    #[test]
    fn test_single_left_movement() {
        // Start with sorted array [1, 3, 5, 7, 9]
        // Update index 3 from 7 to 2 -> needs to move left
        let mut arr = vec![1, 3, 5, 7, 9];
        arr[3] = 2; // Now: [1, 3, 5, 2, 9]
        let updated: HashSet<usize> = [3].into_iter().collect();
        let result = delta_sort_instrumented(&mut arr, &updated, i32::cmp);
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
        let updated: HashSet<usize> = [1].into_iter().collect();
        let result = delta_sort_instrumented(&mut arr, &updated, i32::cmp);
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
        let updated: HashSet<usize> = [1, 3].into_iter().collect();
        let result = delta_sort_instrumented(&mut arr, &updated, i32::cmp);
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
        let updated: HashSet<usize> = [1, 2, 3, 5, 7, 8, 9].into_iter().collect();
        let result = delta_sort_instrumented(&mut arr, &updated, i32::cmp);
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
            // Start with a sorted array
            let mut arr: Vec<i32> = (0..n).collect();
            let mut updated = HashSet::new();

            // Make k random updates
            for _ in 0..k {
                let idx = rng.gen_range(0..n as usize);
                arr[idx] = rng.gen_range(0..n); // Update to random value
                updated.insert(idx);
            }

            let mut expected = arr.clone();
            expected.sort();

            delta_sort_instrumented(&mut arr, &updated, i32::cmp);
            assert_eq!(arr, expected);
        }
    }
}
