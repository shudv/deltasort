//! Parallel DeltaSort: Two-pass algorithm with parallel segment processing.
//!
//! Pass 1: Pre-sort dirty values, compute directions, detect segment boundaries (sequential)
//! Pass 2: Fix each segment independently (parallel)
//!
//! Achieves O(n) time with √k threads, since segments are non-overlapping.

use std::cmp::Ordering;
use std::collections::HashSet;
use rayon::prelude::*;

/// Wrapper to mark raw pointers as Send+Sync.
/// Safety: The user must ensure no data races occur.
struct SendPtr<T>(*mut T);

impl<T> Copy for SendPtr<T> {}
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    fn as_ptr(self) -> *mut T {
        self.0
    }
}

/// Direction for an updated index
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Left,
    Right,
}

/// Parallel DeltaSort for types implementing Ord.
pub fn parallel_delta_sort<T>(arr: &mut [T], updated_indices: &HashSet<usize>)
where
    T: Ord + Send + Sync + Clone,
{
    parallel_delta_sort_by(arr, updated_indices, T::cmp)
}

/// Parallel DeltaSort with custom comparator.
///
/// Two-pass algorithm:
/// 1. Sequential pass: pre-sort dirty values, compute directions based on unchanged neighbors
/// 2. Parallel pass: fix each segment independently based on precomputed directions
pub fn parallel_delta_sort_by<T, F>(arr: &mut [T], updated_indices: &HashSet<usize>, cmp: F)
where
    T: Send + Sync + Clone,
    F: Fn(&T, &T) -> Ordering + Sync + Send,
{
    let k = updated_indices.len();
    if k == 0 {
        return;
    }

    let n = arr.len();

    // Sort dirty indices by position (parallel for large k)
    let mut dirty: Vec<usize> = updated_indices.iter().copied().collect();
    if k > 1000 {
        dirty.par_sort_unstable();
    } else {
        dirty.sort_unstable();
    }

    // Phase 1: Pre-sort dirty values at their scattered positions
    // Use parallel approach for large k (O(k) aux space but parallel)
    if k > 1000 {
        parallel_presort_scattered(arr, &dirty, &cmp);
    } else {
        heapsort_scattered(arr, &dirty, &cmp);
    }

    // Phase 1.5: Compute directions using cluster-based approach
    // Key insight: for a cluster of consecutive dirty indices, the unchanged index
    // just left of it determines the direction of every dirty index in that cluster.
    let directions = compute_directions_cluster(arr, &dirty, updated_indices, &cmp);

    // Phase 1.6: Build segments from L→R transitions
    let segments = build_segments(&dirty, &directions, n);

    // Phase 2: Fix each segment
    // Only parallelize if there's enough work to offset thread overhead
    // Heuristic: need at least 2 segments and total work > threshold
    let total_items: usize = segments.iter().map(|s| s.items.len()).sum();
    let should_parallelize = segments.len() >= 2 && n >= 100_000 && total_items >= 100;
    
    if should_parallelize {
        fix_segments_in_parallel(arr, &segments, &cmp);
    } else {
        for seg in &segments {
            fix_segment_with_directions(arr, seg, &cmp);
        }
    }
}

/// Compute directions using cluster-based approach in O(k) time.
/// For a cluster of consecutive dirty indices, the unchanged index just left of it
/// determines the direction of every dirty index in that cluster.
///
/// Optimized: process dirty indices in order, tracking anchor without repeated scans.
fn compute_directions_cluster<T, F>(
    arr: &[T],
    dirty: &[usize],
    _dirty_set: &HashSet<usize>, // kept for API compatibility
    cmp: &F,
) -> Vec<Direction>
where
    F: Fn(&T, &T) -> Ordering,
{
    let k = dirty.len();
    if k == 0 {
        return Vec::new();
    }
    
    let mut directions = Vec::with_capacity(k);
    
    // Process dirty indices in sorted order
    // Track the current anchor (first unchanged to the left of current cluster)
    let mut current_anchor: Option<usize> = None;
    
    for (idx, &i) in dirty.iter().enumerate() {
        // Check if we're starting a new cluster
        // A new cluster starts when there's a gap between this dirty index and the previous
        if idx == 0 {
            // First dirty index: anchor is i-1 if i > 0, else None
            current_anchor = if i > 0 { Some(i - 1) } else { None };
        } else {
            let prev_dirty = dirty[idx - 1];
            if i > prev_dirty + 1 {
                // Gap between previous dirty and this one
                // New anchor is i-1 (which is unchanged since there's a gap)
                current_anchor = Some(i - 1);
            }
            // If i == prev_dirty + 1, we're in same cluster, keep current_anchor
        }
        
        let dir = match current_anchor {
            Some(a) => {
                if cmp(&arr[a], &arr[i]) == Ordering::Greater {
                    Direction::Left
                } else {
                    Direction::Right
                }
            }
            None => Direction::Right,
        };
        directions.push(dir);
    }

    directions
}

// find_anchor_left removed - no longer needed with O(k) algorithm

/// A segment containing dirty indices with precomputed directions
#[derive(Debug, Clone)]
struct Segment {
    /// (index, direction) pairs for this segment
    items: Vec<(usize, Direction)>,
    /// Left boundary in original array
    left_bound: usize,
    /// Right boundary in original array
    right_bound: usize,
}

/// Build segments with precomputed directions
fn build_segments(dirty: &[usize], directions: &[Direction], n: usize) -> Vec<Segment> {
    if dirty.is_empty() {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut current_items: Vec<(usize, Direction)> = Vec::new();
    let mut seg_left_bound = 0usize;

    for (i, (&idx, &dir)) in dirty.iter().zip(directions.iter()).enumerate() {
        // Detect L→R transition (new segment boundary)
        if i > 0 && directions[i - 1] == Direction::Left && dir == Direction::Right {
            // Close current segment
            if !current_items.is_empty() {
                segments.push(Segment {
                    items: std::mem::take(&mut current_items),
                    left_bound: seg_left_bound,
                    right_bound: idx.saturating_sub(1),
                });
            }
            seg_left_bound = idx;
        }
        current_items.push((idx, dir));
    }

    // Close final segment
    if !current_items.is_empty() {
        segments.push(Segment {
            items: current_items,
            left_bound: seg_left_bound,
            right_bound: n - 1,
        });
    }

    segments
}

/// Fix a segment using precomputed directions
fn fix_segment_with_directions<T, F>(arr: &mut [T], seg: &Segment, cmp: &F)
where
    F: Fn(&T, &T) -> Ordering,
{
    let n = arr.len();
    let mut pending_right: Vec<usize> = Vec::new();
    let mut left_bound = seg.left_bound;

    for &(idx, dir) in &seg.items {
        match dir {
            Direction::Left => {
                // Fix all pending RIGHTs first
                let mut right_bound = idx.saturating_sub(1);
                while let Some(pending_idx) = pending_right.pop() {
                    if pending_idx < n - 1
                        && cmp(&arr[pending_idx], &arr[pending_idx + 1]) == Ordering::Greater
                    {
                        right_bound = fix_right(arr, pending_idx, right_bound, cmp).saturating_sub(1);
                    }
                }
                // Fix LEFT
                left_bound = fix_left(arr, idx, left_bound, cmp) + 1;
            }
            Direction::Right => {
                if pending_right.is_empty() {
                    left_bound = idx;
                }
                pending_right.push(idx);
            }
        }
    }

    // Final flush for this segment
    let mut right_bound = seg.right_bound;
    while let Some(pending_idx) = pending_right.pop() {
        if pending_idx < n - 1
            && cmp(&arr[pending_idx], &arr[pending_idx + 1]) == Ordering::Greater
        {
            right_bound = fix_right(arr, pending_idx, right_bound, cmp).saturating_sub(1);
        }
    }
}

/// Fix segments in parallel using rayon.
/// 
/// Each segment operates on a non-overlapping slice of the array.
fn fix_segments_in_parallel<T, F>(arr: &mut [T], segments: &[Segment], cmp: &F)
where
    T: Send + Sync,
    F: Fn(&T, &T) -> Ordering + Sync + Send,
{
    // Wrap pointer for Send+Sync
    // Safety: segments have non-overlapping [left_bound, right_bound] ranges
    let arr_ptr = SendPtr(arr.as_mut_ptr());
    let arr_len = arr.len();

    segments.par_iter().for_each(move |seg| {
        // Safety: each parallel task only accesses elements within [left_bound, right_bound]
        // which are guaranteed to be non-overlapping across segments
        let slice = unsafe {
            std::slice::from_raw_parts_mut(arr_ptr.as_ptr(), arr_len)
        };
        fix_segment_with_directions(slice, seg, cmp);
    });
}

/// Fix RIGHT: move element at i to its correct position within [i, right_bound]
fn fix_right<T, F>(arr: &mut [T], i: usize, right_bound: usize, cmp: &F) -> usize
where
    F: Fn(&T, &T) -> Ordering,
{
    let mut lo = i + 1;
    let mut hi = right_bound + 1;

    while lo < hi {
        let mid = lo + ((hi - lo) >> 1);
        if cmp(&arr[mid], &arr[i]) != Ordering::Greater {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    let target = lo - 1;
    if target > i {
        arr[i..=target].rotate_left(1);
    }
    target
}

/// Fix LEFT: move element at i to its correct position within [left_bound, i]
fn fix_left<T, F>(arr: &mut [T], i: usize, left_bound: usize, cmp: &F) -> usize
where
    F: Fn(&T, &T) -> Ordering,
{
    let mut lo = left_bound;
    let mut hi = i;

    while lo < hi {
        let mid = lo + ((hi - lo) >> 1);
        if cmp(&arr[i], &arr[mid]) == Ordering::Less {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    if lo < i {
        arr[lo..=i].rotate_right(1);
    }
    lo
}

/// Parallel pre-sort of dirty values using O(k) auxiliary space.
/// Extracts values, sorts with rayon, places back.
fn parallel_presort_scattered<T, F>(arr: &mut [T], indices: &[usize], cmp: &F)
where
    T: Clone + Send,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    // Extract values at dirty indices
    let mut values: Vec<T> = indices.iter().map(|&i| arr[i].clone()).collect();
    
    // Sort values using rayon
    values.par_sort_by(|a, b| cmp(a, b));
    
    // Place sorted values back at indices
    for (i, idx) in indices.iter().enumerate() {
        arr[*idx] = values[i].clone();
    }
}

/// Heapsort values at scattered positions via indirection.
fn heapsort_scattered<T, F>(arr: &mut [T], indices: &[usize], cmp: &F)
where
    F: Fn(&T, &T) -> Ordering,
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

fn sift_down_scattered<T, F>(
    arr: &mut [T],
    indices: &[usize],
    mut root: usize,
    end: usize,
    cmp: &F,
) where
    F: Fn(&T, &T) -> Ordering,
{
    loop {
        let mut child = 2 * root + 1;
        if child >= end {
            break;
        }
        if child + 1 < end && cmp(&arr[indices[child]], &arr[indices[child + 1]]) == Ordering::Less
        {
            child += 1;
        }
        if cmp(&arr[indices[root]], &arr[indices[child]]) != Ordering::Less {
            break;
        }
        arr.swap(indices[root], indices[child]);
        root = child;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_empty() {
        let mut arr: Vec<i32> = vec![1, 2, 3];
        let updated: HashSet<usize> = HashSet::new();
        parallel_delta_sort(&mut arr, &updated);
        assert_eq!(arr, vec![1, 2, 3]);
    }

    #[test]
    fn test_single_left() {
        // Start with sorted array, update one element to smaller value
        let mut arr = vec![1, 3, 5, 7, 9];
        arr[2] = 2; // 5 -> 2, should move left
        let updated: HashSet<usize> = [2].into_iter().collect();
        parallel_delta_sort(&mut arr, &updated);
        assert_eq!(arr, vec![1, 2, 3, 7, 9]);
    }

    #[test]
    fn test_single_right() {
        // Start with sorted array, update one element to larger value
        let mut arr = vec![1, 3, 5, 7, 9];
        arr[2] = 8; // 5 -> 8, should move right
        let updated: HashSet<usize> = [2].into_iter().collect();
        parallel_delta_sort(&mut arr, &updated);
        assert_eq!(arr, vec![1, 3, 7, 8, 9]);
    }

    #[test]
    fn test_two_updates() {
        let mut arr = vec![1, 8, 5, 2, 9];
        let updated: HashSet<usize> = [1, 3].into_iter().collect();
        parallel_delta_sort(&mut arr, &updated);
        assert_eq!(arr, vec![1, 2, 5, 8, 9]);
    }

    #[test]
    fn test_randomized() {
        let mut rng = rand::thread_rng();
        for trial in 0..100 {
            let n = rng.gen_range(10..500);
            let k = rng.gen_range(1..n.min(50));

            // Create a sorted array (this is the precondition for DeltaSort)
            let mut arr: Vec<i32> = (0..n as i32).collect();
            let mut updated = HashSet::new();

            // Modify k random positions
            for _ in 0..k {
                let idx = rng.gen_range(0..n);
                arr[idx] = rng.gen_range(0..n as i32);
                updated.insert(idx);
            }

            let mut expected = arr.clone();
            expected.sort();

            parallel_delta_sort(&mut arr, &updated);
            assert_eq!(arr, expected, "Failed at trial {}, n={}, k={}", trial, n, k);
        }
    }

    #[test]
    fn test_all_dirty() {
        let mut arr = vec![5, 4, 3, 2, 1];
        let updated: HashSet<usize> = (0..5).collect();
        parallel_delta_sort(&mut arr, &updated);
        assert_eq!(arr, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_paper_example() {
        // Classic DeltaSort paper example: values that would cross but pre-sorting avoids it
        let mut arr = vec![1, 8, 5, 2, 9];
        let updated: HashSet<usize> = [1, 3].into_iter().collect();
        parallel_delta_sort(&mut arr, &updated);
        assert_eq!(arr, vec![1, 2, 5, 8, 9]);
    }
}
