use crate::data::User;
use std::collections::HashSet;

pub fn extract_sort_merge(
    arr: &mut Vec<User>,
    dirty_indices: &HashSet<usize>,
    user_comparator: fn(&User, &User) -> std::cmp::Ordering,
) {
    if dirty_indices.is_empty() {
        return;
    }

    let n = arr.len();
    let k = dirty_indices.len();

    // Sort indices ascending for single-pass extraction
    let mut sorted_indices: Vec<usize> = dirty_indices.iter().copied().collect();
    sorted_indices.sort_unstable();

    // Single O(n) pass with index comparison (no HashSet lookups)
    let mut clean_values: Vec<User> = Vec::with_capacity(n - k);
    let mut dirty_values: Vec<User> = Vec::with_capacity(k);
    let mut dirty_ptr = 0;

    for (i, val) in arr.drain(..).enumerate() {
        if dirty_ptr < k && sorted_indices[dirty_ptr] == i {
            dirty_values.push(val);
            dirty_ptr += 1;
        } else {
            clean_values.push(val);
        }
    }

    // Sort dirty values - O(k log k)
    dirty_values.sort_by(user_comparator);

    // Merge - O(n)
    let mut result: Vec<User> = Vec::with_capacity(n);
    let clean_len = clean_values.len();
    let dirty_len = dirty_values.len();
    let mut i = 0;
    let mut j = 0;

    while i < clean_len && j < dirty_len {
        if user_comparator(&clean_values[i], &dirty_values[j]) != std::cmp::Ordering::Greater {
            result.push(std::mem::take(&mut clean_values[i]));
            i += 1;
        } else {
            result.push(std::mem::take(&mut dirty_values[j]));
            j += 1;
        }
    }

    for item in clean_values.drain(i..) {
        result.push(item);
    }
    for item in dirty_values.drain(j..) {
        result.push(item);
    }

    *arr = result;
}
