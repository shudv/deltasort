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

    #[test]
    fn test_empty_dirty_indices() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        let dirty: HashSet<usize> = HashSet::new();
        extract_sort_merge(&mut arr, &dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_single_dirty_index_move_left() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[5] = make_user(2); // 5 -> 2, should move left
        let dirty: HashSet<usize> = [5].into_iter().collect();
        extract_sort_merge(&mut arr, &dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_single_dirty_index_move_right() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[2] = make_user(8); // 2 -> 8, should move right
        let dirty: HashSet<usize> = [2].into_iter().collect();
        extract_sort_merge(&mut arr, &dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_multiple_dirty_indices() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[1] = make_user(9);
        arr[3] = make_user(0);
        arr[7] = make_user(5);
        let dirty: HashSet<usize> = [1, 3, 7].into_iter().collect();
        extract_sort_merge(&mut arr, &dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_all_indices_dirty() {
        let mut arr: Vec<User> = vec![
            make_user(9),
            make_user(3),
            make_user(7),
            make_user(1),
            make_user(5),
        ];
        let dirty: HashSet<usize> = (0..5).collect();
        extract_sort_merge(&mut arr, &dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_adjacent_dirty_indices() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[4] = make_user(8);
        arr[5] = make_user(2);
        let dirty: HashSet<usize> = [4, 5].into_iter().collect();
        extract_sort_merge(&mut arr, &dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_boundary_indices() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[0] = make_user(5);
        arr[9] = make_user(3);
        let dirty: HashSet<usize> = [0, 9].into_iter().collect();
        extract_sort_merge(&mut arr, &dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_duplicates() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[2] = make_user(5);
        arr[7] = make_user(5);
        let dirty: HashSet<usize> = [2, 7].into_iter().collect();
        extract_sort_merge(&mut arr, &dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_randomized_small() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let n = 20;
            let k = rng.gen_range(1..=n);
            let mut arr: Vec<User> = (0..n as u32).map(make_user).collect();
            let mut dirty: HashSet<usize> = HashSet::new();
            for _ in 0..k {
                let idx = rng.gen_range(0..n);
                arr[idx] = make_user(rng.gen_range(0..n as u32 * 2));
                dirty.insert(idx);
            }
            extract_sort_merge(&mut arr, &dirty, user_cmp);
            assert!(is_sorted(&arr), "Failed with dirty indices: {:?}", dirty);
        }
    }

    #[test]
    fn test_randomized_medium() {
        let mut rng = rand::thread_rng();
        for _ in 0..20 {
            let n = 1000;
            let k = rng.gen_range(1..=100);
            let mut arr: Vec<User> = (0..n as u32).map(make_user).collect();
            let mut dirty: HashSet<usize> = HashSet::new();
            for _ in 0..k {
                let idx = rng.gen_range(0..n);
                arr[idx] = make_user(rng.gen_range(0..n as u32 * 2));
                dirty.insert(idx);
            }
            extract_sort_merge(&mut arr, &dirty, user_cmp);
            assert!(is_sorted(&arr));
        }
    }

    #[test]
    fn test_randomized_large_k() {
        // ESM should handle large k well (its strength)
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            let n = 500;
            let k = rng.gen_range(n / 2..n); // 50-100% dirty
            let mut arr: Vec<User> = (0..n as u32).map(make_user).collect();
            let mut dirty: HashSet<usize> = HashSet::new();
            for _ in 0..k {
                let idx = rng.gen_range(0..n);
                arr[idx] = make_user(rng.gen_range(0..n as u32 * 2));
                dirty.insert(idx);
            }
            extract_sort_merge(&mut arr, &dirty, user_cmp);
            assert!(is_sorted(&arr));
        }
    }

    #[test]
    fn test_preserves_all_values() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[2] = make_user(15);
        arr[5] = make_user(20);
        let dirty: HashSet<usize> = [2, 5].into_iter().collect();

        let mut expected_ages: Vec<u32> = arr.iter().map(|u| u.age).collect();
        expected_ages.sort();

        extract_sort_merge(&mut arr, &dirty, user_cmp);

        let actual_ages: Vec<u32> = arr.iter().map(|u| u.age).collect();
        assert_eq!(actual_ages, expected_ages);
    }

    #[test]
    fn test_stability_with_clean_values() {
        // Verify clean values maintain their relative order
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[3] = make_user(100); // Only index 3 is dirty
        let dirty: HashSet<usize> = [3].into_iter().collect();

        extract_sort_merge(&mut arr, &dirty, user_cmp);

        // Clean values should still be in their original sorted order
        assert!(is_sorted(&arr));
        // The dirty value should be at the end
        assert_eq!(arr[9].age, 100);
    }
}
