use crate::data::User;
use std::collections::HashSet;

/// Extract-Sort-Merge with O(k) auxiliary space.
///
/// 1. Swap-partition: move clean elements to front (sorted), dirty to back. O(n), O(1) aux.
/// 2. Sort dirty tail in-place. O(k log k), O(1) aux.
/// 3. Copy dirty tail to O(k) buffer.
/// 4. Merge from right: clean prefix + buffer → final array. O(n).
///
/// Total: O(n + k log k) time, O(k) auxiliary space.
pub fn extract_sort_merge(
    arr: &mut Vec<User>,
    dirty_indices: &HashSet<usize>,
    user_comparator: fn(&User, &User) -> std::cmp::Ordering,
) {
    if dirty_indices.is_empty() {
        return;
    }

    let n = arr.len();

    // Step 1: Swap-partition. Clean elements to front (preserving sorted order),
    // dirty elements to back. O(n) time, O(1) auxiliary space.
    let mut write_pos = 0;
    for read_pos in 0..n {
        if !dirty_indices.contains(&read_pos) {
            if write_pos != read_pos {
                arr.swap(write_pos, read_pos);
            }
            write_pos += 1;
        }
    }
    let clean_count = write_pos;
    // arr[0..clean_count] = sorted clean elements
    // arr[clean_count..n] = dirty elements (unsorted)

    // Step 2: Sort dirty tail in-place. O(k log k).
    arr[clean_count..].sort_by(user_comparator);

    // Step 3: Copy dirty tail to buffer. O(k) auxiliary space.
    let buffer: Vec<User> = arr[clean_count..].to_vec();
    let k = buffer.len();

    // Step 4: Merge from right. O(n) time.
    // Write pointer starts k ahead of clean pointer, so no overwrites occur.
    let mut ci = clean_count; // next clean to consume (counts down)
    let mut di = k;           // next dirty to consume from buffer (counts down)
    let mut wi = n;           // next write position (counts down)

    while ci > 0 && di > 0 {
        if user_comparator(&arr[ci - 1], &buffer[di - 1]) == std::cmp::Ordering::Greater {
            wi -= 1;
            ci -= 1;
            arr.swap(wi, ci);
        } else {
            wi -= 1;
            di -= 1;
            arr[wi] = buffer[di].clone();
        }
    }

    // Remaining dirty values from buffer
    while di > 0 {
        wi -= 1;
        di -= 1;
        arr[wi] = buffer[di].clone();
    }
    // Remaining clean values are already in position (0..ci, and wi == ci here)
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
