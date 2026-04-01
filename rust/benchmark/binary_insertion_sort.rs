use crate::data::User;
use std::collections::HashSet;

/// In-place binary insertion sort using partition and rotation - O(1) auxiliary space.
/// 1. Partition: move dirty elements to end while keeping clean elements sorted - O(n)
/// 2. Binary insert each dirty element using rotation - O(kn)
pub fn binary_insertion_sort(
    arr: &mut Vec<User>,
    dirty_indices: &HashSet<usize>,
    user_comparator: fn(&User, &User) -> std::cmp::Ordering,
) {
    if dirty_indices.is_empty() {
        return;
    }

    let n = arr.len();
    let k = dirty_indices.len();

    // Step 1: Partition - collect dirty to end while maintaining clean order
    // Single O(n) pass instead of k separate O(n) removals
    let mut write_pos = 0;
    for read_pos in 0..n {
        if !dirty_indices.contains(&read_pos) {
            if write_pos != read_pos {
                arr.swap(write_pos, read_pos);
            }
            write_pos += 1;
        }
    }
    // Now arr[0..n-k] contains clean elements in sorted order
    // arr[n-k..n] contains dirty elements in arbitrary order

    // Step 2: Binary insert each dirty element using rotation - O(kn)
    let clean_len = n - k;
    for i in 0..k {
        // Current dirty element is at position clean_len + i
        // The sorted portion is arr[0..clean_len + i]
        let sorted_len = clean_len + i;

        // Binary search in the sorted portion
        let pos = arr[..sorted_len]
            .partition_point(|x| user_comparator(x, &arr[sorted_len]) == std::cmp::Ordering::Less);

        // Rotate to insert: [sorted_part | element] -> insert element at pos
        // This is equivalent to insert but done in-place via rotation
        arr[pos..=sorted_len].rotate_right(1);
    }
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

    // ===== Correctness tests =====

    #[test]
    fn test_small_k() {
        let mut rng = rand::thread_rng();
        for _ in 0..50 {
            let n = 100;
            let k = rng.gen_range(1..=10); // Small k
            let mut arr: Vec<User> = (0..n as u32).map(make_user).collect();
            let mut dirty: HashSet<usize> = HashSet::new();
            while dirty.len() < k {
                let idx = rng.gen_range(0..n);
                arr[idx] = make_user(rng.gen_range(0..n as u32 * 2));
                dirty.insert(idx);
            }
            binary_insertion_sort(&mut arr, &dirty, user_cmp);
            assert!(is_sorted(&arr));
        }
    }

    #[test]
    fn test_large_k() {
        let mut rng = rand::thread_rng();
        for _ in 0..20 {
            let n = 1000;
            let k = rng.gen_range(50..=200); // Large k
            let mut arr: Vec<User> = (0..n as u32).map(make_user).collect();
            let mut dirty: HashSet<usize> = HashSet::new();
            while dirty.len() < k {
                let idx = rng.gen_range(0..n);
                arr[idx] = make_user(rng.gen_range(0..n as u32 * 2));
                dirty.insert(idx);
            }
            binary_insertion_sort(&mut arr, &dirty, user_cmp);
            assert!(is_sorted(&arr));
        }
    }

    // ===== Original tests =====

    #[test]
    fn test_empty_dirty_indices() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        let dirty: HashSet<usize> = HashSet::new();
        binary_insertion_sort(&mut arr, &dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_single_dirty_index_move_left() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[5] = make_user(2); // 5 -> 2, should move left
        let dirty: HashSet<usize> = [5].into_iter().collect();
        binary_insertion_sort(&mut arr, &dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_single_dirty_index_move_right() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[2] = make_user(8); // 2 -> 8, should move right
        let dirty: HashSet<usize> = [2].into_iter().collect();
        binary_insertion_sort(&mut arr, &dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_multiple_dirty_indices() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[1] = make_user(9);
        arr[3] = make_user(0);
        arr[7] = make_user(5);
        let dirty: HashSet<usize> = [1, 3, 7].into_iter().collect();
        binary_insertion_sort(&mut arr, &dirty, user_cmp);
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
        binary_insertion_sort(&mut arr, &dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_adjacent_dirty_indices() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[4] = make_user(8);
        arr[5] = make_user(2);
        let dirty: HashSet<usize> = [4, 5].into_iter().collect();
        binary_insertion_sort(&mut arr, &dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_boundary_indices() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[0] = make_user(5);
        arr[9] = make_user(3);
        let dirty: HashSet<usize> = [0, 9].into_iter().collect();
        binary_insertion_sort(&mut arr, &dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_duplicates() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[2] = make_user(5);
        arr[7] = make_user(5);
        let dirty: HashSet<usize> = [2, 7].into_iter().collect();
        binary_insertion_sort(&mut arr, &dirty, user_cmp);
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
            binary_insertion_sort(&mut arr, &dirty, user_cmp);
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
            binary_insertion_sort(&mut arr, &dirty, user_cmp);
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

        binary_insertion_sort(&mut arr, &dirty, user_cmp);

        let actual_ages: Vec<u32> = arr.iter().map(|u| u.age).collect();
        assert_eq!(actual_ages, expected_ages);
    }
}
