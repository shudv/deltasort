use crate::data::User;

/// In-place binary insertion sort — O(1) auxiliary space, Θ(kn) time.
///
/// 1. Sort dirty indices descending (modifies the input slice).
/// 2. Remove each dirty element right-to-left via rotation, collecting at tail.
/// 3. Binary-insert each collected element back.
///
/// Not stable: equal-keyed elements may not preserve original index order.
///
/// Precondition: dirty_indices contains distinct valid indices into arr.
pub fn binary_insertion_sort<F>(
    arr: &mut Vec<User>,
    dirty_indices: &mut [usize],
    cmp: F,
)
where
    F: Fn(&User, &User) -> std::cmp::Ordering,
{
    if dirty_indices.is_empty() {
        return;
    }

    let n = arr.len();
    let k = dirty_indices.len();

    // Sort dirty indices descending so removals don't shift unprocessed positions
    dirty_indices.sort_unstable_by(|a, b| b.cmp(a));
    debug_assert!(
        dirty_indices.windows(2).all(|w| w[0] > w[1]),
        "dirty indices must be distinct"
    );

    // Phase 1: Remove dirty elements right-to-left, collecting at tail — O(kn)
    let mut clean_end = n;
    for &idx in dirty_indices.iter() {
        arr[idx..clean_end].rotate_left(1);
        clean_end -= 1;
    }
    // arr[0..n-k] = sorted clean elements, arr[n-k..n] = dirty elements

    // Phase 2: Binary insert each dirty element — O(kn)
    let clean_len = n - k;
    for i in 0..k {
        let sorted_len = clean_len + i;
        let pos = arr[..sorted_len]
            .partition_point(|x| cmp(x, &arr[sorted_len]) == std::cmp::Ordering::Less);
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

    fn sample_distinct(rng: &mut impl rand::Rng, n: usize, k: usize) -> Vec<usize> {
        let mut pool: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = rng.gen_range(i..n);
            pool.swap(i, j);
        }
        pool[..k].to_vec()
    }

    #[test]
    fn test_small_k() {
        let mut rng = rand::thread_rng();
        for _ in 0..50 {
            let n = 100;
            let k = rng.gen_range(1..=10);
            let mut arr: Vec<User> = (0..n as u32).map(make_user).collect();
            let mut dirty = sample_distinct(&mut rng, n, k);
            for &idx in &dirty {
                arr[idx] = make_user(rng.gen_range(0..n as u32 * 2));
            }
            binary_insertion_sort(&mut arr, &mut dirty, user_cmp);
            assert!(is_sorted(&arr));
        }
    }

    #[test]
    fn test_large_k() {
        let mut rng = rand::thread_rng();
        for _ in 0..20 {
            let n = 1000;
            let k = rng.gen_range(50..=200);
            let mut arr: Vec<User> = (0..n as u32).map(make_user).collect();
            let mut dirty = sample_distinct(&mut rng, n, k);
            for &idx in &dirty {
                arr[idx] = make_user(rng.gen_range(0..n as u32 * 2));
            }
            binary_insertion_sort(&mut arr, &mut dirty, user_cmp);
            assert!(is_sorted(&arr));
        }
    }

    // ===== Original tests =====

    #[test]
    fn test_empty_dirty_indices() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        let mut dirty: Vec<usize> = vec![];
        binary_insertion_sort(&mut arr, &mut dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_single_dirty_index_move_left() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[5] = make_user(2); // 5 -> 2, should move left
        let mut dirty: Vec<usize> = vec![5];
        binary_insertion_sort(&mut arr, &mut dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_single_dirty_index_move_right() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[2] = make_user(8); // 2 -> 8, should move right
        let mut dirty: Vec<usize> = vec![2];
        binary_insertion_sort(&mut arr, &mut dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_multiple_dirty_indices() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[1] = make_user(9);
        arr[3] = make_user(0);
        arr[7] = make_user(5);
        let mut dirty: Vec<usize> = vec![1, 3, 7];
        binary_insertion_sort(&mut arr, &mut dirty, user_cmp);
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
        let mut dirty: Vec<usize> = (0..5).collect();
        binary_insertion_sort(&mut arr, &mut dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_adjacent_dirty_indices() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[4] = make_user(8);
        arr[5] = make_user(2);
        let mut dirty: Vec<usize> = vec![4, 5];
        binary_insertion_sort(&mut arr, &mut dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_boundary_indices() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[0] = make_user(5);
        arr[9] = make_user(3);
        let mut dirty: Vec<usize> = vec![0, 9];
        binary_insertion_sort(&mut arr, &mut dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_duplicates() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[2] = make_user(5);
        arr[7] = make_user(5);
        let mut dirty: Vec<usize> = vec![2, 7];
        binary_insertion_sort(&mut arr, &mut dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_randomized_small() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let n = 20;
            let k = rng.gen_range(1..=n);
            let mut arr: Vec<User> = (0..n as u32).map(make_user).collect();
            let mut dirty = sample_distinct(&mut rng, n, k);
            for &idx in &dirty {
                arr[idx] = make_user(rng.gen_range(0..n as u32 * 2));
            }
            binary_insertion_sort(&mut arr, &mut dirty, user_cmp);
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
            let mut dirty = sample_distinct(&mut rng, n, k);
            for &idx in &dirty {
                arr[idx] = make_user(rng.gen_range(0..n as u32 * 2));
            }
            binary_insertion_sort(&mut arr, &mut dirty, user_cmp);
            assert!(is_sorted(&arr));
        }
    }

    #[test]
    fn test_preserves_all_values() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[2] = make_user(15);
        arr[5] = make_user(20);
        let mut dirty: Vec<usize> = vec![2, 5];

        let mut expected_ages: Vec<u32> = arr.iter().map(|u| u.age).collect();
        expected_ages.sort();

        binary_insertion_sort(&mut arr, &mut dirty, user_cmp);

        let actual_ages: Vec<u32> = arr.iter().map(|u| u.age).collect();
        assert_eq!(actual_ages, expected_ages);
    }
}
