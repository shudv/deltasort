use crate::data::User;

/// Extract-Sort-Merge using O(k) auxiliary space.
///
/// 1. Sort dirty indices, extract dirty values to a buffer, compact clean left: O(n)
/// 2. Sort dirty buffer: O(k log k)
/// 3. Merge backwards from the right end: O(n)
///
/// Takes unsorted dirty indices; sorts them internally.
pub fn extract_sort_merge<F>(arr: &mut [User], dirty_indices: &mut [usize], cmp: F)
where
    F: Fn(&User, &User) -> std::cmp::Ordering,
{
    if dirty_indices.is_empty() {
        return;
    }

    let n = arr.len();

    // Sort dirty indices for linear scanning — O(k log k)
    dirty_indices.sort_unstable();
    debug_assert!(
        dirty_indices.windows(2).all(|w| w[0] < w[1]),
        "dirty indices must be distinct"
    );
    let k = dirty_indices.len();

    // Step 1: Extract dirty to buffer, compact clean to the left — single O(n) pass
    let mut dirty_buf: Vec<User> = Vec::with_capacity(k);
    let mut write = 0;
    let mut di = 0; // index into dirty_indices
    for read in 0..n {
        if di < k && dirty_indices[di] == read {
            dirty_buf.push(std::mem::take(&mut arr[read]));
            di += 1;
        } else {
            if write != read {
                arr.swap(write, read);
            }
            write += 1;
        }
    }

    // Step 2: Sort dirty values
    dirty_buf.sort_unstable_by(&cmp);

    // Step 3: Backwards merge — pick larger of clean tail / dirty tail
    let mut ci = (n - k) as isize - 1;
    let mut di = k as isize - 1;
    let mut wi = n - 1;

    loop {
        let take_clean = if ci >= 0 && di >= 0 {
            cmp(&arr[ci as usize], &dirty_buf[di as usize]) != std::cmp::Ordering::Less
        } else {
            ci >= 0
        };

        if take_clean {
            if wi != ci as usize {
                arr.swap(wi, ci as usize);
            }
            ci -= 1;
        } else if di >= 0 {
            arr[wi] = std::mem::take(&mut dirty_buf[di as usize]);
            di -= 1;
        } else {
            break;
        }

        if wi == 0 {
            break;
        }
        wi -= 1;
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

    #[test]
    fn test_empty_dirty_indices() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        let mut dirty: Vec<usize> = vec![];
        extract_sort_merge(&mut arr, &mut dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_single_dirty_index_move_left() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[5] = make_user(2); // 5 -> 2, should move left
        let mut dirty: Vec<usize> = vec![5];
        extract_sort_merge(&mut arr, &mut dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_single_dirty_index_move_right() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[2] = make_user(8); // 2 -> 8, should move right
        let mut dirty: Vec<usize> = vec![2];
        extract_sort_merge(&mut arr, &mut dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_multiple_dirty_indices() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[1] = make_user(9);
        arr[3] = make_user(0);
        arr[7] = make_user(5);
        let mut dirty: Vec<usize> = vec![1, 3, 7];
        extract_sort_merge(&mut arr, &mut dirty, user_cmp);
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
        extract_sort_merge(&mut arr, &mut dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_adjacent_dirty_indices() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[4] = make_user(8);
        arr[5] = make_user(2);
        let mut dirty: Vec<usize> = vec![4, 5];
        extract_sort_merge(&mut arr, &mut dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_boundary_indices() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[0] = make_user(5);
        arr[9] = make_user(3);
        let mut dirty: Vec<usize> = vec![0, 9];
        extract_sort_merge(&mut arr, &mut dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_duplicates() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[2] = make_user(5);
        arr[7] = make_user(5);
        let mut dirty: Vec<usize> = vec![2, 7];
        extract_sort_merge(&mut arr, &mut dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    fn sample_distinct(rng: &mut impl rand::Rng, n: usize, k: usize) -> Vec<usize> {
        let mut pool: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = rng.gen_range(i..n);
            pool.swap(i, j);
        }
        pool[..k].to_vec()
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
            extract_sort_merge(&mut arr, &mut dirty, user_cmp);
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
            extract_sort_merge(&mut arr, &mut dirty, user_cmp);
            assert!(is_sorted(&arr));
        }
    }

    #[test]
    fn test_randomized_large_k() {
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            let n = 500;
            let k = rng.gen_range(n / 2..n);
            let mut arr: Vec<User> = (0..n as u32).map(make_user).collect();
            let mut dirty = sample_distinct(&mut rng, n, k);
            for &idx in &dirty {
                arr[idx] = make_user(rng.gen_range(0..n as u32 * 2));
            }
            extract_sort_merge(&mut arr, &mut dirty, user_cmp);
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

        extract_sort_merge(&mut arr, &mut dirty, user_cmp);

        let actual_ages: Vec<u32> = arr.iter().map(|u| u.age).collect();
        assert_eq!(actual_ages, expected_ages);
    }

    #[test]
    fn test_stability_with_clean_values() {
        // Verify clean values maintain their relative order
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[3] = make_user(100); // Only index 3 is dirty
        let mut dirty: Vec<usize> = vec![3];

        extract_sort_merge(&mut arr, &mut dirty, user_cmp);

        // Clean values should still be in their original sorted order
        assert!(is_sorted(&arr));
        // The dirty value should be at the end
        assert_eq!(arr[9].age, 100);
    }
}
