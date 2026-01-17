use crate::data::User;
use std::collections::HashSet;

/// Original binary insertion sort using Vec::remove and Vec::insert
pub fn binary_insertion_sort(
    arr: &mut Vec<User>,
    dirty_indices: &HashSet<usize>,
    user_comparator: fn(&User, &User) -> std::cmp::Ordering,
) {
    if dirty_indices.is_empty() {
        return;
    }

    // Sort indices descending for back-to-front extraction (indices stay valid)
    let mut sorted_desc: Vec<usize> = dirty_indices.iter().copied().collect();
    sorted_desc.sort_unstable_by(|a, b| b.cmp(a));

    // Extract dirty values from back to front - O(kn) but cache-friendly
    // NOTE: Auxiliary space is still O(1) (and not O(k)) because we put a value in extracted only after its removed
    let mut extracted: Vec<User> = Vec::with_capacity(sorted_desc.len());
    for &idx in &sorted_desc {
        extracted.push(arr.remove(idx));
    }

    // Binary insert each dirty value - O(kn)
    for value in extracted {
        let pos = arr.partition_point(|x| user_comparator(x, &value) == std::cmp::Ordering::Less);
        arr.insert(pos, value);
    }
}

/// Partition-and-rotate binary insertion sort
/// 1. Partition: move dirty elements to end while keeping clean elements sorted - O(n)
/// 2. Binary insert each dirty element using rotation - O(kn)
pub fn binary_insertion_sort_rotate(
    arr: &mut [User],
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
    // No need to sort dirty elements first - we just insert them one by one
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

/// Hybrid binary insertion sort
/// Uses original (extract + insert) for small k, rotate version for large k
/// Crossover point is ~10 based on benchmarks
const HYBRID_THRESHOLD: usize = 10;

pub fn binary_insertion_sort_hybrid(
    arr: &mut Vec<User>,
    dirty_indices: &HashSet<usize>,
    user_comparator: fn(&User, &User) -> std::cmp::Ordering,
) {
    if dirty_indices.len() <= HYBRID_THRESHOLD {
        binary_insertion_sort(arr, dirty_indices, user_comparator);
    } else {
        binary_insertion_sort_rotate(arr, dirty_indices, user_comparator);
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

    // ===== Tests for hybrid version =====

    #[test]
    fn test_hybrid_small_k() {
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
            binary_insertion_sort_hybrid(&mut arr, &dirty, user_cmp);
            assert!(is_sorted(&arr));
        }
    }

    #[test]
    fn test_hybrid_large_k() {
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
            binary_insertion_sort_hybrid(&mut arr, &dirty, user_cmp);
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

    // ===== Tests for binary_insertion_sort_rotate =====

    #[test]
    fn test_rotate_empty_dirty_indices() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        let dirty: HashSet<usize> = HashSet::new();
        binary_insertion_sort_rotate(&mut arr, &dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_rotate_single_dirty_index_move_left() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[5] = make_user(2);
        let dirty: HashSet<usize> = [5].into_iter().collect();
        binary_insertion_sort_rotate(&mut arr, &dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_rotate_single_dirty_index_move_right() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[2] = make_user(8);
        let dirty: HashSet<usize> = [2].into_iter().collect();
        binary_insertion_sort_rotate(&mut arr, &dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_rotate_multiple_dirty_indices() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[1] = make_user(9);
        arr[3] = make_user(0);
        arr[7] = make_user(5);
        let dirty: HashSet<usize> = [1, 3, 7].into_iter().collect();
        binary_insertion_sort_rotate(&mut arr, &dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_rotate_all_indices_dirty() {
        let mut arr: Vec<User> = vec![
            make_user(9),
            make_user(3),
            make_user(7),
            make_user(1),
            make_user(5),
        ];
        let dirty: HashSet<usize> = (0..5).collect();
        binary_insertion_sort_rotate(&mut arr, &dirty, user_cmp);
        assert!(is_sorted(&arr));
    }

    #[test]
    fn test_rotate_users_example() {
        // User's example: 1 3 5 7 9 -> 1 8 5 0 9
        let mut arr: Vec<User> = vec![
            make_user(1),
            make_user(3),
            make_user(5),
            make_user(7),
            make_user(9),
        ];
        arr[1] = make_user(8); // 3 -> 8
        arr[3] = make_user(0); // 7 -> 0
        let dirty: HashSet<usize> = [1, 3].into_iter().collect();
        binary_insertion_sort_rotate(&mut arr, &dirty, user_cmp);
        assert!(is_sorted(&arr));
        // Should result in: 0 1 5 8 9
        let ages: Vec<u32> = arr.iter().map(|u| u.age).collect();
        assert_eq!(ages, vec![0, 1, 5, 8, 9]);
    }

    #[test]
    fn test_rotate_randomized_small() {
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
            binary_insertion_sort_rotate(&mut arr, &dirty, user_cmp);
            assert!(is_sorted(&arr), "Failed with dirty indices: {:?}", dirty);
        }
    }

    #[test]
    fn test_rotate_randomized_medium() {
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
            binary_insertion_sort_rotate(&mut arr, &dirty, user_cmp);
            assert!(is_sorted(&arr));
        }
    }

    #[test]
    fn test_rotate_preserves_all_values() {
        let mut arr: Vec<User> = (0..10).map(make_user).collect();
        arr[2] = make_user(15);
        arr[5] = make_user(20);
        let dirty: HashSet<usize> = [2, 5].into_iter().collect();

        let mut expected_ages: Vec<u32> = arr.iter().map(|u| u.age).collect();
        expected_ages.sort();

        binary_insertion_sort_rotate(&mut arr, &dirty, user_cmp);

        let actual_ages: Vec<u32> = arr.iter().map(|u| u.age).collect();
        assert_eq!(actual_ages, expected_ages);
    }

    #[test]
    fn test_both_versions_produce_same_result() {
        let mut rng = rand::thread_rng();
        for _ in 0..50 {
            let n = 100;
            let k = rng.gen_range(1..=20);

            let base: Vec<User> = (0..n as u32).map(make_user).collect();
            let mut arr1 = base.clone();
            let mut arr2 = base.clone();

            let mut dirty: HashSet<usize> = HashSet::new();
            for _ in 0..k {
                let idx = rng.gen_range(0..n);
                let new_val = rng.gen_range(0..n as u32 * 2);
                arr1[idx] = make_user(new_val);
                arr2[idx] = make_user(new_val);
                dirty.insert(idx);
            }

            binary_insertion_sort(&mut arr1, &dirty, user_cmp);
            binary_insertion_sort_rotate(&mut arr2, &dirty, user_cmp);

            let ages1: Vec<u32> = arr1.iter().map(|u| u.age).collect();
            let ages2: Vec<u32> = arr2.iter().map(|u| u.age).collect();
            assert_eq!(
                ages1, ages2,
                "Both versions should produce same sorted result"
            );
        }
    }

    #[test]
    fn benchmark_compare_versions() {
        use std::time::Instant;

        let mut rng = rand::thread_rng();
        let n = 10000;
        let iterations = 10;
        let k_values = [1, 10, 100, 200, 500, 1000, 2000];

        println!(
            "\n=== Binary Insertion Sort Benchmark (n={}, {} iterations) ===",
            n, iterations
        );
        println!(
            "{:>6} | {:>15} | {:>15} | {:>15} | {:>10}",
            "k", "Original (µs)", "Rotate (µs)", "Hybrid (µs)", "Best"
        );
        println!(
            "{:-<6}-+-{:-<15}-+-{:-<15}-+-{:-<15}-+-{:-<10}",
            "", "", "", "", ""
        );

        for &k in &k_values {
            // Prepare test data - create fresh copies for each algorithm
            let mut test_data: Vec<(Vec<User>, HashSet<usize>)> = Vec::with_capacity(iterations);

            for _ in 0..iterations {
                let mut arr: Vec<User> = (0..n as u32).map(make_user).collect();
                let mut dirty: HashSet<usize> = HashSet::new();

                while dirty.len() < k {
                    let idx = rng.gen_range(0..n);
                    arr[idx] = make_user(rng.gen_range(0..n as u32 * 2));
                    dirty.insert(idx);
                }

                test_data.push((arr, dirty));
            }

            // Clone for each algorithm to ensure fair comparison
            let mut original_inputs: Vec<_> = test_data
                .iter()
                .map(|(arr, dirty)| (arr.clone(), dirty.clone()))
                .collect();
            let mut rotate_inputs: Vec<_> = test_data
                .iter()
                .map(|(arr, dirty)| (arr.clone(), dirty.clone()))
                .collect();
            let mut hybrid_inputs: Vec<_> = test_data
                .iter()
                .map(|(arr, dirty)| (arr.clone(), dirty.clone()))
                .collect();

            // Benchmark original version
            let start = Instant::now();
            for (arr, dirty) in &mut original_inputs {
                binary_insertion_sort(arr, dirty, user_cmp);
            }
            let original_time = start.elapsed();

            // Benchmark rotate version
            let start = Instant::now();
            for (arr, dirty) in &mut rotate_inputs {
                binary_insertion_sort_rotate(arr, dirty, user_cmp);
            }
            let rotate_time = start.elapsed();

            // Benchmark hybrid version
            let start = Instant::now();
            for (arr, dirty) in &mut hybrid_inputs {
                binary_insertion_sort_hybrid(arr, dirty, user_cmp);
            }
            let hybrid_time = start.elapsed();

            // Verify all produce same results
            for i in 0..iterations {
                let ages1: Vec<u32> = original_inputs[i].0.iter().map(|u| u.age).collect();
                let ages2: Vec<u32> = rotate_inputs[i].0.iter().map(|u| u.age).collect();
                let ages3: Vec<u32> = hybrid_inputs[i].0.iter().map(|u| u.age).collect();
                assert_eq!(
                    ages1, ages2,
                    "Mismatch original vs rotate at iteration {}",
                    i
                );
                assert_eq!(
                    ages1, ages3,
                    "Mismatch original vs hybrid at iteration {}",
                    i
                );
            }

            let original_us = original_time.as_micros() as f64 / iterations as f64;
            let rotate_us = rotate_time.as_micros() as f64 / iterations as f64;
            let hybrid_us = hybrid_time.as_micros() as f64 / iterations as f64;

            let best = if hybrid_us <= original_us && hybrid_us <= rotate_us {
                "Hybrid"
            } else if original_us <= rotate_us {
                "Original"
            } else {
                "Rotate"
            };

            println!(
                "{:>6} | {:>15.2} | {:>15.2} | {:>15.2} | {:>10}",
                k, original_us, rotate_us, hybrid_us, best
            );
        }
        println!();
    }

    #[test]
    fn benchmark_compare_versions_small() {
        use std::time::Instant;

        let mut rng = rand::thread_rng();
        let n = 10000;
        let iterations = 50;
        let k_values = [1, 5, 10, 20, 50, 100, 500, 1000, 2000];

        println!(
            "\n=== Binary Insertion Sort Benchmark (n={}, {} iterations) ===",
            n, iterations
        );
        println!(
            "{:>6} | {:>15} | {:>15} | {:>15} | {:>10}",
            "k", "Original (µs)", "Rotate (µs)", "Hybrid (µs)", "Best"
        );
        println!(
            "{:-<6}-+-{:-<15}-+-{:-<15}-+-{:-<15}-+-{:-<10}",
            "", "", "", "", ""
        );

        for &k in &k_values {
            let mut test_data: Vec<(Vec<User>, HashSet<usize>)> = Vec::with_capacity(iterations);

            for _ in 0..iterations {
                let mut arr: Vec<User> = (0..n as u32).map(make_user).collect();
                let mut dirty: HashSet<usize> = HashSet::new();

                while dirty.len() < k {
                    let idx = rng.gen_range(0..n);
                    arr[idx] = make_user(rng.gen_range(0..n as u32 * 2));
                    dirty.insert(idx);
                }

                test_data.push((arr, dirty));
            }

            let mut original_inputs: Vec<_> = test_data
                .iter()
                .map(|(arr, dirty)| (arr.clone(), dirty.clone()))
                .collect();
            let mut rotate_inputs: Vec<_> = test_data
                .iter()
                .map(|(arr, dirty)| (arr.clone(), dirty.clone()))
                .collect();
            let mut hybrid_inputs: Vec<_> = test_data
                .iter()
                .map(|(arr, dirty)| (arr.clone(), dirty.clone()))
                .collect();

            let start = Instant::now();
            for (arr, dirty) in &mut original_inputs {
                binary_insertion_sort(arr, dirty, user_cmp);
            }
            let original_time = start.elapsed();

            let start = Instant::now();
            for (arr, dirty) in &mut rotate_inputs {
                binary_insertion_sort_rotate(arr, dirty, user_cmp);
            }
            let rotate_time = start.elapsed();

            let start = Instant::now();
            for (arr, dirty) in &mut hybrid_inputs {
                binary_insertion_sort_hybrid(arr, dirty, user_cmp);
            }
            let hybrid_time = start.elapsed();

            // Verify correctness
            for i in 0..iterations {
                let ages1: Vec<u32> = original_inputs[i].0.iter().map(|u| u.age).collect();
                let ages2: Vec<u32> = rotate_inputs[i].0.iter().map(|u| u.age).collect();
                let ages3: Vec<u32> = hybrid_inputs[i].0.iter().map(|u| u.age).collect();
                assert_eq!(
                    ages1, ages2,
                    "Mismatch original vs rotate at iteration {}",
                    i
                );
                assert_eq!(
                    ages1, ages3,
                    "Mismatch original vs hybrid at iteration {}",
                    i
                );
            }

            let original_us = original_time.as_micros() as f64 / iterations as f64;
            let rotate_us = rotate_time.as_micros() as f64 / iterations as f64;
            let hybrid_us = hybrid_time.as_micros() as f64 / iterations as f64;

            let best = if hybrid_us <= original_us && hybrid_us <= rotate_us {
                "Hybrid"
            } else if original_us <= rotate_us {
                "Original"
            } else {
                "Rotate"
            };

            println!(
                "{:>6} | {:>15.2} | {:>15.2} | {:>15.2} | {:>10}",
                k, original_us, rotate_us, hybrid_us, best
            );
        }
        println!();
    }
}
