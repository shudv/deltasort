use crate::data::User;
use std::collections::HashSet;

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
