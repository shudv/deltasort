//! Integration tests for DeltaSort correctness.
//!
//! These tests verify that DeltaSort produces correct results across
//! various array sizes and delta volumes using randomized testing.

use deltasort::delta_sort_by;
use rand::Rng;
use std::collections::HashSet;

/// Delta volumes as percentages of array size
const DELTA_VOLUMES: [usize; 7] = [0, 1, 5, 10, 20, 50, 80];

/// Number of iterations per configuration
const ITERATIONS: usize = 100;

#[test]
fn test_correctness_scale_10() {
    run_correctness_tests(1);
}

#[test]
fn test_correctness_scale_100() {
    run_correctness_tests(2);
}

#[test]
fn test_correctness_scale_1000() {
    run_correctness_tests(3);
}

#[test]
fn test_correctness_scale_10000() {
    run_correctness_tests(4);
}

fn run_correctness_tests(scale: u32) {
    let mut rng = rand::thread_rng();
    let size = 10_usize.pow(scale);

    for &delta_volume in &DELTA_VOLUMES {
        for iter in 0..ITERATIONS {
            // Calculate delta count: minimum 1, otherwise volume proportional to size
            let delta_count = if delta_volume == 0 {
                0
            } else {
                (delta_volume * size / 100).max(1)
            };

            // Create a sorted array
            let mut arr: Vec<i32> = (0..size as i32).collect();
            let mut dirty_indices = HashSet::new();

            // Randomly modify delta_count values
            for _ in 0..delta_count {
                let idx = rng.gen_range(0..size);
                arr[idx] = rng.gen_range(0..size as i32);
                dirty_indices.insert(idx);
            }

            // Create expected result via native sort
            let mut expected = arr.clone();
            expected.sort();

            // Sort with DeltaSort
            delta_sort_by(&mut arr, &dirty_indices, |a, b| a.cmp(b));

            // Verify correctness
            assert_eq!(
                arr, expected,
                "Mismatch at scale=10^{}, delta_volume={}%, iteration={}",
                scale, delta_volume, iter
            );
        }
    }
}

#[test]
fn test_no_dirty_indices_is_noop() {
    // DeltaSort should not modify array when no dirty indices are provided
    let mut arr = vec![1, 2, 3, 2, 1];
    let dirty: HashSet<usize> = HashSet::new();
    
    delta_sort_by(&mut arr, &dirty, |a, b| a.cmp(b));
    
    assert_eq!(arr, vec![1, 2, 3, 2, 1]);
}

#[test]
fn test_single_dirty_index() {
    let mut arr = vec![1, 2, 3, 4, 5];
    arr[2] = 10; // Change middle element
    
    let dirty: HashSet<usize> = [2].into_iter().collect();
    delta_sort_by(&mut arr, &dirty, |a, b| a.cmp(b));
    
    assert_eq!(arr, vec![1, 2, 4, 5, 10]);
}

#[test]
fn test_all_indices_dirty() {
    let mut rng = rand::thread_rng();
    
    for size in [5, 10, 50, 100] {
        let mut arr: Vec<i32> = (0..size).map(|_| rng.gen_range(0..1000)).collect();
        let dirty: HashSet<usize> = (0..size as usize).collect();
        
        let mut expected = arr.clone();
        expected.sort();
        
        delta_sort_by(&mut arr, &dirty, |a, b| a.cmp(b));
        
        assert_eq!(arr, expected, "Failed with all {} indices dirty", size);
    }
}

#[test]
fn test_adjacent_dirty_indices() {
    let mut arr = vec![1, 5, 4, 3, 2, 6];
    let dirty: HashSet<usize> = [1, 2, 3, 4].into_iter().collect();
    
    delta_sort_by(&mut arr, &dirty, |a, b| a.cmp(b));
    
    assert_eq!(arr, vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_edges_dirty() {
    let mut arr = vec![10, 2, 3, 4, 1];
    let dirty: HashSet<usize> = [0, 4].into_iter().collect();
    
    delta_sort_by(&mut arr, &dirty, |a, b| a.cmp(b));
    
    assert_eq!(arr, vec![1, 2, 3, 4, 10]);
}

#[test]
fn test_with_duplicates() {
    let mut arr = vec![1, 3, 3, 5, 3];
    let dirty: HashSet<usize> = [4].into_iter().collect();
    
    delta_sort_by(&mut arr, &dirty, |a, b| a.cmp(b));
    
    assert_eq!(arr, vec![1, 3, 3, 3, 5]);
}

#[test]
fn test_descending_comparator() {
    let mut arr = vec![1, 2, 3, 4, 5];
    let dirty: HashSet<usize> = [0, 1, 2, 3, 4].into_iter().collect();
    
    // Sort in descending order
    delta_sort_by(&mut arr, &dirty, |a, b| b.cmp(a));
    
    assert_eq!(arr, vec![5, 4, 3, 2, 1]);
}

#[test]
fn test_string_sorting() {
    let mut arr = vec!["banana", "apple", "cherry", "date"];
    let dirty: HashSet<usize> = [0, 1].into_iter().collect();
    
    delta_sort_by(&mut arr, &dirty, |a, b| a.cmp(b));
    
    assert_eq!(arr, vec!["apple", "banana", "cherry", "date"]);
}

#[test]
fn test_struct_with_complex_comparator() {
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Person {
        name: String,
        age: u32,
    }

    let mut people = vec![
        Person { name: "Alice".into(), age: 30 },
        Person { name: "Bob".into(), age: 25 },
        Person { name: "Charlie".into(), age: 35 },
        Person { name: "Diana".into(), age: 28 },
    ];

    // Modify Bob and Diana
    people[1].age = 40;
    people[3].age = 20;

    let dirty: HashSet<usize> = [1, 3].into_iter().collect();
    
    // Sort by age ascending
    delta_sort_by(&mut people, &dirty, |a, b| a.age.cmp(&b.age));

    let ages: Vec<u32> = people.iter().map(|p| p.age).collect();
    assert_eq!(ages, vec![20, 30, 35, 40]);
}

#[test]
fn test_movement_cancellation_example() {
    // This is the example from the paper where pre-sorting eliminates movement
    let mut arr = vec![1, 8, 5, 2, 9];
    let dirty: HashSet<usize> = [1, 3].into_iter().collect();
    
    delta_sort_by(&mut arr, &dirty, |a, b| a.cmp(b));
    
    assert_eq!(arr, vec![1, 2, 5, 8, 9]);
}
