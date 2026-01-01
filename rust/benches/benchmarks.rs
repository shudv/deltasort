//! Benchmarks for DeltaSort comparing against baseline algorithms.
//!
//! Run with: `cargo bench`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use deltasort::deltasort;
use rand::Rng;
use std::collections::HashSet;

// ============================================================================
// BENCHMARK DATA
// ============================================================================

const COUNTRIES: &[&str] = &[
    "USA", "Canada", "UK", "Germany", "France", "Spain", "Italy", "Japan",
    "Australia", "Brazil", "India", "China", "Mexico", "Argentina", "Sweden",
];

const FIRST_NAMES: &[&str] = &[
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael",
    "Linda", "William", "Elizabeth", "David", "Barbara", "Richard", "Susan",
];

const LAST_NAMES: &[&str] = &[
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
];

#[derive(Clone, Debug)]
struct User {
    name: String,
    age: u32,
    country: String,
}

impl User {
    fn generate(seed: usize) -> Self {
        let first_idx = seed % FIRST_NAMES.len();
        let last_idx = (seed / FIRST_NAMES.len()) % LAST_NAMES.len();
        let country_idx = (seed / (FIRST_NAMES.len() * LAST_NAMES.len())) % COUNTRIES.len();

        User {
            name: format!("{} {}", FIRST_NAMES[first_idx], LAST_NAMES[last_idx]),
            age: 18 + (seed % 62) as u32,
            country: COUNTRIES[country_idx].to_string(),
        }
    }

    fn mutate(&mut self, rng: &mut impl Rng, n: usize) {
        let field = rng.gen_range(0..3);
        match field {
            0 => {
                let first = FIRST_NAMES[rng.gen_range(0..FIRST_NAMES.len())];
                let last = LAST_NAMES[rng.gen_range(0..LAST_NAMES.len())];
                self.name = format!("{} {}", first, last);
            }
            1 => self.age = 18 + rng.gen_range(0..62),
            _ => self.country = COUNTRIES[rng.gen_range(0..COUNTRIES.len())].to_string(),
        }
    }
}

fn user_comparator(a: &User, b: &User) -> std::cmp::Ordering {
    a.country
        .cmp(&b.country)
        .then_with(|| a.age.cmp(&b.age))
        .then_with(|| a.name.cmp(&b.name))
}

fn generate_sorted_users(n: usize) -> Vec<User> {
    let mut users: Vec<User> = (0..n).map(User::generate).collect();
    users.sort_by(user_comparator);
    users
}

// ============================================================================
// BASELINE ALGORITHMS
// ============================================================================

/// Binary Insertion: Extract dirty elements, then insert each at correct position.
fn binary_insertion_sort<T, F>(arr: &mut Vec<T>, dirty_indices: &HashSet<usize>, cmp: F)
where
    T: Clone,
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    // Extract in descending order to preserve indices
    let mut sorted_desc: Vec<usize> = dirty_indices.iter().copied().collect();
    sorted_desc.sort_by(|a, b| b.cmp(a));

    let mut elements: Vec<T> = Vec::with_capacity(sorted_desc.len());
    for &idx in &sorted_desc {
        elements.push(arr.remove(idx));
    }

    // Insert each element at correct position
    for value in elements {
        let pos = binary_search_position(arr, &value, &cmp);
        arr.insert(pos, value);
    }
}

/// Extract-Sort-Merge: Extract dirty elements, sort them, merge with clean.
fn extract_sort_merge<T, F>(arr: &mut Vec<T>, dirty_indices: &HashSet<usize>, cmp: F)
where
    T: Clone,
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    // Extract in descending order to preserve indices
    let mut sorted_desc: Vec<usize> = dirty_indices.iter().copied().collect();
    sorted_desc.sort_by(|a, b| b.cmp(a));

    let mut dirty_elements: Vec<T> = Vec::with_capacity(sorted_desc.len());
    for &idx in &sorted_desc {
        dirty_elements.push(arr.remove(idx));
    }

    // Sort dirty elements
    dirty_elements.sort_by(&cmp);

    // Merge
    let mut result: Vec<T> = Vec::with_capacity(arr.len() + dirty_elements.len());
    let mut i = 0;
    let mut j = 0;

    while i < arr.len() && j < dirty_elements.len() {
        if cmp(&arr[i], &dirty_elements[j]) != std::cmp::Ordering::Greater {
            result.push(arr[i].clone());
            i += 1;
        } else {
            result.push(dirty_elements[j].clone());
            j += 1;
        }
    }

    while i < arr.len() {
        result.push(arr[i].clone());
        i += 1;
    }

    while j < dirty_elements.len() {
        result.push(dirty_elements[j].clone());
        j += 1;
    }

    *arr = result;
}

fn binary_search_position<T, F>(arr: &[T], value: &T, cmp: &F) -> usize
where
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    let mut lo = 0;
    let mut hi = arr.len();

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if cmp(value, &arr[mid]) == std::cmp::Ordering::Less {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    lo
}

// ============================================================================
// BENCHMARKS
// ============================================================================

fn benchmark_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("sorting_algorithms");
    
    let n = 50_000;
    let delta_counts = [1, 5, 10, 20, 50, 100, 200];

    for &delta_count in &delta_counts {
        let mut rng = rand::thread_rng();

        // Prepare test data
        let base_users = generate_sorted_users(n);

        // Native sort
        group.bench_with_input(
            BenchmarkId::new("native", delta_count),
            &delta_count,
            |b, &delta_count| {
                b.iter_batched(
                    || {
                        let mut users = base_users.clone();
                        let mut dirty = HashSet::new();
                        for _ in 0..delta_count {
                            let idx = rng.gen_range(0..n);
                            users[idx].mutate(&mut rng, n);
                            dirty.insert(idx);
                        }
                        users
                    },
                    |mut users| {
                        users.sort_by(user_comparator);
                        black_box(users)
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        // Binary insertion
        group.bench_with_input(
            BenchmarkId::new("binary_insertion", delta_count),
            &delta_count,
            |b, &delta_count| {
                b.iter_batched(
                    || {
                        let mut users = base_users.clone();
                        let mut dirty = HashSet::new();
                        for _ in 0..delta_count {
                            let idx = rng.gen_range(0..n);
                            users[idx].mutate(&mut rng, n);
                            dirty.insert(idx);
                        }
                        (users, dirty)
                    },
                    |(mut users, dirty)| {
                        binary_insertion_sort(&mut users, &dirty, user_comparator);
                        black_box(users)
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        // Extract-Sort-Merge
        group.bench_with_input(
            BenchmarkId::new("extract_sort_merge", delta_count),
            &delta_count,
            |b, &delta_count| {
                b.iter_batched(
                    || {
                        let mut users = base_users.clone();
                        let mut dirty = HashSet::new();
                        for _ in 0..delta_count {
                            let idx = rng.gen_range(0..n);
                            users[idx].mutate(&mut rng, n);
                            dirty.insert(idx);
                        }
                        (users, dirty)
                    },
                    |(mut users, dirty)| {
                        extract_sort_merge(&mut users, &dirty, user_comparator);
                        black_box(users)
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        // DeltaSort
        group.bench_with_input(
            BenchmarkId::new("deltasort", delta_count),
            &delta_count,
            |b, &delta_count| {
                b.iter_batched(
                    || {
                        let mut users = base_users.clone();
                        let mut dirty = HashSet::new();
                        for _ in 0..delta_count {
                            let idx = rng.gen_range(0..n);
                            users[idx].mutate(&mut rng, n);
                            dirty.insert(idx);
                        }
                        (users, dirty)
                    },
                    |(mut users, dirty)| {
                        deltasort(&mut users, &dirty, user_comparator);
                        black_box(users)
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn benchmark_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_with_n");
    
    let sizes = [10_000, 50_000, 100_000];
    let delta_count = 50;

    for &n in &sizes {
        let mut rng = rand::thread_rng();
        let base_users = generate_sorted_users(n);

        group.bench_with_input(
            BenchmarkId::new("deltasort", n),
            &n,
            |b, &n| {
                b.iter_batched(
                    || {
                        let mut users = base_users.clone();
                        let mut dirty = HashSet::new();
                        for _ in 0..delta_count {
                            let idx = rng.gen_range(0..n);
                            users[idx].mutate(&mut rng, n);
                            dirty.insert(idx);
                        }
                        (users, dirty)
                    },
                    |(mut users, dirty)| {
                        deltasort(&mut users, &dirty, user_comparator);
                        black_box(users)
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("native", n),
            &n,
            |b, &_n| {
                b.iter_batched(
                    || {
                        let mut users = base_users.clone();
                        for _ in 0..delta_count {
                            let idx = rng.gen_range(0..n);
                            users[idx].mutate(&mut rng, n);
                        }
                        users
                    },
                    |mut users| {
                        users.sort_by(user_comparator);
                        black_box(users)
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_algorithms, benchmark_scaling);
criterion_main!(benches);
