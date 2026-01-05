//! DeltaSort Benchmark Suite
//!
//! Run with: `cargo run --bin benchmark --release`
//! Export CSV: `cargo run --bin benchmark --release -- --export`

use deltasort::delta_sort_by;
use rand::Rng;
use std::cell::Cell;
use std::collections::HashSet;
use std::fs;
use std::io::{self, Write};
use std::time::Instant;

// ============================================================================
// BENCHMARK CONFIGURATION
// ============================================================================

/// Array size for main benchmarks
const N: usize = 50_000;

/// Delta counts to test
const DELTA_COUNTS: &[usize] = &[
    1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000,
];

/// Array sizes for crossover analysis
const CROSSOVER_SIZES: &[usize] = &[
    1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000,
    //5_000_000, 10_000_000,
];

/// Number of iterations per benchmark
const ITERATIONS: usize = 20;

/// Number of iterations for crossover measurements
const CROSSOVER_ITERATIONS: usize = 3;

/// Z-score for 95% confidence interval
const Z_95: f64 = 1.96;

// ============================================================================
// BENCHMARK DATA
// ============================================================================

const COUNTRIES: &[&str] = &[
    "USA",
    "Canada",
    "UK",
    "Germany",
    "France",
    "Spain",
    "Italy",
    "Japan",
    "Australia",
    "Brazil",
    "India",
    "China",
    "Mexico",
    "Argentina",
    "Sweden",
];

const FIRST_NAMES: &[&str] = &[
    "Vijay",
    "Meera",
    "Akash",
    "Kashish",
    "Sunita",
    "Aviral",
    "Saumya",
    "Aman",
    "Sanjay",
    "Kavitha",
    "Radhika",
    "Meenakshi",
    "Suresh",
    "Krishna",
];

const LAST_NAMES: &[&str] = &[
    "Sharma", "Patel", "Dwivedi", "Kumar", "Singh", "Gupta", "Nair", "Iyer", "Rao", "Menon",
    "Pillai", "Joshi", "Verma",
];

#[derive(Clone, Debug, Default)]
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

    fn mutate(&mut self, rng: &mut impl Rng) {
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

thread_local! {
    static COMPARISON_COUNT: Cell<u64> = const { Cell::new(0) };
    static MOVEMENT_COUNT: Cell<u64> = const { Cell::new(0) };
}

fn counting_comparator(a: &User, b: &User) -> std::cmp::Ordering {
    COMPARISON_COUNT.with(|c| c.set(c.get() + 1));
    user_comparator(a, b)
}

fn reset_comparison_count() {
    COMPARISON_COUNT.with(|c| c.set(0));
}

fn get_comparison_count() -> u64 {
    COMPARISON_COUNT.with(|c| c.get())
}

fn add_movements(count: u64) {
    MOVEMENT_COUNT.with(|c| c.set(c.get() + count));
}

fn reset_movement_count() {
    MOVEMENT_COUNT.with(|c| c.set(0));
}

fn get_movement_count() -> u64 {
    MOVEMENT_COUNT.with(|c| c.get())
}

fn generate_sorted_users(n: usize) -> Vec<User> {
    let mut users: Vec<User> = (0..n).map(User::generate).collect();
    users.sort_by(user_comparator);
    users
}

fn sample_distinct_indices(rng: &mut impl Rng, n: usize, k: usize) -> Vec<usize> {
    let mut arr: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = rng.gen_range(i..n);
        arr.swap(i, j);
    }
    arr.truncate(k);
    arr
}

// ============================================================================
// ALGORITHMS (non-counting versions for timing)
// ============================================================================

fn native_sort(arr: &mut [User]) {
    arr.sort_by(user_comparator);
}

fn binary_insertion_sort(arr: &mut Vec<User>, dirty_indices: &HashSet<usize>) {
    if dirty_indices.is_empty() {
        return;
    }
    let mut sorted_desc: Vec<usize> = dirty_indices.iter().copied().collect();
    sorted_desc.sort_unstable_by(|a, b| b.cmp(a));
    let mut extracted: Vec<User> = Vec::with_capacity(sorted_desc.len());
    for &idx in &sorted_desc {
        extracted.push(arr.remove(idx));
    }
    for value in extracted {
        let pos = arr.partition_point(|x| user_comparator(x, &value) == std::cmp::Ordering::Less);
        arr.insert(pos, value);
    }
}

fn extract_sort_merge(arr: &mut Vec<User>, dirty_indices: &HashSet<usize>) {
    if dirty_indices.is_empty() {
        return;
    }
    let mut sorted_desc: Vec<usize> = dirty_indices.iter().copied().collect();
    sorted_desc.sort_unstable_by(|a, b| b.cmp(a));
    let mut dirty_values: Vec<User> = Vec::with_capacity(sorted_desc.len());
    for &idx in &sorted_desc {
        dirty_values.push(arr.remove(idx));
    }
    dirty_values.sort_by(user_comparator);
    let clean_len = arr.len();
    let dirty_len = dirty_values.len();
    let mut result: Vec<User> = Vec::with_capacity(clean_len + dirty_len);
    let mut i = 0;
    let mut j = 0;
    while i < clean_len && j < dirty_len {
        if user_comparator(&arr[i], &dirty_values[j]) != std::cmp::Ordering::Greater {
            result.push(std::mem::take(&mut arr[i]));
            i += 1;
        } else {
            result.push(std::mem::take(&mut dirty_values[j]));
            j += 1;
        }
    }
    for item in arr.drain(i..) {
        result.push(item);
    }
    for item in dirty_values.drain(j..) {
        result.push(item);
    }
    *arr = result;
}

fn deltasort_wrapper(arr: &mut Vec<User>, dirty_indices: &HashSet<usize>) {
    delta_sort_by(arr.as_mut_slice(), dirty_indices, user_comparator);
}

// ============================================================================
// REFERENCE SORT (MergeSort) - Non-counting version
// ============================================================================

fn mergesort(arr: &mut [User]) {
    let len = arr.len();
    if len <= 1 {
        return;
    }
    let mid = len / 2;
    mergesort(&mut arr[..mid]);
    mergesort(&mut arr[mid..]);
    
    // Merge using temporary vectors
    let left: Vec<User> = arr[..mid].to_vec();
    let right: Vec<User> = arr[mid..].to_vec();
    
    let (mut i, mut j, mut k) = (0, 0, 0);
    while i < left.len() && j < right.len() {
        if user_comparator(&left[i], &right[j]) != std::cmp::Ordering::Greater {
            arr[k] = left[i].clone();
            i += 1;
        } else {
            arr[k] = right[j].clone();
            j += 1;
        }
        k += 1;
    }
    while i < left.len() {
        arr[k] = left[i].clone();
        i += 1;
        k += 1;
    }
    while j < right.len() {
        arr[k] = right[j].clone();
        j += 1;
        k += 1;
    }
}

fn mergesort_wrapper(arr: &mut Vec<User>, _dirty_indices: &HashSet<usize>) {
    mergesort(arr.as_mut_slice());
}

// ============================================================================
// ALGORITHMS (counting versions for comparison counts + movement counts)
// ============================================================================

fn native_sort_counting(arr: &mut [User]) {
    // Native sort movements can't be tracked directly; we estimate based on algorithm
    // PDQSort does ~1.4 * n * log2(n) swaps on average, each swap = 2 movements
    let n = arr.len();
    let estimated_swaps = (1.4 * n as f64 * (n as f64).log2()) as u64;
    add_movements(estimated_swaps * 2);
    arr.sort_by(counting_comparator);
}

fn binary_insertion_sort_counting(arr: &mut Vec<User>, dirty_indices: &HashSet<usize>) {
    if dirty_indices.is_empty() {
        return;
    }
    let mut sorted_desc: Vec<usize> = dirty_indices.iter().copied().collect();
    sorted_desc.sort_unstable_by(|a, b| b.cmp(a));
    let mut extracted: Vec<User> = Vec::with_capacity(sorted_desc.len());
    for &idx in &sorted_desc {
        // remove shifts all elements after idx left by 1
        add_movements((arr.len() - idx - 1) as u64);
        extracted.push(arr.remove(idx));
    }
    for value in extracted {
        let pos =
            arr.partition_point(|x| counting_comparator(x, &value) == std::cmp::Ordering::Less);
        // insert shifts all elements from pos right by 1
        add_movements((arr.len() - pos) as u64);
        arr.insert(pos, value);
    }
}

fn extract_sort_merge_counting(arr: &mut Vec<User>, dirty_indices: &HashSet<usize>) {
    if dirty_indices.is_empty() {
        return;
    }
    let mut sorted_desc: Vec<usize> = dirty_indices.iter().copied().collect();
    sorted_desc.sort_unstable_by(|a, b| b.cmp(a));
    let mut dirty_values: Vec<User> = Vec::with_capacity(sorted_desc.len());
    for &idx in &sorted_desc {
        // remove shifts elements
        add_movements((arr.len() - idx - 1) as u64);
        dirty_values.push(arr.remove(idx));
    }
    dirty_values.sort_by(counting_comparator);
    let clean_len = arr.len();
    let dirty_len = dirty_values.len();
    // Merge writes every element once
    add_movements((clean_len + dirty_len) as u64);
    let mut result: Vec<User> = Vec::with_capacity(clean_len + dirty_len);
    let mut i = 0;
    let mut j = 0;
    while i < clean_len && j < dirty_len {
        if counting_comparator(&arr[i], &dirty_values[j]) != std::cmp::Ordering::Greater {
            result.push(std::mem::take(&mut arr[i]));
            i += 1;
        } else {
            result.push(std::mem::take(&mut dirty_values[j]));
            j += 1;
        }
    }
    for item in arr.drain(i..) {
        result.push(item);
    }
    for item in dirty_values.drain(j..) {
        result.push(item);
    }
    *arr = result;
}

fn deltasort_wrapper_counting(arr: &mut Vec<User>, dirty_indices: &HashSet<usize>) {
    // Use our own instrumented version that tracks movements
    deltasort_counting_impl(arr.as_mut_slice(), dirty_indices);
}

/// Instrumented DeltaSort implementation that counts comparisons and movements
fn deltasort_counting_impl(arr: &mut [User], updated_indices: &HashSet<usize>) {
    if updated_indices.is_empty() {
        return;
    }

    // Phase 1: Extract and sort dirty values, write back in index order
    let mut dirty: Vec<usize> = updated_indices.iter().copied().collect();
    dirty.sort_unstable();

    let mut values: Vec<User> = dirty.iter().map(|&i| arr[i].clone()).collect();
    values.sort_by(|a, b| {
        COMPARISON_COUNT.with(|c| c.set(c.get() + 1));
        user_comparator(a, b)
    });

    // Writing sorted values back = k movements
    add_movements(dirty.len() as u64);
    for (i, &idx) in dirty.iter().enumerate() {
        arr[idx] = values[i].clone();
    }

    // Add sentinel to trigger final flush
    dirty.push(arr.len());

    // Phase 2: Scan updated indices left to right
    let mut pending_right_violations: Vec<usize> = Vec::with_capacity(dirty.len());
    let mut left_bound = 0;

    for &i in &dirty {
        let direction = if i == arr.len() {
            true // LEFT (sentinel)
        } else {
            // Check if LEFT violation
            if i > 0 {
                COMPARISON_COUNT.with(|c| c.set(c.get() + 1));
                user_comparator(&arr[i - 1], &arr[i]) == std::cmp::Ordering::Greater
            } else {
                false
            }
        };

        if direction {
            // LEFT violation - fix all pending RIGHTs first
            let mut right_bound = i.saturating_sub(1);
            while let Some(idx) = pending_right_violations.pop() {
                if idx < arr.len() - 1 {
                    COMPARISON_COUNT.with(|c| c.set(c.get() + 1));
                    if user_comparator(&arr[idx], &arr[idx + 1]) == std::cmp::Ordering::Greater {
                        right_bound = fix_right_counting(arr, idx, right_bound).saturating_sub(1);
                    }
                }
            }
            // Fix actual LEFT violation
            if i < arr.len() {
                left_bound = fix_left_counting(arr, i, left_bound) + 1;
            }
        } else {
            pending_right_violations.push(i);
        }
    }
}

fn fix_right_counting(arr: &mut [User], i: usize, right_bound: usize) -> usize {
    let mut lo = i + 1;
    let mut hi = right_bound as isize;

    while lo as isize <= hi {
        let mid = lo + ((hi as usize - lo) >> 1);
        COMPARISON_COUNT.with(|c| c.set(c.get() + 1));
        let cmp_result = user_comparator(&arr[mid], &arr[i]);
        if cmp_result != std::cmp::Ordering::Greater {
            lo = mid + 1;
        } else {
            hi = mid as isize - 1;
        }
    }

    let target = hi as usize;
    if i != target {
        // Movement = number of elements shifted
        add_movements((target - i) as u64);
        arr[i..=target].rotate_left(1);
    }
    target
}

fn fix_left_counting(arr: &mut [User], i: usize, left_bound: usize) -> usize {
    let mut lo = left_bound;
    let mut hi = i.saturating_sub(1) as isize;

    while lo as isize <= hi {
        let mid = lo + ((hi as usize - lo) >> 1);
        COMPARISON_COUNT.with(|c| c.set(c.get() + 1));
        let cmp_result = user_comparator(&arr[i], &arr[mid]);
        if cmp_result == std::cmp::Ordering::Less {
            hi = mid as isize - 1;
        } else {
            lo = mid + 1;
        }
    }

    if i != lo {
        // Movement = number of elements shifted
        add_movements((i - lo) as u64);
        arr[lo..=i].rotate_right(1);
    }
    lo
}

// ============================================================================
// REFERENCE SORT - Counting version
// ============================================================================

fn mergesort_counting(arr: &mut [User]) {
    let len = arr.len();
    if len <= 1 {
        return;
    }
    let mid = len / 2;
    mergesort_counting(&mut arr[..mid]);
    mergesort_counting(&mut arr[mid..]);
    
    // Merge using temporary vectors - count all copies
    let left: Vec<User> = arr[..mid].to_vec();
    let right: Vec<User> = arr[mid..].to_vec();
    add_movements(len as u64); // copy to temp vectors
    
    let (mut i, mut j, mut k) = (0, 0, 0);
    while i < left.len() && j < right.len() {
        COMPARISON_COUNT.with(|c| c.set(c.get() + 1));
        if user_comparator(&left[i], &right[j]) != std::cmp::Ordering::Greater {
            arr[k] = left[i].clone();
            i += 1;
        } else {
            arr[k] = right[j].clone();
            j += 1;
        }
        add_movements(1); // write back
        k += 1;
    }
    while i < left.len() {
        arr[k] = left[i].clone();
        add_movements(1);
        i += 1;
        k += 1;
    }
    while j < right.len() {
        arr[k] = right[j].clone();
        add_movements(1);
        j += 1;
        k += 1;
    }
}

fn mergesort_wrapper_counting(arr: &mut Vec<User>, _dirty_indices: &HashSet<usize>) {
    mergesort_counting(arr.as_mut_slice());
}

// ============================================================================
// STATISTICS
// ============================================================================

struct Stats {
    mean: f64,
    ci_95: f64,
}

fn calculate_stats(values: &[f64]) -> Stats {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();
    let std_error = std_dev / n.sqrt();
    let ci_95 = Z_95 * std_error;
    Stats { mean, ci_95 }
}

fn calculate_stats_u64(values: &[u64]) -> Stats {
    let floats: Vec<f64> = values.iter().map(|&x| x as f64).collect();
    calculate_stats(&floats)
}

// ============================================================================
// BENCHMARK MEASUREMENT
// ============================================================================

struct BenchmarkResult {
    time_us: f64,
    time_ci: f64,
    comparisons: f64,
    comparisons_ci: f64,
    movements: f64,
    movements_ci: f64,
}

/// Measure timing using non-counting comparator (accurate timing)
/// Measure comparisons and movements using counting versions (separate runs)
/// Each iteration generates fresh random mutations for proper variance measurement
fn run_benchmark<F, G>(
    base_users: &[User],
    k: usize,
    mut time_fn: F,
    mut count_fn: G,
) -> BenchmarkResult
where
    F: FnMut(&mut Vec<User>, &HashSet<usize>),
    G: FnMut(&mut Vec<User>, &HashSet<usize>),
{
    let mut rng = rand::thread_rng();
    let n = base_users.len();

    // Phase 1: Measure timing (without counting overhead)
    // Each iteration uses fresh random mutations
    let mut times_us = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let mut users = base_users.to_vec();
        let indices = sample_distinct_indices(&mut rng, n, k);
        let mut dirty_indices = HashSet::with_capacity(k);
        for idx in indices {
            users[idx].mutate(&mut rng);
            dirty_indices.insert(idx);
        }
        let start = Instant::now();
        time_fn(&mut users, &dirty_indices);
        times_us.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }

    // Phase 2: Measure comparisons and movements (separate runs with fresh mutations)
    let mut comparisons = Vec::with_capacity(ITERATIONS);
    let mut movements = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let mut users = base_users.to_vec();
        let indices = sample_distinct_indices(&mut rng, n, k);
        let mut dirty_indices = HashSet::with_capacity(k);
        for idx in indices {
            users[idx].mutate(&mut rng);
            dirty_indices.insert(idx);
        }
        reset_comparison_count();
        reset_movement_count();
        count_fn(&mut users, &dirty_indices);
        comparisons.push(get_comparison_count());
        movements.push(get_movement_count());
    }

    let time_stats = calculate_stats(&times_us);
    let cmp_stats = calculate_stats_u64(&comparisons);
    let mov_stats = calculate_stats_u64(&movements);

    BenchmarkResult {
        time_us: time_stats.mean,
        time_ci: time_stats.ci_95,
        comparisons: cmp_stats.mean,
        comparisons_ci: cmp_stats.ci_95,
        movements: mov_stats.mean,
        movements_ci: mov_stats.ci_95,
    }
}

fn run_native_benchmark(base_users: &[User], k: usize) -> BenchmarkResult {
    let mut rng = rand::thread_rng();
    let n = base_users.len();

    // Phase 1: Timing with fresh mutations each iteration
    let mut times_us = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let mut users = base_users.to_vec();
        let indices = sample_distinct_indices(&mut rng, n, k);
        for idx in indices {
            users[idx].mutate(&mut rng);
        }
        let start = Instant::now();
        native_sort(&mut users);
        times_us.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }

    // Phase 2: Comparisons and movements with fresh mutations
    let mut comparisons = Vec::with_capacity(ITERATIONS);
    let mut movements = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let mut users = base_users.to_vec();
        let indices = sample_distinct_indices(&mut rng, n, k);
        for idx in indices {
            users[idx].mutate(&mut rng);
        }
        reset_comparison_count();
        reset_movement_count();
        native_sort_counting(&mut users);
        comparisons.push(get_comparison_count());
        movements.push(get_movement_count());
    }

    let time_stats = calculate_stats(&times_us);
    let cmp_stats = calculate_stats_u64(&comparisons);
    let mov_stats = calculate_stats_u64(&movements);

    BenchmarkResult {
        time_us: time_stats.mean,
        time_ci: time_stats.ci_95,
        comparisons: cmp_stats.mean,
        comparisons_ci: cmp_stats.ci_95,
        movements: mov_stats.mean,
        movements_ci: mov_stats.ci_95,
    }
}

// ============================================================================
// CROSSOVER ANALYSIS
// ============================================================================

fn deltasort_is_faster(base_users: &[User], k: usize, n: usize) -> bool {
    let mut rng = rand::thread_rng();

    let mut native_time = 0.0;
    let mut ds_time = 0.0;

    for _ in 0..CROSSOVER_ITERATIONS {
        let mut users = base_users.to_vec();
        let mut dirty_indices = HashSet::new();
        for _ in 0..k {
            let idx = rng.gen_range(0..n);
            users[idx].mutate(&mut rng);
            dirty_indices.insert(idx);
        }

        // Use non-counting comparator for accurate timing
        let start = Instant::now();
        let mut test_users = users.clone();
        test_users.sort_by(user_comparator);
        native_time += start.elapsed().as_secs_f64();

        let start = Instant::now();
        let mut test_users = users.clone();
        delta_sort_by(&mut test_users, &dirty_indices, user_comparator);
        ds_time += start.elapsed().as_secs_f64();
    }

    ds_time < native_time
}

fn find_crossover(n: usize) -> usize {
    let base_users = generate_sorted_users(n);

    // Warmup
    for _ in 0..5 {
        let mut users = base_users.clone();
        users.sort_by(user_comparator);
    }

    let mut lo: usize = 1;
    let mut hi: usize = (n * 2) / 5;

    if !deltasort_is_faster(&base_users, 1, n) {
        return 0;
    }

    if deltasort_is_faster(&base_users, n, n) {
        return n;
    }

    let min_range = (n as f64 * 0.001) as usize;

    while lo < hi {
        if hi - lo < min_range {
            break;
        }

        let mid = lo + (hi - lo).div_ceil(2);

        if deltasort_is_faster(&base_users, mid, n) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    lo
}

// ============================================================================
// RESULTS STORAGE
// ============================================================================

struct AlgorithmResult {
    k: usize,
    time_us: f64,
    time_ci: f64,
    comparisons: f64,
    comparisons_ci: f64,
    movements: f64,
    movements_ci: f64,
}

struct BenchmarkResults {
    native: Vec<AlgorithmResult>,
    mergesort: Vec<AlgorithmResult>,
    bis: Vec<AlgorithmResult>,
    esm: Vec<AlgorithmResult>,
    deltasort: Vec<AlgorithmResult>,
}

struct CrossoverResult {
    n: usize,
    crossover_ratio: f64,
}

// ============================================================================
// OUTPUT
// ============================================================================

fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        format!("{}", n)
    }
}

/// Format value ± ci with consistent total width
fn format_with_ci(value: f64, ci: f64, total_width: usize) -> String {
    let val_str = format!("{:.1}", value);
    let ci_str = format!("{:.1}", ci);
    let content = format!("{} ± {}", val_str, ci_str);
    format!("{:>width$}", content, width = total_width)
}

/// Format integer value ± ci with consistent total width  
fn format_int_with_ci(value: f64, ci: f64, total_width: usize) -> String {
    let val_str = format!("{:.0}", value);
    let ci_str = format!("{:.0}", ci);
    let content = format!("{} ± {}", val_str, ci_str);
    format!("{:>width$}", content, width = total_width)
}

fn print_execution_time_table(results: &BenchmarkResults) {
    const COL_WIDTH: usize = 15;

    println!();
    println!("Execution Time (µs) - n={}", format_number(N));
    println!("┌────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┐");
    println!("│   k    │     Native      │    MergeSort    │       BIS       │       ESM       │    DeltaSort    │");
    println!("├────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┤");
    for i in 0..results.native.len() {
        println!(
            "│ {:>6} │ {} │ {} │ {} │ {} │ {} │",
            results.native[i].k,
            format_with_ci(results.native[i].time_us, results.native[i].time_ci, COL_WIDTH),
            format_with_ci(results.mergesort[i].time_us, results.mergesort[i].time_ci, COL_WIDTH),
            format_with_ci(results.bis[i].time_us, results.bis[i].time_ci, COL_WIDTH),
            format_with_ci(results.esm[i].time_us, results.esm[i].time_ci, COL_WIDTH),
            format_with_ci(results.deltasort[i].time_us, results.deltasort[i].time_ci, COL_WIDTH),
        );
    }
    println!("└────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┘");
}

fn print_comparator_count_table(results: &BenchmarkResults) {
    const COL_WIDTH: usize = 15;

    println!();
    println!("Comparator Invocations - n={}", format_number(N));
    println!("┌────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┐");
    println!("│   k    │     Native      │    MergeSort    │       BIS       │       ESM       │    DeltaSort    │");
    println!("├────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┤");
    for i in 0..results.native.len() {
        println!(
            "│ {:>6} │ {} │ {} │ {} │ {} │ {} │",
            results.native[i].k,
            format_int_with_ci(results.native[i].comparisons, results.native[i].comparisons_ci, COL_WIDTH),
            format_int_with_ci(results.mergesort[i].comparisons, results.mergesort[i].comparisons_ci, COL_WIDTH),
            format_int_with_ci(results.bis[i].comparisons, results.bis[i].comparisons_ci, COL_WIDTH),
            format_int_with_ci(results.esm[i].comparisons, results.esm[i].comparisons_ci, COL_WIDTH),
            format_int_with_ci(results.deltasort[i].comparisons, results.deltasort[i].comparisons_ci, COL_WIDTH),
        );
    }
    println!("└────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┘");
}

fn print_movement_count_table(results: &BenchmarkResults) {
    const COL_WIDTH: usize = 15;

    println!();
    println!("Data Movements (writes) - n={}", format_number(N));
    println!("┌────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┐");
    println!("│   k    │  Native (est.)  │    MergeSort    │       BIS       │       ESM       │    DeltaSort    │");
    println!("├────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┤");
    for i in 0..results.native.len() {
        println!(
            "│ {:>6} │ {} │ {} │ {} │ {} │ {} │",
            results.native[i].k,
            format_int_with_ci(results.native[i].movements, results.native[i].movements_ci, COL_WIDTH),
            format_int_with_ci(results.mergesort[i].movements, results.mergesort[i].movements_ci, COL_WIDTH),
            format_int_with_ci(results.bis[i].movements, results.bis[i].movements_ci, COL_WIDTH),
            format_int_with_ci(results.esm[i].movements, results.esm[i].movements_ci, COL_WIDTH),
            format_int_with_ci(results.deltasort[i].movements, results.deltasort[i].movements_ci, COL_WIDTH),
        );
    }
    println!("└────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┘");
}

fn print_crossover_table(results: &[CrossoverResult]) {
    println!();
    println!("Crossover Threshold");
    println!("┌────────────┬──────────────┐");
    println!("│     n      │  k_c/n (%)   │");
    println!("├────────────┼──────────────┤");
    for r in results {
        println!(
            "│ {:>10} │ {:>11.1}% │",
            format_number(r.n),
            r.crossover_ratio
        );
    }
    println!("└────────────┴──────────────┘");
}

// ============================================================================
// CSV EXPORT (values only, no CI)
// ============================================================================

fn export_execution_time_csv(results: &BenchmarkResults, path: &str) {
    let mut csv = String::from("k,native,mergesort,bis,esm,deltasort\n");
    for i in 0..results.native.len() {
        csv.push_str(&format!(
            "{},{:.1},{:.1},{:.1},{:.1},{:.1}\n",
            results.native[i].k,
            results.native[i].time_us,
            results.mergesort[i].time_us,
            results.bis[i].time_us,
            results.esm[i].time_us,
            results.deltasort[i].time_us,
        ));
    }
    fs::write(path, csv).expect("Failed to write execution-time.csv");
    println!("Exported: {}", path);
}

fn export_comparator_count_csv(results: &BenchmarkResults, path: &str) {
    let mut csv = String::from("k,native,mergesort,bis,esm,deltasort\n");
    for i in 0..results.native.len() {
        csv.push_str(&format!(
            "{},{:.0},{:.0},{:.0},{:.0},{:.0}\n",
            results.native[i].k,
            results.native[i].comparisons,
            results.mergesort[i].comparisons,
            results.bis[i].comparisons,
            results.esm[i].comparisons,
            results.deltasort[i].comparisons,
        ));
    }
    fs::write(path, csv).expect("Failed to write comparator-count.csv");
    println!("Exported: {}", path);
}

fn export_movement_count_csv(results: &BenchmarkResults, path: &str) {
    let mut csv = String::from("k,native,mergesort,bis,esm,deltasort\n");
    for i in 0..results.native.len() {
        csv.push_str(&format!(
            "{},{:.0},{:.0},{:.0},{:.0},{:.0}\n",
            results.native[i].k,
            results.native[i].movements,
            results.mergesort[i].movements,
            results.bis[i].movements,
            results.esm[i].movements,
            results.deltasort[i].movements,
        ));
    }
    fs::write(path, csv).expect("Failed to write movement-count.csv");
    println!("Exported: {}", path);
}

fn export_crossover_csv(results: &[CrossoverResult], path: &str) {
    let mut csv = String::from("n,crossover_ratio\n");
    for r in results {
        csv.push_str(&format!("{},{:.1}\n", r.n, r.crossover_ratio));
    }
    fs::write(path, csv).expect("Failed to write crossover-threshold.csv");
    println!("Exported: {}", path);
}

fn export_metadata_csv(path: &str) {
    use std::process::Command;

    // Get current date
    let date = chrono_lite_date();

    // Get OS info
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    // Try to get more detailed system info on macOS
    let machine = if os == "macos" {
        Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| format!("{}/{}", os, arch))
    } else {
        format!("{}/{}", os, arch)
    };

    let csv = format!(
        "key,value\ndate,{}\nmachine,{}\nn,{}\niterations,{}\n",
        date, machine, N, ITERATIONS
    );
    fs::write(path, csv).expect("Failed to write metadata.csv");
    println!("Exported: {}", path);
}

/// Simple date formatter (avoids chrono dependency)
fn chrono_lite_date() -> String {
    use std::process::Command;
    Command::new("date")
        .arg("+%B %Y")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "Unknown".to_string())
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let export = args.iter().any(|a| a == "--export");

    println!();
    println!("DeltaSort Benchmark");
    println!("===================");

    // Warmup
    let base_users = generate_sorted_users(N);
    for _ in 0..5 {
        let mut users = base_users.clone();
        users.sort_by(user_comparator);
    }

    // --- Combined Execution Time & Comparator Count ---
    println!();
    println!("Running benchmarks (time + comparisons + movements)...");
    let mut results = BenchmarkResults {
        native: Vec::new(),
        mergesort: Vec::new(),
        bis: Vec::new(),
        esm: Vec::new(),
        deltasort: Vec::new(),
    };

    for &k in DELTA_COUNTS {
        print!("  k={:>5}...", k);
        io::stdout().flush().unwrap();

        let native = run_native_benchmark(&base_users, k);
        results.native.push(AlgorithmResult {
            k,
            time_us: native.time_us,
            time_ci: native.time_ci,
            comparisons: native.comparisons,
            comparisons_ci: native.comparisons_ci,
            movements: native.movements,
            movements_ci: native.movements_ci,
        });

        let ms = run_benchmark(
            &base_users,
            k,
            mergesort_wrapper,
            mergesort_wrapper_counting,
        );
        results.mergesort.push(AlgorithmResult {
            k,
            time_us: ms.time_us,
            time_ci: ms.time_ci,
            comparisons: ms.comparisons,
            comparisons_ci: ms.comparisons_ci,
            movements: ms.movements,
            movements_ci: ms.movements_ci,
        });

        let bis = run_benchmark(
            &base_users,
            k,
            binary_insertion_sort,
            binary_insertion_sort_counting,
        );
        results.bis.push(AlgorithmResult {
            k,
            time_us: bis.time_us,
            time_ci: bis.time_ci,
            comparisons: bis.comparisons,
            comparisons_ci: bis.comparisons_ci,
            movements: bis.movements,
            movements_ci: bis.movements_ci,
        });

        let esm = run_benchmark(
            &base_users,
            k,
            extract_sort_merge,
            extract_sort_merge_counting,
        );
        results.esm.push(AlgorithmResult {
            k,
            time_us: esm.time_us,
            time_ci: esm.time_ci,
            comparisons: esm.comparisons,
            comparisons_ci: esm.comparisons_ci,
            movements: esm.movements,
            movements_ci: esm.movements_ci,
        });

        let ds = run_benchmark(
            &base_users,
            k,
            deltasort_wrapper,
            deltasort_wrapper_counting,
        );
        results.deltasort.push(AlgorithmResult {
            k,
            time_us: ds.time_us,
            time_ci: ds.time_ci,
            comparisons: ds.comparisons,
            comparisons_ci: ds.comparisons_ci,
            movements: ds.movements,
            movements_ci: ds.movements_ci,
        });

        println!(" done");
    }

    print_execution_time_table(&results);
    print_comparator_count_table(&results);
    print_movement_count_table(&results);

    // --- Crossover Analysis ---
    println!();
    println!("Running crossover analysis (this may take a while)...");
    let mut crossover_results: Vec<CrossoverResult> = Vec::new();

    for &size in CROSSOVER_SIZES {
        print!("  n={:>10}...", format_number(size));
        io::stdout().flush().unwrap();
        let k_c = find_crossover(size);
        let crossover_ratio = (k_c as f64 / size as f64) * 100.0;
        crossover_results.push(CrossoverResult {
            n: size,
            crossover_ratio,
        });
        println!(" k_c={} ({:.1}%)", k_c, crossover_ratio);
    }

    print_crossover_table(&crossover_results);

    // --- Export CSVs ---
    if export {
        println!();
        println!("Exporting CSV files...");
        let base_path = "../paper/figures/rust";
        fs::create_dir_all(base_path).ok();
        export_execution_time_csv(&results, &format!("{}/execution-time.csv", base_path));
        export_comparator_count_csv(&results, &format!("{}/comparator-count.csv", base_path));
        export_movement_count_csv(&results, &format!("{}/movement-count.csv", base_path));
        export_crossover_csv(
            &crossover_results,
            &format!("{}/crossover-threshold.csv", base_path),
        );
        export_metadata_csv(&format!("{}/benchmark_metadata.csv", base_path));
    }

    println!();
    println!("Done!");
    if !export {
        println!("Run with --export to write CSV files to paper/figures/rust/");
    }
}
