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
    5_000_000, 10_000_000,
];

/// Array sizes for segmentation analysis
const SEGMENTATION_SIZES: &[usize] = &[1_000, 10_000, 100_000];

/// Delta percentages for segmentation analysis (as fractions of n)
const SEGMENTATION_K_PERCENTS: &[f64] = &[
    0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
];

/// Iterations for segmentation analysis
const SEGMENTATION_ITERATIONS: usize = 10;

/// Base number of iterations per benchmark (scaled up for small k)
const BASE_ITERATIONS: usize = 100;

/// Number of iterations for crossover measurements
const CROSSOVER_ITERATIONS: usize = 50;

/// Z-score for 95% confidence interval
const Z_95: f64 = 1.96;

/// Get number of iterations for a given k value
/// Small k values need more iterations due to timer resolution
fn iterations_for_k(k: usize) -> usize {
    match k {
        0..=10 => BASE_ITERATIONS * 10,  // 1000 iterations for k <= 10
        11..=50 => BASE_ITERATIONS * 5,  // 500 iterations for k <= 50
        51..=200 => BASE_ITERATIONS * 2, // 200 iterations for k <= 200
        _ => BASE_ITERATIONS,            // 100 iterations for large k
    }
}

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
    sd: f64,    // Standard deviation
    ci_95: f64, // 95% confidence interval half-width
    cv: f64,    // Coefficient of variation (SD/mean as percentage)
}

fn calculate_stats(values: &[f64]) -> Stats {
    let n = values.len() as f64;
    if n < 2.0 {
        let mean = if n > 0.0 { values[0] } else { 0.0 };
        return Stats {
            mean,
            sd: 0.0,
            ci_95: 0.0,
            cv: 0.0,
        };
    }
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let sd = variance.sqrt();
    let std_error = sd / n.sqrt();
    let ci_95 = Z_95 * std_error;
    let cv = if mean > 0.0 { (sd / mean) * 100.0 } else { 0.0 };
    Stats {
        mean,
        sd,
        ci_95,
        cv,
    }
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
    time_sd: f64,
    time_ci: f64,
    time_cv: f64,
    comparisons: f64,
    comparisons_sd: f64,
    comparisons_ci: f64,
    comparisons_cv: f64,
    movements: f64,
    movements_sd: f64,
    movements_ci: f64,
    movements_cv: f64,
    iterations: usize,
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
    let iters = iterations_for_k(k);

    // Phase 1: Measure timing (without counting overhead)
    // Each iteration uses fresh random mutations
    let mut times_us = Vec::with_capacity(iters);
    for _ in 0..iters {
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
    let mut comparisons = Vec::with_capacity(iters);
    let mut movements = Vec::with_capacity(iters);
    for _ in 0..iters {
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
        time_sd: time_stats.sd,
        time_ci: time_stats.ci_95,
        time_cv: time_stats.cv,
        comparisons: cmp_stats.mean,
        comparisons_sd: cmp_stats.sd,
        comparisons_ci: cmp_stats.ci_95,
        comparisons_cv: cmp_stats.cv,
        movements: mov_stats.mean,
        movements_sd: mov_stats.sd,
        movements_ci: mov_stats.ci_95,
        movements_cv: mov_stats.cv,
        iterations: iters,
    }
}

fn run_native_benchmark(base_users: &[User], k: usize) -> BenchmarkResult {
    let mut rng = rand::thread_rng();
    let n = base_users.len();
    let iters = iterations_for_k(k);

    // Phase 1: Timing with fresh mutations each iteration
    let mut times_us = Vec::with_capacity(iters);
    for _ in 0..iters {
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
    let mut comparisons = Vec::with_capacity(iters);
    let mut movements = Vec::with_capacity(iters);
    for _ in 0..iters {
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
        time_sd: time_stats.sd,
        time_ci: time_stats.ci_95,
        time_cv: time_stats.cv,
        comparisons: cmp_stats.mean,
        comparisons_sd: cmp_stats.sd,
        comparisons_ci: cmp_stats.ci_95,
        comparisons_cv: cmp_stats.cv,
        movements: mov_stats.mean,
        movements_sd: mov_stats.sd,
        movements_ci: mov_stats.ci_95,
        movements_cv: mov_stats.cv,
        iterations: iters,
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
// SEGMENTATION ANALYSIS
// ============================================================================

/// Result of segmentation analysis for one (n, k%) configuration
struct SegmentationResult {
    n: usize,
    k_percent: f64,
    k: usize,
    segment_count_mean: f64,
    segment_count_ci: f64,
    segments_per_k_mean: f64, // segment_count / k
    segments_per_k_ci: f64,
}

/// Analyze segmentation structure for a given n and k
/// Returns (segment_count, segments_per_k) over many iterations
fn analyze_segmentation(n: usize, k: usize, iters: usize) -> (Vec<usize>, Vec<f64>) {
    let mut rng = rand::thread_rng();

    let mut segment_counts = Vec::with_capacity(iters);
    let mut segments_per_k = Vec::with_capacity(iters);

    for _ in 0..iters {
        // Create a sorted base array
        let mut arr: Vec<i32> = (0..n as i32).collect();

        // Sample k distinct indices and mutate their values randomly
        let indices = sample_distinct_indices(&mut rng, n, k);
        let dirty_set: HashSet<usize> = indices.iter().copied().collect();

        // Mutate values at dirty indices to random values
        for &idx in &indices {
            arr[idx] = rng.gen_range(0..n as i32);
        }

        // Compute segments
        let (count, _total_size) = compute_segments(&arr, &dirty_set);
        segment_counts.push(count);

        // Segments per k (how many segments per violation)
        let spk = if k > 0 { count as f64 / k as f64 } else { 0.0 };
        segments_per_k.push(spk);
    }

    (segment_counts, segments_per_k)
}

/// Compute segment boundaries for a given array and dirty indices
///
/// A segment is defined by its violation types:
/// - Trailing segment: starts at 0, contains only L violations (no R in between)
/// - Leading segment: ends at n-1, contains only R violations (no L after)
/// - Intermediate segment: contains R violations followed by L violations
///
/// Algorithm: Start at 0, run past all R's, when L is found run past all L's
/// keeping track of farthest L encountered. When you hit R or end of array,
/// record segment from start to farthest L. If hit R, it becomes start of next segment.
///
/// Returns (segment_count, total_segment_size)
fn compute_segments(arr: &[i32], updated_indices: &HashSet<usize>) -> (usize, usize) {
    if updated_indices.is_empty() {
        return (0, 0);
    }

    let n = arr.len();

    // Sort dirty indices
    let mut dirty: Vec<usize> = updated_indices.iter().copied().collect();
    dirty.sort_unstable();

    // Create working array after Phase 1 redistribution
    // (dirty values sorted among themselves and placed back at dirty positions)
    let mut values: Vec<i32> = dirty.iter().map(|&i| arr[i]).collect();
    values.sort_unstable();

    let mut work_arr = arr.to_vec();
    for (i, &idx) in dirty.iter().enumerate() {
        work_arr[idx] = values[i];
    }

    // Classify each dirty index as L (left violation) or R (right violation)
    // L: value is less than its left neighbor (needs to move left)
    // R: value is greater than its right neighbor (needs to move right)
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum ViolationType {
        L,
        R,
    }

    let mut violations: Vec<(usize, ViolationType)> = Vec::new();
    for &idx in &dirty {
        let vtype = if idx > 0 && work_arr[idx] < work_arr[idx - 1] {
            ViolationType::L
        } else {
            ViolationType::R
        };
        violations.push((idx, vtype));
    }

    if violations.is_empty() {
        return (0, 0);
    }

    // Now detect segments by traversing violations
    // Segment starts at leftmost index, run past R's, then past L's, recording farthest L
    let mut segments: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;

    // If first violation index is not 0, then we can just assumy dummy R, add a 0,R to front of violations array
    if violations[0].0 != 0 {
        violations.insert(0, (0, ViolationType::R));
    }

    // If last violation index is not n-1, then we can just assumy dummy L, add a n-1,L to end of violations array
    if violations[violations.len() - 1].0 != n - 1 {
        violations.push((n - 1, ViolationType::L));
    }

    loop {
        let seg_start_idx = violations[i].0;

        // Do an R run first
        while i < violations.len() && violations[i].1 == ViolationType::R {
            i += 1;
        }

        // Then do an L run
        let mut last_l_idx = seg_start_idx; // Initialize to segment start, not i
        while i < violations.len() && violations[i].1 == ViolationType::L {
            last_l_idx = violations[i].0;
            i += 1;
        }

        segments.push((seg_start_idx, last_l_idx));

        if i >= violations.len() {
            break;
        }
    }

    // Count and sum segment sizes
    let segment_count = segments.len();
    let total_size: usize = segments.iter().map(|(l, r)| r - l + 1).sum();

    (segment_count, total_size)
}

/// Run segmentation analysis across all configurations
fn run_segmentation_analysis() -> Vec<SegmentationResult> {
    let mut results = Vec::new();

    for &n in SEGMENTATION_SIZES {
        for &k_percent in SEGMENTATION_K_PERCENTS {
            let k = ((n as f64 * k_percent / 100.0).round() as usize).max(if k_percent > 0.0 {
                1
            } else {
                0
            });

            if k > n {
                continue;
            }

            let (counts, spk) = analyze_segmentation(n, k, SEGMENTATION_ITERATIONS);

            let count_stats =
                calculate_stats_u64(&counts.iter().map(|&x| x as u64).collect::<Vec<_>>());
            let spk_stats = calculate_stats(&spk);

            results.push(SegmentationResult {
                n,
                k_percent,
                k,
                segment_count_mean: count_stats.mean,
                segment_count_ci: count_stats.ci_95,
                segments_per_k_mean: spk_stats.mean,
                segments_per_k_ci: spk_stats.ci_95,
            });
        }
    }

    results
}

fn print_segmentation_table(results: &[SegmentationResult]) {
    println!();
    println!("Segmentation Analysis");
    println!("┌────────────┬──────────┬────────┬────────────────────┬────────────────────┐");
    println!("│     n      │   k (%)  │    k   │   Segment Count    │    Segments/k      │");
    println!("├────────────┼──────────┼────────┼────────────────────┼────────────────────┤");

    for r in results {
        println!(
            "│ {:>10} │ {:>7.1}% │ {:>6} │ {:>7.1} ± {:>7.1} │ {:>7.3} ± {:>7.3} │",
            format_number(r.n),
            r.k_percent,
            r.k,
            r.segment_count_mean,
            r.segment_count_ci,
            r.segments_per_k_mean,
            r.segments_per_k_ci,
        );
    }
    println!("└────────────┴──────────┴────────┴────────────────────┴────────────────────┘");
}

fn export_segmentation_csv(results: &[SegmentationResult], base_path: &str) {
    // Export segment count
    let mut count_csv = String::from("k_percent");
    for &n in SEGMENTATION_SIZES {
        count_csv.push_str(&format!(",n{},n{}_ci", n, n));
    }
    count_csv.push('\n');

    for &k_percent in SEGMENTATION_K_PERCENTS {
        count_csv.push_str(&format!("{:.1}", k_percent));
        for &n in SEGMENTATION_SIZES {
            if let Some(r) = results
                .iter()
                .find(|r| r.n == n && (r.k_percent - k_percent).abs() < 0.01)
            {
                count_csv.push_str(&format!(
                    ",{:.2},{:.2}",
                    r.segment_count_mean, r.segment_count_ci
                ));
            } else {
                count_csv.push_str(",,");
            }
        }
        count_csv.push('\n');
    }
    let count_path = format!("{}/segmentation-count.csv", base_path);
    fs::write(&count_path, count_csv).expect("Failed to write segmentation-count.csv");
    println!("Exported: {}", count_path);

    // Export segments per k
    let mut spk_csv = String::from("k_percent");
    for &n in SEGMENTATION_SIZES {
        spk_csv.push_str(&format!(",n{},n{}_ci", n, n));
    }
    spk_csv.push('\n');

    for &k_percent in SEGMENTATION_K_PERCENTS {
        spk_csv.push_str(&format!("{:.1}", k_percent));
        for &n in SEGMENTATION_SIZES {
            if let Some(r) = results
                .iter()
                .find(|r| r.n == n && (r.k_percent - k_percent).abs() < 0.01)
            {
                spk_csv.push_str(&format!(
                    ",{:.4},{:.4}",
                    r.segments_per_k_mean, r.segments_per_k_ci
                ));
            } else {
                spk_csv.push_str(",,");
            }
        }
        spk_csv.push('\n');
    }
    let spk_path = format!("{}/segments-per-k.csv", base_path);
    fs::write(&spk_path, spk_csv).expect("Failed to write segments-per-k.csv");
    println!("Exported: {}", spk_path);
}

// ============================================================================
// RESULTS STORAGE
// ============================================================================

struct AlgorithmResult {
    k: usize,
    iterations: usize,
    time_us: f64,
    time_sd: f64,
    time_ci: f64,
    time_cv: f64,
    comparisons: f64,
    comparisons_sd: f64,
    comparisons_ci: f64,
    comparisons_cv: f64,
    movements: f64,
    movements_sd: f64,
    movements_ci: f64,
    movements_cv: f64,
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
            format_with_ci(
                results.native[i].time_us,
                results.native[i].time_ci,
                COL_WIDTH
            ),
            format_with_ci(
                results.mergesort[i].time_us,
                results.mergesort[i].time_ci,
                COL_WIDTH
            ),
            format_with_ci(results.bis[i].time_us, results.bis[i].time_ci, COL_WIDTH),
            format_with_ci(results.esm[i].time_us, results.esm[i].time_ci, COL_WIDTH),
            format_with_ci(
                results.deltasort[i].time_us,
                results.deltasort[i].time_ci,
                COL_WIDTH
            ),
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
            format_int_with_ci(
                results.native[i].comparisons,
                results.native[i].comparisons_ci,
                COL_WIDTH
            ),
            format_int_with_ci(
                results.mergesort[i].comparisons,
                results.mergesort[i].comparisons_ci,
                COL_WIDTH
            ),
            format_int_with_ci(
                results.bis[i].comparisons,
                results.bis[i].comparisons_ci,
                COL_WIDTH
            ),
            format_int_with_ci(
                results.esm[i].comparisons,
                results.esm[i].comparisons_ci,
                COL_WIDTH
            ),
            format_int_with_ci(
                results.deltasort[i].comparisons,
                results.deltasort[i].comparisons_ci,
                COL_WIDTH
            ),
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
            format_int_with_ci(
                results.native[i].movements,
                results.native[i].movements_ci,
                COL_WIDTH
            ),
            format_int_with_ci(
                results.mergesort[i].movements,
                results.mergesort[i].movements_ci,
                COL_WIDTH
            ),
            format_int_with_ci(
                results.bis[i].movements,
                results.bis[i].movements_ci,
                COL_WIDTH
            ),
            format_int_with_ci(
                results.esm[i].movements,
                results.esm[i].movements_ci,
                COL_WIDTH
            ),
            format_int_with_ci(
                results.deltasort[i].movements,
                results.deltasort[i].movements_ci,
                COL_WIDTH
            ),
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
// CSV EXPORT (with full statistics: mean, SD, CI, CV, iterations)
// ============================================================================

fn export_execution_time_csv(results: &BenchmarkResults, path: &str) {
    let mut csv = String::from("k,iters,native,native_sd,native_ci,native_cv,mergesort,mergesort_sd,mergesort_ci,mergesort_cv,bis,bis_sd,bis_ci,bis_cv,esm,esm_sd,esm_ci,esm_cv,deltasort,deltasort_sd,deltasort_ci,deltasort_cv\n");
    for i in 0..results.native.len() {
        csv.push_str(&format!(
            "{},{},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}\n",
            results.native[i].k,
            results.native[i].iterations,
            results.native[i].time_us,
            results.native[i].time_sd,
            results.native[i].time_ci,
            results.native[i].time_cv,
            results.mergesort[i].time_us,
            results.mergesort[i].time_sd,
            results.mergesort[i].time_ci,
            results.mergesort[i].time_cv,
            results.bis[i].time_us,
            results.bis[i].time_sd,
            results.bis[i].time_ci,
            results.bis[i].time_cv,
            results.esm[i].time_us,
            results.esm[i].time_sd,
            results.esm[i].time_ci,
            results.esm[i].time_cv,
            results.deltasort[i].time_us,
            results.deltasort[i].time_sd,
            results.deltasort[i].time_ci,
            results.deltasort[i].time_cv,
        ));
    }
    fs::write(path, csv).expect("Failed to write execution-time.csv");
    println!("Exported: {}", path);
}

fn export_comparator_count_csv(results: &BenchmarkResults, path: &str) {
    let mut csv = String::from("k,iters,native,native_sd,native_ci,native_cv,mergesort,mergesort_sd,mergesort_ci,mergesort_cv,bis,bis_sd,bis_ci,bis_cv,esm,esm_sd,esm_ci,esm_cv,deltasort,deltasort_sd,deltasort_ci,deltasort_cv\n");
    for i in 0..results.native.len() {
        csv.push_str(&format!(
            "{},{},{:.0},{:.0},{:.0},{:.1},{:.0},{:.0},{:.0},{:.1},{:.0},{:.0},{:.0},{:.1},{:.0},{:.0},{:.0},{:.1},{:.0},{:.0},{:.0},{:.1}\n",
            results.native[i].k,
            results.native[i].iterations,
            results.native[i].comparisons,
            results.native[i].comparisons_sd,
            results.native[i].comparisons_ci,
            results.native[i].comparisons_cv,
            results.mergesort[i].comparisons,
            results.mergesort[i].comparisons_sd,
            results.mergesort[i].comparisons_ci,
            results.mergesort[i].comparisons_cv,
            results.bis[i].comparisons,
            results.bis[i].comparisons_sd,
            results.bis[i].comparisons_ci,
            results.bis[i].comparisons_cv,
            results.esm[i].comparisons,
            results.esm[i].comparisons_sd,
            results.esm[i].comparisons_ci,
            results.esm[i].comparisons_cv,
            results.deltasort[i].comparisons,
            results.deltasort[i].comparisons_sd,
            results.deltasort[i].comparisons_ci,
            results.deltasort[i].comparisons_cv,
        ));
    }
    fs::write(path, csv).expect("Failed to write comparator-count.csv");
    println!("Exported: {}", path);
}

fn export_movement_count_csv(results: &BenchmarkResults, path: &str) {
    let mut csv = String::from("k,iters,native,native_sd,native_ci,native_cv,mergesort,mergesort_sd,mergesort_ci,mergesort_cv,bis,bis_sd,bis_ci,bis_cv,esm,esm_sd,esm_ci,esm_cv,deltasort,deltasort_sd,deltasort_ci,deltasort_cv\n");
    for i in 0..results.native.len() {
        csv.push_str(&format!(
            "{},{},{:.0},{:.0},{:.0},{:.1},{:.0},{:.0},{:.0},{:.1},{:.0},{:.0},{:.0},{:.1},{:.0},{:.0},{:.0},{:.1},{:.0},{:.0},{:.0},{:.1}\n",
            results.native[i].k,
            results.native[i].iterations,
            results.native[i].movements,
            results.native[i].movements_sd,
            results.native[i].movements_ci,
            results.native[i].movements_cv,
            results.mergesort[i].movements,
            results.mergesort[i].movements_sd,
            results.mergesort[i].movements_ci,
            results.mergesort[i].movements_cv,
            results.bis[i].movements,
            results.bis[i].movements_sd,
            results.bis[i].movements_ci,
            results.bis[i].movements_cv,
            results.esm[i].movements,
            results.esm[i].movements_sd,
            results.esm[i].movements_ci,
            results.esm[i].movements_cv,
            results.deltasort[i].movements,
            results.deltasort[i].movements_sd,
            results.deltasort[i].movements_ci,
            results.deltasort[i].movements_cv,
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
        date, machine, N, "20-1000 (scaled by k)"
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
            iterations: native.iterations,
            time_us: native.time_us,
            time_sd: native.time_sd,
            time_ci: native.time_ci,
            time_cv: native.time_cv,
            comparisons: native.comparisons,
            comparisons_sd: native.comparisons_sd,
            comparisons_ci: native.comparisons_ci,
            comparisons_cv: native.comparisons_cv,
            movements: native.movements,
            movements_sd: native.movements_sd,
            movements_ci: native.movements_ci,
            movements_cv: native.movements_cv,
        });

        let ms = run_benchmark(
            &base_users,
            k,
            mergesort_wrapper,
            mergesort_wrapper_counting,
        );
        results.mergesort.push(AlgorithmResult {
            k,
            iterations: ms.iterations,
            time_us: ms.time_us,
            time_sd: ms.time_sd,
            time_ci: ms.time_ci,
            time_cv: ms.time_cv,
            comparisons: ms.comparisons,
            comparisons_sd: ms.comparisons_sd,
            comparisons_ci: ms.comparisons_ci,
            comparisons_cv: ms.comparisons_cv,
            movements: ms.movements,
            movements_sd: ms.movements_sd,
            movements_ci: ms.movements_ci,
            movements_cv: ms.movements_cv,
        });

        let bis = run_benchmark(
            &base_users,
            k,
            binary_insertion_sort,
            binary_insertion_sort_counting,
        );
        results.bis.push(AlgorithmResult {
            k,
            iterations: bis.iterations,
            time_us: bis.time_us,
            time_sd: bis.time_sd,
            time_ci: bis.time_ci,
            time_cv: bis.time_cv,
            comparisons: bis.comparisons,
            comparisons_sd: bis.comparisons_sd,
            comparisons_ci: bis.comparisons_ci,
            comparisons_cv: bis.comparisons_cv,
            movements: bis.movements,
            movements_sd: bis.movements_sd,
            movements_ci: bis.movements_ci,
            movements_cv: bis.movements_cv,
        });

        let esm = run_benchmark(
            &base_users,
            k,
            extract_sort_merge,
            extract_sort_merge_counting,
        );
        results.esm.push(AlgorithmResult {
            k,
            iterations: esm.iterations,
            time_us: esm.time_us,
            time_sd: esm.time_sd,
            time_ci: esm.time_ci,
            time_cv: esm.time_cv,
            comparisons: esm.comparisons,
            comparisons_sd: esm.comparisons_sd,
            comparisons_ci: esm.comparisons_ci,
            comparisons_cv: esm.comparisons_cv,
            movements: esm.movements,
            movements_sd: esm.movements_sd,
            movements_ci: esm.movements_ci,
            movements_cv: esm.movements_cv,
        });

        let ds = run_benchmark(
            &base_users,
            k,
            deltasort_wrapper,
            deltasort_wrapper_counting,
        );
        results.deltasort.push(AlgorithmResult {
            k,
            iterations: ds.iterations,
            time_us: ds.time_us,
            time_sd: ds.time_sd,
            time_ci: ds.time_ci,
            time_cv: ds.time_cv,
            comparisons: ds.comparisons,
            comparisons_sd: ds.comparisons_sd,
            comparisons_ci: ds.comparisons_ci,
            comparisons_cv: ds.comparisons_cv,
            movements: ds.movements,
            movements_sd: ds.movements_sd,
            movements_ci: ds.movements_ci,
            movements_cv: ds.movements_cv,
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

    // --- Segmentation Analysis ---
    println!();
    println!("Running segmentation analysis...");
    let segmentation_results = run_segmentation_analysis();
    print_segmentation_table(&segmentation_results);

    // --- Export CSVs ---
    if export {
        println!();
        println!("Exporting CSV files...");
        let base_path = "../paper/figures/rust";
        let figures_path = "../paper/figures";
        fs::create_dir_all(base_path).ok();
        fs::create_dir_all(figures_path).ok();
        export_execution_time_csv(&results, &format!("{}/execution-time.csv", base_path));
        export_comparator_count_csv(&results, &format!("{}/comparator-count.csv", base_path));
        export_movement_count_csv(&results, &format!("{}/movement-count.csv", base_path));
        export_crossover_csv(
            &crossover_results,
            &format!("{}/crossover-threshold.csv", base_path),
        );
        export_metadata_csv(&format!("{}/benchmark_metadata.csv", base_path));
        // Segmentation goes in root figures folder (language-independent analysis)
        export_segmentation_csv(&segmentation_results, figures_path);
    }

    println!();
    println!("Done!");
    if !export {
        println!("Run with --export to write CSV files to paper/figures/rust/");
    }
}
