//! DeltaSort Benchmark Suite
//!
//! Run with: `cargo run --bin benchmark --release`
//! Export CSV: `cargo run --bin benchmark --release -- --export`
mod binary_insertion_sort;
mod data;
mod extract_sort_merge;
mod statistics;

use crate::binary_insertion_sort::binary_insertion_sort;
use crate::data::{
    counting_comparator, generate_sorted_users, get_comparison_count, reset_comparison_count,
    sample_distinct_indices, user_comparator, User,
};
use crate::extract_sort_merge::extract_sort_merge;
use crate::statistics::{calculate_stats, calculate_stats_u64};
use deltasort::delta_sort_by;
use std::collections::HashSet;
use std::fs;
use std::io::{self, Write};
use std::time::Instant;

// ============================================================================
// BENCHMARK CONFIGURATION
// ============================================================================

/// Array size for main benchmarks
const N: usize = 100_000;

/// Base number of iterations per benchmark (scaled up for small k)
const BASE_ITERATIONS: usize = 100;

/// Delta counts to test
const DELTA_COUNTS: &[usize] = &[
    1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000,
];

/// Number of iterations for crossover measurements (higher = more stable but slower)
const CROSSOVER_ITERATIONS: usize = 10;

/// Array sizes for crossover analysis
const CROSSOVER_SIZES: &[usize] = &[
    1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000,
];

/// Get number of iterations for a given k value
/// Small k values need more iterations due to timer resolution
fn timing_iterations_for_k(k: usize) -> usize {
    match k {
        0..=5 => BASE_ITERATIONS * 50,   // 1000 iterations for k <= 10
        6..=10 => BASE_ITERATIONS * 10,  // 1000 iterations for k <= 10
        11..=50 => BASE_ITERATIONS * 5,  // 500 iterations for k <= 50
        51..=200 => BASE_ITERATIONS * 2, // 200 iterations for k <= 200
        _ => BASE_ITERATIONS,            // 100 iterations for large k
    }
}

/// Get number of comparison iterations for a given k value
/// Small k values need more iterations due to timer resolution
fn comparison_iterations_for_k(k: usize) -> usize {
    match k {
        0..=50 => BASE_ITERATIONS, // 1000 iterations for k <= 10
        _ => BASE_ITERATIONS / 5,  // 100 iterations for large k
    }
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
    iterations: usize,
}

/// Measure timing using non-counting comparator (accurate timing)
/// Measure comparisons using counting versions (separate runs)
/// Each iteration generates fresh random mutations for proper variance measurement
fn run_benchmark<F>(base_users: &[User], k: usize, mut sort_fn: F) -> BenchmarkResult
where
    F: FnMut(&mut Vec<User>, &HashSet<usize>, fn(&User, &User) -> std::cmp::Ordering),
{
    let mut rng = rand::thread_rng();
    let n = base_users.len();
    let mut iters = timing_iterations_for_k(k);

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
        sort_fn(&mut users, &dirty_indices, user_comparator);
        times_us.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }

    // Phase 2: Measure comparisons (separate runs with fresh mutations)
    iters = comparison_iterations_for_k(k);
    let mut comparisons = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut users = base_users.to_vec();
        let indices = sample_distinct_indices(&mut rng, n, k);
        let mut dirty_indices = HashSet::with_capacity(k);
        for idx in indices {
            users[idx].mutate(&mut rng);
            dirty_indices.insert(idx);
        }
        reset_comparison_count();
        sort_fn(&mut users, &dirty_indices, counting_comparator);
        comparisons.push(get_comparison_count());
    }

    let time_stats = calculate_stats(&times_us);
    let cmp_stats = calculate_stats_u64(&comparisons);

    BenchmarkResult {
        time_us: time_stats.mean,
        time_sd: time_stats.sd,
        time_ci: time_stats.ci_95,
        time_cv: time_stats.cv,
        comparisons: cmp_stats.mean,
        comparisons_sd: cmp_stats.sd,
        comparisons_ci: cmp_stats.ci_95,
        comparisons_cv: cmp_stats.cv,
        iterations: iters,
    }
}

// ============================================================================
// CROSSOVER ANALYSIS
// ============================================================================

/// Compare algorithm timing against native sort
fn algorithm_is_faster<F>(base_users: &[User], k: usize, n: usize, mut algo: F) -> bool
where
    F: FnMut(&mut Vec<User>, &HashSet<usize>),
{
    let mut rng = rand::thread_rng();

    let mut native_time = 0.0;
    let mut algo_time = 0.0;

    for _ in 0..CROSSOVER_ITERATIONS {
        let mut users = base_users.to_vec();
        let indices = sample_distinct_indices(&mut rng, n, k);
        let mut dirty_indices = HashSet::with_capacity(k);
        for idx in indices {
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
        algo(&mut test_users, &dirty_indices);
        algo_time += start.elapsed().as_secs_f64();
    }

    algo_time < native_time
}

/// Compare DeltaSort timing against ESM
fn deltasort_is_faster_than_esm(base_users: &[User], k: usize, n: usize) -> bool {
    let mut rng = rand::thread_rng();

    let mut esm_time = 0.0;
    let mut ds_time = 0.0;

    for _ in 0..CROSSOVER_ITERATIONS {
        let mut users = base_users.to_vec();
        let indices = sample_distinct_indices(&mut rng, n, k);
        let mut dirty_indices = HashSet::with_capacity(k);
        for idx in indices {
            users[idx].mutate(&mut rng);
            dirty_indices.insert(idx);
        }

        let start = Instant::now();
        let mut test_users = users.clone();
        extract_sort_merge(&mut test_users, &dirty_indices, user_comparator);
        esm_time += start.elapsed().as_secs_f64();

        let start = Instant::now();
        let mut test_users = users.clone();
        delta_sort_by(&mut test_users, &dirty_indices, user_comparator);
        ds_time += start.elapsed().as_secs_f64();
    }

    ds_time < esm_time
}

fn deltasort_is_faster(base_users: &[User], k: usize, n: usize) -> bool {
    algorithm_is_faster(base_users, k, n, |arr, indices| {
        delta_sort_by(arr.as_mut_slice(), indices, user_comparator);
    })
}

fn bis_is_faster(base_users: &[User], k: usize, n: usize) -> bool {
    algorithm_is_faster(base_users, k, n, |arr, indices| {
        binary_insertion_sort(arr, indices, user_comparator);
    })
}

fn esm_is_faster(base_users: &[User], k: usize, n: usize) -> bool {
    algorithm_is_faster(base_users, k, n, |arr, indices| {
        extract_sort_merge(arr, indices, user_comparator);
    })
}

/// Generic crossover finder using binary search
/// lo_ratio and hi_ratio define the search range as fractions of n (0.0 to 1.0)
fn find_crossover_generic<F>(n: usize, lo_ratio: f64, hi_ratio: f64, is_faster: F) -> usize
where
    F: Fn(&[User], usize, usize) -> bool,
{
    let base_users = generate_sorted_users(n);

    // Warmup
    for _ in 0..5 {
        let mut users = base_users.clone();
        users.sort_by(user_comparator);
    }

    let mut lo: usize = ((n as f64) * lo_ratio).max(1.0) as usize;
    let mut hi: usize = ((n as f64) * hi_ratio) as usize;

    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if is_faster(&base_users, mid, n) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    lo
}

fn find_crossover(n: usize) -> usize {
    find_crossover_generic(n, 0.0, 0.5, deltasort_is_faster)
}

fn find_crossover_bis(n: usize) -> usize {
    find_crossover_generic(n, 0.0, if n < 5000 { 0.1 } else { 0.01 }, bis_is_faster)
}

fn find_crossover_esm(n: usize) -> usize {
    find_crossover_generic(n, if n > 5000 { 0.7 } else { 0.6 }, 0.95, esm_is_faster)
}

/// Find crossover where DeltaSort becomes slower than ESM
fn find_crossover_deltasort_vs_esm(n: usize) -> usize {
    let base_users = generate_sorted_users(n);

    // Warmup
    for _ in 0..5 {
        let mut users = base_users.clone();
        users.sort_by(user_comparator);
    }

    let mut lo: usize = 1;
    let mut hi: usize = n;

    let min_range = (n as f64 * 0.001) as usize;

    while lo < hi {
        if hi - lo < min_range {
            break;
        }

        let mid = lo + (hi - lo).div_ceil(2);

        if deltasort_is_faster_than_esm(&base_users, mid, n) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    lo
}

// ============================================================================
// SEGMENTATION ANALYSIS (NOT USED)
// ============================================================================

// Compute segment boundaries for a given array and dirty indices
//
// A segment is defined by its violation types:
// - Trailing segment: starts at 0, contains only L violations (no R in between)
// - Leading segment: ends at n-1, contains only R violations (no L after)
// - Intermediate segment: contains R violations followed by L violations
//
// Algorithm: Start at 0, run past all R's, when L is found run past all L's
// keeping track of farthest L encountered. When you hit R or end of array,
// record segment from start to farthest L. If hit R, it becomes start of next segment.
//
// Returns (segment_count, total_segment_size)

/*
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
*/

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
}

struct BenchmarkResults {
    native: Vec<AlgorithmResult>,
    bis: Vec<AlgorithmResult>,
    esm: Vec<AlgorithmResult>,
    deltasort: Vec<AlgorithmResult>,
}

/// Extended crossover results for all algorithms vs native
struct CrossoverResultsAll {
    n: usize,
    bis_k_c: usize,
    bis_ratio: f64,
    esm_k_c: usize,
    esm_ratio: f64,
    deltasort_k_c: usize,
    deltasort_ratio: f64,
}

/// Crossover result for DeltaSort vs ESM
struct CrossoverResultDeltaVsEsm {
    n: usize,
    k_c: usize,
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

/// Format value ± ci% with consistent total width
fn format_with_ci(value: f64, ci: f64, total_width: usize) -> String {
    let val_str = format!("{:.1}", value);
    let ci_percent = if value > 0.0 {
        (ci / value) * 100.0
    } else {
        0.0
    };
    let content = format!("{} ±{:.1}%", val_str, ci_percent);
    format!("{:>width$}", content, width = total_width)
}

/// Format integer value ± ci% with consistent total width  
fn format_int_with_ci(value: f64, ci: f64, total_width: usize) -> String {
    let val_str = format!("{:.0}", value);
    let ci_percent = if value > 0.0 {
        (ci / value) * 100.0
    } else {
        0.0
    };
    let content = format!("{} ±{:.1}%", val_str, ci_percent);
    format!("{:>width$}", content, width = total_width)
}

fn print_execution_time_table(results: &BenchmarkResults) {
    const COL_WIDTH: usize = 15;

    println!();
    println!("Execution Time (µs) - n={}", format_number(N));
    println!("┌────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┐");
    println!("│   k    │     Native      │       BIS       │       ESM       │    DeltaSort    │");
    println!("├────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┤");
    for i in 0..results.native.len() {
        println!(
            "│ {:>6} │ {} │ {} │ {} │ {} │",
            results.native[i].k,
            format_with_ci(
                results.native[i].time_us,
                results.native[i].time_ci,
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
    println!("└────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┘");
}

fn print_comparator_count_table(results: &BenchmarkResults) {
    const COL_WIDTH: usize = 15;

    println!();
    println!("Comparator Invocations - n={}", format_number(N));
    println!("┌────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┐");
    println!("│   k    │     Native      │       BIS       │       ESM       │    DeltaSort    │");
    println!("├────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┤");
    for i in 0..results.native.len() {
        println!(
            "│ {:>6} │ {} │ {} │ {} │ {} │",
            results.native[i].k,
            format_int_with_ci(
                results.native[i].comparisons,
                results.native[i].comparisons_ci,
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
    println!("└────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┘");
}

fn print_crossover_table_all(results: &[CrossoverResultsAll]) {
    println!();
    println!("Crossover Threshold (All Algorithms vs Native)");
    println!("┌────────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐");
    println!("│     n      │  BIS k_c   │  BIS k_c%  │  ESM k_c   │  ESM k_c%  │   DS k_c   │  DS k_c%   │");
    println!("├────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤");
    for r in results {
        println!(
            "│ {:>10} │ {:>10} │ {:>9.3}% │ {:>10} │ {:>9.3}% │ {:>10} │ {:>9.3}% │",
            format_number(r.n),
            format_number(r.bis_k_c),
            r.bis_ratio,
            format_number(r.esm_k_c),
            r.esm_ratio,
            format_number(r.deltasort_k_c),
            r.deltasort_ratio
        );
    }
    println!("└────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘");
}

fn print_crossover_table_ds_vs_esm(results: &[CrossoverResultDeltaVsEsm]) {
    println!();
    println!("Crossover Threshold (DeltaSort vs ESM)");
    println!("┌────────────┬────────────┬────────────┐");
    println!("│     n      │    k_c     │   k_c %    │");
    println!("├────────────┼────────────┼────────────┤");
    for r in results {
        println!(
            "│ {:>10} │ {:>10} │ {:>9.3}% │",
            format_number(r.n),
            format_number(r.k_c),
            r.crossover_ratio
        );
    }
    println!("└────────────┴────────────┴────────────┘");
}

// ============================================================================
// CSV EXPORT (with full statistics: mean, SD, CI, CV, iterations)
// ============================================================================

fn export_execution_time_csv(results: &BenchmarkResults, path: &str) {
    let mut csv = String::from("k,iters,native,native_sd,native_ci,native_cv,bis,bis_sd,bis_ci,bis_cv,esm,esm_sd,esm_ci,esm_cv,deltasort,deltasort_sd,deltasort_ci,deltasort_cv\n");
    for i in 0..results.native.len() {
        csv.push_str(&format!(
            "{},{},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}\n",
            results.native[i].k,
            results.native[i].iterations,
            results.native[i].time_us,
            results.native[i].time_sd,
            results.native[i].time_ci,
            results.native[i].time_cv,
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
    let mut csv = String::from("k,iters,native,native_sd,native_ci,native_cv,bis,bis_sd,bis_ci,bis_cv,esm,esm_sd,esm_ci,esm_cv,deltasort,deltasort_sd,deltasort_ci,deltasort_cv\n");
    for i in 0..results.native.len() {
        csv.push_str(&format!(
            "{},{},{:.0},{:.0},{:.0},{:.1},{:.0},{:.0},{:.0},{:.1},{:.0},{:.0},{:.0},{:.1},{:.0},{:.0},{:.0},{:.1}\n",
            results.native[i].k,
            results.native[i].iterations,
            results.native[i].comparisons,
            results.native[i].comparisons_sd,
            results.native[i].comparisons_ci,
            results.native[i].comparisons_cv,
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

fn export_crossover_all_csv(results: &[CrossoverResultsAll], path: &str) {
    let mut csv = String::from("n,bis_kc,bis,esm_kc,esm,deltasort_kc,deltasort\n");
    for r in results {
        csv.push_str(&format!(
            "{},{},{:.3},{},{:.3},{},{:.3}\n",
            r.n, r.bis_k_c, r.bis_ratio, r.esm_k_c, r.esm_ratio, r.deltasort_k_c, r.deltasort_ratio
        ));
    }
    fs::write(path, csv).expect("Failed to write crossover-all.csv");
    println!("Exported: {}", path);
}

fn export_crossover_ds_vs_esm_csv(results: &[CrossoverResultDeltaVsEsm], path: &str) {
    let mut csv = String::from("n,kc,crossover_ratio\n");
    for r in results {
        csv.push_str(&format!("{},{},{:.3}\n", r.n, r.k_c, r.crossover_ratio));
    }
    fs::write(path, csv).expect("Failed to write crossover-ds-vs-esm.csv");
    println!("Exported: {}", path);
}

fn export_metadata_csv(results: &BenchmarkResults, path: &str) {
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};

    // Get timestamp
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);

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

    // Compute max CI as percentage of mean across all timing measurements
    // This represents our least confident measurement (can be narrowed with more iterations)
    let mut max_ci_percent: f64 = 0.0;
    for r in &results.native {
        if r.time_us > 0.0 {
            max_ci_percent = max_ci_percent.max((r.time_ci / r.time_us) * 100.0);
        }
    }
    for r in &results.bis {
        if r.time_us > 0.0 {
            max_ci_percent = max_ci_percent.max((r.time_ci / r.time_us) * 100.0);
        }
    }
    for r in &results.esm {
        if r.time_us > 0.0 {
            max_ci_percent = max_ci_percent.max((r.time_ci / r.time_us) * 100.0);
        }
    }
    for r in &results.deltasort {
        if r.time_us > 0.0 {
            max_ci_percent = max_ci_percent.max((r.time_ci / r.time_us) * 100.0);
        }
    }

    let csv = format!(
        "key,value\ntimestamp,{}\nmachine,{}\nn,{}\nmax_ci,{:.2}\n",
        timestamp, machine, N, max_ci_percent
    );
    fs::write(path, csv).expect("Failed to write metadata.csv");
    println!("Exported: {}", path);
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
    println!("Running benchmarks (time + comparisons)...");
    let mut results = BenchmarkResults {
        native: Vec::new(),
        bis: Vec::new(),
        esm: Vec::new(),
        deltasort: Vec::new(),
    };

    for &k in DELTA_COUNTS {
        print!("  k={:>5}...", k);
        io::stdout().flush().unwrap();

        let native = run_benchmark(&base_users, k, |arr, _indices, cmp| {
            arr.sort_by(cmp);
        });
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
        });

        let bis = run_benchmark(&base_users, k, binary_insertion_sort);
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
        });

        let esm = run_benchmark(&base_users, k, extract_sort_merge);
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
        });

        let ds = run_benchmark(&base_users, k, |arr, indices, cmp| {
            delta_sort_by(arr.as_mut_slice(), indices, cmp)
        });

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
        });

        println!(" done");
    }

    print_execution_time_table(&results);
    print_comparator_count_table(&results);

    // --- Crossover Analysis (All Algorithms vs Native) ---
    println!();
    println!("Running crossover analysis: All algorithms vs Native...");
    let mut crossover_all_results: Vec<CrossoverResultsAll> = Vec::new();

    for &size in CROSSOVER_SIZES {
        print!("  n={:>10}...", format_number(size));
        io::stdout().flush().unwrap();

        let bis_k_c = find_crossover_bis(size);
        let esm_k_c = find_crossover_esm(size);
        let ds_k_c = find_crossover(size);

        crossover_all_results.push(CrossoverResultsAll {
            n: size,
            bis_k_c,
            bis_ratio: (bis_k_c as f64 / size as f64) * 100.0,
            esm_k_c,
            esm_ratio: (esm_k_c as f64 / size as f64) * 100.0,
            deltasort_k_c: ds_k_c,
            deltasort_ratio: (ds_k_c as f64 / size as f64) * 100.0,
        });
        println!(
            " BIS={:.1}%, ESM={:.1}%, DS={:.1}%",
            (bis_k_c as f64 / size as f64) * 100.0,
            (esm_k_c as f64 / size as f64) * 100.0,
            (ds_k_c as f64 / size as f64) * 100.0
        );
    }

    print_crossover_table_all(&crossover_all_results);

    // --- Crossover Analysis (DeltaSort vs ESM) ---
    println!();
    println!("Running crossover analysis: DeltaSort vs ESM...");
    let mut crossover_ds_vs_esm_results: Vec<CrossoverResultDeltaVsEsm> = Vec::new();

    for &size in CROSSOVER_SIZES {
        print!("  n={:>10}...", format_number(size));
        io::stdout().flush().unwrap();
        let k_c = find_crossover_deltasort_vs_esm(size);
        let crossover_ratio = (k_c as f64 / size as f64) * 100.0;
        crossover_ds_vs_esm_results.push(CrossoverResultDeltaVsEsm {
            n: size,
            k_c,
            crossover_ratio,
        });
        println!(" k_c={} ({:.1}%)", k_c, crossover_ratio);
    }

    print_crossover_table_ds_vs_esm(&crossover_ds_vs_esm_results);

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
        export_crossover_all_csv(
            &crossover_all_results,
            &format!("{}/crossover-all.csv", base_path),
        );
        export_crossover_ds_vs_esm_csv(
            &crossover_ds_vs_esm_results,
            &format!("{}/crossover-ds-vs-esm.csv", base_path),
        );
        export_metadata_csv(&results, &format!("{}/benchmark_metadata.csv", base_path));
    }

    println!();
    println!("Done!");
    if !export {
        println!("Run with --export to write CSV files to paper/figures/rust/");
    }
}
