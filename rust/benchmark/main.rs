//! DeltaSort Benchmark Suite
//!
//! Run with: `cargo run --bin benchmark --release -- [FLAGS]`
//!
//! Flags:
//!   --time       Run execution time benchmark
//!   --crossover  Run crossover analysis
//!   --comparator Run comparator count analysis (instrumented)
//!   --movement   Run movement analysis (instrumented)
//!   --export     Export results to CSV files
//!
//! If no flags are provided, runs all benchmarks.

mod binary_insertion_sort;
mod data;
mod extract_sort_merge;
mod instrumented_deltasort;
mod statistics;

use crate::binary_insertion_sort::binary_insertion_sort_hybrid;
use crate::data::{
    counting_comparator, generate_sorted_users, get_comparison_count, reset_comparison_count,
    sample_distinct_indices, user_comparator, User,
};
use crate::extract_sort_merge::extract_sort_merge;
use crate::instrumented_deltasort::{delta_sort_instrumented, InstrumentedResult};
use crate::statistics::{calculate_stats, calculate_stats_u64};
use deltasort::delta_sort_by;
use rand::Rng;
use std::collections::HashSet;
use std::fs;
use std::io::{self, Write};
use std::time::Instant;

// ============================================================================
// BENCHMARK CONFIGURATION
// ============================================================================

/// Array size for main benchmarks
const N: usize = 100_000;

/// Maximum k value for BIS benchmarks (BIS is O(kn) for moves, too slow for large k)
const BIS_MAX_K: usize = 2000;

/// Base number of iterations per benchmark (scaled up for small k)
const BASE_ITERATIONS: usize = 100;

/// Delta counts to test for time benchmarks
const DELTA_COUNTS: &[usize] = &[
    1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000,
];

/// Number of iterations for crossover measurements (higher = more stable but slower)
const CROSSOVER_ITERATIONS: usize = 10;

/// Array sizes for crossover analysis
const CROSSOVER_SIZES: &[usize] = &[
    1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000,
];

/// Array sizes for movement/comparator analysis
const ANALYSIS_N_VALUES: &[usize] = &[1_000, 10_000, 100_000, 1_000_000];

/// k percentages for movement/comparator analysis
const ANALYSIS_K_FRACTIONS: &[f64] = &[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2];

/// Iterations for analysis
const ANALYSIS_ITERATIONS: usize = 100;

/// Get number of iterations for a given k value
fn timing_iterations_for_k(k: usize) -> usize {
    match k {
        1 => BASE_ITERATIONS * 100,
        2..=5 => BASE_ITERATIONS * 50,
        6..=10 => BASE_ITERATIONS * 10,
        11..=50 => BASE_ITERATIONS * 5,
        51..=1000 => BASE_ITERATIONS * 2,
        _ => BASE_ITERATIONS,
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
    iterations: usize,
}

/// Measure timing using non-counting comparator (accurate timing)
fn run_benchmark<F>(base_users: &[User], k: usize, mut sort_fn: F) -> BenchmarkResult
where
    F: FnMut(&mut Vec<User>, &HashSet<usize>, fn(&User, &User) -> std::cmp::Ordering),
{
    let mut rng = rand::thread_rng();
    let n = base_users.len();
    let iters = timing_iterations_for_k(k);

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

    let time_stats = calculate_stats(&times_us);

    BenchmarkResult {
        time_us: time_stats.mean,
        time_sd: time_stats.sd,
        time_ci: time_stats.ci_95,
        time_cv: time_stats.cv,
        iterations: iters,
    }
}

// ============================================================================
// CROSSOVER ANALYSIS
// ============================================================================

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

fn deltasort_is_faster(base_users: &[User], k: usize, n: usize) -> bool {
    algorithm_is_faster(base_users, k, n, |arr, indices| {
        delta_sort_by(arr.as_mut_slice(), indices, user_comparator);
    })
}

fn bis_is_faster(base_users: &[User], k: usize, n: usize) -> bool {
    algorithm_is_faster(base_users, k, n, |arr, indices| {
        binary_insertion_sort_hybrid(arr, indices, user_comparator);
    })
}

fn esm_is_faster(base_users: &[User], k: usize, n: usize) -> bool {
    algorithm_is_faster(base_users, k, n, |arr, indices| {
        extract_sort_merge(arr, indices, user_comparator);
    })
}

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
    find_crossover_generic(n, 0.0, 0.1, bis_is_faster)
}

fn find_crossover_esm(n: usize) -> usize {
    find_crossover_generic(n, if n > 5000 { 0.7 } else { 0.6 }, 0.95, esm_is_faster)
}

// ============================================================================
// INSTRUMENTED ANALYSIS (MOVEMENT & COMPARATOR)
// ============================================================================

#[derive(Debug)]
struct AnalysisResult {
    n: usize,
    k: usize,
    k_percent: f64,
    // Movement analysis
    movement_mean: f64,
    movement_sd: f64,
    movement_normalized: f64, // movement / (n * sqrt(k))
    // Segment analysis
    segments_mean: f64,
    segments_sd: f64,
    segments_normalized: f64, // segments / sqrt(k)
    // Comparator analysis (for DeltaSort)
    comparisons_mean: f64,
    comparisons_sd: f64,
    comparisons_normalized: f64, // comparisons / (k * log(n * sqrt(k)))
}

fn run_analysis(n: usize, k: usize) -> AnalysisResult {
    let mut rng = rand::thread_rng();
    let base_arr: Vec<i32> = (0..n as i32).collect();

    let mut movements: Vec<usize> = Vec::with_capacity(ANALYSIS_ITERATIONS);
    let mut segments: Vec<usize> = Vec::with_capacity(ANALYSIS_ITERATIONS);
    let mut comparisons: Vec<u64> = Vec::with_capacity(ANALYSIS_ITERATIONS);

    for _ in 0..ANALYSIS_ITERATIONS {
        let mut arr = base_arr.clone();
        let indices: Vec<usize> = {
            let mut idx: Vec<usize> = (0..n).collect();
            for i in 0..k {
                let j = rng.gen_range(i..n);
                idx.swap(i, j);
            }
            idx.truncate(k);
            idx
        };
        let mut dirty_indices = HashSet::with_capacity(k);

        for idx in indices {
            arr[idx] = rng.gen_range(0..n as i32);
            dirty_indices.insert(idx);
        }

        let result: InstrumentedResult =
            delta_sort_instrumented(&mut arr, &dirty_indices, i32::cmp);
        movements.push(result.movement);
        segments.push(result.segments);

        // For comparisons, use the counting comparator on User data
        let base_users = generate_sorted_users(n);
        let mut users = base_users.clone();
        let indices: Vec<usize> = {
            let mut idx: Vec<usize> = (0..n).collect();
            for i in 0..k {
                let j = rng.gen_range(i..n);
                idx.swap(i, j);
            }
            idx.truncate(k);
            idx
        };
        let mut dirty_indices = HashSet::with_capacity(k);
        for idx in indices {
            users[idx].mutate(&mut rng);
            dirty_indices.insert(idx);
        }
        reset_comparison_count();
        delta_sort_by(&mut users, &dirty_indices, counting_comparator);
        comparisons.push(get_comparison_count());
    }

    let mov_stats = calculate_stats_usize(&movements);
    let seg_stats = calculate_stats_usize(&segments);
    let cmp_stats = calculate_stats_u64(&comparisons);

    let k_sqrt = (k as f64).sqrt();
    let log_n_sqrt_k = (n as f64 * k_sqrt).ln();

    AnalysisResult {
        n,
        k,
        k_percent: (k as f64 / n as f64) * 100.0,
        movement_mean: mov_stats.mean,
        movement_sd: mov_stats.sd,
        movement_normalized: mov_stats.mean / (n as f64 * k_sqrt),
        segments_mean: seg_stats.mean,
        segments_sd: seg_stats.sd,
        segments_normalized: seg_stats.mean / k_sqrt,
        comparisons_mean: cmp_stats.mean,
        comparisons_sd: cmp_stats.sd,
        comparisons_normalized: cmp_stats.mean / (k as f64 * log_n_sqrt_k),
    }
}

fn calculate_stats_usize(values: &[usize]) -> statistics::Stats {
    let floats: Vec<f64> = values.iter().map(|&x| x as f64).collect();
    calculate_stats(&floats)
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
}

struct BenchmarkResults {
    native: Vec<AlgorithmResult>,
    bis: Vec<Option<AlgorithmResult>>,
    esm: Vec<AlgorithmResult>,
    deltasort: Vec<AlgorithmResult>,
}

struct CrossoverResultsAll {
    n: usize,
    bis_k_c: usize,
    bis_ratio: f64,
    esm_k_c: usize,
    esm_ratio: f64,
    deltasort_k_c: usize,
    deltasort_ratio: f64,
}

// ============================================================================
// OUTPUT HELPERS
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

// ============================================================================
// EXECUTION TIME BENCHMARK
// ============================================================================

fn run_time_benchmark(export: bool) {
    println!();
    println!("Execution Time Benchmark");
    println!("========================");

    let base_users = generate_sorted_users(N);

    // Warmup
    for _ in 0..5 {
        let mut users = base_users.clone();
        users.sort_by(user_comparator);
    }

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
        });

        if k <= BIS_MAX_K {
            let bis = run_benchmark(&base_users, k, binary_insertion_sort_hybrid);
            results.bis.push(Some(AlgorithmResult {
                k,
                iterations: bis.iterations,
                time_us: bis.time_us,
                time_sd: bis.time_sd,
                time_ci: bis.time_ci,
                time_cv: bis.time_cv,
            }));
        } else {
            results.bis.push(None);
        }

        let esm = run_benchmark(&base_users, k, extract_sort_merge);
        results.esm.push(AlgorithmResult {
            k,
            iterations: esm.iterations,
            time_us: esm.time_us,
            time_sd: esm.time_sd,
            time_ci: esm.time_ci,
            time_cv: esm.time_cv,
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
        });

        println!(" done");
    }

    print_execution_time_table(&results);

    if export {
        let base_path = "../paper/figures/rust";
        fs::create_dir_all(base_path).ok();
        export_execution_time_csv(&results, &format!("{}/execution-time.csv", base_path));
        export_metadata_csv(&results, &format!("{}/benchmark_metadata.csv", base_path));
    }
}

fn print_execution_time_table(results: &BenchmarkResults) {
    const COL_WIDTH: usize = 15;

    println!();
    println!("Execution Time (µs) - n={}", format_number(N));
    println!("┌────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┐");
    println!("│   k    │     Native      │       BIS       │       ESM       │    DeltaSort    │");
    println!("├────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┤");
    for i in 0..results.native.len() {
        let bis_str = match &results.bis[i] {
            Some(bis) => format_with_ci(bis.time_us, bis.time_ci, COL_WIDTH),
            None => format!("{:>width$}", "-", width = COL_WIDTH),
        };
        println!(
            "│ {:>6} │ {} │ {} │ {} │ {} │",
            results.native[i].k,
            format_with_ci(
                results.native[i].time_us,
                results.native[i].time_ci,
                COL_WIDTH
            ),
            bis_str,
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

fn export_execution_time_csv(results: &BenchmarkResults, path: &str) {
    let mut csv = String::from("k,iters,native,native_sd,native_ci,native_cv,bis,bis_sd,bis_ci,bis_cv,esm,esm_sd,esm_ci,esm_cv,deltasort,deltasort_sd,deltasort_ci,deltasort_cv\n");
    for i in 0..results.native.len() {
        let bis_cols = match &results.bis[i] {
            Some(bis) => format!(
                "{:.1},{:.1},{:.1},{:.1}",
                bis.time_us, bis.time_sd, bis.time_ci, bis.time_cv
            ),
            None => ",,,".to_string(),
        };
        csv.push_str(&format!(
            "{},{},{:.1},{:.1},{:.1},{:.1},{},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}\n",
            results.native[i].k,
            results.native[i].iterations,
            results.native[i].time_us,
            results.native[i].time_sd,
            results.native[i].time_ci,
            results.native[i].time_cv,
            bis_cols,
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

fn export_metadata_csv(results: &BenchmarkResults, path: &str) {
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);

    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

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

    let mut max_ci_percent: f64 = 0.0;
    for r in &results.native {
        if r.time_us > 0.0 {
            max_ci_percent = max_ci_percent.max((r.time_ci / r.time_us) * 100.0);
        }
    }
    for r in results.bis.iter().flatten() {
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
// CROSSOVER BENCHMARK
// ============================================================================

fn run_crossover_benchmark(export: bool) {
    println!();
    println!("Crossover Benchmark");
    println!("===================");

    // --- Crossover Analysis (All Algorithms vs Native) ---
    println!();
    println!("Running crossover analysis: All algorithms vs Native...");
    let mut crossover_all: Vec<CrossoverResultsAll> = Vec::new();

    for &size in CROSSOVER_SIZES {
        print!("  n={:>10}...", format_number(size));
        io::stdout().flush().unwrap();

        let bis_k_c = find_crossover_bis(size);
        let esm_k_c = find_crossover_esm(size);
        let ds_k_c = find_crossover(size);

        crossover_all.push(CrossoverResultsAll {
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

    print_crossover_table_all(&crossover_all);

    if export {
        let base_path = "../paper/figures/rust";
        fs::create_dir_all(base_path).ok();
        export_crossover_all_csv(&crossover_all, &format!("{}/crossover-all.csv", base_path));
    }
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

// ============================================================================
// COMPARATOR COUNT ANALYSIS
// ============================================================================

fn run_comparator_analysis(export: bool) {
    println!();
    println!("Comparator Count Analysis");
    println!("=========================");
    println!("Normalized by k·log(n·√k) to validate O(k·log(n·√k)) complexity");
    println!();

    let mut results: Vec<AnalysisResult> = Vec::new();

    for &n in ANALYSIS_N_VALUES {
        for &k_pct in ANALYSIS_K_FRACTIONS {
            let k = ((n as f64) * k_pct).round() as usize;
            if k < 1 {
                continue;
            }
            print!(
                "  n={:>7}, k={:>7} ({:>5.1}%)...",
                format_number(n),
                format_number(k),
                k_pct * 100.0
            );
            io::stdout().flush().unwrap();
            let result = run_analysis(n, k);
            println!(
                " cmp={:.0}, norm={:.4}",
                result.comparisons_mean, result.comparisons_normalized
            );
            results.push(result);
        }
    }

    print_comparator_table(&results);

    if export {
        let base_path = "../paper/figures";
        fs::create_dir_all(base_path).ok();
        export_comparator_csv(&results, &format!("{}/comparator-analysis.csv", base_path));
    }
}

fn print_comparator_table(results: &[AnalysisResult]) {
    println!();
    println!("Comparator Count (normalized by k·log(n·√k))");
    println!("┌────────────┬────────────┬──────────┬─────────────────┬────────────┐");
    println!("│     n      │     k      │   k %    │   Comparisons   │ Normalized │");
    println!("├────────────┼────────────┼──────────┼─────────────────┼────────────┤");
    for r in results {
        println!(
            "│ {:>10} │ {:>10} │ {:>7.2}% │ {:>15.0} │ {:>10.4} │",
            format_number(r.n),
            format_number(r.k),
            r.k_percent,
            r.comparisons_mean,
            r.comparisons_normalized
        );
    }
    println!("└────────────┴────────────┴──────────┴─────────────────┴────────────┘");
}

fn export_comparator_csv(results: &[AnalysisResult], path: &str) {
    let mut csv = String::from("n,k,k_percent,comparisons,comparisons_sd,normalized\n");
    for r in results {
        csv.push_str(&format!(
            "{},{},{:.4},{:.0},{:.0},{:.6}\n",
            r.n, r.k, r.k_percent, r.comparisons_mean, r.comparisons_sd, r.comparisons_normalized
        ));
    }
    fs::write(path, csv).expect("Failed to write comparator-analysis.csv");
    println!("Exported: {}", path);
}

// ============================================================================
// MOVEMENT ANALYSIS
// ============================================================================

fn run_movement_analysis(export: bool) {
    println!();
    println!("Movement Analysis");
    println!("=================");
    println!("Normalized by n·√k to validate O(n·√k) complexity");
    println!();

    let mut results: Vec<AnalysisResult> = Vec::new();

    for &n in ANALYSIS_N_VALUES {
        for &k_pct in ANALYSIS_K_FRACTIONS {
            let k = ((n as f64) * k_pct).round() as usize;
            if k < 1 {
                continue;
            }
            print!(
                "  n={:>7}, k={:>7} ({:>5.1}%)...",
                format_number(n),
                format_number(k),
                k_pct * 100.0
            );
            io::stdout().flush().unwrap();
            let result = run_analysis(n, k);
            println!(
                " mov={:.0}, seg={:.1}, mov_norm={:.4}, seg_norm={:.4}",
                result.movement_mean,
                result.segments_mean,
                result.movement_normalized,
                result.segments_normalized
            );
            results.push(result);
        }
    }

    print_movement_table(&results);
    print_segments_table(&results);

    if export {
        let base_path = "../paper/figures";
        fs::create_dir_all(base_path).ok();
        export_movement_csv(&results, &format!("{}/movement-analysis.csv", base_path));
    }
}

fn print_movement_table(results: &[AnalysisResult]) {
    println!();
    println!("Movement (normalized by n·√k)");
    println!("┌────────────┬────────────┬──────────┬─────────────────┬────────────┐");
    println!("│     n      │     k      │   k %    │    Movement     │ Normalized │");
    println!("├────────────┼────────────┼──────────┼─────────────────┼────────────┤");
    for r in results {
        println!(
            "│ {:>10} │ {:>10} │ {:>7.2}% │ {:>15.0} │ {:>10.4} │",
            format_number(r.n),
            format_number(r.k),
            r.k_percent,
            r.movement_mean,
            r.movement_normalized
        );
    }
    println!("└────────────┴────────────┴──────────┴─────────────────┴────────────┘");
}

fn print_segments_table(results: &[AnalysisResult]) {
    println!();
    println!("Segments (normalized by √k)");
    println!("┌────────────┬────────────┬──────────┬─────────────────┬────────────┐");
    println!("│     n      │     k      │   k %    │    Segments     │ Normalized │");
    println!("├────────────┼────────────┼──────────┼─────────────────┼────────────┤");
    for r in results {
        println!(
            "│ {:>10} │ {:>10} │ {:>7.2}% │ {:>15.1} │ {:>10.4} │",
            format_number(r.n),
            format_number(r.k),
            r.k_percent,
            r.segments_mean,
            r.segments_normalized
        );
    }
    println!("└────────────┴────────────┴──────────┴─────────────────┴────────────┘");
}

fn export_movement_csv(results: &[AnalysisResult], path: &str) {
    let mut csv = String::from("n,k,k_percent,movement,movement_sd,movement_normalized,segments,segments_sd,segments_normalized\n");
    for r in results {
        csv.push_str(&format!(
            "{},{},{:.4},{:.0},{:.0},{:.6},{:.1},{:.1},{:.6}\n",
            r.n,
            r.k,
            r.k_percent,
            r.movement_mean,
            r.movement_sd,
            r.movement_normalized,
            r.segments_mean,
            r.segments_sd,
            r.segments_normalized
        ));
    }
    fs::write(path, csv).expect("Failed to write movement-analysis.csv");
    println!("Exported: {}", path);
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let run_time = args.iter().any(|a| a == "--time");
    let run_crossover = args.iter().any(|a| a == "--crossover");
    let run_comparator = args.iter().any(|a| a == "--comparator");
    let run_movement = args.iter().any(|a| a == "--movement");
    let export = args.iter().any(|a| a == "--export");

    // If no specific flags, run all
    let run_all = !run_time && !run_crossover && !run_comparator && !run_movement;

    println!();
    println!("DeltaSort Benchmark");
    println!("===================");

    if run_all || run_time {
        run_time_benchmark(export);
    }

    if run_all || run_crossover {
        run_crossover_benchmark(export);
    }

    if run_all || run_comparator {
        run_comparator_analysis(export);
    }

    if run_all || run_movement {
        run_movement_analysis(export);
    }

    println!();
    println!("Done!");
    if !export {
        println!("Run with --export to write CSV files");
    }
}
