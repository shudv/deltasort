//! Parallel vs Serial DeltaSort Benchmark
//!
//! Compares execution time of serial DeltaSort against parallel DeltaSort.
//!
//! Run with: `cargo run --bin parallel_benchmark --release`

use deltasort::delta_sort_by;
use deltasort::parallel::parallel_delta_sort_by;
use rand::Rng;
use std::collections::HashSet;
use std::io::{self, Write};
use std::time::Instant;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Array sizes to benchmark
const N_VALUES: &[usize] = &[10_000, 50_000, 100_000, 500_000, 1_000_000];

/// k as fractions of n
const K_FRACTIONS: &[f64] = &[0.001, 0.005, 0.01, 0.02, 0.05, 0.1];

/// Iterations per measurement
const ITERATIONS: usize = 10;

/// Warmup iterations
const WARMUP: usize = 3;

// ============================================================================
// DATA GENERATION
// ============================================================================

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
// BENCHMARKING
// ============================================================================

struct BenchResult {
    n: usize,
    k: usize,
    serial_us: f64,
    parallel_us: f64,
    speedup: f64,
    num_threads: usize,
}

fn benchmark_config(n: usize, k: usize) -> BenchResult {
    let mut rng = rand::thread_rng();
    let num_threads = rayon::current_num_threads();

    // Generate base sorted array
    let base_arr: Vec<i32> = (0..n as i32).collect();

    // Warmup
    for _ in 0..WARMUP {
        let mut arr = base_arr.clone();
        let indices = sample_distinct_indices(&mut rng, n, k);
        let mut dirty: HashSet<usize> = HashSet::with_capacity(k);
        for idx in indices {
            arr[idx] = rng.gen_range(0..n as i32);
            dirty.insert(idx);
        }
        delta_sort_by(&mut arr, &dirty, i32::cmp);
    }

    // Benchmark serial
    let mut serial_times = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let mut arr = base_arr.clone();
        let indices = sample_distinct_indices(&mut rng, n, k);
        let mut dirty: HashSet<usize> = HashSet::with_capacity(k);
        for idx in &indices {
            arr[*idx] = rng.gen_range(0..n as i32);
            dirty.insert(*idx);
        }

        let start = Instant::now();
        delta_sort_by(&mut arr, &dirty, i32::cmp);
        serial_times.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }

    // Benchmark parallel
    let mut parallel_times = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let mut arr = base_arr.clone();
        let indices = sample_distinct_indices(&mut rng, n, k);
        let mut dirty: HashSet<usize> = HashSet::with_capacity(k);
        for idx in &indices {
            arr[*idx] = rng.gen_range(0..n as i32);
            dirty.insert(*idx);
        }

        let start = Instant::now();
        parallel_delta_sort_by(&mut arr, &dirty, i32::cmp);
        parallel_times.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }

    let serial_mean = serial_times.iter().sum::<f64>() / serial_times.len() as f64;
    let parallel_mean = parallel_times.iter().sum::<f64>() / parallel_times.len() as f64;
    let speedup = serial_mean / parallel_mean;

    BenchResult {
        n,
        k,
        serial_us: serial_mean,
        parallel_us: parallel_mean,
        speedup,
        num_threads,
    }
}

fn verify_correctness() -> bool {
    let mut rng = rand::thread_rng();
    let mut all_pass = true;

    for &n in &[100, 1_000, 10_000] {
        for &k_frac in &[0.01, 0.05, 0.1, 0.5] {
            let k = ((n as f64) * k_frac).round() as usize;
            for _ in 0..10 {
                let mut arr: Vec<i32> = (0..n as i32).collect();
                let indices = sample_distinct_indices(&mut rng, n, k);
                let mut dirty: HashSet<usize> = HashSet::with_capacity(k);
                for idx in &indices {
                    arr[*idx] = rng.gen_range(0..n as i32);
                    dirty.insert(*idx);
                }

                let mut expected = arr.clone();
                expected.sort();

                parallel_delta_sort_by(&mut arr, &dirty, i32::cmp);

                if arr != expected {
                    eprintln!("FAIL: n={}, k={}", n, k);
                    all_pass = false;
                }
            }
        }
    }

    all_pass
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

fn print_results(results: &[BenchResult]) {
    println!();
    println!("┌────────────┬────────────┬─────────────────┬─────────────────┬───────────┐");
    println!("│     n      │     k      │   Serial (µs)   │  Parallel (µs)  │  Speedup  │");
    println!("├────────────┼────────────┼─────────────────┼─────────────────┼───────────┤");

    for r in results {
        let speedup_str = if r.speedup >= 1.0 {
            format!("{:.2}x", r.speedup)
        } else {
            format!("{:.2}x", r.speedup)
        };

        println!(
            "│ {:>10} │ {:>10} │ {:>15.1} │ {:>15.1} │ {:>9} │",
            format_number(r.n),
            format_number(r.k),
            r.serial_us,
            r.parallel_us,
            speedup_str
        );
    }

    println!("└────────────┴────────────┴─────────────────┴─────────────────┴───────────┘");
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    println!();
    println!("Parallel DeltaSort Benchmark");
    println!("============================");
    println!("Threads: {}", rayon::current_num_threads());
    println!();

    // Verify correctness first
    print!("Verifying correctness... ");
    io::stdout().flush().unwrap();
    if verify_correctness() {
        println!("PASS");
    } else {
        println!("FAIL");
        return;
    }

    println!();
    println!("Running benchmarks ({} iterations each)...", ITERATIONS);

    let mut results = Vec::new();

    for &n in N_VALUES {
        for &k_frac in K_FRACTIONS {
            let k = ((n as f64) * k_frac).round() as usize;
            if k < 1 {
                continue;
            }

            print!("  n={:>7}, k={:>7} ({:.1}%)...", format_number(n), format_number(k), k_frac * 100.0);
            io::stdout().flush().unwrap();

            let result = benchmark_config(n, k);
            println!(
                " serial={:.1}µs, parallel={:.1}µs, speedup={:.2}x",
                result.serial_us, result.parallel_us, result.speedup
            );

            results.push(result);
        }
    }

    print_results(&results);

    // Summary statistics
    let speedups: Vec<f64> = results.iter().map(|r| r.speedup).collect();
    let avg_speedup = speedups.iter().sum::<f64>() / speedups.len() as f64;
    let max_speedup = speedups.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let wins = speedups.iter().filter(|&&s| s > 1.0).count();

    println!();
    println!("Summary:");
    println!("  Average speedup: {:.2}x", avg_speedup);
    println!("  Max speedup:     {:.2}x", max_speedup);
    println!("  Parallel wins:   {}/{}", wins, results.len());
    println!();
}
