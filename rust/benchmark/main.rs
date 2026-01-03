//! Performance comparison tool for DeltaSort.
//!
//! Run with: `cargo run --bin benchmark --release`
//!
//! This outputs a formatted comparison table showing speedups.

use deltasort::deltasort;
use rand::Rng;
use std::cell::Cell;
use std::collections::HashSet;
use std::time::Instant;

// ============================================================================
// BENCHMARK DATA
// ============================================================================

const COUNTRIES: &[&str] = &[
    "USA", "Canada", "UK", "Germany", "France", "Spain", "Italy", "Japan",
    "Australia", "Brazil", "India", "China", "Mexico", "Argentina", "Sweden",
];

const FIRST_NAMES: &[&str] = &[
    "Vijay", "Meera", "Akash", "Kashish", "Sunita", "Aviral", "Saumya",
    "Aman", "Sanjay", "Kavitha", "Radhika", "Meenakshi", "Suresh", "Krishna",
];

const LAST_NAMES: &[&str] = &[
    "Sharma", "Patel", "Dwivedi", "Kumar", "Singh", "Gupta", "Nair",
    "Iyer", "Rao", "Menon", "Pillai", "Joshi", "Verma",
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

fn generate_sorted_users(n: usize) -> Vec<User> {
    let mut users: Vec<User> = (0..n).map(User::generate).collect();
    users.sort_by(user_comparator);
    users
}

/// Generate k distinct random indices from [0, n-1] using Fisher-Yates partial shuffle.
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
// ALGORITHMS
// ============================================================================

fn native_sort(arr: &mut Vec<User>) {
    arr.sort_by(user_comparator);
}

fn native_sort_counting(arr: &mut Vec<User>) {
    arr.sort_by(counting_comparator);
}

/// Binary Insertion Sort
/// 
/// Algorithm: Extract dirty elements (preserving array order), then insert each 
/// at correct position via binary search.
/// 
/// Complexity: O(k) extractions × O(n) shift each + O(k) insertions × O(n) shift each
///           = O(k*n) overall
/// 
/// Note: The O(n) cost per remove/insert is inherent to this approach when working
/// with contiguous arrays. This is a faithful implementation.
fn binary_insertion_sort(arr: &mut Vec<User>, dirty_indices: &HashSet<usize>) {
    if dirty_indices.is_empty() {
        return;
    }

    // Extract dirty elements in descending index order (so indices stay valid)
    let mut sorted_desc: Vec<usize> = dirty_indices.iter().copied().collect();
    sorted_desc.sort_unstable_by(|a, b| b.cmp(a));

    let mut extracted: Vec<User> = Vec::with_capacity(sorted_desc.len());
    for &idx in &sorted_desc {
        extracted.push(arr.remove(idx));  // O(n) shift - inherent to algorithm
    }

    // Insert each element at correct position
    for value in extracted {
        let pos = arr.partition_point(|x| user_comparator(x, &value) == std::cmp::Ordering::Less);
        arr.insert(pos, value);  // O(n) shift - inherent to algorithm
    }
}

/// Binary Insertion Sort with comparison counting
fn binary_insertion_sort_counting(arr: &mut Vec<User>, dirty_indices: &HashSet<usize>) {
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
        let pos = arr.partition_point(|x| counting_comparator(x, &value) == std::cmp::Ordering::Less);
        arr.insert(pos, value);
    }
}

/// Extract-Sort-Merge
/// 
/// Algorithm: Extract dirty elements, sort them, merge back with clean portion.
/// 
/// Complexity: O(k*n) extraction + O(k log k) sort + O(n) merge = O(k*n + n) overall
/// 
/// Note: The extraction cost dominates. A linked-list version would be O(k log k + n),
/// but for contiguous arrays this O(k*n) extraction is unavoidable without auxiliary
/// data structures.
fn extract_sort_merge(arr: &mut Vec<User>, dirty_indices: &HashSet<usize>) {
    if dirty_indices.is_empty() {
        return;
    }

    // Extract dirty elements in descending index order
    let mut sorted_desc: Vec<usize> = dirty_indices.iter().copied().collect();
    sorted_desc.sort_unstable_by(|a, b| b.cmp(a));

    let mut dirty_elements: Vec<User> = Vec::with_capacity(sorted_desc.len());
    for &idx in &sorted_desc {
        dirty_elements.push(arr.remove(idx));  // O(n) shift each
    }

    // Sort the extracted dirty elements
    dirty_elements.sort_by(user_comparator);

    // Merge: both arrays are now sorted
    let clean_len = arr.len();
    let dirty_len = dirty_elements.len();
    let mut result: Vec<User> = Vec::with_capacity(clean_len + dirty_len);
    
    let mut i = 0;
    let mut j = 0;

    // Standard merge of two sorted arrays
    while i < clean_len && j < dirty_len {
        // Move rather than clone for efficiency
        if user_comparator(&arr[i], &dirty_elements[j]) != std::cmp::Ordering::Greater {
            result.push(std::mem::take(&mut arr[i]));
            i += 1;
        } else {
            result.push(std::mem::take(&mut dirty_elements[j]));
            j += 1;
        }
    }

    // Drain remaining
    for item in arr.drain(i..) {
        result.push(item);
    }
    for item in dirty_elements.drain(j..) {
        result.push(item);
    }

    *arr = result;
}

/// Extract-Sort-Merge with comparison counting
fn extract_sort_merge_counting(arr: &mut Vec<User>, dirty_indices: &HashSet<usize>) {
    if dirty_indices.is_empty() {
        return;
    }

    let mut sorted_desc: Vec<usize> = dirty_indices.iter().copied().collect();
    sorted_desc.sort_unstable_by(|a, b| b.cmp(a));

    let mut dirty_elements: Vec<User> = Vec::with_capacity(sorted_desc.len());
    for &idx in &sorted_desc {
        dirty_elements.push(arr.remove(idx));
    }

    dirty_elements.sort_by(counting_comparator);

    let clean_len = arr.len();
    let dirty_len = dirty_elements.len();
    let mut result: Vec<User> = Vec::with_capacity(clean_len + dirty_len);
    
    let mut i = 0;
    let mut j = 0;

    while i < clean_len && j < dirty_len {
        if counting_comparator(&arr[i], &dirty_elements[j]) != std::cmp::Ordering::Greater {
            result.push(std::mem::take(&mut arr[i]));
            i += 1;
        } else {
            result.push(std::mem::take(&mut dirty_elements[j]));
            j += 1;
        }
    }

    for item in arr.drain(i..) {
        result.push(item);
    }
    for item in dirty_elements.drain(j..) {
        result.push(item);
    }

    *arr = result;
}

fn deltasort_wrapper(arr: &mut Vec<User>, dirty_indices: &HashSet<usize>) {
    deltasort(arr.as_mut_slice(), dirty_indices, user_comparator);
}

fn deltasort_wrapper_counting(arr: &mut Vec<User>, dirty_indices: &HashSet<usize>) {
    deltasort(arr.as_mut_slice(), dirty_indices, counting_comparator);
}

// ============================================================================
// BENCHMARK RUNNER
// ============================================================================

struct BenchResult {
    times_us: Vec<f64>,
}

impl BenchResult {
    fn mean(&self) -> f64 {
        self.times_us.iter().sum::<f64>() / self.times_us.len() as f64
    }

    fn std_dev(&self) -> f64 {
        let mean = self.mean();
        let variance = self.times_us.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / self.times_us.len() as f64;
        variance.sqrt()
    }

    /// 95% confidence interval half-width
    fn ci_95(&self) -> f64 {
        // t-value for 95% CI with ~100 samples ≈ 1.96
        1.96 * self.std_dev() / (self.times_us.len() as f64).sqrt()
    }

    fn format_with_ci(&self) -> String {
        let mean = self.mean();
        let ci = self.ci_95();
        if mean >= 10000.0 {
            format!("{:>5.0}±{:<4.0}", mean, ci)
        } else if mean >= 1000.0 {
            format!("{:>5.0}±{:<4.0}", mean, ci)
        } else if mean >= 100.0 {
            format!("{:>5.0}±{:<4.0}", mean, ci)
        } else {
            format!("{:>5.1}±{:<4.1}", mean, ci)
        }
    }
}

fn run_benchmark<F>(
    _name: &'static str,
    _base_users: &[User],
    dirty_indices: &HashSet<usize>,
    mutated_users: &[User],
    iterations: usize,
    mut f: F,
) -> BenchResult
where
    F: FnMut(&mut Vec<User>, &HashSet<usize>),
{
    let mut times_us = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let mut users = mutated_users.to_vec();
        let dirty = dirty_indices.clone();

        let start = Instant::now();
        f(&mut users, &dirty);
        let elapsed = start.elapsed();

        times_us.push(elapsed.as_secs_f64() * 1_000_000.0); // Convert to microseconds
    }

    BenchResult { times_us }
}

fn run_native_benchmark(
    _name: &'static str,
    mutated_users: &[User],
    iterations: usize,
) -> BenchResult {
    let mut times_us = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let mut users = mutated_users.to_vec();

        let start = Instant::now();
        native_sort(&mut users);
        let elapsed = start.elapsed();

        times_us.push(elapsed.as_secs_f64() * 1_000_000.0);
    }

    BenchResult { times_us }
}

// ============================================================================
// COMPARISON COUNTING
// ============================================================================

struct ComparisonResult {
    native: u64,
    bi: u64,
    esm: u64,
    ds: u64,
}

fn count_comparisons(
    mutated_users: &[User],
    dirty_indices: &HashSet<usize>,
) -> ComparisonResult {
    // Native sort
    reset_comparison_count();
    let mut users = mutated_users.to_vec();
    native_sort_counting(&mut users);
    let native = get_comparison_count();

    // Binary Insertion Sort
    reset_comparison_count();
    let mut users = mutated_users.to_vec();
    binary_insertion_sort_counting(&mut users, dirty_indices);
    let bi = get_comparison_count();

    // Extract-Sort-Merge
    reset_comparison_count();
    let mut users = mutated_users.to_vec();
    extract_sort_merge_counting(&mut users, dirty_indices);
    let esm = get_comparison_count();

    // DeltaSort
    reset_comparison_count();
    let mut users = mutated_users.to_vec();
    deltasort_wrapper_counting(&mut users, dirty_indices);
    let ds = get_comparison_count();

    ComparisonResult { native, bi, esm, ds }
}

// ============================================================================
// OUTPUT FORMATTING
// ============================================================================

fn print_header() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                     DeltaSort Performance Comparison                         ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║  n = 50,000 elements  │  100 iterations per config  │  Times in microseconds ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();
}

fn print_table_header() {
    println!("┌────────┬───────────────┬───────────────┬───────────────┬───────────────┬─────────────────┐");
    println!("│   k    │    Native     │      BI       │      ESM      │   DeltaSort   │   DS vs Best    │");
    println!("│        │    (µs)       │     (µs)      │     (µs)      │     (µs)      │                 │");
    println!("├────────┼───────────────┼───────────────┼───────────────┼───────────────┼─────────────────┤");
}

fn print_table_row(k: usize, native: &BenchResult, bi: &BenchResult, esm: &BenchResult, ds: &BenchResult) {
    let native_mean = native.mean();
    let bi_mean = bi.mean();
    let esm_mean = esm.mean();
    let ds_mean = ds.mean();

    // Compare DS against best of others (Native, BI, ESM)
    let best_other = native_mean.min(bi_mean).min(esm_mean);
    let ds_vs_best = best_other / ds_mean;

    // Format speedup with color indicator
    let speedup_str = if ds_vs_best >= 1.0 {
        format!("{:>6.2}x faster ✓", ds_vs_best)
    } else {
        format!("{:>6.2}x slower ✗", 1.0 / ds_vs_best)
    };

    println!(
        "│ {:>6} │ {:>13} │ {:>13} │ {:>13} │ {:>13} │ {:>15} │",
        k,
        native.format_with_ci(),
        bi.format_with_ci(),
        esm.format_with_ci(),
        ds.format_with_ci(),
        speedup_str
    );
}

fn print_table_footer() {
    println!("└────────┴───────────────┴───────────────┴───────────────┴───────────────┴─────────────────┘");
    println!("  Note: Values shown as mean ± 95% CI");
}

// ============================================================================
// COMPARISON COUNT OUTPUT FORMATTING
// ============================================================================

fn print_comparison_header() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                      Comparator Invocation Count                             ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║  n = 50,000 elements  │  Exact counts (no variance)                          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();
}

fn print_comparison_table_header() {
    println!("┌────────┬────────────┬────────────┬────────────┬────────────┬─────────────────┐");
    println!("│   k    │   Native   │     BI     │    ESM     │  DeltaSort │   DS vs Best    │");
    println!("├────────┼────────────┼────────────┼────────────┼────────────┼─────────────────┤");
}

fn format_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:>7.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:>7.2}K", n as f64 / 1_000.0)
    } else {
        format!("{:>8}", n)
    }
}

fn print_comparison_table_row(k: usize, cmp: &ComparisonResult) {
    let best_other = cmp.native.min(cmp.bi).min(cmp.esm);
    let ratio = best_other as f64 / cmp.ds as f64;

    let ratio_str = if ratio >= 1.0 {
        format!("{:>6.2}x fewer ✓", ratio)
    } else {
        format!("{:>6.2}x more  ✗", 1.0 / ratio)
    };

    println!(
        "│ {:>6} │ {:>10} │ {:>10} │ {:>10} │ {:>10} │ {:>15} │",
        k,
        format_count(cmp.native),
        format_count(cmp.bi),
        format_count(cmp.esm),
        format_count(cmp.ds),
        ratio_str
    );
}

fn print_comparison_table_footer() {
    println!("└────────┴────────────┴────────────┴────────────┴────────────┴─────────────────┘");
}

fn print_comparison_summary(results: &[(usize, ComparisonResult)]) {
    println!();
    println!("Comparison Count Analysis:");
    
    // Find where DS uses fewest comparisons
    let best_k = results
        .iter()
        .filter(|(_, c)| c.ds < c.native && c.ds < c.bi && c.ds < c.esm)
        .map(|(k, c)| {
            let best_other = c.native.min(c.bi).min(c.esm);
            (*k, best_other as f64 / c.ds as f64)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    if let Some((k, ratio)) = best_k {
        println!("  • Best reduction: {:.2}x fewer comparisons at k={}", ratio, k);
    }

    // Theoretical bound check: O(k log n) for DeltaSort
    println!();
    println!("  Theoretical bounds (k log n for n=50,000):");
    for (k, cmp) in results.iter().take(8) {
        let theoretical = (*k as f64) * (50_000f64).log2();
        let ratio = cmp.ds as f64 / theoretical;
        println!("    k={:>5}: actual={:>8}, k·log₂(n)={:>8.0}, ratio={:.2}x", 
                 k, cmp.ds, theoretical, ratio);
    }
    println!();
}

fn print_summary(results: &[(usize, f64, f64, f64, f64)]) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                                  Summary                                     ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Find sweet spot (best speedup vs BI)
    let best_vs_bi = results
        .iter()
        .map(|(k, _, bi, _, ds)| (*k, bi / ds))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    // Find crossover point with ESM
    let esm_crossover = results
        .iter()
        .find(|(_, _, _, esm, ds)| ds > esm)
        .map(|(k, _, _, _, _)| *k);

    println!("• Best speedup vs Binary Insertion: {:.2}x at k={}", best_vs_bi.1, best_vs_bi.0);

    if let Some(k) = esm_crossover {
        println!("• ESM becomes faster than DeltaSort around k={}", k);
    } else {
        println!("• DeltaSort faster than ESM for all tested k values");
    }

    // Calculate average speedup in sweet spot (k=20 to k=100)
    let sweet_spot: Vec<_> = results
        .iter()
        .filter(|(k, _, _, _, _)| *k >= 20 && *k <= 100)
        .collect();

    if !sweet_spot.is_empty() {
        let avg_vs_bi: f64 = sweet_spot.iter().map(|(_, _, bi, _, ds)| bi / ds).sum::<f64>()
            / sweet_spot.len() as f64;
        let avg_vs_esm: f64 = sweet_spot.iter().map(|(_, _, _, esm, ds)| esm / ds).sum::<f64>()
            / sweet_spot.len() as f64;

        println!();
        println!("• Average speedup in sweet spot (k=20-100):");
        println!("  - vs Binary Insertion: {:.2}x", avg_vs_bi);
        println!("  - vs Extract-Sort-Merge: {:.2}x", avg_vs_esm);
    }

    // Find crossover with native sort
    let native_crossover = results
        .iter()
        .find(|(_, native, _, _, ds)| ds > native)
        .map(|(k, _, _, _, _)| *k);

    if let Some(k) = native_crossover {
        println!("• Native sort becomes faster than DeltaSort around k={}", k);
    } else {
        println!("• DeltaSort faster than Native sort for all tested k values");
    }

    println!();
    println!("Recommendation:");
    if let Some(crossover) = native_crossover {
        println!("  • k ≤ 5:        Use Binary Insertion (simpler, similar speed)");
        println!("  • 5 < k < {}:  Use DeltaSort ✓", crossover);
        println!("  • k ≥ {}:      Use Native Sort", crossover);
    } else {
        println!("  • k ≤ 5:    Use Binary Insertion (simpler, similar speed)");
        println!("  • k > 5:    Use DeltaSort ✓");
    }
    println!();
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    let n = 50_000;
    let delta_counts = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];
    let iterations = 500;
    let warmup = 5;

    print_header();

    // Warmup
    let base_users = generate_sorted_users(n);
    for _ in 0..warmup {
        let mut users = base_users.clone();
        native_sort(&mut users);
    }

    let mut all_results: Vec<(usize, f64, f64, f64, f64)> = Vec::new();
    let mut comparison_results: Vec<(usize, ComparisonResult)> = Vec::new();

    print_table_header();

    for &k in &delta_counts {
        let mut rng = rand::thread_rng();

        // Generate k distinct random indices, then mutate each
        let indices_to_mutate = sample_distinct_indices(&mut rng, n, k);
        let mut mutated_users = base_users.clone();
        let mut dirty_indices = HashSet::with_capacity(k);
        for idx in indices_to_mutate {
            mutated_users[idx].mutate(&mut rng);
            dirty_indices.insert(idx);
        }

        // Run timing benchmarks
        let native = run_native_benchmark("Native", &mutated_users, iterations);
        let bi = run_benchmark("BI", &base_users, &dirty_indices, &mutated_users, iterations, binary_insertion_sort);
        let esm = run_benchmark("ESM", &base_users, &dirty_indices, &mutated_users, iterations, extract_sort_merge);
        let ds = run_benchmark("DS", &base_users, &dirty_indices, &mutated_users, iterations, deltasort_wrapper);

        print_table_row(k, &native, &bi, &esm, &ds);

        all_results.push((k, native.mean(), bi.mean(), esm.mean(), ds.mean()));

        // Count comparisons (single run - exact count)
        let cmp_result = count_comparisons(&mutated_users, &dirty_indices);
        comparison_results.push((k, cmp_result));
    }

    print_table_footer();
    print_summary(&all_results);

    // Print comparison count table
    print_comparison_header();
    print_comparison_table_header();
    for (k, cmp) in &comparison_results {
        print_comparison_table_row(*k, cmp);
    }
    print_comparison_table_footer();
    print_comparison_summary(&comparison_results);
}
