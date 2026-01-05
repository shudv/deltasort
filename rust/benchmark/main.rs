//! DeltaSort Benchmark Suite
//!
//! Run with: `cargo run --bin benchmark --release`
//! Export CSV: `cargo run --bin benchmark --release -- --export`

use deltasort::delta_sort_by;
use rand::Rng;
use std::cell::Cell;
use std::collections::HashSet;
use std::fs;
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

fn extract_sort_merge_counting(arr: &mut Vec<User>, dirty_indices: &HashSet<usize>) {
    if dirty_indices.is_empty() {
        return;
    }

    let mut sorted_desc: Vec<usize> = dirty_indices.iter().copied().collect();
    sorted_desc.sort_unstable_by(|a, b| b.cmp(a));

    let mut dirty_values: Vec<User> = Vec::with_capacity(sorted_desc.len());
    for &idx in &sorted_desc {
        dirty_values.push(arr.remove(idx));
    }

    dirty_values.sort_by(counting_comparator);

    let clean_len = arr.len();
    let dirty_len = dirty_values.len();
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

fn deltasort_wrapper(arr: &mut Vec<User>, dirty_indices: &HashSet<usize>) {
    delta_sort_by(arr.as_mut_slice(), dirty_indices, user_comparator);
}

fn deltasort_wrapper_counting(arr: &mut Vec<User>, dirty_indices: &HashSet<usize>) {
    delta_sort_by(arr.as_mut_slice(), dirty_indices, counting_comparator);
}

// ============================================================================
// BENCHMARK MEASUREMENT
// ============================================================================

const ITERATIONS: usize = 100;
const CROSSOVER_ITERATIONS: usize = 10;

fn measure_mean_time<F>(
    mutated_users: &[User],
    dirty_indices: &HashSet<usize>,
    iterations: usize,
    mut f: F,
) -> f64
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

        times_us.push(elapsed.as_secs_f64() * 1_000_000.0);
    }

    times_us.iter().sum::<f64>() / times_us.len() as f64
}

fn measure_native_time(mutated_users: &[User], iterations: usize) -> f64 {
    let mut times_us = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let mut users = mutated_users.to_vec();

        let start = Instant::now();
        native_sort(&mut users);
        let elapsed = start.elapsed();

        times_us.push(elapsed.as_secs_f64() * 1_000_000.0);
    }

    times_us.iter().sum::<f64>() / times_us.len() as f64
}

fn count_comparisons(
    mutated_users: &[User],
    dirty_indices: &HashSet<usize>,
) -> (u64, u64, u64, u64) {
    reset_comparison_count();
    let mut users = mutated_users.to_vec();
    native_sort_counting(&mut users);
    let native = get_comparison_count();

    reset_comparison_count();
    let mut users = mutated_users.to_vec();
    binary_insertion_sort_counting(&mut users, dirty_indices);
    let bi = get_comparison_count();

    reset_comparison_count();
    let mut users = mutated_users.to_vec();
    extract_sort_merge_counting(&mut users, dirty_indices);
    let esm = get_comparison_count();

    reset_comparison_count();
    let mut users = mutated_users.to_vec();
    deltasort_wrapper_counting(&mut users, dirty_indices);
    let ds = get_comparison_count();

    (native, bi, esm, ds)
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

        let mid = lo + (hi - lo + 1) / 2;

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

struct ExecutionTimeResult {
    k: usize,
    native: f64,
    bis: f64,
    esm: f64,
    deltasort: f64,
}

struct ComparatorCountResult {
    k: usize,
    native: u64,
    bis: u64,
    esm: u64,
    deltasort: u64,
}

struct CrossoverResult {
    n: usize,
    crossover_ratio: f64,
}

// ============================================================================
// OUTPUT
// ============================================================================

fn print_execution_time_table(results: &[ExecutionTimeResult]) {
    println!();
    println!("Execution Time (µs) - n=50,000");
    println!("┌────────┬──────────┬──────────┬──────────┬──────────┐");
    println!("│   k    │  Native  │   BIS    │   ESM    │ DeltaSort│");
    println!("├────────┼──────────┼──────────┼──────────┼──────────┤");
    for r in results {
        println!(
            "│ {:>6} │ {:>8.1} │ {:>8.1} │ {:>8.1} │ {:>8.1} │",
            r.k, r.native, r.bis, r.esm, r.deltasort
        );
    }
    println!("└────────┴──────────┴──────────┴──────────┴──────────┘");
}

fn print_comparator_count_table(results: &[ComparatorCountResult]) {
    println!();
    println!("Comparator Invocations - n=50,000");
    println!("┌────────┬──────────┬──────────┬──────────┬──────────┐");
    println!("│   k    │  Native  │   BIS    │   ESM    │ DeltaSort│");
    println!("├────────┼──────────┼──────────┼──────────┼──────────┤");
    for r in results {
        println!(
            "│ {:>6} │ {:>8} │ {:>8} │ {:>8} │ {:>8} │",
            r.k, r.native, r.bis, r.esm, r.deltasort
        );
    }
    println!("└────────┴──────────┴──────────┴──────────┴──────────┘");
}

fn print_crossover_table(results: &[CrossoverResult]) {
    println!();
    println!("Crossover Threshold Analysis");
    println!("┌────────────┬──────────────┐");
    println!("│     n      │  k_c/n (%)   │");
    println!("├────────────┼──────────────┤");
    for r in results {
        println!("│ {:>10} │ {:>10.1}% │", format_number(r.n), r.crossover_ratio);
    }
    println!("└────────────┴──────────────┘");
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        format!("{}", n)
    }
}

// ============================================================================
// CSV EXPORT
// ============================================================================

fn export_execution_time_csv(results: &[ExecutionTimeResult], path: &str) {
    let mut csv = String::from("k,native,bis,esm,deltasort\n");
    for r in results {
        csv.push_str(&format!(
            "{},{:.1},{:.1},{:.1},{:.1}\n",
            r.k, r.native, r.bis, r.esm, r.deltasort
        ));
    }
    fs::write(path, csv).expect("Failed to write execution-time.csv");
    println!("Exported: {}", path);
}

fn export_comparator_count_csv(results: &[ComparatorCountResult], path: &str) {
    let mut csv = String::from("k,native,bis,esm,deltasort\n");
    for r in results {
        csv.push_str(&format!(
            "{},{},{},{},{}\n",
            r.k, r.native, r.bis, r.esm, r.deltasort
        ));
    }
    fs::write(path, csv).expect("Failed to write comparator-count.csv");
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

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let export = args.iter().any(|a| a == "--export");

    let n = 50_000;
    let delta_counts = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];
    let crossover_sizes = [
        1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000,
        1_000_000, 2_000_000, 5_000_000, 10_000_000,
    ];

    println!();
    println!("DeltaSort Benchmark");
    println!("===================");

    // Warmup
    let base_users = generate_sorted_users(n);
    for _ in 0..5 {
        let mut users = base_users.clone();
        native_sort(&mut users);
    }

    // --- Execution Time ---
    println!();
    println!("Running execution time benchmarks...");
    let mut exec_results: Vec<ExecutionTimeResult> = Vec::new();

    for &k in &delta_counts {
        print!("  k={:>5}...", k);
        let mut rng = rand::thread_rng();

        let indices = sample_distinct_indices(&mut rng, n, k);
        let mut mutated_users = base_users.clone();
        let mut dirty_indices = HashSet::with_capacity(k);
        for idx in indices {
            mutated_users[idx].mutate(&mut rng);
            dirty_indices.insert(idx);
        }

        let native = measure_native_time(&mutated_users, ITERATIONS);
        let bis = measure_mean_time(&mutated_users, &dirty_indices, ITERATIONS, binary_insertion_sort);
        let esm = measure_mean_time(&mutated_users, &dirty_indices, ITERATIONS, extract_sort_merge);
        let deltasort = measure_mean_time(&mutated_users, &dirty_indices, ITERATIONS, deltasort_wrapper);

        exec_results.push(ExecutionTimeResult { k, native, bis, esm, deltasort });
        println!(" done");
    }

    print_execution_time_table(&exec_results);

    // --- Comparator Count ---
    println!();
    println!("Running comparator count benchmarks...");
    let mut cmp_results: Vec<ComparatorCountResult> = Vec::new();

    for &k in &delta_counts {
        print!("  k={:>5}...", k);
        let mut rng = rand::thread_rng();

        let indices = sample_distinct_indices(&mut rng, n, k);
        let mut mutated_users = base_users.clone();
        let mut dirty_indices = HashSet::with_capacity(k);
        for idx in indices {
            mutated_users[idx].mutate(&mut rng);
            dirty_indices.insert(idx);
        }

        let (native, bis, esm, deltasort) = count_comparisons(&mutated_users, &dirty_indices);
        cmp_results.push(ComparatorCountResult { k, native, bis, esm, deltasort });
        println!(" done");
    }

    print_comparator_count_table(&cmp_results);

    // --- Crossover Analysis ---
    println!();
    println!("Running crossover analysis (this may take a while)...");
    let mut crossover_results: Vec<CrossoverResult> = Vec::new();

    for &size in &crossover_sizes {
        print!("  n={:>10}...", format_number(size));
        let k_c = find_crossover(size);
        let crossover_ratio = (k_c as f64 / size as f64) * 100.0;
        crossover_results.push(CrossoverResult { n: size, crossover_ratio });
        println!(" k_c={} ({:.1}%)", k_c, crossover_ratio);
    }

    print_crossover_table(&crossover_results);

    // --- Export CSVs ---
    if export {
        println!();
        println!("Exporting CSV files...");
        let base_path = "../paper/benchmarks/rust";
        fs::create_dir_all(base_path).ok();
        export_execution_time_csv(&exec_results, &format!("{}/execution-time.csv", base_path));
        export_comparator_count_csv(&cmp_results, &format!("{}/comparator-count.csv", base_path));
        export_crossover_csv(&crossover_results, &format!("{}/crossover-threshold.csv", base_path));
    }

    println!();
    println!("Done!");
    if !export {
        println!("Run with --export to write CSV files to paper/benchmarks/rust/");
    }
}
