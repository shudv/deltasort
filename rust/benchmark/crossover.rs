//! Crossover Point Finder for DeltaSort
//!
//! Finds the critical delta size k_c where Native sort becomes faster than DeltaSort.
//! Uses binary search for efficient discovery.
//!
//! Run with: `cargo run --bin crossover --release`

use deltasort::deltasort;
use rand::Rng;
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

fn generate_sorted_users(n: usize) -> Vec<User> {
    let mut users: Vec<User> = (0..n).map(User::generate).collect();
    users.sort_by(user_comparator);
    users
}

// ============================================================================
// MEASUREMENT
// ============================================================================

const ITERATIONS: usize = 10;

/// Measure mean time for native sort (in microseconds)
fn measure_native(base_users: &[User], k: usize, n: usize) -> f64 {
    let mut rng = rand::thread_rng();
    let mut times = Vec::with_capacity(ITERATIONS);

    for _ in 0..ITERATIONS {
        // Prepare mutated data
        let mut users = base_users.to_vec();
        for _ in 0..k {
            let idx = rng.gen_range(0..n);
            users[idx].mutate(&mut rng);
        }

        let start = Instant::now();
        users.sort_by(user_comparator);
        let elapsed = start.elapsed();

        times.push(elapsed.as_secs_f64() * 1_000_000.0);
    }

    times.iter().sum::<f64>() / times.len() as f64
}

/// Measure mean time for deltasort (in microseconds)
fn measure_deltasort(base_users: &[User], k: usize, n: usize) -> f64 {
    let mut rng = rand::thread_rng();
    let mut times = Vec::with_capacity(ITERATIONS);

    for _ in 0..ITERATIONS {
        // Prepare mutated data with tracked dirty indices
        let mut users = base_users.to_vec();
        let mut dirty_indices = HashSet::new();
        for _ in 0..k {
            let idx = rng.gen_range(0..n);
            users[idx].mutate(&mut rng);
            dirty_indices.insert(idx);
        }

        let start = Instant::now();
        deltasort(&mut users, &dirty_indices, user_comparator);
        let elapsed = start.elapsed();

        times.push(elapsed.as_secs_f64() * 1_000_000.0);
    }

    times.iter().sum::<f64>() / times.len() as f64
}

/// Check if DeltaSort is faster than Native at given k
fn deltasort_is_faster(base_users: &[User], k: usize, n: usize) -> bool {
    let native_time = measure_native(base_users, k, n);
    let ds_time = measure_deltasort(base_users, k, n);
    ds_time < native_time
}

// ============================================================================
// BINARY SEARCH FOR CROSSOVER
// ============================================================================

/// Find the crossover point k_c where DeltaSort becomes slower than Native.
/// Returns the largest k where DeltaSort is still faster.
fn find_crossover(n: usize) -> usize {
    println!("  Finding crossover for N = {}...", n);
    
    let base_users = generate_sorted_users(n);
    
    // Warmup
    for _ in 0..5 {
        let mut users = base_users.clone();
        users.sort_by(user_comparator);
    }

    let mut lo: usize = 1;
    let mut hi: usize = (n*2)/5; // Crossover is usually not beyond 40% of n
    
    // First, check if DeltaSort is faster at k=1 (should be)
    if !deltasort_is_faster(&base_users, 1, n) {
        println!("    Warning: DeltaSort slower than Native even at k=1");
        return 0;
    }
    
    // Check if DeltaSort is still faster at k=n (unlikely for large n)
    if deltasort_is_faster(&base_users, n, n) {
        println!("    DeltaSort faster than Native even at k=N");
        return n;
    }

    // Binary search for the crossover point
    // We want the largest k where DeltaSort is still faster
    // Stop early when range size drops below 0.1% of n
    let min_range = (n as f64 * 0.001) as usize;
    
    while lo < hi {
        // Early termination: if range is small enough, we have a good approximation
        if hi - lo < min_range {
            println!("    Crossover: ~{}", lo);
            break;
        }
        
        let mid = lo + (hi - lo + 1) / 2;  // Bias towards higher value
        
        print!("    Testing k = {}... ", mid);
        
        if deltasort_is_faster(&base_users, mid, n) {
            print!("DS faster");
            lo = mid;
        } else {
            print!("Native faster");
            hi = mid - 1;
        }
        println!();
    }

    lo
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    DeltaSort Crossover Point Finder                          ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Finding k_c where Native sort becomes faster than DeltaSort                 ║");
    println!("║  Using binary search with {} iterations per measurement                       ║", ITERATIONS);
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let test_sizes = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000];
    let mut results: Vec<(usize, usize, f64)> = Vec::new();

    for &n in &test_sizes {
        let k_c = find_crossover(n);
        let percentage = (k_c as f64 / n as f64) * 100.0;
        results.push((n, k_c, percentage));
        println!();
    }

    // Print results table
    println!();
    println!("┌──────────────┬──────────────┬──────────────┐");
    println!("│      N       │     k_c      │   k_c / N    │");
    println!("├──────────────┼──────────────┼──────────────┤");

    for (n, k_c, pct) in &results {
        println!(
            "│ {:>10}   │ {:>10}   │ {:>9.1}%   │",
            format_number(*n),
            format_number(*k_c),
            pct
        );
    }

    println!("└──────────────┴──────────────┴──────────────┘");
    println!();
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
