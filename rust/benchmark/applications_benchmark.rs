use rand::Rng;
use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

#[path = "data.rs"]
mod data;

#[path = "extract_sort_merge.rs"]
mod extract_sort_merge;

#[path = "binary_insertion_sort.rs"]
mod binary_insertion_sort;

#[path = "statistics.rs"]
mod statistics;

use data::{generate_sorted_users, sample_distinct_indices, user_comparator, User};
use statistics::{calculate_stats, Stats};

// =============================================================================
// TRACK 3: CDC Sorted Materialized View Benchmark
// =============================================================================

/// Product record for CDC simulation — small variant (64 bytes)
#[derive(Clone, Debug)]
struct ProductSmall {
    id: u64,
    price: f64,
    rating: f32,
    stock: u32,
}

/// Product record for CDC simulation — large variant (~1KB)
#[derive(Clone, Debug)]
struct ProductLarge {
    id: u64,
    price: f64,
    rating: f32,
    stock: u32,
    name: String,
    description: String, // padded to ~1KB total
}

fn price_cmp_small(a: &ProductSmall, b: &ProductSmall) -> std::cmp::Ordering {
    a.price
        .partial_cmp(&b.price)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a.id.cmp(&b.id))
}

fn price_cmp_large(a: &ProductLarge, b: &ProductLarge) -> std::cmp::Ordering {
    a.price
        .partial_cmp(&b.price)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a.id.cmp(&b.id))
}

fn generate_sorted_products_small(n: usize) -> Vec<ProductSmall> {
    let mut rng = rand::thread_rng();
    let mut products: Vec<ProductSmall> = (0..n)
        .map(|i| ProductSmall {
            id: i as u64,
            price: rng.gen_range(1.0..1000.0),
            rating: rng.gen_range(1.0..5.0),
            stock: rng.gen_range(0..10000),
        })
        .collect();
    products.sort_by(price_cmp_small);
    products
}

fn generate_sorted_products_large(n: usize) -> Vec<ProductLarge> {
    let mut rng = rand::thread_rng();
    let pad: String = "X".repeat(900); // pad to ~1KB per record
    let mut products: Vec<ProductLarge> = (0..n)
        .map(|i| ProductLarge {
            id: i as u64,
            price: rng.gen_range(1.0..1000.0),
            rating: rng.gen_range(1.0..5.0),
            stock: rng.gen_range(0..10000),
            name: format!("Product_{}", i),
            description: pad.clone(),
        })
        .collect();
    products.sort_by(price_cmp_large);
    products
}

struct CdcResult {
    label: String,
    n: usize,
    k: usize,
    time_stats: Stats,
}

fn run_cdc_benchmark_small(n: usize, k: usize, iters: usize) -> Vec<CdcResult> {
    let base = generate_sorted_products_small(n);
    let mut rng = rand::thread_rng();
    let mut results = Vec::new();

    // --- FullSort ---
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut arr = base.clone();
        let indices = sample_distinct_indices(&mut rng, n, k);
        for &idx in &indices {
            arr[idx].price = rng.gen_range(1.0..1000.0);
        }
        let start = Instant::now();
        arr.sort_by(price_cmp_small);
        times.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    results.push(CdcResult {
        label: "FullSort".into(),
        n, k,
        time_stats: calculate_stats(&times),
    });

    // --- ESM (Extract-Sort-Merge) ---
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut arr = base.clone();
        let indices = sample_distinct_indices(&mut rng, n, k);
        let dirty: HashSet<usize> = indices.iter().copied().collect();
        for &idx in &indices {
            arr[idx].price = rng.gen_range(1.0..1000.0);
        }
        let start = Instant::now();
        // ESM: partition clean/dirty, sort dirty, merge
        esm_small(&mut arr, &dirty);
        times.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    results.push(CdcResult {
        label: "ESM".into(),
        n, k,
        time_stats: calculate_stats(&times),
    });

    // --- BIS (Binary Insertion Sort) ---
    if k <= n / 10 {
        // BIS is too slow for large k, skip
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let mut arr = base.clone();
            let indices = sample_distinct_indices(&mut rng, n, k);
            let dirty: HashSet<usize> = indices.iter().copied().collect();
            for &idx in &indices {
                arr[idx].price = rng.gen_range(1.0..1000.0);
            }
            let start = Instant::now();
            bis_small(&mut arr, &dirty);
            times.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        }
        results.push(CdcResult {
            label: "BIS".into(),
            n, k,
            time_stats: calculate_stats(&times),
        });
    }

    // --- DeltaSort ---
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut arr = base.clone();
        let indices = sample_distinct_indices(&mut rng, n, k);
        let dirty: HashSet<usize> = indices.iter().copied().collect();
        for &idx in &indices {
            arr[idx].price = rng.gen_range(1.0..1000.0);
        }
        let start = Instant::now();
        deltasort::delta_sort_by(&mut arr, &dirty, price_cmp_small);
        times.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    results.push(CdcResult {
        label: "DeltaSort".into(),
        n, k,
        time_stats: calculate_stats(&times),
    });

    results
}

// Simple ESM for ProductSmall
fn esm_small(arr: &mut Vec<ProductSmall>, dirty: &HashSet<usize>) {
    let n = arr.len();
    let mut write = 0;
    for i in 0..n {
        if !dirty.contains(&i) {
            if write != i { arr.swap(write, i); }
            write += 1;
        }
    }
    let clean_count = write;
    arr[clean_count..].sort_by(price_cmp_small);
    let buffer: Vec<ProductSmall> = arr[clean_count..].to_vec();
    let bk = buffer.len();
    let mut ci = clean_count;
    let mut di = bk;
    let mut wi = n;
    while ci > 0 && di > 0 {
        if price_cmp_small(&arr[ci - 1], &buffer[di - 1]) == std::cmp::Ordering::Greater {
            wi -= 1; ci -= 1; arr.swap(wi, ci);
        } else {
            wi -= 1; di -= 1; arr[wi] = buffer[di].clone();
        }
    }
    while di > 0 { wi -= 1; di -= 1; arr[wi] = buffer[di].clone(); }
}

// Simple BIS for ProductSmall
fn bis_small(arr: &mut Vec<ProductSmall>, dirty: &HashSet<usize>) {
    let mut sorted_dirty: Vec<usize> = dirty.iter().copied().collect();
    sorted_dirty.sort_unstable();
    for &idx in sorted_dirty.iter().rev() {
        let elem = arr.remove(idx);
        let pos = arr.partition_point(|x| price_cmp_small(x, &elem) != std::cmp::Ordering::Greater);
        arr.insert(pos, elem);
    }
}

// =============================================================================
// TRACK 1: Sorted Array vs. BTreeMap Crossover
// =============================================================================

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SortKey {
    country: String,
    age: u32,
    name: String,
}

impl SortKey {
    fn from_user(u: &User) -> Self {
        SortKey { country: u.country.clone(), age: u.age, name: u.name.clone() }
    }
}

struct CrossoverResult {
    n: usize,
    k: usize,
    q: usize,
    array_deltasort_us: f64,
    btree_us: f64,
    array_fullsort_us: f64,
}

fn run_crossover_benchmark(n: usize, k: usize, q: usize, iters: usize) -> CrossoverResult {
    let base_users = generate_sorted_users(n);
    let mut rng = rand::thread_rng();

    // Build BTreeMap
    let mut base_tree: BTreeMap<SortKey, usize> = BTreeMap::new();
    for (i, u) in base_users.iter().enumerate() {
        base_tree.insert(SortKey::from_user(u), i);
    }

    // Pre-generate lookup keys from the sorted array
    let lookup_indices: Vec<usize> = (0..q).map(|_| rng.gen_range(0..n)).collect();

    // --- Sorted Vec + DeltaSort ---
    let mut times_ds = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut arr = base_users.clone();
        let indices = sample_distinct_indices(&mut rng, n, k);
        let dirty: HashSet<usize> = indices.iter().copied().collect();
        for &idx in &indices {
            arr[idx].mutate(&mut rng);
        }

        let start = Instant::now();
        // Update phase
        deltasort::delta_sort_by(&mut arr, &dirty, user_comparator);
        // Lookup phase
        for &li in &lookup_indices {
            let target = &arr[li % arr.len()];
            let _ = arr.binary_search_by(|probe| user_comparator(probe, target));
        }
        times_ds.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }

    // --- BTreeMap ---
    let mut times_bt = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut tree = base_tree.clone();
        let mut users_copy = base_users.clone();
        let indices = sample_distinct_indices(&mut rng, n, k);
        // Build old keys before mutation
        let old_keys: Vec<SortKey> = indices.iter().map(|&i| SortKey::from_user(&users_copy[i])).collect();
        for &idx in &indices {
            users_copy[idx].mutate(&mut rng);
        }

        let start = Instant::now();
        // Update phase: remove old keys, insert new keys
        for (j, &idx) in indices.iter().enumerate() {
            tree.remove(&old_keys[j]);
            tree.insert(SortKey::from_user(&users_copy[idx]), idx);
        }
        // Lookup phase
        for &li in &lookup_indices {
            let target = SortKey::from_user(&users_copy[li % users_copy.len()]);
            let _ = tree.get(&target);
        }
        times_bt.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }

    // --- Sorted Vec + FullSort (baseline) ---
    let mut times_fs = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut arr = base_users.clone();
        let indices = sample_distinct_indices(&mut rng, n, k);
        for &idx in &indices {
            arr[idx].mutate(&mut rng);
        }

        let start = Instant::now();
        arr.sort_by(user_comparator);
        for &li in &lookup_indices {
            let target = &arr[li % arr.len()];
            let _ = arr.binary_search_by(|probe| user_comparator(probe, target));
        }
        times_fs.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }

    CrossoverResult {
        n, k, q,
        array_deltasort_us: calculate_stats(&times_ds).mean,
        btree_us: calculate_stats(&times_bt).mean,
        array_fullsort_us: calculate_stats(&times_fs).mean,
    }
}

// =============================================================================
// TRACK 2: Particle Depth Sorting
// =============================================================================

#[derive(Clone, Debug)]
struct Particle {
    x: f32,
    y: f32,
    z: f32,
    camera_dist: f64,
}

impl Particle {
    fn recompute_distance(&mut self) {
        self.camera_dist =
            ((self.x as f64).powi(2) + (self.y as f64).powi(2) + (self.z as f64).powi(2)).sqrt();
    }
}

fn dist_cmp_cheap(a: &Particle, b: &Particle) -> std::cmp::Ordering {
    a.camera_dist
        .partial_cmp(&b.camera_dist)
        .unwrap_or(std::cmp::Ordering::Equal)
}

fn dist_cmp_expensive(a: &Particle, b: &Particle) -> std::cmp::Ordering {
    // Recompute distance on the fly (simulates expensive comparator)
    let da = ((a.x as f64).powi(2) + (a.y as f64).powi(2) + (a.z as f64).powi(2)).sqrt();
    let db = ((b.x as f64).powi(2) + (b.y as f64).powi(2) + (b.z as f64).powi(2)).sqrt();
    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
}

fn generate_particles(n: usize) -> Vec<Particle> {
    let mut rng = rand::thread_rng();
    let mut particles: Vec<Particle> = (0..n)
        .map(|_| {
            let mut p = Particle {
                x: rng.gen_range(-100.0..100.0f32),
                y: rng.gen_range(-100.0..100.0f32),
                z: rng.gen_range(-100.0..100.0f32),
                camera_dist: 0.0,
            };
            p.recompute_distance();
            p
        })
        .collect();
    particles.sort_by(dist_cmp_cheap);
    particles
}

struct ParticleResult {
    label: String,
    n: usize,
    k: usize,
    comparator: String,
    time_stats: Stats,
}

fn run_particle_benchmark(
    n: usize,
    k: usize,
    frames: usize,
    use_expensive_cmp: bool,
) -> Vec<ParticleResult> {
    let mut base = generate_particles(n);
    let mut rng = rand::thread_rng();
    let cmp_label = if use_expensive_cmp { "expensive" } else { "cheap" };
    let cmp_fn = if use_expensive_cmp { dist_cmp_expensive } else { dist_cmp_cheap };
    let mut results = Vec::new();

    // --- FullSort ---
    let mut arr = base.clone();
    let mut times = Vec::with_capacity(frames);
    for _ in 0..frames {
        let indices = sample_distinct_indices(&mut rng, n, k);
        // Move dynamic particles
        for &idx in &indices {
            arr[idx].x += rng.gen_range(-2.0..2.0f32);
            arr[idx].y += rng.gen_range(-2.0..2.0f32);
            arr[idx].z += rng.gen_range(-2.0..2.0f32);
            arr[idx].recompute_distance();
        }
        let start = Instant::now();
        arr.sort_by(cmp_fn);
        times.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    results.push(ParticleResult {
        label: "FullSort".into(), n, k,
        comparator: cmp_label.into(),
        time_stats: calculate_stats(&times),
    });

    // --- NearlySort (just call sort — TimSort/DriftSort detects near-sorted) ---
    let mut arr = base.clone();
    let mut times = Vec::with_capacity(frames);
    for _ in 0..frames {
        let indices = sample_distinct_indices(&mut rng, n, k);
        for &idx in &indices {
            arr[idx].x += rng.gen_range(-2.0..2.0f32);
            arr[idx].y += rng.gen_range(-2.0..2.0f32);
            arr[idx].z += rng.gen_range(-2.0..2.0f32);
            arr[idx].recompute_distance();
        }
        let start = Instant::now();
        arr.sort_by(cmp_fn);
        times.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    results.push(ParticleResult {
        label: "NearlySort".into(), n, k,
        comparator: cmp_label.into(),
        time_stats: calculate_stats(&times),
    });

    // --- DeltaSort ---
    let mut arr = base.clone();
    let mut times = Vec::with_capacity(frames);
    for _ in 0..frames {
        let indices = sample_distinct_indices(&mut rng, n, k);
        let dirty: HashSet<usize> = indices.iter().copied().collect();
        for &idx in &indices {
            arr[idx].x += rng.gen_range(-2.0..2.0f32);
            arr[idx].y += rng.gen_range(-2.0..2.0f32);
            arr[idx].z += rng.gen_range(-2.0..2.0f32);
            arr[idx].recompute_distance();
        }
        let start = Instant::now();
        deltasort::delta_sort_by(&mut arr, &dirty, cmp_fn);
        times.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    results.push(ParticleResult {
        label: "DeltaSort".into(), n, k,
        comparator: cmp_label.into(),
        time_stats: calculate_stats(&times),
    });

    results
}

// =============================================================================
// MAIN
// =============================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let export = args.iter().any(|a| a == "--export");

    println!("═══════════════════════════════════════════════════════════════");
    println!("  DeltaSort Applications Benchmark");
    println!("═══════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // TRACK 3: CDC Sorted Materialized View
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  TRACK 3: CDC Sorted Materialized View (n=100K, small records)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let n_cdc = 100_000;
    let k_values_cdc: Vec<usize> = vec![1, 10, 50, 100, 500, 1000, 5000, 10000, 20000];
    let iters_cdc = 30;
    let mut cdc_all: Vec<Vec<CdcResult>> = Vec::new();

    for &k in &k_values_cdc {
        let pct = k as f64 / n_cdc as f64 * 100.0;
        eprint!("  k={:<6} ({:>6.3}%) ... ", k, pct);
        let res = run_cdc_benchmark_small(n_cdc, k, iters_cdc);
        // Print one-liner
        let ds = res.iter().find(|r| r.label == "DeltaSort").unwrap();
        let fs = res.iter().find(|r| r.label == "FullSort").unwrap();
        let speedup = fs.time_stats.mean / ds.time_stats.mean;
        eprintln!("DeltaSort {:.1}µs  FullSort {:.1}µs  ({:.1}x)", ds.time_stats.mean, fs.time_stats.mean, speedup);
        cdc_all.push(res);
    }

    // Print CDC summary table
    println!("\n{:<10} {:>14} {:>14} {:>14} {:>14}", "k", "FullSort(µs)", "ESM(µs)", "BIS(µs)", "DeltaSort(µs)");
    println!("{}", "─".repeat(70));
    for group in &cdc_all {
        let k = group[0].k;
        let fs = group.iter().find(|r| r.label == "FullSort").map(|r| format!("{:.1} ±{:.1}%", r.time_stats.mean, r.time_stats.cv)).unwrap_or("-".into());
        let esm = group.iter().find(|r| r.label == "ESM").map(|r| format!("{:.1} ±{:.1}%", r.time_stats.mean, r.time_stats.cv)).unwrap_or("-".into());
        let bis = group.iter().find(|r| r.label == "BIS").map(|r| format!("{:.1} ±{:.1}%", r.time_stats.mean, r.time_stats.cv)).unwrap_or("🐢".into());
        let ds = group.iter().find(|r| r.label == "DeltaSort").map(|r| format!("{:.1} ±{:.1}%", r.time_stats.mean, r.time_stats.cv)).unwrap_or("-".into());
        println!("{:<10} {:>14} {:>14} {:>14} {:>14}", k, fs, esm, bis, ds);
    }

    // =========================================================================
    // TRACK 1: Sorted Array vs. BTreeMap Crossover
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  TRACK 1: Sorted Array vs. BTreeMap Crossover (n=100K)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let n_cross = 100_000;
    let k_values_cross: Vec<usize> = vec![1, 10, 100, 1000];
    let q_values: Vec<usize> = vec![100, 1000, 10000, 50000, 100000];
    let iters_cross = 10;
    let mut cross_results: Vec<CrossoverResult> = Vec::new();

    for &k in &k_values_cross {
        for &q in &q_values {
            eprint!("  k={:<5} Q={:<7} ... ", k, q);
            let res = run_crossover_benchmark(n_cross, k, q, iters_cross);
            let winner = if res.array_deltasort_us < res.btree_us { "Array+DS ⚡" } else { "BTreeMap ⚡" };
            eprintln!("Array+DS {:.0}µs  BTree {:.0}µs  FullSort {:.0}µs  {}", res.array_deltasort_us, res.btree_us, res.array_fullsort_us, winner);
            cross_results.push(res);
        }
    }

    // Print crossover summary table
    println!("\n{:<6} {:>8} {:>16} {:>16} {:>16} {:>10}", "k", "Q", "Array+DS(µs)", "BTreeMap(µs)", "FullSort(µs)", "Winner");
    println!("{}", "─".repeat(80));
    for r in &cross_results {
        let winner = if r.array_deltasort_us < r.btree_us { "Array+DS" } else { "BTreeMap" };
        println!("{:<6} {:>8} {:>16.1} {:>16.1} {:>16.1} {:>10}", r.k, r.q, r.array_deltasort_us, r.btree_us, r.array_fullsort_us, winner);
    }

    // =========================================================================
    // TRACK 2: Particle Depth Sorting
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  TRACK 2: Game Engine Particle Depth Sorting");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let n_particle = 100_000;
    let k_values_particle: Vec<usize> = vec![10, 50, 100, 500, 1000, 5000];
    let frames = 200;

    // --- Cheap comparator ---
    println!("  Cheap comparator (pre-computed f64 distance):");
    let mut particle_cheap: Vec<Vec<ParticleResult>> = Vec::new();
    for &k in &k_values_particle {
        eprint!("    k={:<5} ... ", k);
        let res = run_particle_benchmark(n_particle, k, frames, false);
        let ds = res.iter().find(|r| r.label == "DeltaSort").unwrap();
        let fs = res.iter().find(|r| r.label == "FullSort").unwrap();
        eprintln!("DeltaSort {:.1}µs  FullSort {:.1}µs  ({:.1}x)", ds.time_stats.mean, fs.time_stats.mean, fs.time_stats.mean / ds.time_stats.mean);
        particle_cheap.push(res);
    }

    // --- Expensive comparator ---
    println!("\n  Expensive comparator (recompute 3D distance per comparison):");
    let mut particle_expensive: Vec<Vec<ParticleResult>> = Vec::new();
    for &k in &k_values_particle {
        eprint!("    k={:<5} ... ", k);
        let res = run_particle_benchmark(n_particle, k, frames, true);
        let ds = res.iter().find(|r| r.label == "DeltaSort").unwrap();
        let fs = res.iter().find(|r| r.label == "FullSort").unwrap();
        eprintln!("DeltaSort {:.1}µs  FullSort {:.1}µs  ({:.1}x)", ds.time_stats.mean, fs.time_stats.mean, fs.time_stats.mean / ds.time_stats.mean);
        particle_expensive.push(res);
    }

    // Print particle summary tables
    for (label, data) in [("Cheap Comparator", &particle_cheap), ("Expensive Comparator", &particle_expensive)] {
        println!("\n  {} (n={}):", label, n_particle);
        println!("  {:<8} {:>14} {:>14} {:>14}", "k", "FullSort(µs)", "NearlySort(µs)", "DeltaSort(µs)");
        println!("  {}", "─".repeat(56));
        for group in data {
            let k = group[0].k;
            let fs = group.iter().find(|r| r.label == "FullSort").unwrap();
            let ns = group.iter().find(|r| r.label == "NearlySort").unwrap();
            let ds = group.iter().find(|r| r.label == "DeltaSort").unwrap();
            println!("  {:<8} {:>11.1} ±{:.1}% {:>11.1} ±{:.1}% {:>11.1} ±{:.1}%",
                k, fs.time_stats.mean, fs.time_stats.cv,
                ns.time_stats.mean, ns.time_stats.cv,
                ds.time_stats.mean, ds.time_stats.cv);
        }
    }

    // =========================================================================
    // CSV Export
    // =========================================================================
    if export {
        // CDC
        let mut csv = String::from("k,fullsort_us,fullsort_cv,esm_us,esm_cv,bis_us,bis_cv,deltasort_us,deltasort_cv\n");
        for group in &cdc_all {
            let k = group[0].k;
            let fs = group.iter().find(|r| r.label == "FullSort").unwrap();
            let esm = group.iter().find(|r| r.label == "ESM").unwrap();
            let bis_us = group.iter().find(|r| r.label == "BIS").map(|r| r.time_stats.mean).unwrap_or(f64::NAN);
            let bis_cv = group.iter().find(|r| r.label == "BIS").map(|r| r.time_stats.cv).unwrap_or(f64::NAN);
            let ds = group.iter().find(|r| r.label == "DeltaSort").unwrap();
            csv += &format!("{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}\n",
                k, fs.time_stats.mean, fs.time_stats.cv, esm.time_stats.mean, esm.time_stats.cv,
                bis_us, bis_cv, ds.time_stats.mean, ds.time_stats.cv);
        }
        std::fs::write("cdc_benchmark.csv", csv).unwrap();

        // Crossover
        let mut csv = String::from("k,q,array_deltasort_us,btree_us,array_fullsort_us\n");
        for r in &cross_results {
            csv += &format!("{},{},{:.2},{:.2},{:.2}\n", r.k, r.q, r.array_deltasort_us, r.btree_us, r.array_fullsort_us);
        }
        std::fs::write("crossover_benchmark.csv", csv).unwrap();

        // Particles
        for (fname, data) in [("particle_cheap.csv", &particle_cheap), ("particle_expensive.csv", &particle_expensive)] {
            let mut csv = String::from("k,fullsort_us,fullsort_cv,nearlysort_us,nearlysort_cv,deltasort_us,deltasort_cv\n");
            for group in data {
                let k = group[0].k;
                let fs = group.iter().find(|r| r.label == "FullSort").unwrap();
                let ns = group.iter().find(|r| r.label == "NearlySort").unwrap();
                let ds = group.iter().find(|r| r.label == "DeltaSort").unwrap();
                csv += &format!("{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}\n",
                    k, fs.time_stats.mean, fs.time_stats.cv, ns.time_stats.mean, ns.time_stats.cv,
                    ds.time_stats.mean, ds.time_stats.cv);
            }
            std::fs::write(fname, csv).unwrap();
        }

        println!("\n✅ Exported: cdc_benchmark.csv, crossover_benchmark.csv, particle_cheap.csv, particle_expensive.csv");
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Benchmark complete.");
    println!("═══════════════════════════════════════════════════════════════");
}
