//! Track 1 — sorted Vec + DeltaSort vs BTreeMap crossover benchmark.
//!
//! Workload (one "epoch"):
//!   1. k items have their sort keys mutated in place
//!   2. Q indexed-rank reads (`get element at rank r`)
//!
//! Strategies compared:
//!   - VecFullSort:  Vec<User> + `sort_unstable_by` per epoch, O(1) rank read
//!   - VecDeltaSort: Vec<User> + DeltaSort per epoch,        O(1) rank read
//!   - VecESM:       Vec<User> + extract-sort-merge per epoch, O(1) rank read
//!   - BTreeNth:     BTreeMap<(SortKey, Id), ()> with k remove+insert, `iter().nth(r)` rank read
//!   - BTreeIdeal:   BTreeMap update cost + simulated O(log n) rank read (augmented OST upper bound)
//!
//! Output: `rust/results/track1_*.csv` for downstream plotting.

mod binary_insertion_sort;
mod data;
mod esm_variants;
mod extract_sort_merge;
mod instrumented_deltasort;
mod statistics;

use crate::data::{
    generate_sorted_users, sample_distinct_indices, user_comparator, User,
};
use crate::extract_sort_merge::extract_sort_merge;
use deltasort::delta_sort_by;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SortKey {
    country: String,
    age: u32,
    name: String,
}

impl SortKey {
    fn from_user(u: &User) -> Self {
        SortKey {
            country: u.country.clone(),
            age: u.age,
            name: u.name.clone(),
        }
    }
}

// =============================================================================
// Strategies
// =============================================================================

fn epoch_full_sort(arr: &mut Vec<User>, dirty: &[usize], read_ranks: &[usize]) -> Duration {
    // dirty[] already applied to arr by the caller. We just re-sort and read.
    let _ = dirty;
    let t0 = Instant::now();
    arr.sort_unstable_by(user_comparator);
    let mut sink: u64 = 0;
    for &r in read_ranks {
        sink ^= arr[r].age as u64;
    }
    std::hint::black_box(sink);
    t0.elapsed()
}

fn epoch_delta_sort(arr: &mut Vec<User>, dirty: &[usize], read_ranks: &[usize]) -> Duration {
    let mut d = dirty.to_vec();
    let t0 = Instant::now();
    delta_sort_by(arr, &mut d, user_comparator);
    let mut sink: u64 = 0;
    for &r in read_ranks {
        sink ^= arr[r].age as u64;
    }
    std::hint::black_box(sink);
    t0.elapsed()
}

fn epoch_esm(arr: &mut Vec<User>, dirty: &[usize], read_ranks: &[usize]) -> Duration {
    let mut d = dirty.to_vec();
    let t0 = Instant::now();
    extract_sort_merge(arr, &mut d, user_comparator);
    let mut sink: u64 = 0;
    for &r in read_ranks {
        sink ^= arr[r].age as u64;
    }
    std::hint::black_box(sink);
    t0.elapsed()
}

/// BTreeMap maintaining (sort_key, id) → user.
/// Updates: remove old (key,id), insert (newKey,id).
/// Rank reads: `iter().nth(r)` — O(r) walk.
fn epoch_btree_nth(
    btree: &mut BTreeMap<(SortKey, u32), User>,
    dirty_old_keys: &[(SortKey, u32)],
    dirty_new_users: &[User],
    read_ranks: &[usize],
) -> Duration {
    let t0 = Instant::now();
    for (i, old_key) in dirty_old_keys.iter().enumerate() {
        let new_user = dirty_new_users[i].clone();
        btree.remove(old_key);
        let new_key = (SortKey::from_user(&new_user), old_key.1);
        btree.insert(new_key, new_user);
    }
    let mut sink: u64 = 0;
    for &r in read_ranks {
        if let Some((_, u)) = btree.iter().nth(r) {
            sink ^= u.age as u64;
        }
    }
    std::hint::black_box(sink);
    t0.elapsed()
}

/// BTreeMap variant: do k remove+insert, then materialize once to Vec<&User>
/// and index it Q times. This is strictly faster than `iter().nth(r)` whenever
/// Q > sqrt(n)·log_factor. Represents what a real engineer would do.
fn epoch_btree_materialized(
    btree: &mut BTreeMap<(SortKey, u32), User>,
    dirty_old_keys: &[(SortKey, u32)],
    dirty_new_users: &[User],
    read_ranks: &[usize],
) -> Duration {
    let t0 = Instant::now();
    for (i, old_key) in dirty_old_keys.iter().enumerate() {
        let new_user = dirty_new_users[i].clone();
        btree.remove(old_key);
        let new_key = (SortKey::from_user(&new_user), old_key.1);
        btree.insert(new_key, new_user);
    }
    let materialized: Vec<&User> = btree.values().collect();
    let mut sink: u64 = 0;
    for &r in read_ranks {
        sink ^= materialized[r].age as u64;
    }
    std::hint::black_box(sink);
    t0.elapsed()
}

// =============================================================================
// Workload generation
// =============================================================================

fn make_btree(arr: &[User]) -> BTreeMap<(SortKey, u32), User> {
    let mut btree = BTreeMap::new();
    for (i, u) in arr.iter().enumerate() {
        btree.insert((SortKey::from_user(u), i as u32), u.clone());
    }
    btree
}

/// Snapshot old (key,id) for each dirty index BEFORE mutation, so we can
/// remove them from the BTreeMap and insert the new ones after Vec mutation.
fn snapshot_dirty(arr: &[User], dirty: &[usize]) -> Vec<(SortKey, u32)> {
    dirty
        .iter()
        .map(|&i| (SortKey::from_user(&arr[i]), i as u32))
        .collect()
}

// =============================================================================
// Per-epoch benchmark
// =============================================================================

#[derive(Clone, Debug)]
struct EpochResult {
    n: usize,
    k: usize,
    q: usize,
    full_us: f64,
    delta_us: f64,
    esm_us: f64,
    btree_nth_us: f64,
    btree_mat_us: f64,
}

fn run_epoch(n: usize, k: usize, q: usize, iterations: usize, seed: u64) -> EpochResult {
    let mut rng = StdRng::seed_from_u64(seed);

    let base = generate_sorted_users(n);

    let mut full_total = Duration::ZERO;
    let mut delta_total = Duration::ZERO;
    let mut esm_total = Duration::ZERO;
    let mut btree_nth_total = Duration::ZERO;
    let mut btree_mat_total = Duration::ZERO;

    // Heuristic: skip btree-nth when total work would be prohibitive.
    // iter().nth(r) is O(r); total cost ~ q * n / 2. Skip when > 5e8 op-equiv per iter.
    let skip_btree_nth = (q as u128) * (n as u128) > 500_000_000;

    for it in 0..iterations {
        let mut arr = base.clone();

        let dirty = sample_distinct_indices(&mut rng, n, k);
        let read_ranks: Vec<usize> = (0..q).map(|_| rng.gen_range(0..n)).collect();

        let old_keys = snapshot_dirty(&arr, &dirty);
        for &i in &dirty {
            arr[i].mutate(&mut rng);
        }
        let new_users: Vec<User> = dirty.iter().map(|&i| arr[i].clone()).collect();

        {
            let mut a = arr.clone();
            full_total += epoch_full_sort(&mut a, &dirty, &read_ranks);
        }
        {
            let mut a = arr.clone();
            delta_total += epoch_delta_sort(&mut a, &dirty, &read_ranks);
        }
        {
            let mut a = arr.clone();
            esm_total += epoch_esm(&mut a, &dirty, &read_ranks);
        }
        if !skip_btree_nth {
            let mut btree = make_btree(&base);
            btree_nth_total += epoch_btree_nth(&mut btree, &old_keys, &new_users, &read_ranks);
        }
        {
            let mut btree = make_btree(&base);
            btree_mat_total +=
                epoch_btree_materialized(&mut btree, &old_keys, &new_users, &read_ranks);
        }

        std::hint::black_box(it);
    }

    let to_us = |d: Duration| d.as_nanos() as f64 / 1000.0 / iterations as f64;

    EpochResult {
        n,
        k,
        q,
        full_us: to_us(full_total),
        delta_us: to_us(delta_total),
        esm_us: to_us(esm_total),
        btree_nth_us: if skip_btree_nth { f64::NAN } else { to_us(btree_nth_total) },
        btree_mat_us: to_us(btree_mat_total),
    }
}

// =============================================================================
// Crossover finder
// =============================================================================

/// Given fixed (n,k), sweep q and find the smallest q where DeltaSort beats
/// the BEST BTreeMap strategy (min of nth and materialized).
fn find_q_break(n: usize, k: usize, iterations: usize, seed: u64) -> Option<usize> {
    let q_candidates: &[usize] = &[0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000];
    for &q in q_candidates {
        let res = run_epoch(n, k, q, iterations, seed.wrapping_add(q as u64));
        let btree_best = if res.btree_nth_us.is_nan() {
            res.btree_mat_us
        } else {
            res.btree_nth_us.min(res.btree_mat_us)
        };
        if res.delta_us < btree_best {
            return Some(q);
        }
    }
    None
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let fast = args.iter().any(|a| a == "--fast");

    fs::create_dir_all("results").ok();

    let n_values: &[usize] = if fast {
        &[1_000, 10_000, 100_000]
    } else {
        &[1_000, 10_000, 100_000, 1_000_000]
    };
    let k_values: &[usize] = &[1, 10, 100, 1_000, 10_000];
    let q_values: &[usize] = &[0, 1, 10, 100, 1_000, 10_000];
    let iters_for_n = |n: usize, _q: usize| -> usize {
        if fast {
            2
        } else if n >= 1_000_000 {
            3
        } else if n >= 100_000 {
            8
        } else {
            20
        }
    };

    let mut sweep_csv =
        String::from("n,k,q,full_us,delta_us,esm_us,btree_nth_us,btree_mat_us\n");
    println!("# Track 1 sweep — (n, k, q) → wall time per epoch (µs, mean)");
    println!(
        "{:>9} {:>7} {:>7} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "n", "k", "q", "full", "delta", "esm", "btree_nth", "btree_mat"
    );

    let mut total_runs = 0usize;
    for &n in n_values {
        for &k in k_values {
            if k > n {
                continue;
            }
            for &q in q_values {
                let iters = iters_for_n(n, q);
                let res = run_epoch(n, k, q, iters, 0xC0FFEE_u64.wrapping_add(total_runs as u64));
                let fmt = |v: f64| {
                    if v.is_nan() {
                        "      n/a".to_string()
                    } else {
                        format!("{:>12.2}", v)
                    }
                };
                println!(
                    "{:>9} {:>7} {:>7} {:>12.2} {:>12.2} {:>12.2} {} {:>12.2}",
                    res.n,
                    res.k,
                    res.q,
                    res.full_us,
                    res.delta_us,
                    res.esm_us,
                    fmt(res.btree_nth_us),
                    res.btree_mat_us
                );
                let nth_csv = if res.btree_nth_us.is_nan() {
                    "".to_string()
                } else {
                    format!("{:.3}", res.btree_nth_us)
                };
                sweep_csv.push_str(&format!(
                    "{},{},{},{:.3},{:.3},{:.3},{},{:.3}\n",
                    res.n, res.k, res.q, res.full_us, res.delta_us, res.esm_us, nth_csv, res.btree_mat_us
                ));
                total_runs += 1;
            }
        }
    }
    let sweep_path = Path::new("results/track1_sweep.csv");
    fs::File::create(sweep_path)
        .and_then(|mut f| f.write_all(sweep_csv.as_bytes()))
        .expect("write sweep csv");
    println!("\nWrote {}", sweep_path.display());

    let qbreak_n: &[usize] = if fast {
        &[1_000, 10_000]
    } else {
        &[1_000, 10_000, 100_000, 1_000_000]
    };
    let qbreak_k: &[usize] = &[1, 10, 100, 1_000, 10_000];
    let mut qbreak_csv = String::from("n,k,q_break_delta\n");
    println!("\n# Q_break — smallest q where DeltaSort beats best BTree variant");
    println!("{:>9} {:>7} {:>12}", "n", "k", "q_break");
    for &n in qbreak_n {
        for &k in qbreak_k {
            if k > n {
                continue;
            }
            let iters = if fast {
                2
            } else if n >= 100_000 {
                3
            } else {
                10
            };
            let q = find_q_break(n, k, iters, 0xBABE_u64.wrapping_add((n * k) as u64));
            let q_str = q.map(|x| x.to_string()).unwrap_or_else(|| "NA".to_string());
            println!("{:>9} {:>7} {:>12}", n, k, q_str);
            qbreak_csv.push_str(&format!("{},{},{}\n", n, k, q_str));
        }
    }
    let qbreak_path = Path::new("results/track1_qbreak.csv");
    fs::File::create(qbreak_path)
        .and_then(|mut f| f.write_all(qbreak_csv.as_bytes()))
        .expect("write qbreak csv");
    println!("\nWrote {}", qbreak_path.display());
}
