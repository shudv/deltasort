//! Hypothesis: DeltaSort beats pdqsort when per-comparison cost is HIGH.
//!
//! At k/n ≪ 1%, DeltaSort does O(n + k log n) comparisons vs O(n log n) for pdqsort.
//! For n=10K, k=10: 10140 vs 140K comparisons — 14× fewer.
//! With cheap integer compares (~1 ns each), this doesn't translate because pdqsort's
//! per-comparison constants are tiny (branchless, SIMD-friendly). But if a comparison
//! costs 50-200 ns (long strings, computed keys, multi-field comparator), the 14× gap
//! should materialize.
//!
//! Three workloads:
//!   1. STRINGS: sort Vec<String> of 64-byte strings (prefix-similar, so compare is full scan)
//!   2. EXPENSIVE_KEY: sort by f64 key computed from expensive function (simulating ML score)
//!   3. MULTIFIELD: sort by tuple comparator over 5 fields with NULLs

use deltasort::delta_sort_by;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn gen_strings(n: usize, len: usize, prefix_len: usize, seed: u64) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let prefix: String = (0..prefix_len).map(|_| (b'a' + rng.gen_range(0..26)) as char).collect();
    (0..n).map(|_| {
        let suffix: String = (0..(len - prefix_len)).map(|_| (b'a' + rng.gen_range(0..26)) as char).collect();
        format!("{}{}", prefix, suffix)
    }).collect()
}

fn inject_k_dirty<T: Clone>(arr: &mut [T], k: usize, new_values: &[T], seed: u64) -> Vec<usize> {
    let n = arr.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut dirty: Vec<usize> = Vec::with_capacity(k);
    let mut used = std::collections::HashSet::new();
    while dirty.len() < k {
        let idx = rng.gen_range(0..n);
        if used.insert(idx) {
            dirty.push(idx);
        }
    }
    for (i, &d) in dirty.iter().enumerate() {
        arr[d] = new_values[i % new_values.len()].clone();
    }
    dirty.sort_unstable();
    dirty
}

fn bench_strings(n: usize, k: usize, str_len: usize, prefix_len: usize, reps: usize) -> (f64, f64) {
    let mut ds_times = Vec::with_capacity(reps);
    let mut pdq_times = Vec::with_capacity(reps);
    for r in 0..reps {
        let seed = 0xC0FFEE_u64.wrapping_add(r as u64 * 7919);
        let mut data = gen_strings(n, str_len, prefix_len, seed);
        data.sort_unstable();

        // Inject k dirty: replace values at k positions with fresh random strings
        let new_vals = gen_strings(k, str_len, prefix_len, seed.wrapping_add(2));
        let mut data_ds = data.clone();
        let dirty = inject_k_dirty(&mut data_ds, k, &new_vals, seed.wrapping_add(1));

        let mut data_pdq = data_ds.clone();

        // DeltaSort
        let mut dirty_for_ds = dirty.clone();
        let t = Instant::now();
        delta_sort_by(&mut data_ds, &mut dirty_for_ds, |a, b| a.cmp(b));
        let ds_ns = t.elapsed().as_nanos() as f64;
        ds_times.push(ds_ns);

        // pdqsort (sort_unstable)
        let t = Instant::now();
        data_pdq.sort_unstable();
        let pdq_ns = t.elapsed().as_nanos() as f64;
        pdq_times.push(pdq_ns);

        assert_eq!(data_ds, data_pdq, "outputs disagree at n={}, k={}", n, k);
    }
    ds_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    pdq_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (ds_times[ds_times.len() / 2], pdq_times[pdq_times.len() / 2])
}

fn expensive_hash(x: u64) -> u64 {
    // Simulate expensive key computation: 100 rounds of a mixing function
    let mut h = x;
    for _ in 0..100 {
        h = h.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0xBF58476D1CE4E5B9);
        h ^= h >> 30;
    }
    h
}

#[derive(Clone, Default)]
struct ExpItem {
    raw: u64,
    key: u64,
}

fn bench_expensive_key(n: usize, k: usize, reps: usize) -> (f64, f64, f64) {
    let mut ds_times = Vec::with_capacity(reps);
    let mut pdq_times_recompute = Vec::with_capacity(reps);
    let mut pdq_times_precomputed = Vec::with_capacity(reps);

    for r in 0..reps {
        let seed = 0xC0FFEE_u64.wrapping_add(r as u64 * 7919);
        let mut rng = StdRng::seed_from_u64(seed);

        let raws: Vec<u64> = (0..n).map(|_| rng.gen()).collect();
        let mut items: Vec<ExpItem> = raws.iter().map(|&r| ExpItem { raw: r, key: expensive_hash(r) }).collect();
        items.sort_unstable_by_key(|x| x.key);

        // Choose k dirty indices, update raw, and recompute key for those k.
        let mut dirty: Vec<usize> = (0..k).map(|_| rng.gen_range(0..n)).collect();
        dirty.sort_unstable();
        dirty.dedup();
        for &d in &dirty {
            items[d].raw = rng.gen();
        }
        let mut items_ds = items.clone();
        // For DeltaSort: only recompute k keys
        let t = Instant::now();
        for &d in &dirty {
            items_ds[d].key = expensive_hash(items_ds[d].raw);
        }
        delta_sort_by(&mut items_ds, &mut dirty.clone(), |a, b| a.key.cmp(&b.key));
        let ds_ns = t.elapsed().as_nanos() as f64;
        ds_times.push(ds_ns);

        // For pdqsort with key recomputation (fair: it doesn't know dirty set)
        let mut items_pdq1 = items.clone();
        let t = Instant::now();
        for it in items_pdq1.iter_mut() {
            it.key = expensive_hash(it.raw);
        }
        items_pdq1.sort_unstable_by_key(|x| x.key);
        let pdq1_ns = t.elapsed().as_nanos() as f64;
        pdq_times_recompute.push(pdq1_ns);

        // For pdqsort with precomputed keys (assuming someone else fixed the keys for us)
        let mut items_pdq2 = items.clone();
        for &d in &dirty {
            items_pdq2[d].key = expensive_hash(items_pdq2[d].raw);
        }
        let t = Instant::now();
        items_pdq2.sort_unstable_by_key(|x| x.key);
        let pdq2_ns = t.elapsed().as_nanos() as f64;
        pdq_times_precomputed.push(pdq2_ns);

        // Sanity check
        let ks_ds: Vec<u64> = items_ds.iter().map(|x| x.key).collect();
        let ks_pdq: Vec<u64> = items_pdq2.iter().map(|x| x.key).collect();
        assert_eq!(ks_ds, ks_pdq, "outputs disagree n={}, k={}", n, k);
    }
    ds_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    pdq_times_recompute.sort_by(|a, b| a.partial_cmp(b).unwrap());
    pdq_times_precomputed.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (
        ds_times[ds_times.len() / 2],
        pdq_times_recompute[pdq_times_recompute.len() / 2],
        pdq_times_precomputed[pdq_times_precomputed.len() / 2],
    )
}

#[derive(Clone, Default)]
struct MultiField {
    a: i32,
    b: Option<i32>,
    c: i64,
    d: u8,
    e: i16,
}

fn multifield_cmp(x: &MultiField, y: &MultiField) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    // NULLS LAST on b, then sort by a, c, d, e
    match x.a.cmp(&y.a) {
        Equal => {}
        ord => return ord,
    }
    match (x.b, y.b) {
        (None, None) => {}
        (None, Some(_)) => return Greater,
        (Some(_), None) => return Less,
        (Some(p), Some(q)) => match p.cmp(&q) {
            Equal => {}
            ord => return ord,
        },
    }
    match x.c.cmp(&y.c) {
        Equal => {}
        ord => return ord,
    }
    match x.d.cmp(&y.d) {
        Equal => {}
        ord => return ord,
    }
    x.e.cmp(&y.e)
}

fn bench_multifield(n: usize, k: usize, reps: usize) -> (f64, f64) {
    let mut ds_times = Vec::with_capacity(reps);
    let mut pdq_times = Vec::with_capacity(reps);
    for r in 0..reps {
        let seed = 0xC0FFEE_u64.wrapping_add(r as u64 * 7919);
        let mut rng = StdRng::seed_from_u64(seed);

        // Bias: a is the dominant field but has many duplicates, so the comparator
        // typically descends to b/c/d/e — making the comparison expensive on average.
        let gen = |rng: &mut StdRng| MultiField {
            a: rng.gen_range(0..10),
            b: if rng.gen_bool(0.2) { None } else { Some(rng.gen_range(0..1000)) },
            c: rng.gen_range(0..1000000),
            d: rng.gen(),
            e: rng.gen(),
        };

        let mut data: Vec<MultiField> = (0..n).map(|_| gen(&mut rng)).collect();
        data.sort_unstable_by(multifield_cmp);

        let mut dirty: Vec<usize> = (0..k).map(|_| rng.gen_range(0..n)).collect();
        dirty.sort_unstable();
        dirty.dedup();
        for &d in &dirty {
            data[d] = gen(&mut rng);
        }

        let mut data_ds = data.clone();
        let mut data_pdq = data.clone();

        let t = Instant::now();
        delta_sort_by(&mut data_ds, &mut dirty.clone(), multifield_cmp);
        let ds_ns = t.elapsed().as_nanos() as f64;
        ds_times.push(ds_ns);

        let t = Instant::now();
        data_pdq.sort_unstable_by(multifield_cmp);
        let pdq_ns = t.elapsed().as_nanos() as f64;
        pdq_times.push(pdq_ns);
    }
    ds_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    pdq_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (ds_times[ds_times.len() / 2], pdq_times[pdq_times.len() / 2])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps: usize = args
        .iter()
        .position(|a| a == "--reps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    fs::create_dir_all("results").ok();
    let path = Path::new("results/track_b_expensive_compare.csv");
    let mut f = fs::File::create(path).expect("create csv");
    writeln!(f, "experiment,n,k,variant,detail,ns_total,ns_per_item,speedup_vs_pdq").unwrap();

    println!("\n=== Experiment 1: STRING SORT (64-byte strings, 32-byte common prefix) ===\n");
    println!("{:>10} {:>10} {:>14} {:>14} {:>10}", "n", "k", "DS ns", "pdq ns", "speedup");
    for &n in &[1_000, 10_000, 100_000, 1_000_000] {
        for &k_pct in &[0.001, 0.01, 0.1, 1.0] {
            let k = ((n as f64) * k_pct / 100.0).ceil().max(1.0) as usize;
            let (ds, pdq) = bench_strings(n, k, 64, 32, reps);
            let speedup = pdq / ds;
            println!("{:>10} {:>10} {:>14.0} {:>14.0} {:>10.2}x", n, k, ds, pdq, speedup);
            writeln!(f, "strings,{},{},DeltaSort,prefix32_len64,{:.0},{:.2},{:.2}", n, k, ds, ds / n as f64, speedup).unwrap();
            writeln!(f, "strings,{},{},pdqsort,prefix32_len64,{:.0},{:.2},1.00", n, k, pdq, pdq / n as f64).unwrap();
        }
    }

    println!("\n=== Experiment 2: EXPENSIVE COMPUTED KEY (100-round hash, simulating ML score) ===\n");
    println!("{:>10} {:>10} {:>14} {:>14} {:>14} {:>10} {:>10}", "n", "k", "DS ns", "pdq+recomp", "pdq+precomp", "vs recomp", "vs precomp");
    for &n in &[1_000, 10_000, 100_000, 1_000_000] {
        for &k_pct in &[0.001, 0.01, 0.1, 1.0] {
            let k = ((n as f64) * k_pct / 100.0).ceil().max(1.0) as usize;
            let (ds, pdq_r, pdq_p) = bench_expensive_key(n, k, reps);
            let sp_r = pdq_r / ds;
            let sp_p = pdq_p / ds;
            println!("{:>10} {:>10} {:>14.0} {:>14.0} {:>14.0} {:>9.2}x {:>9.2}x", n, k, ds, pdq_r, pdq_p, sp_r, sp_p);
            writeln!(f, "expensive_key,{},{},DeltaSort,key_recompute_k,{:.0},{:.2},{:.2}", n, k, ds, ds / n as f64, sp_r).unwrap();
            writeln!(f, "expensive_key,{},{},pdqsort_recompute_all,key_recompute_n,{:.0},{:.2},1.00", n, k, pdq_r, pdq_r / n as f64).unwrap();
            writeln!(f, "expensive_key,{},{},pdqsort_precomputed,key_recompute_k,{:.0},{:.2},{:.2}", n, k, pdq_p, pdq_p / n as f64, pdq_p / ds).unwrap();
        }
    }

    println!("\n=== Experiment 3: MULTI-FIELD COMPARATOR (5 columns w/ NULLs, SQL-like ORDER BY) ===\n");
    println!("{:>10} {:>10} {:>14} {:>14} {:>10}", "n", "k", "DS ns", "pdq ns", "speedup");
    for &n in &[1_000, 10_000, 100_000, 1_000_000] {
        for &k_pct in &[0.001, 0.01, 0.1, 1.0] {
            let k = ((n as f64) * k_pct / 100.0).ceil().max(1.0) as usize;
            let (ds, pdq) = bench_multifield(n, k, reps);
            let speedup = pdq / ds;
            println!("{:>10} {:>10} {:>14.0} {:>14.0} {:>10.2}x", n, k, ds, pdq, speedup);
            writeln!(f, "multifield,{},{},DeltaSort,5cols_w_nulls,{:.0},{:.2},{:.2}", n, k, ds, ds / n as f64, speedup).unwrap();
            writeln!(f, "multifield,{},{},pdqsort,5cols_w_nulls,{:.0},{:.2},1.00", n, k, pdq, pdq / n as f64).unwrap();
        }
    }

    println!("\nWrote {}", path.display());
}
