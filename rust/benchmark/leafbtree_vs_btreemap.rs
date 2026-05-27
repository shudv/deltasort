//! Track A — LeafBTree vs std::collections::BTreeMap.
//!
//! Compares insertion, lookup, range scan throughput across:
//!   - n in {1e4, 1e5, 1e6, 1e7}
//!   - leaf_capacity in {64, 256, 1024, 4096, 16384}
//!   - buffer_capacity = sqrt(leaf_capacity) (DeltaSort sweet spot)
//!
//! Workloads:
//!   - bulk_insert: insert n random keys, measure total time
//!   - point_lookup: lookup n random keys after build, throughput
//!   - range_scan: full in-order iteration throughput
//!   - mixed_5050: 50% inserts + 50% lookups interleaved

use deltasort::leaf_btree::{Compactor, LeafBTree};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

#[derive(Clone, Copy)]
enum Workload {
    BulkInsert,
    PointLookup,
    RangeScan,
    Mixed5050,
}

impl Workload {
    fn name(self) -> &'static str {
        match self {
            Workload::BulkInsert => "bulk_insert",
            Workload::PointLookup => "point_lookup",
            Workload::RangeScan => "range_scan",
            Workload::Mixed5050 => "mixed_5050",
        }
    }
}

fn gen_keys(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen()).collect()
}

fn bench_btreemap(workload: Workload, n: usize, seed: u64) -> (f64, BTreeMap<u64, u64>) {
    let keys = gen_keys(n, seed);
    let mut m = BTreeMap::<u64, u64>::new();
    match workload {
        Workload::BulkInsert => {
            let t = Instant::now();
            for (i, &k) in keys.iter().enumerate() {
                m.insert(k, i as u64);
            }
            (t.elapsed().as_nanos() as f64 / n as f64, m)
        }
        Workload::PointLookup => {
            for (i, &k) in keys.iter().enumerate() {
                m.insert(k, i as u64);
            }
            let lookup_keys = gen_keys(n, seed.wrapping_add(1));
            let t = Instant::now();
            let mut hit = 0u64;
            for &k in &lookup_keys {
                if let Some(v) = m.get(&k) {
                    hit = hit.wrapping_add(*v);
                }
            }
            std::hint::black_box(hit);
            (t.elapsed().as_nanos() as f64 / n as f64, m)
        }
        Workload::RangeScan => {
            for (i, &k) in keys.iter().enumerate() {
                m.insert(k, i as u64);
            }
            let t = Instant::now();
            let mut sum = 0u64;
            for (_k, v) in m.iter() {
                sum = sum.wrapping_add(*v);
            }
            std::hint::black_box(sum);
            (t.elapsed().as_nanos() as f64 / n as f64, m)
        }
        Workload::Mixed5050 => {
            let lookup_keys = gen_keys(n, seed.wrapping_add(1));
            let t = Instant::now();
            for i in 0..n {
                if i & 1 == 0 {
                    m.insert(keys[i], i as u64);
                } else {
                    let _ = m.get(&lookup_keys[i]);
                }
            }
            (t.elapsed().as_nanos() as f64 / n as f64, m)
        }
    }
}

fn bench_leafbtree(
    workload: Workload,
    n: usize,
    leaf_cap: usize,
    buf_cap: usize,
    compactor: Compactor,
    seed: u64,
) -> f64 {
    let keys = gen_keys(n, seed);
    let mut t = LeafBTree::<u64, u64>::with_config_and_compactor(leaf_cap, buf_cap, 16, compactor);
    match workload {
        Workload::BulkInsert => {
            let start = Instant::now();
            for (i, &k) in keys.iter().enumerate() {
                t.insert(k, i as u64);
            }
            t.compact_all();
            start.elapsed().as_nanos() as f64 / n as f64
        }
        Workload::PointLookup => {
            for (i, &k) in keys.iter().enumerate() {
                t.insert(k, i as u64);
            }
            t.compact_all();
            let lookup_keys = gen_keys(n, seed.wrapping_add(1));
            let start = Instant::now();
            let mut hit = 0u64;
            for &k in &lookup_keys {
                if let Some(v) = t.get(&k) {
                    hit = hit.wrapping_add(*v);
                }
            }
            std::hint::black_box(hit);
            start.elapsed().as_nanos() as f64 / n as f64
        }
        Workload::RangeScan => {
            for (i, &k) in keys.iter().enumerate() {
                t.insert(k, i as u64);
            }
            t.compact_all();
            let start = Instant::now();
            let mut sum = 0u64;
            for (_k, v) in t.iter() {
                sum = sum.wrapping_add(*v);
            }
            std::hint::black_box(sum);
            start.elapsed().as_nanos() as f64 / n as f64
        }
        Workload::Mixed5050 => {
            let lookup_keys = gen_keys(n, seed.wrapping_add(1));
            let start = Instant::now();
            for i in 0..n {
                if i & 1 == 0 {
                    t.insert(keys[i], i as u64);
                } else {
                    let _ = t.get(&lookup_keys[i]);
                }
            }
            start.elapsed().as_nanos() as f64 / n as f64
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let fast = args.iter().any(|a| a == "--fast");
    let reps: usize = args
        .iter()
        .position(|a| a == "--reps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let only_n: Option<usize> = args
        .iter()
        .position(|a| a == "--n")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());

    fs::create_dir_all("results").ok();

    let n_values: Vec<usize> = if let Some(n) = only_n {
        vec![n]
    } else if fast {
        vec![10_000, 100_000]
    } else {
        vec![10_000, 100_000, 1_000_000, 10_000_000]
    };

    let leaf_configs: &[(usize, usize)] = &[
        (64, 8),
        (256, 16),
        (1024, 32),
        (4096, 64),
        (16384, 128),
    ];

    let workloads = [
        Workload::BulkInsert,
        Workload::PointLookup,
        Workload::RangeScan,
        Workload::Mixed5050,
    ];

    let path = Path::new("results/track_a_leafbtree.csv");
    let append = only_n.is_some() && path.exists();
    let mut f: Box<dyn Write> = if append {
        Box::new(fs::OpenOptions::new().append(true).open(path).expect("open csv"))
    } else {
        let mut x = fs::File::create(path).expect("create csv");
        writeln!(x, "workload,n,structure,leaf_cap,buf_cap,rep,ns_per_op").unwrap();
        Box::new(x)
    };

    println!(
        "{:>14} {:>10} {:>20} {:>10} {:>10} {:>14} {:>14}",
        "workload", "n", "structure", "leaf_cap", "buf_cap", "median ns/op", "min ns/op"
    );

    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };

    for &n in &n_values {
        for &wl in &workloads {
            let mut samples = Vec::with_capacity(reps);
            for r in 0..reps {
                let seed = 0xC0FFEE_u64.wrapping_add(r as u64 * 7919);
                let (ns, _m) = bench_btreemap(wl, n, seed);
                samples.push(ns);
                writeln!(f, "{},{},std::BTreeMap,,,{},{:.1}", wl.name(), n, r, ns).unwrap();
                f.flush().ok();
            }
            let med = median(samples.clone());
            let mn = samples.iter().cloned().fold(f64::INFINITY, f64::min);
            println!(
                "{:>14} {:>10} {:>20} {:>10} {:>10} {:>14.1} {:>14.1}",
                wl.name(), n, "std::BTreeMap", "-", "-", med, mn
            );

            for &(leaf_cap, buf_cap) in leaf_configs {
                for (cstr, comp) in [("LeafBTree-DS", Compactor::DeltaSort), ("LeafBTree-FS", Compactor::FullSort)] {
                    let mut samples = Vec::with_capacity(reps);
                    for r in 0..reps {
                        let seed = 0xC0FFEE_u64.wrapping_add(r as u64 * 7919);
                        let ns = bench_leafbtree(wl, n, leaf_cap, buf_cap, comp, seed);
                        samples.push(ns);
                        writeln!(
                            f, "{},{},{},{},{},{},{:.1}",
                            wl.name(), n, cstr, leaf_cap, buf_cap, r, ns
                        ).unwrap();
                        f.flush().ok();
                    }
                    let med = median(samples.clone());
                    let mn = samples.iter().cloned().fold(f64::INFINITY, f64::min);
                    println!(
                        "{:>14} {:>10} {:>20} {:>10} {:>10} {:>14.1} {:>14.1}",
                        wl.name(), n, cstr, leaf_cap, buf_cap, med, mn
                    );
                }
            }
        }
    }
    println!("\nWrote {}", path.display());
}
