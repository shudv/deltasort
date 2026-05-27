//! MIXED workload: scan-heavy + occasional inserts.
//!
//! The regime we missed in prior LeafBTree experiments: long range scans
//! (favors large leaves) + small-k compactions (DS's sweet spot).
//!
//! Tests whether LeafBTree+DeltaSort opens a *new design point* with much
//! larger leaves than std::BTreeMap can sustain.

use deltasort::leaf_btree::{Compactor, LeafBTree};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::time::Instant;

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
struct UrlKey(String);

fn gen_url(rng: &mut StdRng) -> UrlKey {
    let prefix = "https://example.com/api/v1/users/";
    let suffix: String = (0..(64 - prefix.len()))
        .map(|_| (b'a' + rng.gen_range(0..26)) as char)
        .collect();
    UrlKey(format!("{}{}", prefix, suffix))
}

enum Op {
    Insert(UrlKey),
    Scan(UrlKey, usize), // start key, max len
}

fn gen_workload(
    initial: &[UrlKey],
    n_ops: usize,
    scan_pct: f64,
    scan_len: usize,
    rng: &mut StdRng,
) -> Vec<Op> {
    let mut ops = Vec::with_capacity(n_ops);
    for _ in 0..n_ops {
        if rng.gen_bool(scan_pct) {
            let idx = rng.gen_range(0..initial.len());
            ops.push(Op::Scan(initial[idx].clone(), scan_len));
        } else {
            ops.push(Op::Insert(gen_url(rng)));
        }
    }
    ops
}

fn run_btreemap(initial: &[UrlKey], ops: &[Op]) -> f64 {
    let mut m: BTreeMap<UrlKey, u64> = initial.iter().enumerate().map(|(i, k)| (k.clone(), i as u64)).collect();
    let mut sink: u64 = 0;
    let t = Instant::now();
    for op in ops {
        match op {
            Op::Insert(k) => { m.insert(k.clone(), 0); }
            Op::Scan(start, len) => {
                let mut n = 0usize;
                for (_k, v) in m.range(start.clone()..) {
                    sink = sink.wrapping_add(*v);
                    n += 1;
                    if n >= *len { break; }
                }
            }
        }
    }
    let ns = t.elapsed().as_nanos() as f64;
    std::hint::black_box(sink);
    ns
}

fn run_leafbtree(
    initial: &[UrlKey],
    ops: &[Op],
    leaf_cap: usize,
    buf_cap: usize,
    comp: Compactor,
) -> f64 {
    let mut t = LeafBTree::<UrlKey, u64>::with_config_and_compactor(leaf_cap, buf_cap, 16, comp);
    for (i, k) in initial.iter().enumerate() { t.insert(k.clone(), i as u64); }
    t.compact_all();
    let mut sink: u64 = 0;
    let start = Instant::now();
    for op in ops {
        match op {
            Op::Insert(k) => { t.insert(k.clone(), 0); }
            Op::Scan(start_key, len) => {
                let mut n = 0usize;
                for (_k, v) in t.range_from(start_key) {
                    sink = sink.wrapping_add(*v);
                    n += 1;
                    if n >= *len { break; }
                }
            }
        }
    }
    let ns = start.elapsed().as_nanos() as f64;
    std::hint::black_box(sink);
    ns
}

fn main() {
    // Quick correctness check for range_from against BTreeMap
    {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 5000usize;
        let mut keys: Vec<UrlKey> = (0..n).map(|_| gen_url(&mut rng)).collect();
        keys.sort(); keys.dedup();
        let bt: BTreeMap<UrlKey, u64> = keys.iter().enumerate().map(|(i, k)| (k.clone(), i as u64)).collect();
        let mut lb = LeafBTree::<UrlKey, u64>::with_config(256, 32, 16);
        for (i, k) in keys.iter().enumerate() { lb.insert(k.clone(), i as u64); }
        // Pick 100 random start keys
        for _ in 0..100 {
            let start = &keys[rng.gen_range(0..keys.len())];
            let bt_v: Vec<u64> = bt.range(start.clone()..).take(50).map(|(_, v)| *v).collect();
            let lb_v: Vec<u64> = lb.range_from(start).take(50).map(|(_, v)| *v).collect();
            assert_eq!(bt_v, lb_v, "range_from mismatch");
        }
        println!("[sanity] range_from matches BTreeMap range for 100 random starts.");
    }

    fs::create_dir_all("results").ok();
    let mut f = fs::File::create("results/track_f_mixed.csv").expect("create csv");
    writeln!(f, "n_init,n_ops,scan_pct,scan_len,structure,leaf_cap,buf_cap,total_ns,ns_per_op").unwrap();

    let n_init = 1_000_000usize;
    let n_ops = 100_000usize;
    let scan_pct = 0.8;

    let mut rng_keys = StdRng::seed_from_u64(0xDEADBEEFu64);
    println!("Generating {} initial URL keys...", n_init);
    let mut initial: Vec<UrlKey> = (0..n_init).map(|_| gen_url(&mut rng_keys)).collect();
    initial.sort(); initial.dedup();
    while initial.len() < n_init { initial.push(gen_url(&mut rng_keys)); initial.sort(); initial.dedup(); }
    initial.sort();
    initial.truncate(n_init);
    println!("Initial set ready: {} keys", initial.len());

    println!(
        "\n{:>9}  {:>22}  {:>10} {:>10}  {:>12}  {:>12}",
        "scan_len", "structure", "leaf_cap", "buf_cap", "total_ms", "ns/op",
    );

    for &scan_len in &[10usize, 100, 1000, 10000] {
        let mut rng_ops = StdRng::seed_from_u64(0xC0DEu64.wrapping_add(scan_len as u64));
        let ops = gen_workload(&initial, n_ops, scan_pct, scan_len, &mut rng_ops);

        let configs: &[(usize, usize)] = &[
            (256, 4),
            (1024, 16),
            (4096, 64),
            (16384, 256),
            (65536, 1024),
        ];

        // BTreeMap baseline
        let bt_ns = run_btreemap(&initial, &ops);
        println!("{:>9}  {:>22}  {:>10} {:>10}  {:>12.3}  {:>12.0}",
            scan_len, "std::BTreeMap", 0, 0, bt_ns / 1_000_000.0, bt_ns / n_ops as f64);
        writeln!(f, "{},{},{},{},std::BTreeMap,0,0,{:.0},{:.0}",
            n_init, n_ops, scan_pct, scan_len, bt_ns, bt_ns / n_ops as f64).unwrap();

        for &(lc, bc) in configs {
            for &(comp, name) in &[(Compactor::DeltaSort, "LeafBTree-DS"), (Compactor::FullSort, "LeafBTree-FS")] {
                let ns = run_leafbtree(&initial, &ops, lc, bc, comp);
                let speedup = bt_ns / ns;
                println!("{:>9}  {:>22}  {:>10} {:>10}  {:>12.3}  {:>12.0}  ({:.2}x vs BT)",
                    scan_len, name, lc, bc, ns / 1_000_000.0, ns / n_ops as f64, speedup);
                writeln!(f, "{},{},{},{},{},{},{},{:.0},{:.0}",
                    n_init, n_ops, scan_pct, scan_len, name, lc, bc, ns, ns / n_ops as f64).unwrap();
            }
        }
        println!();
    }
    println!("Wrote results/track_f_mixed.csv");
}
