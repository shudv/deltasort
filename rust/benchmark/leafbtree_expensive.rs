//! Re-evaluation of LeafBTree (DeltaSort compaction) vs std::BTreeMap and LeafBTree-FullSort
//! with EXPENSIVE COMPARATORS.
//!
//! Hypothesis: With int keys, pdqsort's branchless constants dominate. With string or
//! multi-field keys, comparison cost rises 50-100x, and DeltaSort's "fewer comparisons"
//! advantage may finally translate into wall-clock wins inside leaf compaction.

use deltasort::leaf_btree::{Compactor, LeafBTree};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::time::Instant;

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
struct UrlKey(String);

fn gen_url(rng: &mut StdRng) -> UrlKey {
    let prefix = "https://example.com/api/v1/users/";
    let suffix: String = (0..(64 - prefix.len()))
        .map(|_| (b'a' + rng.gen_range(0..26)) as char)
        .collect();
    UrlKey(format!("{}{}", prefix, suffix))
}

#[derive(Clone, Default, PartialEq, Eq)]
struct MultiKey {
    a: i32,
    b: Option<i32>,
    c: i64,
    d: u8,
    e: i16,
}

impl PartialOrd for MultiKey { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }
impl Ord for MultiKey {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        match self.a.cmp(&o.a) { Equal => {}, ord => return ord }
        match (&self.b, &o.b) {
            (None, None) => {}, (None, Some(_)) => return Greater, (Some(_), None) => return Less,
            (Some(p), Some(q)) => match p.cmp(q) { Equal => {}, ord => return ord },
        }
        match self.c.cmp(&o.c) { Equal => {}, ord => return ord }
        match self.d.cmp(&o.d) { Equal => {}, ord => return ord }
        self.e.cmp(&o.e)
    }
}

fn gen_multi(rng: &mut StdRng) -> MultiKey {
    MultiKey {
        a: rng.gen_range(0..10),
        b: if rng.gen_bool(0.2) { None } else { Some(rng.gen_range(0..1000)) },
        c: rng.gen_range(0..1000000),
        d: rng.gen(),
        e: rng.gen(),
    }
}

enum Workload { BulkInsert, PointLookup, RangeScan }

impl Workload { fn name(&self) -> &'static str { match self { Workload::BulkInsert => "bulk_insert", Workload::PointLookup => "point_lookup", Workload::RangeScan => "range_scan" } } }

fn bench_btreemap_url(wl: &Workload, keys: &[UrlKey], top_m: usize) -> f64 {
    let mut m: BTreeMap<UrlKey, u64> = BTreeMap::new();
    match wl {
        Workload::BulkInsert => {
            let t = Instant::now();
            for (i, k) in keys.iter().enumerate() { m.insert(k.clone(), i as u64); }
            t.elapsed().as_nanos() as f64 / keys.len() as f64
        }
        Workload::PointLookup => {
            for (i, k) in keys.iter().enumerate() { m.insert(k.clone(), i as u64); }
            let t = Instant::now();
            let mut hit = 0u64;
            for k in keys.iter() { if let Some(v) = m.get(k) { hit = hit.wrapping_add(*v); } }
            std::hint::black_box(hit);
            t.elapsed().as_nanos() as f64 / keys.len() as f64
        }
        Workload::RangeScan => {
            for (i, k) in keys.iter().enumerate() { m.insert(k.clone(), i as u64); }
            let t = Instant::now();
            let mut sum = 0u64;
            for (_k, v) in m.iter().take(top_m) { sum = sum.wrapping_add(*v); }
            std::hint::black_box(sum);
            t.elapsed().as_nanos() as f64 / top_m as f64
        }
    }
}

fn bench_leafbtree_url(wl: &Workload, keys: &[UrlKey], top_m: usize, leaf_cap: usize, buf_cap: usize, comp: Compactor) -> f64 {
    let mut t = LeafBTree::<UrlKey, u64>::with_config_and_compactor(leaf_cap, buf_cap, 16, comp);
    match wl {
        Workload::BulkInsert => {
            let start = Instant::now();
            for (i, k) in keys.iter().enumerate() { t.insert(k.clone(), i as u64); }
            t.compact_all();
            start.elapsed().as_nanos() as f64 / keys.len() as f64
        }
        Workload::PointLookup => {
            for (i, k) in keys.iter().enumerate() { t.insert(k.clone(), i as u64); }
            t.compact_all();
            let start = Instant::now();
            let mut hit = 0u64;
            for k in keys.iter() { if let Some(v) = t.get(k) { hit = hit.wrapping_add(*v); } }
            std::hint::black_box(hit);
            start.elapsed().as_nanos() as f64 / keys.len() as f64
        }
        Workload::RangeScan => {
            for (i, k) in keys.iter().enumerate() { t.insert(k.clone(), i as u64); }
            t.compact_all();
            let start = Instant::now();
            let mut sum = 0u64;
            for (_k, v) in t.iter().take(top_m) { sum = sum.wrapping_add(*v); }
            std::hint::black_box(sum);
            start.elapsed().as_nanos() as f64 / top_m as f64
        }
    }
}

fn bench_btreemap_multi(wl: &Workload, keys: &[MultiKey], top_m: usize) -> f64 {
    let mut m: BTreeMap<MultiKey, u64> = BTreeMap::new();
    match wl {
        Workload::BulkInsert => {
            let t = Instant::now();
            for (i, k) in keys.iter().enumerate() { m.insert(k.clone(), i as u64); }
            t.elapsed().as_nanos() as f64 / keys.len() as f64
        }
        Workload::PointLookup => {
            for (i, k) in keys.iter().enumerate() { m.insert(k.clone(), i as u64); }
            let t = Instant::now();
            let mut hit = 0u64;
            for k in keys.iter() { if let Some(v) = m.get(k) { hit = hit.wrapping_add(*v); } }
            std::hint::black_box(hit);
            t.elapsed().as_nanos() as f64 / keys.len() as f64
        }
        Workload::RangeScan => {
            for (i, k) in keys.iter().enumerate() { m.insert(k.clone(), i as u64); }
            let t = Instant::now();
            let mut sum = 0u64;
            for (_k, v) in m.iter().take(top_m) { sum = sum.wrapping_add(*v); }
            std::hint::black_box(sum);
            t.elapsed().as_nanos() as f64 / top_m as f64
        }
    }
}

fn bench_leafbtree_multi(wl: &Workload, keys: &[MultiKey], top_m: usize, leaf_cap: usize, buf_cap: usize, comp: Compactor) -> f64 {
    let mut t = LeafBTree::<MultiKey, u64>::with_config_and_compactor(leaf_cap, buf_cap, 16, comp);
    match wl {
        Workload::BulkInsert => {
            let start = Instant::now();
            for (i, k) in keys.iter().enumerate() { t.insert(k.clone(), i as u64); }
            t.compact_all();
            start.elapsed().as_nanos() as f64 / keys.len() as f64
        }
        Workload::PointLookup => {
            for (i, k) in keys.iter().enumerate() { t.insert(k.clone(), i as u64); }
            t.compact_all();
            let start = Instant::now();
            let mut hit = 0u64;
            for k in keys.iter() { if let Some(v) = t.get(k) { hit = hit.wrapping_add(*v); } }
            std::hint::black_box(hit);
            start.elapsed().as_nanos() as f64 / keys.len() as f64
        }
        Workload::RangeScan => {
            for (i, k) in keys.iter().enumerate() { t.insert(k.clone(), i as u64); }
            t.compact_all();
            let start = Instant::now();
            let mut sum = 0u64;
            for (_k, v) in t.iter().take(top_m) { sum = sum.wrapping_add(*v); }
            std::hint::black_box(sum);
            start.elapsed().as_nanos() as f64 / top_m as f64
        }
    }
}

fn main() {
    let reps = 3usize;
    fs::create_dir_all("results").ok();
    let mut f = fs::File::create("results/track_c_leafbtree_expensive.csv").expect("create csv");
    writeln!(f, "keytype,workload,n,structure,leaf_cap,buf_cap,rep,ns_per_op").unwrap();

    let n_values = [100_000usize, 1_000_000];
    let leaf_configs: &[(usize, usize)] = &[(64, 8), (256, 16), (1024, 32), (4096, 64), (16384, 128)];
    let workloads = [Workload::BulkInsert, Workload::PointLookup, Workload::RangeScan];

    let median = |mut v: Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };

    println!("\n=== UrlKey (64-byte URL strings, 32-byte common prefix) ===\n");
    println!("{:>14} {:>10} {:>20} {:>10} {:>10} {:>14}", "workload", "n", "structure", "leaf_cap", "buf_cap", "ns/op (med)");
    for &n in &n_values {
        for wl in workloads.iter() {
            let mut samples = Vec::with_capacity(reps);
            for r in 0..reps {
                let mut rng = StdRng::seed_from_u64(0xC0FFEE_u64.wrapping_add(r as u64 * 7919));
                let keys: Vec<UrlKey> = (0..n).map(|_| gen_url(&mut rng)).collect();
                let ns = bench_btreemap_url(wl, &keys, 100);
                samples.push(ns);
                writeln!(f, "url,{},{},std::BTreeMap,,,{},{:.1}", wl.name(), n, r, ns).unwrap();
            }
            println!("{:>14} {:>10} {:>20} {:>10} {:>10} {:>14.1}", wl.name(), n, "std::BTreeMap", "-", "-", median(samples));

            for &(lc, bc) in leaf_configs {
                for (cname, comp) in [("LeafBTree-DS", Compactor::DeltaSort), ("LeafBTree-FS", Compactor::FullSort)] {
                    let mut samples = Vec::with_capacity(reps);
                    for r in 0..reps {
                        let mut rng = StdRng::seed_from_u64(0xC0FFEE_u64.wrapping_add(r as u64 * 7919));
                        let keys: Vec<UrlKey> = (0..n).map(|_| gen_url(&mut rng)).collect();
                        let ns = bench_leafbtree_url(wl, &keys, 100, lc, bc, comp);
                        samples.push(ns);
                        writeln!(f, "url,{},{},{},{},{},{},{:.1}", wl.name(), n, cname, lc, bc, r, ns).unwrap();
                    }
                    println!("{:>14} {:>10} {:>20} {:>10} {:>10} {:>14.1}", wl.name(), n, cname, lc, bc, median(samples));
                }
            }
            f.flush().ok();
        }
    }

    println!("\n=== MultiKey (5-field SQL ORDER BY with NULLs) ===\n");
    println!("{:>14} {:>10} {:>20} {:>10} {:>10} {:>14}", "workload", "n", "structure", "leaf_cap", "buf_cap", "ns/op (med)");
    for &n in &n_values {
        for wl in workloads.iter() {
            let mut samples = Vec::with_capacity(reps);
            for r in 0..reps {
                let mut rng = StdRng::seed_from_u64(0xC0FFEE_u64.wrapping_add(r as u64 * 7919));
                let keys: Vec<MultiKey> = (0..n).map(|_| gen_multi(&mut rng)).collect();
                let ns = bench_btreemap_multi(wl, &keys, 100);
                samples.push(ns);
                writeln!(f, "multi,{},{},std::BTreeMap,,,{},{:.1}", wl.name(), n, r, ns).unwrap();
            }
            println!("{:>14} {:>10} {:>20} {:>10} {:>10} {:>14.1}", wl.name(), n, "std::BTreeMap", "-", "-", median(samples));

            for &(lc, bc) in leaf_configs {
                for (cname, comp) in [("LeafBTree-DS", Compactor::DeltaSort), ("LeafBTree-FS", Compactor::FullSort)] {
                    let mut samples = Vec::with_capacity(reps);
                    for r in 0..reps {
                        let mut rng = StdRng::seed_from_u64(0xC0FFEE_u64.wrapping_add(r as u64 * 7919));
                        let keys: Vec<MultiKey> = (0..n).map(|_| gen_multi(&mut rng)).collect();
                        let ns = bench_leafbtree_multi(wl, &keys, 100, lc, bc, comp);
                        samples.push(ns);
                        writeln!(f, "multi,{},{},{},{},{},{},{:.1}", wl.name(), n, cname, lc, bc, r, ns).unwrap();
                    }
                    println!("{:>14} {:>10} {:>20} {:>10} {:>10} {:>14.1}", wl.name(), n, cname, lc, bc, median(samples));
                }
            }
            f.flush().ok();
        }
    }

    println!("\nWrote results/track_c_leafbtree_expensive.csv");
}
