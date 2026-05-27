//! Realistic workload trace: streaming updates with periodic queries.
//!
//! Models a "maintained sorted index" scenario:
//!   - Initial population: n entries, sorted
//!   - Stream: arrives in epochs. Each epoch = k UPDATEs + Q range scans (top-M)
//!   - Compare 3 strategies:
//!     A. std::BTreeMap incremental update + iter() top-M (production baseline)
//!     B. Flat Vec + DeltaSort batched re-sort + binary scan top-M (ours)
//!     C. Flat Vec + pdqsort batched re-sort + binary scan top-M (naive baseline)
//!
//! 3 key types with realistic-ish distributions:
//!   - URL-LIKE STRINGS (64 bytes, common prefix "https://example.com/")
//!   - COMPOUND ORDER BY (5 cols with nulls, simulating SQL ORDER BY)
//!   - EXPENSIVE COMPUTED KEY (100-round hash on raw u64)
//!
//! Per-epoch cost measured, then mean and p95.

use deltasort::delta_sort_by;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::time::Instant;

// ---------- Trait abstraction ----------
trait Key: Ord + Clone + Default + std::fmt::Debug {
    fn gen(rng: &mut StdRng) -> Self;
    fn label() -> &'static str;
}

// URL-like strings: 64 bytes, 32-byte common prefix
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct UrlKey(String);

impl Key for UrlKey {
    fn gen(rng: &mut StdRng) -> Self {
        let prefix = "https://example.com/api/v1/users/";
        let suffix: String = (0..(64 - prefix.len()))
            .map(|_| (b'a' + rng.gen_range(0..26)) as char)
            .collect();
        UrlKey(format!("{}{}", prefix, suffix))
    }
    fn label() -> &'static str { "url_strings" }
}

// Multi-field key with NULLs
#[derive(Clone, Default, PartialEq, Eq, Debug)]
struct MultiKey {
    a: i32,
    b: Option<i32>,
    c: i64,
    d: u8,
    e: i16,
}

impl PartialOrd for MultiKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MultiKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        match self.a.cmp(&other.a) {
            Equal => {}
            ord => return ord,
        }
        match (&self.b, &other.b) {
            (None, None) => {}
            (None, Some(_)) => return Greater,
            (Some(_), None) => return Less,
            (Some(p), Some(q)) => match p.cmp(q) {
                Equal => {}
                ord => return ord,
            },
        }
        match self.c.cmp(&other.c) {
            Equal => {}
            ord => return ord,
        }
        match self.d.cmp(&other.d) {
            Equal => {}
            ord => return ord,
        }
        self.e.cmp(&other.e)
    }
}

impl Key for MultiKey {
    fn gen(rng: &mut StdRng) -> Self {
        MultiKey {
            a: rng.gen_range(0..10),
            b: if rng.gen_bool(0.2) { None } else { Some(rng.gen_range(0..1000)) },
            c: rng.gen_range(0..1000000),
            d: rng.gen(),
            e: rng.gen(),
        }
    }
    fn label() -> &'static str { "multifield" }
}

// Expensive-key: stores raw u64 + computed key (100-round hash)
fn expensive_hash(x: u64) -> u64 {
    let mut h = x;
    for _ in 0..100 {
        h = h.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0xBF58476D1CE4E5B9);
        h ^= h >> 30;
    }
    h
}

#[derive(Clone, Default, PartialEq, Eq, Debug)]
struct ExpKey {
    raw: u64,
    cached: u64,
}

impl PartialOrd for ExpKey {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) }
}

impl Ord for ExpKey {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering { self.cached.cmp(&o.cached) }
}

impl Key for ExpKey {
    fn gen(rng: &mut StdRng) -> Self {
        let raw = rng.gen();
        ExpKey { raw, cached: expensive_hash(raw) }
    }
    fn label() -> &'static str { "expensive_key" }
}

// ---------- Strategies ----------
struct EpochResult {
    update_ns: f64,
    query_ns: f64,
}

fn run_btreemap<K: Key>(initial: &[K], dirty_keys: &[(usize, K)], top_m: usize) -> EpochResult {
    let mut map: BTreeMap<K, u64> = initial.iter().enumerate().map(|(i, k)| (k.clone(), i as u64)).collect();
    // Need to "update" — find old keys at index positions and replace.
    // Approach: maintain a parallel Vec<K> for "current key at index i". Build it.
    let mut idx_keys: Vec<K> = initial.to_vec();

    let t = Instant::now();
    for (i, new_k) in dirty_keys {
        let old = idx_keys[*i].clone();
        map.remove(&old);
        idx_keys[*i] = new_k.clone();
        map.insert(new_k.clone(), *i as u64);
    }
    let update_ns = t.elapsed().as_nanos() as f64;

    let t = Instant::now();
    let mut sum = 0u64;
    let mut count = 0;
    for (_k, v) in map.iter() {
        sum = sum.wrapping_add(*v);
        count += 1;
        if count >= top_m { break; }
    }
    std::hint::black_box(sum);
    let query_ns = t.elapsed().as_nanos() as f64;

    EpochResult { update_ns, query_ns }
}

fn run_deltasort<K: Key>(initial: &[K], dirty_keys: &[(usize, K)], top_m: usize) -> EpochResult {
    let mut data: Vec<K> = initial.to_vec();
    // We have to maintain a mapping from "logical index" to "current array position".
    // Simplest model for this benchmark: dirty indices refer to POSITIONS in the sorted array
    // (i.e., physical indices). That's the same setup the original Track 1 sweep uses.

    let t = Instant::now();
    let mut dirty: Vec<usize> = dirty_keys.iter().map(|(i, _)| *i).collect();
    for (i, new_k) in dirty_keys {
        data[*i] = new_k.clone();
    }
    dirty.sort_unstable();
    dirty.dedup();
    delta_sort_by(&mut data, &mut dirty, |a, b| a.cmp(b));
    let update_ns = t.elapsed().as_nanos() as f64;

    let t = Instant::now();
    let mut sum = 0u64;
    for k in data.iter().take(top_m) {
        // Use a hash of the key as the "value" for the sum
        sum = sum.wrapping_add(simple_hash(k));
    }
    std::hint::black_box(sum);
    let query_ns = t.elapsed().as_nanos() as f64;

    EpochResult { update_ns, query_ns }
}

fn run_pdqsort<K: Key>(initial: &[K], dirty_keys: &[(usize, K)], top_m: usize) -> EpochResult {
    let mut data: Vec<K> = initial.to_vec();

    let t = Instant::now();
    for (i, new_k) in dirty_keys {
        data[*i] = new_k.clone();
    }
    data.sort_unstable();
    let update_ns = t.elapsed().as_nanos() as f64;

    let t = Instant::now();
    let mut sum = 0u64;
    for k in data.iter().take(top_m) {
        sum = sum.wrapping_add(simple_hash(k));
    }
    std::hint::black_box(sum);
    let query_ns = t.elapsed().as_nanos() as f64;

    EpochResult { update_ns, query_ns }
}

fn simple_hash<K>(_k: &K) -> u64 { 1 }

// ---------- Runner ----------
fn run_workload<K: Key>(n: usize, k: usize, top_m: usize, reps: usize) -> Vec<(String, f64, f64)> {
    let mut out: Vec<(String, f64, f64)> = Vec::new();
    let mut bt_u = Vec::new();
    let mut bt_q = Vec::new();
    let mut ds_u = Vec::new();
    let mut ds_q = Vec::new();
    let mut pq_u = Vec::new();
    let mut pq_q = Vec::new();

    for r in 0..reps {
        let mut rng = StdRng::seed_from_u64(0xC0FFEE_u64.wrapping_add(r as u64 * 7919));
        let mut initial: Vec<K> = (0..n).map(|_| K::gen(&mut rng)).collect();
        initial.sort();
        initial.dedup();
        if initial.len() < n {
            // Top up to n
            while initial.len() < n {
                initial.push(K::gen(&mut rng));
            }
            initial.sort();
            // Truncate to n
            initial.truncate(n);
        }

        // Build dirty key set
        let mut used = std::collections::HashSet::new();
        let mut dirty_keys: Vec<(usize, K)> = Vec::with_capacity(k);
        while dirty_keys.len() < k {
            let idx = rng.gen_range(0..n);
            if used.insert(idx) {
                dirty_keys.push((idx, K::gen(&mut rng)));
            }
        }

        let r1 = run_btreemap(&initial, &dirty_keys, top_m);
        bt_u.push(r1.update_ns);
        bt_q.push(r1.query_ns);

        let r2 = run_deltasort(&initial, &dirty_keys, top_m);
        ds_u.push(r2.update_ns);
        ds_q.push(r2.query_ns);

        let r3 = run_pdqsort(&initial, &dirty_keys, top_m);
        pq_u.push(r3.update_ns);
        pq_q.push(r3.query_ns);
    }

    let med = |mut v: Vec<f64>| { v.sort_by(|a,b| a.partial_cmp(b).unwrap()); v[v.len()/2] };

    out.push(("BTreeMap".into(), med(bt_u), med(bt_q)));
    out.push(("DeltaSort".into(), med(ds_u), med(ds_q)));
    out.push(("pdqsort".into(), med(pq_u), med(pq_q)));
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps: usize = args.iter().position(|a| a == "--reps").and_then(|i| args.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(5);

    fs::create_dir_all("results").ok();
    let mut f = fs::File::create("results/track_b_realistic.csv").expect("create csv");
    writeln!(f, "keytype,n,k,k_pct,top_m,strategy,update_ns,query_ns,total_ns").unwrap();

    println!("\n{:>14} {:>10} {:>10} {:>10} {:>16} {:>14} {:>14} {:>14}",
        "keytype", "n", "k", "top_m", "strategy", "update ns", "query ns", "total ns");

    macro_rules! sweep {
        ($K:ty) => {{
            for &n in &[10_000usize, 100_000, 1_000_000] {
                for &k_pct in &[0.001f64, 0.01, 0.1, 1.0] {
                    let k = ((n as f64) * k_pct / 100.0).ceil().max(1.0) as usize;
                    let top_m = 100;
                    let rs = run_workload::<$K>(n, k, top_m, reps);
                    for (strat, u, q) in &rs {
                        let total = u + q;
                        println!("{:>14} {:>10} {:>10} {:>10} {:>16} {:>14.0} {:>14.0} {:>14.0}",
                            <$K>::label(), n, k, top_m, strat, u, q, total);
                        writeln!(f, "{},{},{},{},{},{},{:.0},{:.0},{:.0}",
                            <$K>::label(), n, k, k_pct, top_m, strat, u, q, total).unwrap();
                    }
                }
            }
        }};
    }

    println!("\n--- URL strings ---");
    sweep!(UrlKey);

    println!("\n--- Multi-field ---");
    sweep!(MultiKey);

    println!("\n--- Expensive key ---");
    sweep!(ExpKey);

    println!("\nWrote results/track_b_realistic.csv");
}
