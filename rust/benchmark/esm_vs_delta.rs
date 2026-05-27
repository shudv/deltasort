//! THE critical experiment: DeltaSort vs ESM vs std::BTreeMap with EXPENSIVE COMPARATORS.
//!
//! Both DeltaSort and ESM exploit the known dirty-index hint.
//!   - DeltaSort: O(n + k log n) comparisons, in-place sweep
//!   - ESM:       O(n + k log k) comparisons, requires O(k) buffer
//!
//! Theoretical: ESM has FEWER comparisons (log k < log n for k < n), so for
//! expensive comparators ESM should be the strongest baseline.
//!
//! Compared against std::BTreeMap incremental maintenance (the production baseline).

use deltasort::delta_sort_by;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::time::Instant;

// Generic ESM: backwards-merge variant for O(k) extra space
fn esm_sort_by<T: Default + Clone, F: Fn(&T, &T) -> Ordering>(
    arr: &mut [T],
    dirty_indices: &mut [usize],
    cmp: F,
) {
    if dirty_indices.is_empty() { return; }
    let n = arr.len();
    let k = dirty_indices.len();

    dirty_indices.sort_unstable();

    // Phase 1: extract dirty into buffer, compact clean to left.
    let mut dirty_buf: Vec<T> = Vec::with_capacity(k);
    let mut write = 0;
    let mut di = 0;
    for read in 0..n {
        if di < k && dirty_indices[di] == read {
            dirty_buf.push(std::mem::take(&mut arr[read]));
            di += 1;
        } else {
            if write != read { arr.swap(write, read); }
            write += 1;
        }
    }

    // Phase 2: sort dirty buffer
    dirty_buf.sort_by(&cmp);

    // Phase 3: backwards merge into arr
    let mut ci = (n - k) as isize - 1;
    let mut di = k as isize - 1;
    let mut wi = n - 1;
    loop {
        let take_clean = if ci >= 0 && di >= 0 {
            cmp(&arr[ci as usize], &dirty_buf[di as usize]) != Ordering::Less
        } else {
            ci >= 0
        };
        if take_clean {
            if wi != ci as usize { arr.swap(wi, ci as usize); }
            ci -= 1;
        } else if di >= 0 {
            arr[wi] = std::mem::take(&mut dirty_buf[di as usize]);
            di -= 1;
        } else { break; }
        if wi == 0 { break; }
        wi -= 1;
    }
}

// ---------- 3 keytype generators ----------

fn gen_url(rng: &mut StdRng) -> String {
    let prefix = "https://example.com/api/v1/users/";
    let suffix: String = (0..(64 - prefix.len()))
        .map(|_| (b'a' + rng.gen_range(0..26)) as char)
        .collect();
    format!("{}{}", prefix, suffix)
}

#[derive(Clone, Default, PartialEq, Eq)]
struct MultiKey { a: i32, b: Option<i32>, c: i64, d: u8, e: i16 }
impl PartialOrd for MultiKey { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for MultiKey {
    fn cmp(&self, o: &Self) -> Ordering {
        match self.a.cmp(&o.a) { Ordering::Equal => {}, ord => return ord }
        match (&self.b, &o.b) {
            (None, None) => {}, (None, Some(_)) => return Ordering::Greater, (Some(_), None) => return Ordering::Less,
            (Some(p), Some(q)) => match p.cmp(q) { Ordering::Equal => {}, ord => return ord },
        }
        match self.c.cmp(&o.c) { Ordering::Equal => {}, ord => return ord }
        match self.d.cmp(&o.d) { Ordering::Equal => {}, ord => return ord }
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

fn expensive_hash(x: u64) -> u64 {
    let mut h = x;
    for _ in 0..100 {
        h = h.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0xBF58476D1CE4E5B9);
        h ^= h >> 30;
    }
    h
}

#[derive(Clone, Default, PartialEq, Eq)]
struct ExpKey { raw: u64, cached: u64 }
impl PartialOrd for ExpKey { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for ExpKey { fn cmp(&self, o: &Self) -> Ordering { self.cached.cmp(&o.cached) } }

fn gen_exp(rng: &mut StdRng) -> ExpKey {
    let raw = rng.gen();
    ExpKey { raw, cached: expensive_hash(raw) }
}

// ---------- Three head-to-heads per workload ----------

struct EpochOut { delta_ns: f64, esm_ns: f64, btreemap_ns: f64 }

fn run_epoch<T, F, G>(
    initial: &[T],
    dirty_keys: &[(usize, T)],
    cmp: F,
    cmp_box: G,
) -> EpochOut
where
    T: Clone + Default + Ord,
    F: Fn(&T, &T) -> Ordering + Copy,
    G: Fn(&T, &T) -> Ordering + Copy,
{
    let n = initial.len();
    let _k = dirty_keys.len();

    // --- DeltaSort ---
    let mut data_ds = initial.to_vec();
    let mut dirty_ds: Vec<usize> = dirty_keys.iter().map(|(i, _)| *i).collect();
    for (i, k) in dirty_keys { data_ds[*i] = k.clone(); }
    dirty_ds.sort_unstable();
    dirty_ds.dedup();
    let t = Instant::now();
    delta_sort_by(&mut data_ds, &mut dirty_ds, cmp);
    let delta_ns = t.elapsed().as_nanos() as f64;

    // --- ESM ---
    let mut data_esm = initial.to_vec();
    let mut dirty_esm: Vec<usize> = dirty_keys.iter().map(|(i, _)| *i).collect();
    for (i, k) in dirty_keys { data_esm[*i] = k.clone(); }
    dirty_esm.sort_unstable();
    dirty_esm.dedup();
    let t = Instant::now();
    esm_sort_by(&mut data_esm, &mut dirty_esm, cmp_box);
    let esm_ns = t.elapsed().as_nanos() as f64;

    // Sanity (compare without Debug requirement)
    assert!(data_ds == data_esm, "DS != ESM");

    // --- BTreeMap ---
    let mut idx_keys: Vec<T> = initial.to_vec();
    let mut map: BTreeMap<T, u64> = initial.iter().enumerate().map(|(i, k)| (k.clone(), i as u64)).collect();
    let t = Instant::now();
    for (i, new_k) in dirty_keys {
        let old = idx_keys[*i].clone();
        map.remove(&old);
        idx_keys[*i] = new_k.clone();
        map.insert(new_k.clone(), *i as u64);
    }
    let _ = std::hint::black_box(&map);
    let btreemap_ns = t.elapsed().as_nanos() as f64;

    let _ = n;
    EpochOut { delta_ns, esm_ns, btreemap_ns }
}

fn run_workload<T, FG, F>(
    keytype: &str,
    n: usize,
    k: usize,
    reps: usize,
    gen: FG,
    cmp: F,
    csv: &mut impl Write,
)
where
    T: Clone + Default + Ord,
    FG: Fn(&mut StdRng) -> T,
    F: Fn(&T, &T) -> Ordering + Copy,
{
    let mut ds = Vec::new();
    let mut esm = Vec::new();
    let mut bt = Vec::new();
    for r in 0..reps {
        let mut rng = StdRng::seed_from_u64(0xC0FFEEu64.wrapping_add(r as u64 * 7919));
        let mut initial: Vec<T> = (0..n).map(|_| gen(&mut rng)).collect();
        initial.sort();
        initial.dedup();
        while initial.len() < n { initial.push(gen(&mut rng)); }
        initial.sort();
        initial.truncate(n);

        let mut used = std::collections::HashSet::new();
        let mut dirty_keys: Vec<(usize, T)> = Vec::with_capacity(k);
        while dirty_keys.len() < k {
            let idx = rng.gen_range(0..n);
            if used.insert(idx) { dirty_keys.push((idx, gen(&mut rng))); }
        }

        let out = run_epoch(&initial, &dirty_keys, cmp, cmp);
        ds.push(out.delta_ns);
        esm.push(out.esm_ns);
        bt.push(out.btreemap_ns);
    }
    let med = |mut v: Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len()/2] };
    let mds = med(ds);
    let mesm = med(esm);
    let mbt = med(bt);
    let best_flat = mds.min(mesm);
    let speedup_bt_over_best = mbt / best_flat;
    println!("{:>14} {:>10} {:>10} {:>16.0} {:>16.0} {:>16.0} {:>10.2} {:>10.2}",
        keytype, n, k, mds, mesm, mbt,
        mesm/mds, // ESM speedup over DS (>1 means ESM wins)
        speedup_bt_over_best, // BTreeMap speedup over best of DS,ESM (>1 means BT wins)
    );
    writeln!(csv, "{},{},{},DeltaSort,{:.0}", keytype, n, k, mds).unwrap();
    writeln!(csv, "{},{},{},ESM,{:.0}", keytype, n, k, mesm).unwrap();
    writeln!(csv, "{},{},{},BTreeMap,{:.0}", keytype, n, k, mbt).unwrap();
}

fn main() {
    let reps = 5usize;
    fs::create_dir_all("results").ok();
    let mut f = fs::File::create("results/track_d_esm_vs_delta.csv").expect("create csv");
    writeln!(f, "keytype,n,k,strategy,ns").unwrap();

    println!("{:>14} {:>10} {:>10} {:>16} {:>16} {:>16} {:>10} {:>10}",
        "keytype", "n", "k", "DeltaSort ns", "ESM ns", "BTreeMap ns", "ESM/DS", "BT/best");

    println!("\n--- URL strings ---");
    for &n in &[10_000usize, 100_000, 1_000_000] {
        for &k_pct in &[0.001f64, 0.01, 0.1, 1.0] {
            let k = ((n as f64) * k_pct / 100.0).ceil().max(1.0) as usize;
            run_workload::<String, _, _>("url", n, k, reps, gen_url, |a, b| a.cmp(b), &mut f);
        }
    }
    println!("\n--- MultiField ---");
    for &n in &[10_000usize, 100_000, 1_000_000] {
        for &k_pct in &[0.001f64, 0.01, 0.1, 1.0] {
            let k = ((n as f64) * k_pct / 100.0).ceil().max(1.0) as usize;
            run_workload::<MultiKey, _, _>("multi", n, k, reps, gen_multi, |a, b| a.cmp(b), &mut f);
        }
    }
    println!("\n--- ExpensiveKey ---");
    for &n in &[10_000usize, 100_000, 1_000_000] {
        for &k_pct in &[0.001f64, 0.01, 0.1, 1.0] {
            let k = ((n as f64) * k_pct / 100.0).ceil().max(1.0) as usize;
            run_workload::<ExpKey, _, _>("exp", n, k, reps, gen_exp, |a, b| a.cmp(b), &mut f);
        }
    }

    println!("\nWrote results/track_d_esm_vs_delta.csv");
}
