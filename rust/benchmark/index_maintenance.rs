//! Database secondary index maintenance under batch UPDATE.
//!
//! Realistic scenario: a relational table has a secondary index on column `c`.
//! The index is maintained as a flat sorted array of (c_val, row_id) pairs.
//! An UPDATE statement modifies k rows. For each modified row, the old
//! index entry must be removed and a new one inserted.
//!
//! Production approaches today:
//!   - PostgreSQL: HOT (Heap-Only Tuple) avoids index update when index columns
//!     unchanged; otherwise delete+insert into B-tree: O(k log n) per tx.
//!   - DuckDB / analytical engines: sort-based index rebuild: O(n log n).
//!   - Some OLAP systems: deferred batch maintenance (merge at query time).
//!
//! The flat-array / sorted-run model applies to:
//!   - LSM-tree level-0 run maintenance after memtable flush
//!   - DuckDB ART (adaptive radix trie) with sorted key vector
//!   - Analytical sorted materialized views (ClickHouse, CrateDB)
//!   - In-memory analytics (DuckDB, Polars sort)
//!
//! Here we model it as:
//!   - One sorted array of (sort_key, row_id) pairs (the index)
//!   - k entries are replaced per "transaction" (dirty indices known)
//!   - Two comparator modes: i64 (cheap, numeric column) and String (expensive,
//!     varchar column)
//!
//! Key claim: for analytical workloads where the index is a sorted flat array
//! (e.g., LSM run, sorted materialized view), DS beats ESM for k/n < ~2%.

use deltasort::delta_sort_by;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::fs;
use std::io::Write;
use std::time::Instant;

// ============================================================
//  Numeric index entry (cheap comparator)
// ============================================================

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
struct NumEntry {
    val: i64,
    row_id: u64,
}

fn cmp_num(a: &NumEntry, b: &NumEntry) -> Ordering { a.cmp(b) }

// ============================================================
//  String index entry (expensive comparator, varchar column)
// ============================================================

#[derive(Clone, Default, PartialEq, Eq)]
struct StrEntry {
    val: String,
    row_id: u64,
}
impl Ord for StrEntry {
    fn cmp(&self, o: &Self) -> Ordering {
        self.val.cmp(&o.val).then(self.row_id.cmp(&o.row_id))
    }
}
impl PartialOrd for StrEntry { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
fn cmp_str(a: &StrEntry, b: &StrEntry) -> Ordering { a.cmp(b) }

// ============================================================
//  Generic ESM
// ============================================================

fn esm_sort<T: Default + Clone, F: Fn(&T, &T) -> Ordering>(
    arr: &mut [T], dirty: &mut [usize], cmp: F,
) {
    if dirty.is_empty() { return; }
    let n = arr.len();
    let k = dirty.len();
    dirty.sort_unstable();
    let mut buf: Vec<T> = Vec::with_capacity(k);
    let mut write = 0; let mut di = 0;
    for read in 0..n {
        if di < k && dirty[di] == read {
            buf.push(std::mem::take(&mut arr[read])); di += 1;
        } else {
            if write != read { arr.swap(write, read); }
            write += 1;
        }
    }
    buf.sort_by(&cmp);
    let mut ci = (n - k) as isize - 1;
    let mut di2 = k as isize - 1;
    let mut wi = n - 1;
    loop {
        let take_clean = if ci >= 0 && di2 >= 0 {
            cmp(&arr[ci as usize], &buf[di2 as usize]) != Ordering::Less
        } else { ci >= 0 };
        if take_clean { if wi != ci as usize { arr.swap(wi, ci as usize); } ci -= 1; }
        else if di2 >= 0 { arr[wi] = std::mem::take(&mut buf[di2 as usize]); di2 -= 1; }
        else { break; }
        if wi == 0 { break; }
        wi -= 1;
    }
}

// ============================================================
//  Numeric workload
// ============================================================

fn gen_num_index(n: usize, seed: u64) -> Vec<NumEntry> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v: Vec<NumEntry> = (0..n).map(|i| NumEntry {
        val: rng.gen_range(0..1_000_000_000_i64),
        row_id: i as u64,
    }).collect();
    v.sort();
    v
}

fn perturb_num(index: &mut [NumEntry], k: usize, rng: &mut StdRng) -> Vec<usize> {
    let n = index.len();
    let mut picks: Vec<usize> = (0..n).collect();
    for i in 0..k { let j = rng.gen_range(i..n); picks.swap(i, j); }
    let mut dirty: Vec<usize> = picks[..k].to_vec();
    dirty.sort_unstable();
    for &i in &dirty {
        index[i].val = rng.gen_range(0..1_000_000_000_i64);
    }
    dirty
}

fn run_num(baseline: &[NumEntry], dirty: &[usize]) -> (f64, f64, f64) {
    let mut data = baseline.to_vec();
    let mut dv = dirty.to_vec(); dv.sort_unstable(); dv.dedup();
    let t = Instant::now();
    delta_sort_by(&mut data, &mut dv, cmp_num);
    let ds = t.elapsed().as_nanos() as f64;
    let ds_result = data;

    let mut data = baseline.to_vec();
    let mut dv = dirty.to_vec();
    let t = Instant::now();
    esm_sort(&mut data, &mut dv, cmp_num);
    let esm = t.elapsed().as_nanos() as f64;
    assert!(data == ds_result, "num ESM mismatch");

    let mut data = baseline.to_vec();
    let t = Instant::now();
    data.sort_unstable_by(cmp_num);
    let full = t.elapsed().as_nanos() as f64;
    assert!(data == ds_result, "num full mismatch");
    (ds, esm, full)
}

// ============================================================
//  String workload
// ============================================================

fn rand_str(rng: &mut StdRng, len: usize) -> String {
    (0..len).map(|_| (b'a' + rng.gen_range(0..26u8)) as char).collect()
}

fn gen_str_index(n: usize, str_len: usize, seed: u64) -> Vec<StrEntry> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v: Vec<StrEntry> = (0..n).map(|i| StrEntry {
        val: rand_str(&mut rng, str_len),
        row_id: i as u64,
    }).collect();
    v.sort();
    v
}

fn perturb_str(index: &mut [StrEntry], k: usize, str_len: usize, rng: &mut StdRng) -> Vec<usize> {
    let n = index.len();
    let mut picks: Vec<usize> = (0..n).collect();
    for i in 0..k { let j = rng.gen_range(i..n); picks.swap(i, j); }
    let mut dirty: Vec<usize> = picks[..k].to_vec();
    dirty.sort_unstable();
    for &i in &dirty {
        index[i].val = rand_str(rng, str_len);
    }
    dirty
}

fn run_str(baseline: &[StrEntry], dirty: &[usize]) -> (f64, f64, f64) {
    let mut data = baseline.to_vec();
    let mut dv = dirty.to_vec(); dv.sort_unstable(); dv.dedup();
    let t = Instant::now();
    delta_sort_by(&mut data, &mut dv, cmp_str);
    let ds = t.elapsed().as_nanos() as f64;
    let ds_result = data;

    let mut data = baseline.to_vec();
    let mut dv = dirty.to_vec();
    let t = Instant::now();
    esm_sort(&mut data, &mut dv, cmp_str);
    let esm = t.elapsed().as_nanos() as f64;
    assert!(data == ds_result, "str ESM mismatch");

    let mut data = baseline.to_vec();
    let t = Instant::now();
    data.sort_unstable_by(cmp_str);
    let full = t.elapsed().as_nanos() as f64;
    assert!(data == ds_result, "str full mismatch");
    (ds, esm, full)
}

// ============================================================
//  main
// ============================================================

fn med(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }

fn run_num_wl(n: usize, k_frac: f64, n_tx: usize, csv: &mut impl Write) {
    let mut index = gen_num_index(n, 0x1DE0u64.wrapping_add(n as u64));
    let mut rng = StdRng::seed_from_u64(0xDB01);
    let k = ((n as f64) * k_frac).ceil().max(1.0) as usize;
    let mut ds_v = Vec::new(); let mut esm_v = Vec::new(); let mut full_v = Vec::new();
    for _ in 0..n_tx {
        let dirty = perturb_num(&mut index, k, &mut rng);
        let (ds, esm, full) = run_num(&index, &dirty);
        ds_v.push(ds); esm_v.push(esm); full_v.push(full);
        index.sort();
    }
    let dsm = med(ds_v); let esmm = med(esm_v); let fm = med(full_v);
    let kpct = 100.0 * k as f64 / n as f64;
    println!("NUM {:>10} {:>7} {:>6.3}%  {:>9.1} {:>9.1} {:>9.1}   {:>6.2}× {:>6.2}×",
        n, k, kpct, dsm/1e3, esmm/1e3, fm/1e3, esmm/dsm, fm/dsm);
    writeln!(csv, "num,{},{},DeltaSort,{:.0}", n, k, dsm).unwrap();
    writeln!(csv, "num,{},{},ESM,{:.0}", n, k, esmm).unwrap();
    writeln!(csv, "num,{},{},Full,{:.0}", n, k, fm).unwrap();
}

fn run_str_wl(n: usize, k_frac: f64, str_len: usize, n_tx: usize, csv: &mut impl Write) {
    let mut index = gen_str_index(n, str_len, 0x5700u64.wrapping_add(n as u64));
    let mut rng = StdRng::seed_from_u64(0xDB02);
    let k = ((n as f64) * k_frac).ceil().max(1.0) as usize;
    let mut ds_v = Vec::new(); let mut esm_v = Vec::new(); let mut full_v = Vec::new();
    for _ in 0..n_tx {
        let dirty = perturb_str(&mut index, k, str_len, &mut rng);
        let (ds, esm, full) = run_str(&index, &dirty);
        ds_v.push(ds); esm_v.push(esm); full_v.push(full);
        index.sort();
    }
    let dsm = med(ds_v); let esmm = med(esm_v); let fm = med(full_v);
    let kpct = 100.0 * k as f64 / n as f64;
    println!("STR {:>10} {:>7} {:>6.3}%  {:>9.1} {:>9.1} {:>9.1}   {:>6.2}× {:>6.2}×  (len={})",
        n, k, kpct, dsm/1e3, esmm/1e3, fm/1e3, esmm/dsm, fm/dsm, str_len);
    writeln!(csv, "str{},{},{},DeltaSort,{:.0}", str_len, n, k, dsm).unwrap();
    writeln!(csv, "str{},{},{},ESM,{:.0}", str_len, n, k, esmm).unwrap();
    writeln!(csv, "str{},{},{},Full,{:.0}", str_len, n, k, fm).unwrap();
}

fn main() {
    fs::create_dir_all("results").ok();
    let mut f = fs::File::create("results/track_l_index_maint.csv").expect("csv");
    writeln!(f, "cmp,n,k,strategy,ns").unwrap();

    println!("{:>3} {:>10} {:>7} {:>7}  {:>9} {:>9} {:>9}   {:>7} {:>7}",
        "cmp", "n", "k", "k/n", "DS us", "ESM us", "Full us", "ESM/DS", "Full/DS");

    println!("\n=== Numeric index (i64 sort key) ===");
    for &n in &[10_000usize, 100_000, 1_000_000] {
        for &kf in &[0.0001f64, 0.001, 0.005, 0.01, 0.05, 0.1] {
            run_num_wl(n, kf, 21, &mut f);
        }
        println!();
    }

    println!("=== String index (varchar-20 sort key) ===");
    for &n in &[10_000usize, 100_000, 1_000_000] {
        for &kf in &[0.0001f64, 0.001, 0.005, 0.01, 0.05] {
            run_str_wl(n, kf, 20, 21, &mut f);
        }
        println!();
    }

    println!("=== String index (varchar-100, realistic URL/path column) ===");
    for &n in &[100_000usize, 1_000_000] {
        for &kf in &[0.0001f64, 0.001, 0.005, 0.01] {
            run_str_wl(n, kf, 100, 21, &mut f);
        }
        println!();
    }

    println!("Wrote results/track_l_index_maint.csv");
}
