//! Adversarial validation:
//!
//! Q1: The original DeltaSort paper reports ~2.5× over ESM at k=0.1%, n=100K.
//!     Our application benchmarks report 30-346×. WHY?
//!
//! Hypotheses to test:
//!   H1: k/n in our "headline" numbers is much smaller than 0.1%
//!       (e.g., k=5 at n=100K is 0.005%, 20× smaller)
//!   H2: Our application comparators differ from the original paper's MultiKey
//!   H3: Our cargo profile (LTO + codegen-units=1) differs
//!   H4: Our harness measures something different
//!
//! This binary runs the ORIGINAL paper's experimental setup
//! (n=100K, k swept 0.01%..1%, multi-key user comparator) with the
//! NEW validated harness so we can directly compare to:
//!   (a) the published 2.5× number
//!   (b) our application headlines
//!
//! Q2: Is anything we measured an artifact?
//!     Run a tail-latency stress test specifically on the worst (n,k,dist)
//!     to see if the 92× p99/p50 outlier reproduces.

mod common;
use common::{bench, esm_sort, print_machine_info, BenchStats};
use deltasort::delta_sort_by;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::fs;
use std::io::Write;

// ============================================================
//  Original paper's MultiKey user-record comparator
// ============================================================

#[derive(Clone, Default, PartialEq)]
struct UserRec {
    name_hash: i64,
    age: i32,
    country_code: u16,
    score: f64,
    id: u32,
}
impl Eq for UserRec {}
impl PartialOrd for UserRec {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
}
impl Ord for UserRec {
    fn cmp(&self, o: &Self) -> Ordering {
        self.name_hash
            .cmp(&o.name_hash)
            .then(self.age.cmp(&o.age))
            .then(self.country_code.cmp(&o.country_code))
            .then(self.score.partial_cmp(&o.score).unwrap_or(Ordering::Equal))
            .then(self.id.cmp(&o.id))
    }
}
fn cmp_user(a: &UserRec, b: &UserRec) -> Ordering { a.cmp(b) }

fn init_users(n: usize, seed: u64) -> Vec<UserRec> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v: Vec<UserRec> = (0..n as u32).map(|i| UserRec {
        name_hash: rng.gen(),
        age: rng.gen_range(18..90),
        country_code: rng.gen_range(0..250),
        score: rng.gen_range(0.0..100.0),
        id: i,
    }).collect();
    v.sort();
    v
}

fn perturb_users(arr: &mut [UserRec], dirty: &[usize], rng: &mut StdRng) {
    for &di in dirty {
        arr[di].name_hash = rng.gen();
        arr[di].age = rng.gen_range(18..90);
        arr[di].score = rng.gen_range(0.0..100.0);
    }
}

fn pick_dirty(n: usize, k: usize, rng: &mut StdRng) -> Vec<usize> {
    let mut picks: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = rng.gen_range(i..n);
        picks.swap(i, j);
    }
    let mut d: Vec<usize> = picks[..k].to_vec();
    d.sort_unstable();
    d
}

// ============================================================
//  Experiment 1: Reproduce the original paper's ~2.5× claim
// ============================================================

fn reproduce_original_claim(csv: &mut impl Write) {
    println!("\n=== Experiment 1: Reproducing original paper setup ===");
    println!("n=100K, multi-key user comparator, k sweep");
    println!("Original paper reports: at k=0.1%, DS is ~2.5× over ESM, ~13× over Full\n");
    println!("{:>8} {:>6} {:>5} {:>12} {:>12} {:>12} {:>8} {:>8}",
             "n", "k", "k/n%", "DS_p50_ns", "ESM_p50_ns", "Full_p50_ns", "ESM/DS", "Full/DS");
    println!("{}", "-".repeat(90));

    let n = 100_000usize;
    let base = init_users(n, 0xDEAD);
    let ks = [10usize, 50, 100, 200, 500, 1000, 5000, 10000];

    for k in ks {
        let mut rng = StdRng::seed_from_u64(0xC0FE + k as u64);
        let dirty = pick_dirty(n, k, &mut rng);
        let perturbed = {
            let mut b = base.clone();
            perturb_users(&mut b, &dirty, &mut rng);
            b
        };
        let correct = {
            let mut x = perturbed.clone();
            x.sort_by(cmp_user);
            x
        };

        // DS
        let stats_ds = bench(
            "ds",
            || (perturbed.clone(), dirty.clone()),
            |(arr, d)| delta_sort_by(arr.as_mut_slice(), d.as_mut_slice(), cmp_user),
            |(arr, _)| arr == &correct,
        );
        // ESM
        let stats_esm = bench(
            "esm",
            || (perturbed.clone(), dirty.clone()),
            |(arr, d)| esm_sort(arr, d.as_mut_slice(), cmp_user),
            |(arr, _)| arr == &correct,
        );
        // Full
        let stats_full = bench(
            "full",
            || perturbed.clone(),
            |arr| arr.sort_unstable_by(cmp_user),
            |arr| arr == &correct,
        );

        let kpct = k as f64 / n as f64 * 100.0;
        println!("{:>8} {:>6} {:>5.3} {:>12.0} {:>12.0} {:>12.0} {:>8.2} {:>8.2}",
            n, k, kpct, stats_ds.p50, stats_esm.p50, stats_full.p50,
            stats_esm.p50 / stats_ds.p50,
            stats_full.p50 / stats_ds.p50);
        writeln!(csv, "reproduce,multi_user,{},{},{:.4},DS,{}",
            n, k, kpct, stats_ds.to_csv()).unwrap();
        writeln!(csv, "reproduce,multi_user,{},{},{:.4},ESM,{}",
            n, k, kpct, stats_esm.to_csv()).unwrap();
        writeln!(csv, "reproduce,multi_user,{},{},{:.4},Full,{}",
            n, k, kpct, stats_full.to_csv()).unwrap();
    }
}

// ============================================================
//  Experiment 2: Force k/n to be a fair comparison across apps
//   For every app, hold k/n = 0.1% (the original paper's headline).
//   Question: at k/n=0.1%, are our application speedups really
//   in the 30-346× range, or were we cherry-picking very low k?
// ============================================================

fn fair_kn_comparison(csv: &mut impl Write) {
    println!("\n=== Experiment 2: Fair k/n=0.1% comparison ===");
    println!("Holding k/n = 0.1% across all applications.\n");
    println!("If app speedups stay 30×+ at fair k/n, the regime is real.");
    println!("If they collapse to 2-5×, the headlines were cherry-picked.\n");

    // Each row: (workload_name, fn pointer to a workload runner with given k/n)
    // We'll inline this with closures rather than building dynamic dispatch.

    // --- Depth sort (cheap f32 cmp) ---
    #[derive(Clone, Default, PartialEq)]
    struct Sprite { z: f32, id: u32 }
    impl Eq for Sprite {}
    impl Ord for Sprite {
        fn cmp(&self, o: &Self) -> Ordering {
            self.z.partial_cmp(&o.z).unwrap_or(Ordering::Equal).then(self.id.cmp(&o.id))
        }
    }
    impl PartialOrd for Sprite { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }

    let depth_runner = |n: usize, k: usize| -> (BenchStats, BenchStats, BenchStats) {
        let mut rng = StdRng::seed_from_u64(0xDEDE + n as u64 + k as u64);
        let mut base: Vec<Sprite> = (0..n).map(|i| Sprite {
            z: rng.gen_range(-100.0..100.0), id: i as u32,
        }).collect();
        base.sort();
        let dirty = pick_dirty(n, k, &mut rng);
        let perturbed = {
            let mut b = base.clone();
            for &di in &dirty {
                b[di].z = (b[di].z + rng.gen_range(-5.0f32..5.0)).clamp(-100.0, 100.0);
            }
            b
        };
        let correct = { let mut x = perturbed.clone(); x.sort(); x };

        let s_ds = bench("dep_ds",
            || (perturbed.clone(), dirty.clone()),
            |(arr, d)| delta_sort_by(arr.as_mut_slice(), d.as_mut_slice(), Sprite::cmp),
            |(arr, _)| arr == &correct);
        let s_esm = bench("dep_esm",
            || (perturbed.clone(), dirty.clone()),
            |(arr, d)| esm_sort(arr, d.as_mut_slice(), Sprite::cmp),
            |(arr, _)| arr == &correct);
        let s_full = bench("dep_full",
            || perturbed.clone(),
            |arr| arr.sort_unstable(),
            |arr| arr == &correct);
        (s_ds, s_esm, s_full)
    };

    println!("{:>14} {:>8} {:>6} {:>12} {:>12} {:>10} {:>10}",
             "workload", "n", "k", "DS_p50_ns", "ESM_p50_ns", "ESM/DS", "Full/DS");
    println!("{}", "-".repeat(86));

    // Force k = max(1, n/1000) so k/n = 0.1%
    for &n in &[1_000usize, 10_000, 100_000] {
        let k = (n / 1000).max(1);
        let (ds, esm, full) = depth_runner(n, k);
        println!("{:>14} {:>8} {:>6} {:>12.0} {:>12.0} {:>10.2} {:>10.2}",
                 "depth_sort", n, k, ds.p50, esm.p50, esm.p50/ds.p50, full.p50/ds.p50);
        writeln!(csv, "fair_kn,depth_sort,{},{},0.1000,DS,{}", n, k, ds.to_csv()).unwrap();
        writeln!(csv, "fair_kn,depth_sort,{},{},0.1000,ESM,{}", n, k, esm.to_csv()).unwrap();
        writeln!(csv, "fair_kn,depth_sort,{},{},0.1000,Full,{}", n, k, full.to_csv()).unwrap();
    }

    // Same workload but at k/n = 1% (the original paper's "ESM wins" boundary)
    println!("\n--- Same workload at k/n = 1% ---");
    for &n in &[10_000usize, 100_000] {
        let k = (n / 100).max(1);
        let (ds, esm, full) = depth_runner(n, k);
        println!("{:>14} {:>8} {:>6} {:>12.0} {:>12.0} {:>10.2} {:>10.2}",
                 "depth_sort", n, k, ds.p50, esm.p50, esm.p50/ds.p50, full.p50/ds.p50);
        writeln!(csv, "fair_kn,depth_sort,{},{},1.0000,DS,{}", n, k, ds.to_csv()).unwrap();
        writeln!(csv, "fair_kn,depth_sort,{},{},1.0000,ESM,{}", n, k, esm.to_csv()).unwrap();
        writeln!(csv, "fair_kn,depth_sort,{},{},1.0000,Full,{}", n, k, full.to_csv()).unwrap();
    }
}

// ============================================================
//  Experiment 3: Tail latency stress test
//   The validated.csv flagged a 92× p99/p50 ratio at one cell.
//   Reproduce with 10,000 samples to confirm it's real, not noise.
// ============================================================

fn tail_latency_stress(csv: &mut impl Write) {
    println!("\n=== Experiment 3: Tail latency stress test ===");
    println!("Reproducing the 92× p99/p50 outlier with 10,000 samples\n");

    // Tracking n=10K, k=50, uniform was the outlier
    use crate::common::bench_with_config;
    #[derive(Clone, Default, PartialEq)]
    struct Track {
        threat: f32, ttc: f32, range_km: f32,
        callsign: u64, alt: i32, bearing: u16, id: u32,
    }
    impl Eq for Track {}
    impl Ord for Track {
        fn cmp(&self, o: &Self) -> Ordering {
            o.threat.partial_cmp(&self.threat).unwrap_or(Ordering::Equal)
                .then(self.ttc.partial_cmp(&o.ttc).unwrap_or(Ordering::Equal))
                .then(self.range_km.partial_cmp(&o.range_km).unwrap_or(Ordering::Equal))
                .then(self.callsign.cmp(&o.callsign))
                .then(self.alt.cmp(&o.alt))
                .then(self.bearing.cmp(&o.bearing))
                .then(self.id.cmp(&o.id))
        }
    }
    impl PartialOrd for Track { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }

    let n = 10_000usize;
    let k = 50usize;
    let mut rng = StdRng::seed_from_u64(0x71AC + n as u64);
    let mut base: Vec<Track> = (0..n as u32).map(|i| Track {
        threat: rng.gen(), ttc: rng.gen_range(0.0..3600.0),
        range_km: rng.gen_range(0.0..500.0),
        callsign: rng.gen(),
        alt: rng.gen_range(0..50_000),
        bearing: rng.gen_range(0..6400),
        id: i,
    }).collect();
    base.sort();
    let mut rng2 = StdRng::seed_from_u64(0xA5A5 + n as u64);
    let dirty = pick_dirty(n, k, &mut rng2);
    let perturbed = {
        let mut b = base.clone();
        for &di in &dirty {
            b[di].threat = (b[di].threat + rng2.gen_range(-0.1f32..0.1)).clamp(0.0, 1.0);
        }
        b
    };
    let correct = { let mut x = perturbed.clone(); x.sort(); x };

    println!("Running 10,000 samples (was 500)...");
    let stats = bench_with_config("tail_stress",
        500, 10_000,
        || (perturbed.clone(), dirty.clone()),
        |(arr, d)| delta_sort_by(arr.as_mut_slice(), d.as_mut_slice(), Track::cmp),
        |(arr, _)| arr == &correct);
    println!("\nDS p50  = {:>9.0} ns", stats.p50);
    println!("DS p95  = {:>9.0} ns", stats.p95);
    println!("DS p99  = {:>9.0} ns", stats.p99);
    println!("DS p999 = (not computed; would need percentile fn)");
    println!("DS mean = {:>9.1} ns", stats.mean);
    println!("DS std  = {:>9.1} ns", stats.std);
    println!("DS min  = {:>9.0} ns", stats.min);
    println!("DS max  = {:>9.0} ns", stats.max);
    println!("p99/p50 = {:.1}×", stats.p99 / stats.p50);
    writeln!(csv, "tail_stress,tracking,{},{},,DS,{}", n, k, stats.to_csv()).unwrap();
}

fn main() {
    print_machine_info();
    fs::create_dir_all("results").ok();
    let mut f = fs::File::create("results/adversarial.csv").expect("csv");
    writeln!(f, "experiment,workload,n,k,k_n_pct,strategy,{}", BenchStats::csv_header()).unwrap();

    reproduce_original_claim(&mut f);
    fair_kn_comparison(&mut f);
    tail_latency_stress(&mut f);

    println!("\nWrote results/adversarial.csv");
}
