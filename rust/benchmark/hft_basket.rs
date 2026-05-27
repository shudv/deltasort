//! HFT alpha-signal basket benchmark.
//!
//! Realistic HFT scenario: a market-making / stat-arb strategy maintains a
//! basket of N instruments sorted by expected alpha (signal strength × risk
//! coefficient). After each market data tick a small batch of k signals are
//! recalculated (e.g., because new quotes arrived for those instruments).
//! The strategy must re-rank the basket to route execution to top signals.
//!
//! Why a flat sorted array:
//!   - O(1) indexed reads: "give me rank-3 instrument" for routing
//!   - Cache-friendly iteration for risk aggregation
//!   - No allocation on update (critical in latency-sensitive path)
//!
//! Baselines:
//!   - FullSort: `sort_unstable_by` — what most non-HFT algo systems do
//!   - ESM: extract-sort-merge — optimal for medium k
//!   - DeltaSort: our algorithm
//!
//! Note on HFT latency context: top exchange colocation boxes see 100K–5M
//! ticks/sec. At 100K ticks/sec: 1μs savings = 0.1 CPU-seconds saved per
//! trading second. At 1M ticks/sec: same 1μs savings → 1 CPU-second/sec.
//! For a strategy running on 4 cores, this can matter.

use deltasort::delta_sort_by;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::fs;
use std::io::Write;
use std::time::Instant;

/// An instrument in the basket with its computed alpha signal.
#[derive(Clone, Default, PartialEq)]
struct Instrument {
    /// Composite alpha: signal × vol_adj × capacity_adj (float, higher = better)
    alpha: f64,
    /// Secondary: risk-adjusted return
    risk_adj: f64,
    /// Instrument id (tiebreaker, also used to look up by name)
    id: u32,
}
impl Eq for Instrument {}
impl Ord for Instrument {
    fn cmp(&self, o: &Self) -> Ordering {
        // Descending alpha, then descending risk_adj, then id
        o.alpha.partial_cmp(&self.alpha).unwrap_or(Ordering::Equal)
            .then(o.risk_adj.partial_cmp(&self.risk_adj).unwrap_or(Ordering::Equal))
            .then(self.id.cmp(&o.id))
    }
}
impl PartialOrd for Instrument { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }

fn cmp_inst(a: &Instrument, b: &Instrument) -> Ordering { a.cmp(b) }

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
    let mut di = k as isize - 1;
    let mut wi = n - 1;
    loop {
        let take_clean = if ci >= 0 && di >= 0 {
            cmp(&arr[ci as usize], &buf[di as usize]) != Ordering::Less
        } else { ci >= 0 };
        if take_clean { if wi != ci as usize { arr.swap(wi, ci as usize); } ci -= 1; }
        else if di >= 0 { arr[wi] = std::mem::take(&mut buf[di as usize]); di -= 1; }
        else { break; }
        if wi == 0 { break; }
        wi -= 1;
    }
}

fn init_basket(n: usize, seed: u64) -> Vec<Instrument> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v: Vec<Instrument> = (0..n).map(|i| Instrument {
        alpha: rng.gen_range(-1.0..1.0),
        risk_adj: rng.gen_range(0.0..10.0),
        id: i as u32,
    }).collect();
    v.sort();
    v
}

/// Simulate a market data tick: k instruments get their alpha recomputed.
/// Returns (dirty_indices, updated basket).
fn simulate_tick(basket: &mut [Instrument], k: usize, rng: &mut StdRng) -> Vec<usize> {
    let n = basket.len();
    let mut picks: Vec<usize> = (0..n).collect();
    for i in 0..k { let j = rng.gen_range(i..n); picks.swap(i, j); }
    let mut dirty: Vec<usize> = picks[..k].to_vec();
    dirty.sort_unstable();
    for &i in &dirty {
        // Realistic: alpha changes by a small perturbation (most ticks just nudge)
        // Occasionally a full signal flip (rare event)
        let noise: f64 = rng.gen_range(-0.3..0.3);
        basket[i].alpha = (basket[i].alpha + noise).clamp(-1.0, 1.0);
        basket[i].risk_adj = rng.gen_range(0.0..10.0);
    }
    dirty
}

struct Out { ds: f64, esm: f64, full: f64 }

fn run_tick(baseline: &[Instrument], dirty: &[usize]) -> Out {
    let mut data = baseline.to_vec();
    let mut dirty_v = dirty.to_vec();
    dirty_v.sort_unstable(); dirty_v.dedup();
    let t = Instant::now();
    delta_sort_by(&mut data, &mut dirty_v, cmp_inst);
    let ds = t.elapsed().as_nanos() as f64;
    let ds_result = data;

    let mut data = baseline.to_vec();
    let mut dirty_v = dirty.to_vec();
    let t = Instant::now();
    esm_sort(&mut data, &mut dirty_v, cmp_inst);
    let esm = t.elapsed().as_nanos() as f64;
    assert!(data == ds_result, "ESM mismatch");

    let mut data = baseline.to_vec();
    let t = Instant::now();
    data.sort_unstable_by(cmp_inst);
    let full = t.elapsed().as_nanos() as f64;
    assert!(data == ds_result, "full mismatch");

    Out { ds, esm, full }
}

fn run_workload(n: usize, k_frac: f64, n_ticks: usize, csv: &mut impl Write) {
    let mut basket = init_basket(n, 0xA1B2u64.wrapping_add(n as u64));
    let mut rng = StdRng::seed_from_u64(0xA1B3);
    let k = ((n as f64) * k_frac).ceil().max(1.0) as usize;

    let mut ds_v = Vec::new(); let mut esm_v = Vec::new(); let mut full_v = Vec::new();
    for _ in 0..n_ticks {
        let dirty = simulate_tick(&mut basket, k, &mut rng);
        let r = run_tick(&basket, &dirty);
        ds_v.push(r.ds); esm_v.push(r.esm); full_v.push(r.full);
        basket.sort();
    }
    let med = |mut v: Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
    let dsm = med(ds_v); let esmm = med(esm_v); let fm = med(full_v);
    let kpct = 100.0 * k as f64 / n as f64;

    // Context: latency budget per tick
    let tick_budget_us = match n {
        n if n <= 500  => 1.0_f64,   // 1μs: colocation HFT
        n if n <= 5000 => 10.0_f64,  // 10μs: low-latency
        _              => 100.0_f64, // 100μs: algorithmic
    };
    let ds_pct = 100.0 * dsm / 1000.0 / tick_budget_us;
    let full_pct = 100.0 * fm / 1000.0 / tick_budget_us;

    println!(
        "{:>7} {:>5} {:>6.3}%  {:>8.1} {:>8.1} {:>8.1}   {:>6.2}× {:>6.2}×  budget:{:.0}μs  DS={:.1}% Full={:.1}%",
        n, k, kpct,
        dsm / 1000.0, esmm / 1000.0, fm / 1000.0,
        esmm / dsm, fm / dsm,
        tick_budget_us, ds_pct, full_pct,
    );
    writeln!(csv, "{},{},DeltaSort,{:.0}", n, k, dsm).unwrap();
    writeln!(csv, "{},{},ESM,{:.0}", n, k, esmm).unwrap();
    writeln!(csv, "{},{},Full,{:.0}", n, k, fm).unwrap();
}

fn main() {
    fs::create_dir_all("results").ok();
    let mut f = fs::File::create("results/track_k_hft.csv").expect("csv");
    writeln!(f, "n,k,strategy,ns").unwrap();

    println!("=== HFT Alpha Basket Repricing ===");
    println!("(DS ns | ESM ns | Full ns | ESM/DS | Full/DS | budget context)");
    println!("{:>7} {:>5} {:>7}  {:>8} {:>8} {:>8}   {:>7} {:>7}", "n", "k", "k/n", "DS ns", "ESM ns", "Full ns", "ESM/DS", "Full/DS");

    // HFT colocation: ultra-small baskets, single-digit k
    println!("\n--- Colocation HFT (basket 100-500, 1μs budget) ---");
    for &n in &[100usize, 200, 500] {
        for &kf in &[0.01f64, 0.02, 0.05, 0.1, 0.2] {
            run_workload(n, kf, 51, &mut f);
        }
        println!();
    }

    // Low-latency: medium baskets
    println!("--- Low-latency algo (basket 1K-5K, 10μs budget) ---");
    for &n in &[1_000usize, 5_000] {
        for &kf in &[0.005f64, 0.01, 0.02, 0.05, 0.1] {
            run_workload(n, kf, 51, &mut f);
        }
        println!();
    }

    // Algorithmic: larger baskets
    println!("--- Algorithmic trading (basket 10K-100K, 100μs budget) ---");
    for &n in &[10_000usize, 100_000] {
        for &kf in &[0.001f64, 0.005, 0.01, 0.05] {
            run_workload(n, kf, 51, &mut f);
        }
        println!();
    }

    println!("Wrote results/track_k_hft.csv");
}
