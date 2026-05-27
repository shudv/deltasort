//! Multiple Hypothesis Tracking (MHT) / Sensor Fusion benchmark.
//!
//! In modern radar/lidar/EO sensor fusion (military, autonomous vehicles,
//! drone swarms), the Multiple Hypothesis Tracking algorithm maintains a
//! sorted hypothesis list, each hypothesis representing one candidate
//! association of measurements to tracks. The hypothesis score is the
//! product of Kalman filter likelihoods (numerically: a log-sum), which is
//! an EXPENSIVE comparator: each comparison involves comparing 8–16
//! likelihood components.
//!
//! After each sensor scan:
//!   - k hypotheses get their scores updated (the ones consistent with the
//!     new measurements)
//!   - Sometimes new hypotheses are spawned (track births)
//!   - Sometimes hypotheses are pruned (track deaths or low scores)
//!
//! The MHT algorithm requires the top-N hypotheses to be reported every scan
//! (for sensor management / cueing decisions), so a sorted view is needed.
//!
//! Reference systems:
//!   - Reid's classic MHT algorithm
//!   - Stone-Bar-Shalom hypothesis-oriented MHT (HOMHT)
//!   - Kurien's track-oriented MHT (TOMHT)
//!   - Production: BAE PHALANX, Lockheed COBRA JUDY radar, Tesla autopilot
//!     sensor fusion
//!
//! Latency context:
//!   - Air-defense radar: 1-10 ms per scan
//!   - Automotive radar: 50 ms per scan
//!   - Drone swarm: 10-100 ms per scan
//!   - Each MHT iteration must complete well under the scan period

use deltasort::delta_sort_by;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::fs;
use std::io::Write;
use std::time::Instant;

/// A track hypothesis with 12-component log-likelihood signature.
/// Real MHT systems use Mahalanobis distance over the innovation covariance,
/// which decomposes into a sum over individual components. The comparator
/// must traverse and sum all components to break ties — modeling the
/// "expensive structured comparator" use case.
#[derive(Clone, Default, PartialEq)]
struct Hypothesis {
    /// Aggregate score (log-likelihood sum). Higher = more probable.
    score: f64,
    /// Per-channel log-likelihoods (for tiebreak)
    ll_components: [f64; 12],
    /// Track lineage id (for stable ordering)
    lineage_id: u32,
}

impl Eq for Hypothesis {}
impl Ord for Hypothesis {
    fn cmp(&self, o: &Self) -> Ordering {
        // Descending score, then component-wise tiebreak (expensive)
        let primary = o.score.partial_cmp(&self.score).unwrap_or(Ordering::Equal);
        if primary != Ordering::Equal { return primary; }
        for i in 0..12 {
            let c = o.ll_components[i].partial_cmp(&self.ll_components[i]).unwrap_or(Ordering::Equal);
            if c != Ordering::Equal { return c; }
        }
        self.lineage_id.cmp(&o.lineage_id)
    }
}
impl PartialOrd for Hypothesis { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
fn cmp_hyp(a: &Hypothesis, b: &Hypothesis) -> Ordering { a.cmp(b) }

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

fn init_hypotheses(n: usize, seed: u64) -> Vec<Hypothesis> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v: Vec<Hypothesis> = (0..n as u32).map(|i| {
        let mut components = [0.0f64; 12];
        let mut score = 0.0;
        for c in components.iter_mut() {
            *c = rng.gen_range(-10.0..0.0); // log-likelihoods are negative
            score += *c;
        }
        Hypothesis { score, ll_components: components, lineage_id: i }
    }).collect();
    v.sort();
    v
}

/// One sensor scan: k hypotheses get their scores updated (a new measurement
/// arrives consistent with them). Innovation = small perturbation per component.
fn process_scan(hyps: &mut [Hypothesis], k: usize, rng: &mut StdRng) -> Vec<usize> {
    let n = hyps.len();
    let mut picks: Vec<usize> = (0..n).collect();
    for i in 0..k { let j = rng.gen_range(i..n); picks.swap(i, j); }
    let mut dirty: Vec<usize> = picks[..k].to_vec();
    dirty.sort_unstable();
    for &i in &dirty {
        let mut new_score = 0.0;
        for c in hyps[i].ll_components.iter_mut() {
            let innov: f64 = rng.gen_range(-2.0..2.0);
            *c = (*c + innov).clamp(-50.0, 0.0);
            new_score += *c;
        }
        hyps[i].score = new_score;
    }
    dirty
}

struct Out { ds: f64, esm: f64, full: f64 }

fn run_scan(baseline: &[Hypothesis], dirty: &[usize]) -> Out {
    let mut data = baseline.to_vec();
    let mut dv = dirty.to_vec(); dv.sort_unstable(); dv.dedup();
    let t = Instant::now();
    delta_sort_by(&mut data, &mut dv, cmp_hyp);
    let ds = t.elapsed().as_nanos() as f64;
    let ds_result = data;

    let mut data = baseline.to_vec();
    let mut dv = dirty.to_vec();
    let t = Instant::now();
    esm_sort(&mut data, &mut dv, cmp_hyp);
    let esm = t.elapsed().as_nanos() as f64;
    assert!(data == ds_result, "ESM mismatch");

    let mut data = baseline.to_vec();
    let t = Instant::now();
    data.sort_unstable_by(cmp_hyp);
    let full = t.elapsed().as_nanos() as f64;
    assert!(data == ds_result, "Full mismatch");

    Out { ds, esm, full }
}

fn med(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }

fn run_workload(n: usize, k: usize, n_scans: usize, csv: &mut impl Write) {
    let mut hyps = init_hypotheses(n, 0xA1A2u64.wrapping_add(n as u64));
    let mut rng = StdRng::seed_from_u64(0x5EED);
    let mut ds_v = Vec::new(); let mut esm_v = Vec::new(); let mut full_v = Vec::new();
    for _ in 0..n_scans {
        let dirty = process_scan(&mut hyps, k, &mut rng);
        let r = run_scan(&hyps, &dirty);
        ds_v.push(r.ds); esm_v.push(r.esm); full_v.push(r.full);
        hyps.sort();
    }
    let dsm = med(ds_v); let esmm = med(esm_v); let fm = med(full_v);
    let kpct = 100.0 * k as f64 / n as f64;
    println!(
        "{:>6} {:>5} {:>6.2}%  {:>8.1} {:>8.1} {:>8.1}   {:>5.2}× {:>5.2}×",
        n, k, kpct,
        dsm/1e3, esmm/1e3, fm/1e3,
        esmm/dsm, fm/dsm,
    );
    for (strat, val) in [("DeltaSort", dsm), ("ESM", esmm), ("Full", fm)] {
        writeln!(csv, "{},{},{},{:.0}", n, k, strat, val).unwrap();
    }
}

fn main() {
    fs::create_dir_all("results").ok();
    let mut f = fs::File::create("results/track_r_mht.csv").expect("csv");
    writeln!(f, "n,k,strategy,ns").unwrap();

    println!("=== Multiple Hypothesis Tracking (Sensor Fusion) ===");
    println!("(comparator: 12-component log-likelihood vector with full tiebreak)");
    println!("{:>6} {:>5} {:>7}  {:>8} {:>8} {:>8}", "n", "k", "k/n", "DS μs", "ESM μs", "Full μs");

    println!("\n--- Small surveillance (automotive radar/drone, n=100-1K) ---");
    for &n in &[100usize, 500, 1_000] {
        for &k in &[1usize, 5, 10, 25, 50, 100] {
            if k <= n / 2 {
                run_workload(n, k, 51, &mut f);
            }
        }
        println!();
    }

    println!("--- Medium tactical (air-defense radar, n=5K-20K) ---");
    for &n in &[5_000usize, 10_000, 20_000] {
        for &k in &[1usize, 10, 50, 100, 500, 1000] {
            if k <= n / 4 {
                run_workload(n, k, 31, &mut f);
            }
        }
        println!();
    }

    println!("--- Large strategic (theater BMD, n=100K hypotheses) ---");
    for &n in &[50_000usize, 100_000] {
        for &k in &[1usize, 100, 1000, 5000] {
            if k <= n / 4 {
                run_workload(n, k, 11, &mut f);
            }
        }
        println!();
    }

    println!("Wrote results/track_r_mht.csv");
}
