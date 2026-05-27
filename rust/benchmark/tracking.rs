//! Multi-target tracking: real-time sorted track list maintenance.
//!
//! Realistic scenario: air-traffic / maritime tracking system. The system
//! maintains a sorted list of tracks ranked by composite priority (e.g.,
//! threat-level, time-to-conflict, range). Each radar scan (1-10 Hz) updates
//! a subset of tracks; track priorities must be re-sorted between scans.
//!
//! Why flat array, not BTreeMap:
//!   - Operators query "top-K threats" by index — flat array gives O(1) per
//!   - SIMD/AVX-512 operators process contiguous track buffers
//!   - Tactical displays render contiguous track records (cache locality)
//!   - DO-178C / safety certification prefers deterministic memory layouts
//!
//! Comparator: multi-field (threat_score, ttc, range, callsign) — moderately
//! expensive (~50 ns per compare).

use deltasort::delta_sort_by;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::fs;
use std::io::Write;
use std::time::Instant;

#[derive(Clone, Default, PartialEq)]
struct Track {
    threat: u8,      // primary key (0=safe, 255=critical)
    ttc_sec: f32,    // time-to-conflict (negative = past)
    range_nm: f32,   // nautical miles
    altitude: i32,
    bearing: i16,
    callsign: [u8; 8],
}
impl Eq for Track {}
impl Ord for Track {
    fn cmp(&self, o: &Self) -> Ordering {
        // Sort by descending threat → ascending ttc → ascending range → callsign
        o.threat.cmp(&self.threat)
            .then(self.ttc_sec.partial_cmp(&o.ttc_sec).unwrap_or(Ordering::Equal))
            .then(self.range_nm.partial_cmp(&o.range_nm).unwrap_or(Ordering::Equal))
            .then(self.callsign.cmp(&o.callsign))
            .then(self.altitude.cmp(&o.altitude))
            .then(self.bearing.cmp(&o.bearing))
    }
}
impl PartialOrd for Track { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }

fn cmp_t(a: &Track, b: &Track) -> Ordering { a.cmp(b) }

/// Directional bubble: each dirty track bubbles to its correct position.
fn directional_bubble<F: Fn(&Track, &Track) -> Ordering + Copy>(
    arr: &mut [Track], dirty: &[usize], old_keys: &[(usize, u8, f32, f32)], cmp: F,
) {
    let n = arr.len();
    let mut right_movers: Vec<usize> = Vec::new();
    let mut left_movers: Vec<usize> = Vec::new();
    for &(i, old_threat, old_ttc, old_range) in old_keys {
        // Determine direction by comparing old vs new keys lexicographically
        let old = (old_threat, old_ttc, old_range);
        let new = (arr[i].threat, arr[i].ttc_sec, arr[i].range_nm);
        use std::cmp::Ordering::*;
        // Primary sort: descending threat. So if new.threat > old.threat, track moves LEFT (higher priority = earlier index)
        let dir = match new.0.cmp(&old.0).reverse() {
            Less => Less,
            Greater => Greater,
            Equal => match new.1.partial_cmp(&old.1).unwrap_or(Equal) {
                Less => Less,
                Greater => Greater,
                Equal => new.2.partial_cmp(&old.2).unwrap_or(Equal),
            }
        };
        if dir == Greater { right_movers.push(i); }
        else if dir == Less { left_movers.push(i); }
    }
    right_movers.sort_by_key(|&x| std::cmp::Reverse(x));
    left_movers.sort_by_key(|&x| x);
    for i in right_movers {
        let mut j = i;
        while j + 1 < n && cmp(&arr[j], &arr[j + 1]) == Ordering::Greater {
            arr.swap(j, j + 1); j += 1;
        }
    }
    for i in left_movers {
        let mut j = i;
        while j > 0 && cmp(&arr[j - 1], &arr[j]) == Ordering::Greater {
            arr.swap(j - 1, j); j -= 1;
        }
    }
}

fn esm_sort<T: Default + Clone, F: Fn(&T, &T) -> Ordering>(
    arr: &mut [T], dirty: &mut [usize], cmp: F,
) {
    if dirty.is_empty() { return; }
    let n = arr.len();
    let k = dirty.len();
    dirty.sort_unstable();
    let mut buf: Vec<T> = Vec::with_capacity(k);
    let mut write = 0;
    let mut di = 0;
    for read in 0..n {
        if di < k && dirty[di] == read {
            buf.push(std::mem::take(&mut arr[read]));
            di += 1;
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
        if take_clean {
            if wi != ci as usize { arr.swap(wi, ci as usize); }
            ci -= 1;
        } else if di >= 0 {
            arr[wi] = std::mem::take(&mut buf[di as usize]);
            di -= 1;
        } else { break; }
        if wi == 0 { break; }
        wi -= 1;
    }
}

fn init_tracks(n: usize, seed: u64) -> Vec<Track> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v: Vec<Track> = (0..n).map(|i| Track {
        threat: rng.gen(),
        ttc_sec: rng.gen_range(-300.0..300.0),
        range_nm: rng.gen_range(0.0..500.0),
        altitude: rng.gen_range(0..50000),
        bearing: rng.gen_range(0..3600),
        callsign: {
            let mut c = [0u8; 8];
            for b in &mut c { *b = b'A' + rng.gen_range(0..26); }
            let _ = i; c
        },
    }).collect();
    v.sort();
    v
}

fn perturb(tracks: &mut [Track], k: usize, rng: &mut StdRng) -> (Vec<usize>, Vec<(usize, u8, f32, f32)>) {
    let n = tracks.len();
    let mut picks: Vec<usize> = (0..n).collect();
    for i in 0..k { let j = rng.gen_range(i..n); picks.swap(i, j); }
    let mut dirty: Vec<usize> = picks[..k].to_vec();
    dirty.sort_unstable();
    let mut old_keys = Vec::with_capacity(k);
    for &i in &dirty {
        old_keys.push((i, tracks[i].threat, tracks[i].ttc_sec, tracks[i].range_nm));
        tracks[i].threat = rng.gen();
        tracks[i].ttc_sec = rng.gen_range(-300.0..300.0);
        tracks[i].range_nm = rng.gen_range(0.0..500.0);
    }
    (dirty, old_keys)
}

struct Out { ds: f64, esm: f64, full: f64 }

fn run_frame(baseline: &[Track], dirty: &[usize], _old_keys: &[(usize, u8, f32, f32)]) -> Out {
    let mut data = baseline.to_vec();
    let mut dirty_v: Vec<usize> = dirty.to_vec();
    dirty_v.sort_unstable();
    dirty_v.dedup();
    let t = Instant::now();
    delta_sort_by(&mut data, &mut dirty_v, cmp_t);
    let ds = t.elapsed().as_nanos() as f64;
    let ds_result = data;

    let mut data = baseline.to_vec();
    let mut dirty_v: Vec<usize> = dirty.to_vec();
    let t = Instant::now();
    esm_sort(&mut data, &mut dirty_v, cmp_t);
    let esm = t.elapsed().as_nanos() as f64;
    assert!(data == ds_result, "ESM mismatch");

    let mut data = baseline.to_vec();
    let t = Instant::now();
    data.sort_unstable_by(cmp_t);
    let full = t.elapsed().as_nanos() as f64;
    assert!(data == ds_result, "full mismatch");

    Out { ds, esm, full }
}

fn run_workload(n: usize, k_frac: f64, n_scans: usize, csv: &mut impl Write) {
    let mut tracks = init_tracks(n, 0xBEEFu64.wrapping_add(n as u64));
    let mut rng = StdRng::seed_from_u64(0xFADEu64);
    let k = ((n as f64) * k_frac).ceil().max(1.0) as usize;

    let mut ds_v = Vec::new(); let mut esm_v = Vec::new(); let mut full_v = Vec::new();
    for _ in 0..n_scans {
        let (dirty, old_keys) = perturb(&mut tracks, k, &mut rng);
        let r = run_frame(&tracks, &dirty, &old_keys);
        ds_v.push(r.ds); esm_v.push(r.esm); full_v.push(r.full);
        tracks.sort();
    }
    let med = |mut v: Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
    let dsm = med(ds_v); let esmm = med(esm_v); let fm = med(full_v);
    let kpct = 100.0 * k as f64 / n as f64;
    println!(
        "{:>8} {:>8} {:>6.3}%  {:>10.1} {:>10.1} {:>10.1}   {:>7.2}× {:>7.2}×",
        n, k, kpct,
        dsm / 1000.0, esmm / 1000.0, fm / 1000.0,
        esmm / dsm, fm / dsm,
    );
    writeln!(csv, "{},{},DeltaSort,{:.0}", n, k, dsm).unwrap();
    writeln!(csv, "{},{},ESM,{:.0}", n, k, esmm).unwrap();
    writeln!(csv, "{},{},Full,{:.0}", n, k, fm).unwrap();
}

fn main() {
    fs::create_dir_all("results").ok();
    let mut f = fs::File::create("results/track_j_tracking.csv").expect("csv");
    writeln!(f, "n,k,strategy,ns").unwrap();
    println!(
        "{:>8} {:>8} {:>7}  {:>10} {:>10} {:>10}   {:>8} {:>8}",
        "n_tracks", "k", "k/n",
        "DS us", "ESM us", "Full us",
        "ESM/DS", "Full/DS",
    );

    for &n in &[1_000usize, 10_000, 50_000] {
        for &kf in &[0.001f64, 0.005, 0.01, 0.05, 0.1] {
            run_workload(n, kf, 21, &mut f);
        }
        println!();
    }
    println!("Wrote results/track_j_tracking.csv");
}
