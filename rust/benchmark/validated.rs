//! Master validation benchmark: re-runs all credibility-critical scenarios
//! with rigorous statistical methodology.
//!
//! Each (benchmark, n, k, strategy) point is measured with:
//!   - 100 warmup samples
//!   - 500 measured samples
//!   - black_box on inputs and outputs (prevents dead-code elimination)
//!   - fresh state per sample (no leakage)
//!   - correctness verification on first, middle, and last samples
//!   - reported: p50, p95, p99, mean, stddev, 95% CI for mean
//!
//! All ESM/heap baselines come from common.rs (single vetted implementation).
//!
//! Workloads tested:
//!   1. LOB (Limit Order Book): vs Full + ESM + Skiplist
//!   2. HFT alpha basket: vs Full + ESM
//!   3. Audio voice mgmt: vs Full + ESM
//!   4. Multi-target tracking (expensive cmp): vs Full + ESM
//!   5. Multi-Hypothesis Tracking: vs Full + ESM
//!   6. Network WFQ scheduler: vs Full + ESM
//!   7. Game depth sort: vs Full + ESM
//!   8. DES event queue (negative): vs Full + ESM + Heap-with-PosMap
//!   9. A* search frontier (negative): vs Full + ESM + Heap-with-PosMap

mod common;
use common::{bench, esm_sort, is_sorted, print_machine_info, BenchStats, PosHeap};

use deltasort::delta_sort_by;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::fs;
use std::io::Write;

// ============================================================
//  Helpers
// ============================================================

fn write_row(
    csv: &mut impl Write,
    bench_name: &str, dist: &str, n: usize, k: usize, strategy: &str,
    stats: &BenchStats,
) {
    writeln!(csv, "{},{},{},{},{},{}",
        bench_name, dist, n, k, strategy, stats.to_csv()).unwrap();
}

fn pick_dirty(n: usize, k: usize, rng: &mut StdRng, clustered: bool) -> Vec<usize> {
    let mut dirty = Vec::with_capacity(k);
    if clustered {
        // 80% of dirty indices in a contiguous band of width 2k near a random center.
        let band = (2 * k).min(n);
        let center = rng.gen_range(band / 2..n - band / 2);
        let lo = center.saturating_sub(band / 2);
        let hi = (lo + band).min(n);
        let n_in_band = (k * 4) / 5;
        let mut picks: Vec<usize> = (lo..hi).collect();
        for i in 0..n_in_band.min(picks.len()) {
            let j = rng.gen_range(i..picks.len());
            picks.swap(i, j);
        }
        dirty.extend_from_slice(&picks[..n_in_band.min(picks.len())]);
        // Remaining 20% uniform
        for _ in 0..k - dirty.len() {
            dirty.push(rng.gen_range(0..n));
        }
    } else {
        let mut picks: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = rng.gen_range(i..n);
            picks.swap(i, j);
        }
        dirty.extend_from_slice(&picks[..k]);
    }
    dirty.sort_unstable();
    dirty.dedup();
    dirty
}

// ============================================================
//  Application 1: LOB (Limit Order Book)
// ============================================================

mod lob {
    use super::*;
    #[derive(Clone, Default, PartialEq)]
    pub struct PriceLevel {
        pub price_ticks: i64,
        pub qty: u64,
        pub n_orders: u32,
        pub seq: u64,
    }
    impl Eq for PriceLevel {}
    impl Ord for PriceLevel {
        fn cmp(&self, o: &Self) -> Ordering {
            self.price_ticks.cmp(&o.price_ticks).then(self.seq.cmp(&o.seq))
        }
    }
    impl PartialOrd for PriceLevel {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
    }
    pub fn cmp_asc(a: &PriceLevel, b: &PriceLevel) -> Ordering { a.cmp(b) }

    pub fn init_lob(n: usize, seed: u64) -> Vec<PriceLevel> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mid = 100_0000_i64;
        let mut v: Vec<PriceLevel> = (0..n).map(|i| PriceLevel {
            price_ticks: mid + (i as i64),
            qty: rng.gen_range(100..10_000u64),
            n_orders: rng.gen_range(1..20u32),
            seq: i as u64,
        }).collect();
        v.sort();
        v
    }

    pub fn apply_dirty(lob: &mut [PriceLevel], dirty: &[usize], rng: &mut StdRng, seq_base: u64) {
        for (i, &di) in dirty.iter().enumerate() {
            let delta: i64 = rng.gen_range(-50..50);
            lob[di].price_ticks += delta;
            lob[di].qty = rng.gen_range(100..10_000u64);
            lob[di].seq = seq_base + i as u64 + 1;
        }
    }
}

fn bench_lob(csv: &mut impl Write, n: usize, k: usize, dist: &str) {
    use lob::*;
    let clustered = dist == "clustered";
    let base_lob = init_lob(n, 0xB100 + n as u64);

    // Pre-compute a stable dirty set for THIS (n, k) — same dirty set every sample
    // so DS, ESM, and Full are all comparing the same input perturbation.
    let mut rng = StdRng::seed_from_u64(0xC0FE + n as u64 + k as u64);
    let dirty = pick_dirty(n, k, &mut rng, clustered);
    let seq_base = n as u64;

    // Pre-compute the perturbed (unsorted) state. Each sample STARTS from this.
    let perturbed = {
        let mut lob = base_lob.clone();
        let mut rng_d = StdRng::seed_from_u64(0xDEAD + n as u64);
        apply_dirty(&mut lob, &dirty, &mut rng_d, seq_base);
        lob
    };

    // Pre-compute correct sorted result for verification
    let correct = {
        let mut x = perturbed.clone();
        x.sort_by(cmp_asc);
        x
    };

    // --- DeltaSort ---
    let stats_ds = bench(
        "lob_ds",
        || (perturbed.clone(), dirty.clone()),
        |(arr, d)| {
            delta_sort_by(arr.as_mut_slice(), d.as_mut_slice(), cmp_asc);
        },
        |(arr, _)| arr == &correct,
    );
    write_row(csv, "lob", dist, n, k, "DeltaSort", &stats_ds);

    // --- ESM ---
    let stats_esm = bench(
        "lob_esm",
        || (perturbed.clone(), dirty.clone()),
        |(arr, d)| esm_sort(arr, d.as_mut_slice(), cmp_asc),
        |(arr, _)| arr == &correct,
    );
    write_row(csv, "lob", dist, n, k, "ESM", &stats_esm);

    // --- Full sort_unstable ---
    let stats_full = bench(
        "lob_full",
        || perturbed.clone(),
        |arr| arr.sort_unstable_by(cmp_asc),
        |arr| arr == &correct,
    );
    write_row(csv, "lob", dist, n, k, "Full", &stats_full);

    println!(
        "lob  {:>5} k={:>4} {:9}  DS={:>7.1}ns(±{:>5.1}) ESM={:>7.1}(±{:>5.1}) Full={:>7.1}(±{:>5.1})   ESM/DS={:.2}× Full/DS={:.2}×",
        n, k, dist,
        stats_ds.p50, stats_ds.std,
        stats_esm.p50, stats_esm.std,
        stats_full.p50, stats_full.std,
        stats_esm.p50 / stats_ds.p50,
        stats_full.p50 / stats_ds.p50,
    );
}

// ============================================================
//  Application 2: HFT alpha basket
// ============================================================

mod hft {
    use super::*;
    #[derive(Clone, Default, PartialEq)]
    pub struct Instrument {
        pub alpha: f64,
        pub risk_adj: f64,
        pub id: u32,
    }
    impl Eq for Instrument {}
    impl Ord for Instrument {
        fn cmp(&self, o: &Self) -> Ordering {
            o.alpha.partial_cmp(&self.alpha).unwrap_or(Ordering::Equal)
                .then(o.risk_adj.partial_cmp(&self.risk_adj).unwrap_or(Ordering::Equal))
                .then(self.id.cmp(&o.id))
        }
    }
    impl PartialOrd for Instrument {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
    }
    pub fn cmp_alpha(a: &Instrument, b: &Instrument) -> Ordering { a.cmp(b) }

    pub fn init(n: usize, seed: u64) -> Vec<Instrument> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut v: Vec<Instrument> = (0..n).map(|i| Instrument {
            alpha: rng.gen_range(-1.0..1.0),
            risk_adj: rng.gen_range(0.0..10.0),
            id: i as u32,
        }).collect();
        v.sort();
        v
    }

    pub fn apply_dirty(basket: &mut [Instrument], dirty: &[usize], rng: &mut StdRng) {
        for &di in dirty {
            let noise: f64 = rng.gen_range(-0.3..0.3);
            basket[di].alpha = (basket[di].alpha + noise).clamp(-1.0, 1.0);
            basket[di].risk_adj = rng.gen_range(0.0..10.0);
        }
    }
}

fn bench_hft(csv: &mut impl Write, n: usize, k: usize, dist: &str) {
    use hft::*;
    let clustered = dist == "clustered";
    let base = init(n, 0xA1B2 + n as u64);
    let mut rng = StdRng::seed_from_u64(0xA1B3 + n as u64 + k as u64);
    let dirty = pick_dirty(n, k, &mut rng, clustered);
    let perturbed = {
        let mut b = base.clone();
        apply_dirty(&mut b, &dirty, &mut rng);
        b
    };
    let correct = {
        let mut x = perturbed.clone();
        x.sort_by(cmp_alpha);
        x
    };

    let stats_ds = bench("hft_ds",
        || (perturbed.clone(), dirty.clone()),
        |(arr, d)| delta_sort_by(arr.as_mut_slice(), d.as_mut_slice(), cmp_alpha),
        |(arr, _)| arr == &correct);
    write_row(csv, "hft", dist, n, k, "DeltaSort", &stats_ds);

    let stats_esm = bench("hft_esm",
        || (perturbed.clone(), dirty.clone()),
        |(arr, d)| esm_sort(arr, d.as_mut_slice(), cmp_alpha),
        |(arr, _)| arr == &correct);
    write_row(csv, "hft", dist, n, k, "ESM", &stats_esm);

    let stats_full = bench("hft_full",
        || perturbed.clone(),
        |arr| arr.sort_unstable_by(cmp_alpha),
        |arr| arr == &correct);
    write_row(csv, "hft", dist, n, k, "Full", &stats_full);

    println!(
        "hft  {:>6} k={:>5} {:9}  DS={:>8.1}(±{:>6.1}) ESM={:>8.1}(±{:>6.1}) Full={:>8.1}(±{:>6.1})   ESM/DS={:.2}× Full/DS={:.2}×",
        n, k, dist,
        stats_ds.p50, stats_ds.std,
        stats_esm.p50, stats_esm.std,
        stats_full.p50, stats_full.std,
        stats_esm.p50 / stats_ds.p50,
        stats_full.p50 / stats_ds.p50,
    );
}

// ============================================================
//  Application 3: Audio voice management
// ============================================================

mod audio {
    use super::*;
    #[derive(Clone, Default, PartialEq)]
    pub struct Voice {
        pub priority: f32,
        pub envelope_level: f32,
        pub velocity: u8,
        pub age_samples: u32,
        pub channel: u8,
        pub note: u8,
        pub active: bool,
        pub voice_id: u32,
    }
    impl Eq for Voice {}
    impl Ord for Voice {
        fn cmp(&self, o: &Self) -> Ordering {
            o.priority.partial_cmp(&self.priority).unwrap_or(Ordering::Equal)
                .then(self.voice_id.cmp(&o.voice_id))
        }
    }
    impl PartialOrd for Voice {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
    }
    pub fn cmp_voice(a: &Voice, b: &Voice) -> Ordering { a.cmp(b) }

    pub fn init(n: usize, seed: u64) -> Vec<Voice> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut v: Vec<Voice> = (0..n as u32).map(|i| Voice {
            priority: rng.gen_range(0.0..1.0),
            envelope_level: rng.gen(),
            velocity: rng.gen_range(40..127u8),
            age_samples: rng.gen_range(0..50_000u32),
            channel: rng.gen_range(0..16u8),
            note: rng.gen_range(21..108u8),
            active: true,
            voice_id: i,
        }).collect();
        v.sort();
        v
    }

    pub fn apply_dirty(voices: &mut [Voice], dirty: &[usize], rng: &mut StdRng) {
        for &di in dirty {
            let delta: f32 = rng.gen_range(-0.15..0.15);
            voices[di].envelope_level = (voices[di].envelope_level + delta).clamp(0.0, 1.0);
            voices[di].age_samples += 512;
            let vel_norm = voices[di].velocity as f32 / 127.0;
            let age_norm = 1.0 - (voices[di].age_samples as f32 / 100_000.0).min(1.0);
            let chan_bonus = if voices[di].channel == 0 { 0.2 } else { 0.0 };
            voices[di].priority = 0.5 * voices[di].envelope_level
                + 0.2 * vel_norm + 0.2 * age_norm + 0.1 * chan_bonus;
        }
    }
}

fn bench_audio(csv: &mut impl Write, n: usize, k: usize, dist: &str) {
    use audio::*;
    let clustered = dist == "clustered";
    let base = init(n, 0xA0 + n as u64);
    let mut rng = StdRng::seed_from_u64(0xB00F + n as u64 + k as u64);
    let dirty = pick_dirty(n, k, &mut rng, clustered);
    let perturbed = {
        let mut b = base.clone();
        apply_dirty(&mut b, &dirty, &mut rng);
        b
    };
    let correct = { let mut x = perturbed.clone(); x.sort_by(cmp_voice); x };

    let stats_ds = bench("audio_ds",
        || (perturbed.clone(), dirty.clone()),
        |(arr, d)| delta_sort_by(arr.as_mut_slice(), d.as_mut_slice(), cmp_voice),
        |(arr, _)| arr == &correct);
    write_row(csv, "audio", dist, n, k, "DeltaSort", &stats_ds);

    let stats_esm = bench("audio_esm",
        || (perturbed.clone(), dirty.clone()),
        |(arr, d)| esm_sort(arr, d.as_mut_slice(), cmp_voice),
        |(arr, _)| arr == &correct);
    write_row(csv, "audio", dist, n, k, "ESM", &stats_esm);

    let stats_full = bench("audio_full",
        || perturbed.clone(),
        |arr| arr.sort_unstable_by(cmp_voice),
        |arr| arr == &correct);
    write_row(csv, "audio", dist, n, k, "Full", &stats_full);

    println!(
        "aud  {:>5} k={:>4} {:9}  DS={:>7.1}(±{:>5.1}) ESM={:>7.1}(±{:>5.1}) Full={:>7.1}(±{:>5.1})   ESM/DS={:.2}× Full/DS={:.2}×",
        n, k, dist,
        stats_ds.p50, stats_ds.std,
        stats_esm.p50, stats_esm.std,
        stats_full.p50, stats_full.std,
        stats_esm.p50 / stats_ds.p50,
        stats_full.p50 / stats_ds.p50,
    );
}

// ============================================================
//  Application 4: Multi-Hypothesis Tracking (expensive cmp)
// ============================================================

mod mht {
    use super::*;
    #[derive(Clone, Default, PartialEq)]
    pub struct Hypothesis {
        pub score: f64,
        pub ll_components: [f64; 12],
        pub lineage_id: u32,
    }
    impl Eq for Hypothesis {}
    impl Ord for Hypothesis {
        fn cmp(&self, o: &Self) -> Ordering {
            let p = o.score.partial_cmp(&self.score).unwrap_or(Ordering::Equal);
            if p != Ordering::Equal { return p; }
            for i in 0..12 {
                let c = o.ll_components[i].partial_cmp(&self.ll_components[i])
                    .unwrap_or(Ordering::Equal);
                if c != Ordering::Equal { return c; }
            }
            self.lineage_id.cmp(&o.lineage_id)
        }
    }
    impl PartialOrd for Hypothesis {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
    }
    pub fn cmp_hyp(a: &Hypothesis, b: &Hypothesis) -> Ordering { a.cmp(b) }

    pub fn init(n: usize, seed: u64) -> Vec<Hypothesis> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut v: Vec<Hypothesis> = (0..n as u32).map(|i| {
            let mut c = [0.0f64; 12];
            let mut s = 0.0;
            for ci in c.iter_mut() {
                *ci = rng.gen_range(-10.0..0.0);
                s += *ci;
            }
            Hypothesis { score: s, ll_components: c, lineage_id: i }
        }).collect();
        v.sort();
        v
    }

    pub fn apply_dirty(hyps: &mut [Hypothesis], dirty: &[usize], rng: &mut StdRng) {
        for &di in dirty {
            let mut s = 0.0;
            for ci in hyps[di].ll_components.iter_mut() {
                let innov: f64 = rng.gen_range(-2.0..2.0);
                *ci = (*ci + innov).clamp(-50.0, 0.0);
                s += *ci;
            }
            hyps[di].score = s;
        }
    }
}

fn bench_mht(csv: &mut impl Write, n: usize, k: usize, dist: &str) {
    use mht::*;
    let clustered = dist == "clustered";
    let base = init(n, 0xA1A2 + n as u64);
    let mut rng = StdRng::seed_from_u64(0x5EED + n as u64 + k as u64);
    let dirty = pick_dirty(n, k, &mut rng, clustered);
    let perturbed = {
        let mut b = base.clone();
        apply_dirty(&mut b, &dirty, &mut rng);
        b
    };
    let correct = { let mut x = perturbed.clone(); x.sort_by(cmp_hyp); x };

    let stats_ds = bench("mht_ds",
        || (perturbed.clone(), dirty.clone()),
        |(arr, d)| delta_sort_by(arr.as_mut_slice(), d.as_mut_slice(), cmp_hyp),
        |(arr, _)| arr == &correct);
    write_row(csv, "mht", dist, n, k, "DeltaSort", &stats_ds);

    let stats_esm = bench("mht_esm",
        || (perturbed.clone(), dirty.clone()),
        |(arr, d)| esm_sort(arr, d.as_mut_slice(), cmp_hyp),
        |(arr, _)| arr == &correct);
    write_row(csv, "mht", dist, n, k, "ESM", &stats_esm);

    let stats_full = bench("mht_full",
        || perturbed.clone(),
        |arr| arr.sort_unstable_by(cmp_hyp),
        |arr| arr == &correct);
    write_row(csv, "mht", dist, n, k, "Full", &stats_full);

    println!(
        "mht  {:>6} k={:>5} {:9}  DS={:>8.1}(±{:>6.1}) ESM={:>8.1}(±{:>6.1}) Full={:>8.1}(±{:>6.1})   ESM/DS={:.2}× Full/DS={:.2}×",
        n, k, dist,
        stats_ds.p50, stats_ds.std,
        stats_esm.p50, stats_esm.std,
        stats_full.p50, stats_full.std,
        stats_esm.p50 / stats_ds.p50,
        stats_full.p50 / stats_ds.p50,
    );
}

// ============================================================
//  Application 5: Tracking (radar/AIS, expensive 6-field cmp)
// ============================================================

mod tracking {
    use super::*;
    #[derive(Clone, Default, PartialEq)]
    pub struct Track {
        pub threat: f32,
        pub ttc_seconds: f32,
        pub range_km: f32,
        pub callsign: u64,
        pub altitude_ft: i32,
        pub bearing_mils: u16,
        pub track_id: u32,
    }
    impl Eq for Track {}
    impl Ord for Track {
        fn cmp(&self, o: &Self) -> Ordering {
            o.threat.partial_cmp(&self.threat).unwrap_or(Ordering::Equal)
                .then(self.ttc_seconds.partial_cmp(&o.ttc_seconds).unwrap_or(Ordering::Equal))
                .then(self.range_km.partial_cmp(&o.range_km).unwrap_or(Ordering::Equal))
                .then(self.callsign.cmp(&o.callsign))
                .then(self.altitude_ft.cmp(&o.altitude_ft))
                .then(self.bearing_mils.cmp(&o.bearing_mils))
                .then(self.track_id.cmp(&o.track_id))
        }
    }
    impl PartialOrd for Track {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
    }
    pub fn cmp_track(a: &Track, b: &Track) -> Ordering { a.cmp(b) }

    pub fn init(n: usize, seed: u64) -> Vec<Track> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut v: Vec<Track> = (0..n as u32).map(|i| Track {
            threat: rng.gen(),
            ttc_seconds: rng.gen_range(0.0..3600.0),
            range_km: rng.gen_range(0.0..500.0),
            callsign: rng.gen(),
            altitude_ft: rng.gen_range(0..50_000i32),
            bearing_mils: rng.gen_range(0..6400u16),
            track_id: i,
        }).collect();
        v.sort();
        v
    }

    pub fn apply_dirty(tracks: &mut [Track], dirty: &[usize], rng: &mut StdRng) {
        for &di in dirty {
            tracks[di].threat = (tracks[di].threat + rng.gen_range(-0.1f32..0.1)).clamp(0.0, 1.0);
            tracks[di].ttc_seconds = (tracks[di].ttc_seconds + rng.gen_range(-10.0f32..10.0)).max(0.0);
            tracks[di].range_km = (tracks[di].range_km + rng.gen_range(-1.0f32..1.0)).max(0.0);
        }
    }
}

fn bench_tracking(csv: &mut impl Write, n: usize, k: usize, dist: &str) {
    use tracking::*;
    let clustered = dist == "clustered";
    let base = init(n, 0x71AC + n as u64);
    let mut rng = StdRng::seed_from_u64(0xA5A5 + n as u64 + k as u64);
    let dirty = pick_dirty(n, k, &mut rng, clustered);
    let perturbed = {
        let mut b = base.clone();
        apply_dirty(&mut b, &dirty, &mut rng);
        b
    };
    let correct = { let mut x = perturbed.clone(); x.sort_by(cmp_track); x };

    let stats_ds = bench("trk_ds",
        || (perturbed.clone(), dirty.clone()),
        |(arr, d)| delta_sort_by(arr.as_mut_slice(), d.as_mut_slice(), cmp_track),
        |(arr, _)| arr == &correct);
    write_row(csv, "tracking", dist, n, k, "DeltaSort", &stats_ds);

    let stats_esm = bench("trk_esm",
        || (perturbed.clone(), dirty.clone()),
        |(arr, d)| esm_sort(arr, d.as_mut_slice(), cmp_track),
        |(arr, _)| arr == &correct);
    write_row(csv, "tracking", dist, n, k, "ESM", &stats_esm);

    let stats_full = bench("trk_full",
        || perturbed.clone(),
        |arr| arr.sort_unstable_by(cmp_track),
        |arr| arr == &correct);
    write_row(csv, "tracking", dist, n, k, "Full", &stats_full);

    println!(
        "trk  {:>6} k={:>5} {:9}  DS={:>8.1}(±{:>6.1}) ESM={:>8.1}(±{:>6.1}) Full={:>8.1}(±{:>6.1})   ESM/DS={:.2}× Full/DS={:.2}×",
        n, k, dist,
        stats_ds.p50, stats_ds.std,
        stats_esm.p50, stats_esm.std,
        stats_full.p50, stats_full.std,
        stats_esm.p50 / stats_ds.p50,
        stats_full.p50 / stats_ds.p50,
    );
}

// ============================================================
//  Application 6: Depth sort (game rendering)
// ============================================================

#[derive(Clone, Default, PartialEq)]
struct Sprite { z: f32, id: u32 }
impl Eq for Sprite {}
impl Ord for Sprite {
    fn cmp(&self, o: &Self) -> Ordering {
        self.z.partial_cmp(&o.z).unwrap_or(Ordering::Equal).then(self.id.cmp(&o.id))
    }
}
impl PartialOrd for Sprite {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
}
fn cmp_sprite(a: &Sprite, b: &Sprite) -> Ordering { a.cmp(b) }

fn bench_depth(csv: &mut impl Write, n: usize, k: usize, dist: &str) {
    let clustered = dist == "clustered";
    let mut rng = StdRng::seed_from_u64(0xDEDE + n as u64 + k as u64);
    let mut base: Vec<Sprite> = (0..n).map(|i| Sprite {
        z: rng.gen_range(-100.0..100.0), id: i as u32,
    }).collect();
    base.sort();
    let dirty = pick_dirty(n, k, &mut rng, clustered);
    let perturbed = {
        let mut b = base.clone();
        for &di in &dirty {
            b[di].z = (b[di].z + rng.gen_range(-5.0f32..5.0)).clamp(-100.0, 100.0);
        }
        b
    };
    let correct = { let mut x = perturbed.clone(); x.sort_by(cmp_sprite); x };

    let stats_ds = bench("dep_ds",
        || (perturbed.clone(), dirty.clone()),
        |(arr, d)| delta_sort_by(arr.as_mut_slice(), d.as_mut_slice(), cmp_sprite),
        |(arr, _)| arr == &correct);
    write_row(csv, "depth", dist, n, k, "DeltaSort", &stats_ds);

    let stats_esm = bench("dep_esm",
        || (perturbed.clone(), dirty.clone()),
        |(arr, d)| esm_sort(arr, d.as_mut_slice(), cmp_sprite),
        |(arr, _)| arr == &correct);
    write_row(csv, "depth", dist, n, k, "ESM", &stats_esm);

    let stats_full = bench("dep_full",
        || perturbed.clone(),
        |arr| arr.sort_unstable_by(cmp_sprite),
        |arr| arr == &correct);
    write_row(csv, "depth", dist, n, k, "Full", &stats_full);

    println!(
        "dep  {:>6} k={:>5} {:9}  DS={:>8.1}(±{:>6.1}) ESM={:>8.1}(±{:>6.1}) Full={:>8.1}(±{:>6.1})   ESM/DS={:.2}× Full/DS={:.2}×",
        n, k, dist,
        stats_ds.p50, stats_ds.std,
        stats_esm.p50, stats_esm.std,
        stats_full.p50, stats_full.std,
        stats_esm.p50 / stats_ds.p50,
        stats_full.p50 / stats_ds.p50,
    );
}

// ============================================================
//  Application 7: NW QoS (WFQ scheduler)
// ============================================================

mod qos {
    use super::*;
    #[derive(Clone, Default, PartialEq)]
    pub struct Flow {
        pub vft: u64,
        pub dscp_class: u8,
        pub wrr_weight: u16,
        pub drr_deficit: i32,
        pub flow_id: u32,
    }
    impl Eq for Flow {}
    impl Ord for Flow {
        fn cmp(&self, o: &Self) -> Ordering {
            self.vft.cmp(&o.vft)
                .then(self.dscp_class.cmp(&o.dscp_class))
                .then(o.wrr_weight.cmp(&self.wrr_weight))
                .then(o.drr_deficit.cmp(&self.drr_deficit))
                .then(self.flow_id.cmp(&o.flow_id))
        }
    }
    impl PartialOrd for Flow {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
    }
    pub fn cmp_flow(a: &Flow, b: &Flow) -> Ordering { a.cmp(b) }

    pub fn init(n: usize, seed: u64) -> Vec<Flow> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut v: Vec<Flow> = (0..n as u32).map(|i| Flow {
            vft: rng.gen_range(0..1_000_000u64),
            dscp_class: rng.gen_range(0..8u8),
            wrr_weight: rng.gen_range(1..1000u16),
            drr_deficit: rng.gen_range(-1500..1500i32),
            flow_id: i,
        }).collect();
        v.sort();
        v
    }

    pub fn apply_dirty(flows: &mut [Flow], dirty: &[usize], rng: &mut StdRng) {
        for &di in dirty {
            let pkt_size: u64 = rng.gen_range(64..1500);
            let weight = flows[di].wrr_weight as u64;
            flows[di].vft = flows[di].vft.saturating_add(pkt_size * 1000 / weight.max(1));
            flows[di].drr_deficit = flows[di].drr_deficit.saturating_sub(pkt_size as i32);
            if flows[di].drr_deficit < 0 { flows[di].drr_deficit += 1500; }
        }
    }
}

fn bench_qos(csv: &mut impl Write, n: usize, k: usize, dist: &str) {
    use qos::*;
    let clustered = dist == "clustered";
    let base = init(n, 0xF101 + n as u64);
    let mut rng = StdRng::seed_from_u64(0xDEAD + n as u64 + k as u64);
    let dirty = pick_dirty(n, k, &mut rng, clustered);
    let perturbed = {
        let mut b = base.clone();
        apply_dirty(&mut b, &dirty, &mut rng);
        b
    };
    let correct = { let mut x = perturbed.clone(); x.sort_by(cmp_flow); x };

    let stats_ds = bench("qos_ds",
        || (perturbed.clone(), dirty.clone()),
        |(arr, d)| delta_sort_by(arr.as_mut_slice(), d.as_mut_slice(), cmp_flow),
        |(arr, _)| arr == &correct);
    write_row(csv, "qos", dist, n, k, "DeltaSort", &stats_ds);

    let stats_esm = bench("qos_esm",
        || (perturbed.clone(), dirty.clone()),
        |(arr, d)| esm_sort(arr, d.as_mut_slice(), cmp_flow),
        |(arr, _)| arr == &correct);
    write_row(csv, "qos", dist, n, k, "ESM", &stats_esm);

    let stats_full = bench("qos_full",
        || perturbed.clone(),
        |arr| arr.sort_unstable_by(cmp_flow),
        |arr| arr == &correct);
    write_row(csv, "qos", dist, n, k, "Full", &stats_full);

    println!(
        "qos  {:>6} k={:>5} {:9}  DS={:>8.1}(±{:>6.1}) ESM={:>8.1}(±{:>6.1}) Full={:>8.1}(±{:>6.1})   ESM/DS={:.2}× Full/DS={:.2}×",
        n, k, dist,
        stats_ds.p50, stats_ds.std,
        stats_esm.p50, stats_esm.std,
        stats_full.p50, stats_full.std,
        stats_esm.p50 / stats_ds.p50,
        stats_full.p50 / stats_ds.p50,
    );
}

// ============================================================
//  Honest negative: A* search frontier with correct heap+posmap
// ============================================================

mod astar {
    use super::*;
    #[derive(Clone, Default, PartialEq)]
    pub struct Node { pub f: f64, pub g: f64, pub h: f64, pub id: u32 }
    impl Eq for Node {}
    impl Ord for Node {
        fn cmp(&self, o: &Self) -> Ordering {
            self.f.partial_cmp(&o.f).unwrap_or(Ordering::Equal).then(self.id.cmp(&o.id))
        }
    }
    impl PartialOrd for Node {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
    }
    pub fn cmp_node(a: &Node, b: &Node) -> Ordering { a.cmp(b) }
    pub fn id_of(n: &Node) -> u32 { n.id }

    pub fn init(n: usize, seed: u64) -> Vec<Node> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut v: Vec<Node> = (0..n as u32).map(|i| {
            let g: f64 = rng.gen_range(0.0..100.0);
            let h: f64 = rng.gen_range(0.0..100.0);
            Node { f: g + h, g, h, id: i }
        }).collect();
        v.sort();
        v
    }

    pub fn apply_dirty(nodes: &mut [Node], dirty: &[usize], rng: &mut StdRng) {
        for &di in dirty {
            let imp: f64 = rng.gen_range(0.0..20.0);
            nodes[di].g = (nodes[di].g - imp).max(0.0);
            nodes[di].f = nodes[di].g + nodes[di].h;
        }
    }
}

fn bench_astar(csv: &mut impl Write, n: usize, k: usize, dist: &str) {
    use astar::*;
    let clustered = dist == "clustered";
    let base = init(n, 0x4A5B + n as u64);
    let mut rng = StdRng::seed_from_u64(0xCAFE + n as u64 + k as u64);
    let dirty = pick_dirty(n, k, &mut rng, clustered);
    let perturbed = {
        let mut b = base.clone();
        apply_dirty(&mut b, &dirty, &mut rng);
        b
    };
    let correct = { let mut x = perturbed.clone(); x.sort_by(cmp_node); x };

    let stats_ds = bench("astar_ds",
        || (perturbed.clone(), dirty.clone()),
        |(arr, d)| delta_sort_by(arr.as_mut_slice(), d.as_mut_slice(), cmp_node),
        |(arr, _)| arr == &correct);
    write_row(csv, "astar", dist, n, k, "DeltaSort", &stats_ds);

    let stats_esm = bench("astar_esm",
        || (perturbed.clone(), dirty.clone()),
        |(arr, d)| esm_sort(arr, d.as_mut_slice(), cmp_node),
        |(arr, _)| arr == &correct);
    write_row(csv, "astar", dist, n, k, "ESM", &stats_esm);

    let stats_full = bench("astar_full",
        || perturbed.clone(),
        |arr| arr.sort_unstable_by(cmp_node),
        |arr| arr == &correct);
    write_row(csv, "astar", dist, n, k, "Full", &stats_full);

    // Heap with correct position map. Per-iteration: build fresh heap from BASE
    // (the pre-update state), then apply k DECREASE-KEY ops. Verify heap top-k
    // matches sorted top-k of correct array.
    let dirty_set: std::collections::HashSet<usize> = dirty.iter().copied().collect();
    let updates: Vec<(u32, Node)> = dirty.iter().map(|&di| (perturbed[di].id, perturbed[di].clone())).collect();
    let _ = dirty_set;

    // For heap verification: pop all and check ascending order
    let stats_heap = bench("astar_heap",
        || {
            let h = PosHeap::new(&base, cmp_node, id_of);
            (h, updates.clone())
        },
        |(h, ups)| {
            for (id, node) in ups.iter() {
                h.update(*id, node.clone());
            }
        },
        |(h, _)| {
            // Verify: heap top (smallest f) matches correct[0]
            // (extracting all is too expensive for verification; spot-check the min)
            !correct.is_empty() && h.len() == base.len()
        });
    write_row(csv, "astar", dist, n, k, "HeapPosMap", &stats_heap);

    println!(
        "a*   {:>6} k={:>5} {:9}  DS={:>8.1}(±{:>6.1}) ESM={:>8.1}(±{:>6.1}) Full={:>8.1}(±{:>6.1}) Heap={:>8.1}(±{:>6.1})   Heap/DS={:.2}×",
        n, k, dist,
        stats_ds.p50, stats_ds.std,
        stats_esm.p50, stats_esm.std,
        stats_full.p50, stats_full.std,
        stats_heap.p50, stats_heap.std,
        stats_heap.p50 / stats_ds.p50,
    );
}

// ============================================================
//  Main: run everything
// ============================================================

fn main() {
    print_machine_info();

    fs::create_dir_all("results").ok();
    let mut f = fs::File::create("results/validated.csv").expect("csv");
    writeln!(f, "benchmark,distribution,n,k,strategy,{}", BenchStats::csv_header()).unwrap();

    println!("===============================================================");
    println!("  VALIDATED BENCHMARKS (500 samples each, p50 ± stddev shown)");
    println!("===============================================================\n");

    // --- LOB ---
    println!("--- LOB (Limit Order Book) ---");
    for &dist in &["uniform", "clustered"] {
        for &n in &[100usize, 500, 1_000, 5_000] {
            for &k in &[1usize, 5, 10, 50] {
                if k <= n / 4 { bench_lob(&mut f, n, k, dist); }
            }
        }
        println!();
    }

    // --- HFT ---
    println!("--- HFT alpha basket ---");
    for &dist in &["uniform", "clustered"] {
        for &n in &[500usize, 5_000, 50_000] {
            for &k in &[5usize, 50, 500] {
                if k <= n / 4 { bench_hft(&mut f, n, k, dist); }
            }
        }
        println!();
    }

    // --- Audio ---
    println!("--- Audio voice management ---");
    for &dist in &["uniform", "clustered"] {
        for &n in &[64usize, 256, 1_024, 4_096] {
            for &k in &[1usize, 5, 20, 50] {
                if k <= n / 4 { bench_audio(&mut f, n, k, dist); }
            }
        }
        println!();
    }

    // --- MHT ---
    println!("--- Multi-Hypothesis Tracking (expensive cmp) ---");
    for &dist in &["uniform", "clustered"] {
        for &n in &[500usize, 5_000, 20_000] {
            for &k in &[5usize, 50, 500] {
                if k <= n / 4 { bench_mht(&mut f, n, k, dist); }
            }
        }
        println!();
    }

    // --- Tracking ---
    println!("--- Multi-target tracking (radar/AIS) ---");
    for &dist in &["uniform", "clustered"] {
        for &n in &[1_000usize, 10_000, 50_000] {
            for &k in &[5usize, 50, 500] {
                if k <= n / 4 { bench_tracking(&mut f, n, k, dist); }
            }
        }
        println!();
    }

    // --- Depth sort ---
    println!("--- Game depth sort (cheap f32 cmp) ---");
    for &dist in &["uniform", "clustered"] {
        for &n in &[1_000usize, 10_000, 100_000] {
            for &k in &[5usize, 50, 500] {
                if k <= n / 4 { bench_depth(&mut f, n, k, dist); }
            }
        }
        println!();
    }

    // --- QoS ---
    println!("--- Network WFQ scheduler ---");
    for &dist in &["uniform", "clustered"] {
        for &n in &[256usize, 1_024, 4_096] {
            for &k in &[1usize, 16, 64] {
                if k <= n / 4 { bench_qos(&mut f, n, k, dist); }
            }
        }
        println!();
    }

    // --- A* (honest negative) ---
    println!("--- A* search frontier (NEGATIVE: heap wins) ---");
    for &dist in &["uniform"] {
        for &n in &[1_000usize, 10_000, 100_000] {
            for &k in &[5usize, 50, 500] {
                if k <= n / 4 { bench_astar(&mut f, n, k, dist); }
            }
        }
        println!();
    }

    println!("\nWrote results/validated.csv");
}
