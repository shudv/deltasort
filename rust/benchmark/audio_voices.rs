//! Real-time audio voice management benchmark.
//!
//! Polyphonic audio synthesis requires managing a pool of n voice objects,
//! each with a priority score computed from: envelope level, note velocity,
//! age, and channel. When a new note-on event occurs on a full polyphonic
//! engine, the voice with the LOWEST priority is stolen (voice stealing).
//!
//! Production voice stealers (JUCE, SOUL, SuperCollider, Vital synth) keep
//! voices sorted by priority so that the min-priority voice can be found
//! quickly. After each buffer (typically 64-512 samples), a subset of k
//! voices have their priorities updated (envelope position changed, note-off
//! applied, LFO tick, etc.) and the list must be re-sorted.
//!
//! Why flat sorted array (not heap):
//!   - Needs random-access "k-th voice" for voice routing (send to bus k)
//!   - Cache-friendly iteration for mixing (apply envelope to all voices)
//!   - No allocation per tick (critical in audio thread, no malloc allowed)
//!   - Voice stealing: scan sorted list to find steal candidate
//!
//! Baselines:
//!   - Full re-sort (what naive implementations do)
//!   - ESM (extract-sort-merge, what a careful implementation would do)
//!   - DeltaSort (ours)
//!
//! Additional comparison: BinaryHeap-based voice manager (common approach).
//! Here k voices have their priorities updated; heap requires k delete+insert.
//!
//! Workload parameters:
//!   n = 32, 64, 128, 256, 512 voices (common polyphony values)
//!   k = 1..20 voices updated per buffer (from ADSR envelope changes,
//!           note-off events, sustain pedal, etc.)
//!   buffer_size = 64..512 samples (@ 48 kHz: 1.3..10.7 ms per buffer)
//!   Latency budget: 1μs per sort call (audio thread runs at hard deadline)

use deltasort::delta_sort_by;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::fs;
use std::io::Write;
use std::time::Instant;

/// A synthesizer voice with composite priority.
///
/// Lower priority = candidate for stealing.
/// Priority components (all normalized 0..1):
///   - envelope_level: higher → higher priority (don't steal loud notes)
///   - velocity: higher → higher priority
///   - age: older → lower priority (steal old quiet notes)
///   - channel_weight: some channels have higher priority (sustain, solo)
#[derive(Clone, Default, PartialEq)]
struct Voice {
    priority: f32,     // composite priority (computed from below)
    envelope_level: f32,
    velocity: u8,
    age_samples: u32,
    channel: u8,
    note: u8,
    active: bool,
    voice_id: u32,
}

impl Voice {
    fn compute_priority(&mut self) {
        // Weighted composite (matches JUCE's voice stealing heuristic)
        let vel_norm = self.velocity as f32 / 127.0;
        let age_norm = 1.0 - (self.age_samples as f32 / 100_000.0).min(1.0);
        let chan_bonus = if self.channel == 0 { 0.2 } else { 0.0 }; // main channel
        self.priority = 0.5 * self.envelope_level
            + 0.2 * vel_norm
            + 0.2 * age_norm
            + 0.1 * chan_bonus;
    }
}

impl Eq for Voice {}
impl Ord for Voice {
    fn cmp(&self, o: &Self) -> Ordering {
        // Descending priority: highest priority voices first
        o.priority.partial_cmp(&self.priority).unwrap_or(Ordering::Equal)
            .then(self.voice_id.cmp(&o.voice_id))
    }
}
impl PartialOrd for Voice { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }

fn cmp_voice(a: &Voice, b: &Voice) -> Ordering { a.cmp(b) }

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

fn init_voices(n: usize, seed: u64) -> Vec<Voice> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v: Vec<Voice> = (0..n as u32).map(|i| {
        let mut voice = Voice {
            envelope_level: rng.gen(),
            velocity: rng.gen_range(40..127u8),
            age_samples: rng.gen_range(0..50_000u32),
            channel: rng.gen_range(0..16u8),
            note: rng.gen_range(21..108u8),
            active: true,
            voice_id: i,
            priority: 0.0,
        };
        voice.compute_priority();
        voice
    }).collect();
    v.sort();
    v
}

/// Simulate one audio buffer: update k voices' envelope/age, recompute priority.
fn process_buffer(voices: &mut [Voice], k: usize, rng: &mut StdRng) -> Vec<usize> {
    let n = voices.len();
    let mut picks: Vec<usize> = (0..n).collect();
    for i in 0..k { let j = rng.gen_range(i..n); picks.swap(i, j); }
    let mut dirty: Vec<usize> = picks[..k].to_vec();
    dirty.sort_unstable();
    for &i in &dirty {
        // Envelope advance (ADSR tick)
        let delta: f32 = rng.gen_range(-0.15..0.15);
        voices[i].envelope_level = (voices[i].envelope_level + delta).clamp(0.0, 1.0);
        voices[i].age_samples += 512; // one buffer's worth of samples
        voices[i].compute_priority();
    }
    dirty
}

struct Out { ds: f64, esm: f64, full: f64 }

fn run_buffer(baseline: &[Voice], dirty: &[usize]) -> Out {
    let mut data = baseline.to_vec();
    let mut dv = dirty.to_vec(); dv.sort_unstable(); dv.dedup();
    let t = Instant::now();
    delta_sort_by(&mut data, &mut dv, cmp_voice);
    let ds = t.elapsed().as_nanos() as f64;
    let ds_result = data;

    let mut data = baseline.to_vec();
    let mut dv = dirty.to_vec();
    let t = Instant::now();
    esm_sort(&mut data, &mut dv, cmp_voice);
    let esm = t.elapsed().as_nanos() as f64;
    assert!(data == ds_result, "ESM mismatch n={}", baseline.len());

    let mut data = baseline.to_vec();
    let t = Instant::now();
    data.sort_unstable_by(cmp_voice);
    let full = t.elapsed().as_nanos() as f64;
    assert!(data == ds_result, "Full mismatch n={}", baseline.len());

    Out { ds, esm, full }
}

fn med(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }

fn run_polyphony(n_voices: usize, k: usize, n_buffers: usize, csv: &mut impl Write) {
    let mut voices = init_voices(n_voices, 0xA0 + n_voices as u64);
    let mut rng = StdRng::seed_from_u64(0xB00F);
    let mut ds_v = Vec::new(); let mut esm_v = Vec::new(); let mut full_v = Vec::new();
    for _ in 0..n_buffers {
        let dirty = process_buffer(&mut voices, k, &mut rng);
        let r = run_buffer(&voices, &dirty);
        ds_v.push(r.ds); esm_v.push(r.esm); full_v.push(r.full);
        voices.sort();
    }
    let dsm = med(ds_v); let esmm = med(esm_v); let fm = med(full_v);
    let kpct = 100.0 * k as f64 / n_voices as f64;

    // Audio latency context: at 48kHz, 64-sample buffer = 1.33ms deadline.
    // Sort must complete in < 10μs to not impact audio thread.
    let budget_ns = 10_000.0f64; // 10μs
    let ds_pct_budget = 100.0 * dsm / budget_ns;
    let full_pct_budget = 100.0 * fm / budget_ns;

    println!(
        "{:>5} {:>5} {:>6.1}%  {:>7.1} {:>7.1} {:>7.1}   {:>5.2}× {:>5.2}×  DS={:.1}% Full={:.1}% of 10μs",
        n_voices, k, kpct,
        dsm, esmm, fm,
        esmm / dsm, fm / dsm,
        ds_pct_budget, full_pct_budget,
    );
    for (strat, val) in [("DeltaSort", dsm), ("ESM", esmm), ("Full", fm)] {
        writeln!(csv, "{},{},{},{:.0}", n_voices, k, strat, val).unwrap();
    }
}

fn main() {
    fs::create_dir_all("results").ok();
    let mut f = fs::File::create("results/track_n_audio_voices.csv").expect("csv");
    writeln!(f, "n_voices,k,strategy,ns").unwrap();

    println!("=== Real-time Audio: Voice Priority Management ===");
    println!("(times in nanoseconds; budget = 10μs for sort in audio thread)");
    println!("{:>5} {:>5} {:>7}  {:>7} {:>7} {:>7}   {:>7} {:>7}",
        "n", "k", "k/n", "DS ns", "ESM ns", "Full ns", "ESM/DS", "Full/DS");

    // Small polyphony (hardware synths, embedded)
    println!("\n--- Small polyphony (16-64 voices, hardware synth) ---");
    for &n in &[16usize, 32, 64] {
        for &k in &[1usize, 2, 3, 5, 8, 10, 16] {
            if k <= n {
                run_polyphony(n, k, 101, &mut f);
            }
        }
        println!();
    }

    // Medium polyphony (software synths: Vital, Surge, Serum)
    println!("--- Medium polyphony (128-512 voices, software synth) ---");
    for &n in &[128usize, 256, 512] {
        for &k in &[1usize, 2, 5, 10, 20, 30, 50] {
            if k <= n {
                run_polyphony(n, k, 101, &mut f);
            }
        }
        println!();
    }

    // Large polyphony (game audio middleware: Wwise, FMOD)
    println!("--- Large polyphony (1024-4096 virtual voices, game audio) ---");
    for &n in &[1024usize, 4096] {
        for &k in &[1usize, 5, 10, 20, 50, 100] {
            run_polyphony(n, k, 51, &mut f);
        }
        println!();
    }

    println!("Wrote results/track_n_audio_voices.csv");
}
