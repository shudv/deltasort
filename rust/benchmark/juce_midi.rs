//! JUCE MidiMessageSequence — production-faithful editing benchmark.
//!
//! JUCE (https://juce.com) is the dominant open-source C++ audio framework,
//! used by Tracktion Waveform, FL Studio plugin SDKs, Cubase tooling, ROLI,
//! and thousands of commercial audio products. Its `MidiMessageSequence`
//! class stores MIDI events in a persistent timestamp-sorted
//! `OwnedArray<MidiEventHolder*>` and exposes `sort()` which calls
//! `std::stable_sort` with a `getTimeStamp() < getTimeStamp()` comparator.
//!
//! Source: JUCE/modules/juce_audio_basics/midi/juce_MidiMessageSequence.cpp
//!
//! When a DAW user edits a selection of k notes (drag in time, quantize,
//! transpose, etc.), the dirty indices are known exactly from the UI. This
//! is the canonical DeltaSort workload.
//!
//! Data layout matches JUCE faithfully:
//!   - `Vec<Box<MidiMessage>>`  ≡  `OwnedArray<MidiEventHolder*>`
//!   - Comparator dereferences a Box to compare a `f64` timestamp
//!     (matches JUCE's pointer-indirect double compare)
//!   - Sort is STABLE (Rust's `sort_by` is Timsort-style stable, matching
//!     C++'s `std::stable_sort`)
//!
//! Operations modeled after Tracktion Engine / Ardour / Reaper edit patterns:
//!   - selection_drag(k):   contiguous time range shift (most common)
//!   - quantize(k):         scattered selection, snap-to-grid
//!   - bulk_paste(k):       append k events from clipboard, re-sort
//!
//! Output: `results/juce_midi.csv`

use deltasort::delta_sort_by;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::fs;
use std::io::Write;
use std::time::Instant;

#[path = "common.rs"]
mod common;
use common::{bench_with_config, esm_sort, BenchStats};

// =============================================================================
//  JUCE-faithful MIDI event layout
// =============================================================================
//
// JUCE's MidiMessage is ~32 bytes:
//   double timeStamp;          // 8 bytes
//   union { uint8 preallocatedData[14]; uint8* allocatedData; };
//   uint32 size;               // packed metadata
// We model the same: 8-byte timestamp + 16 bytes of MIDI data + 8 bytes meta = 32B.

#[derive(Clone)]
struct MidiMessage {
    time_stamp: f64,    // matches JUCE MidiMessage::timeStamp
    data: [u8; 16],     // matches preallocatedData + size
    note_off_idx: i32,  // matches MidiEventHolder::noteOffObject (we use index not ptr)
    id: u32,            // stable identifier for verification
}

impl Default for MidiMessage {
    fn default() -> Self {
        MidiMessage { time_stamp: f64::NAN, data: [0; 16], note_off_idx: -1, id: u32::MAX }
    }
}

// JUCE comparator: `a->message.getTimeStamp() < b->message.getTimeStamp()`
// In Rust we use `partial_cmp` because timestamps are f64 (JUCE also assumes finite).
#[inline]
fn cmp_msg(a: &Box<MidiMessage>, b: &Box<MidiMessage>) -> Ordering {
    a.time_stamp.partial_cmp(&b.time_stamp).unwrap_or(Ordering::Equal)
}

// =============================================================================
//  Workload generator (synthetic Type-1 MIDI track)
// =============================================================================

/// Generate a sorted sequence of `n` MIDI events spanning a realistic 4-minute
/// track. Note timestamps are in PPQ ticks (480 PPQ * 120 BPM * 240 s).
fn generate_track(n: usize, seed: u64) -> Vec<Box<MidiMessage>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let total_ticks: f64 = (n as f64) * 240.0;
    let mut events: Vec<Box<MidiMessage>> = (0..n)
        .map(|i| {
            let ts = rng.gen::<f64>() * total_ticks;
            let mut data = [0u8; 16];
            data[0] = 0x90 | (rng.gen::<u8>() & 0x0F);     // note-on, channel
            data[1] = 24 + (rng.gen::<u8>() % 72);          // note
            data[2] = 32 + (rng.gen::<u8>() % 96);          // velocity
            data[3] = 3;                                    // size
            Box::new(MidiMessage { time_stamp: ts, data, note_off_idx: -1, id: i as u32 })
        })
        .collect();
    events.sort_by(cmp_msg);
    events
}

// =============================================================================
//  Realistic edit operations (return dirty indices)
// =============================================================================

/// Selection drag: user selects a contiguous time range of k events and
/// shifts them by `delta` ticks. Very common DAW operation.
/// Returns (dirty_indices, edited_events).
fn op_selection_drag(
    events: &mut Vec<Box<MidiMessage>>, k: usize, delta: f64, rng: &mut StdRng,
) -> Vec<usize> {
    let n = events.len();
    let start = rng.gen_range(0..(n - k).max(1));
    let dirty: Vec<usize> = (start..start + k).collect();
    for &i in &dirty {
        events[i].time_stamp += delta;
    }
    dirty
}

/// Quantize: user selects k events scattered through the track and snaps
/// their timestamps to a beat grid (e.g., 16th notes).
fn op_quantize(
    events: &mut Vec<Box<MidiMessage>>, k: usize, grid: f64, rng: &mut StdRng,
) -> Vec<usize> {
    let n = events.len();
    let mut idx_set = std::collections::HashSet::with_capacity(k);
    while idx_set.len() < k {
        idx_set.insert(rng.gen_range(0..n));
    }
    let mut dirty: Vec<usize> = idx_set.into_iter().collect();
    dirty.sort_unstable();
    for &i in &dirty {
        let t = events[i].time_stamp;
        events[i].time_stamp = (t / grid).round() * grid;
    }
    dirty
}

/// Bulk paste: append k new events to the end, then sort. Dirty indices are
/// the appended positions. (Models user paste-clipboard or import.)
fn op_bulk_paste(
    events: &mut Vec<Box<MidiMessage>>, k: usize, rng: &mut StdRng,
) -> Vec<usize> {
    let n = events.len();
    let max_ts = events.last().map(|e| e.time_stamp).unwrap_or(0.0);
    for i in 0..k {
        let ts = rng.gen::<f64>() * max_ts;
        let mut data = [0u8; 16];
        data[0] = 0x90;
        data[1] = 60 + (rng.gen::<u8>() % 24);
        data[2] = 80;
        data[3] = 3;
        events.push(Box::new(MidiMessage {
            time_stamp: ts, data,
            note_off_idx: -1,
            id: (n + i) as u32,
        }));
    }
    (n..n + k).collect()
}

// =============================================================================
//  Strategies under test
// =============================================================================

/// JUCE's exact implementation: std::stable_sort on the whole sequence.
fn juce_stock_sort(events: &mut Vec<Box<MidiMessage>>, _dirty: &mut [usize]) {
    events.sort_by(cmp_msg);
}

/// Rust's sort_unstable (pdqsort) — what JUCE *could* use if stability wasn't
/// required. Included to bound the comparison.
fn full_unstable_sort(events: &mut Vec<Box<MidiMessage>>, _dirty: &mut [usize]) {
    events.sort_unstable_by(cmp_msg);
}

/// DeltaSort with known dirty indices (the proposal).
fn deltasort_sort(events: &mut Vec<Box<MidiMessage>>, dirty: &mut [usize]) {
    delta_sort_by(events, dirty, cmp_msg);
}

/// Extract-Sort-Merge baseline.
fn esm_strategy(events: &mut Vec<Box<MidiMessage>>, dirty: &mut [usize]) {
    esm_sort(events, dirty, cmp_msg);
}

// =============================================================================
//  Verification
// =============================================================================

fn is_sorted(events: &[Box<MidiMessage>]) -> bool {
    events.windows(2).all(|w| cmp_msg(&w[0], &w[1]) != Ordering::Greater)
}

// =============================================================================
//  Main experiment
// =============================================================================

#[derive(Clone, Copy)]
enum Op { Drag, Quantize, Paste }

impl Op {
    fn name(&self) -> &'static str {
        match self {
            Op::Drag => "drag",
            Op::Quantize => "quantize",
            Op::Paste => "paste",
        }
    }
}

fn run_cell(
    op: Op, n: usize, k: usize, seed: u64,
    writer: &mut impl Write,
) -> std::io::Result<()> {
    // Drag delta = 1 beat (480 ticks) — typical small move
    // Quantize grid = 120 ticks (16th note)
    // Paste: k new events
    let strategies: [(&str, fn(&mut Vec<Box<MidiMessage>>, &mut [usize])); 4] = [
        ("juce_stable", juce_stock_sort),
        ("unstable",    full_unstable_sort),
        ("esm",         esm_strategy),
        ("delta",       deltasort_sort),
    ];

    for (sname, strategy) in strategies.iter() {
        // setup() produces a fresh edited sequence and its dirty index list
        let setup = || {
            let mut events = generate_track(n, seed);
            let mut op_rng = StdRng::seed_from_u64(seed.wrapping_add(0x9E37_79B9_7F4A_7C15));
            let dirty = match op {
                Op::Drag     => op_selection_drag(&mut events, k, 480.0, &mut op_rng),
                Op::Quantize => op_quantize(&mut events, k, 120.0, &mut op_rng),
                Op::Paste    => op_bulk_paste(&mut events, k, &mut op_rng),
            };
            (events, dirty)
        };
        let bench_fn = |s: &mut (Vec<Box<MidiMessage>>, Vec<usize>)| {
            strategy(&mut s.0, &mut s.1);
        };
        let verify = |s: &(Vec<Box<MidiMessage>>, Vec<usize>)| is_sorted(&s.0);
        let warmup = if n <= 5_000 { 100 } else { 30 };
        let samples = if n <= 5_000 { 500 } else { 200 };
        let stats: BenchStats = bench_with_config(
            "juce-midi", warmup, samples, setup, bench_fn, verify,
        );
        writeln!(
            writer,
            "{},{},{},{},{}",
            op.name(), n, k, sname, stats.to_csv(),
        )?;
        writer.flush()?;
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    fs::create_dir_all("rust/results").ok();
    let out_path = "rust/results/juce_midi.csv";
    let mut writer = std::io::BufWriter::new(fs::File::create(out_path)?);
    writeln!(writer, "op,n,k,strategy,{}", BenchStats::csv_header())?;

    // Realistic sizes:
    //   - n = 1K     small loop / pattern
    //   - n = 5K     verse or section
    //   - n = 20K    full song
    //   - n = 100K   long composition / merged multi-track
    let n_values = [1_000usize, 5_000, 20_000, 100_000];
    // Realistic k for selections:
    //   - k = 1      single-note move
    //   - k = 5      small chord drag
    //   - k = 25     phrase
    //   - k = 100    section
    //   - k = 500    bulk transform
    let k_ratios = [0.001, 0.005, 0.025, 0.10, 0.25];

    let t0 = Instant::now();
    for &op in &[Op::Drag, Op::Quantize, Op::Paste] {
        for &n in &n_values {
            for &r in &k_ratios {
                let k = ((n as f64) * r).round() as usize;
                let k = k.max(1).min(n / 2);
                eprintln!(
                    "[{:>4}s] op={} n={:>6} k={:>5} ({}%)",
                    t0.elapsed().as_secs(),
                    op.name(), n, k, (r * 100.0) as u32,
                );
                run_cell(op, n, k, 0x5EED_0123_u64 ^ (n as u64) ^ ((k as u64) << 16), &mut writer)?;
            }
        }
    }

    eprintln!("Wrote {}", out_path);
    Ok(())
}
