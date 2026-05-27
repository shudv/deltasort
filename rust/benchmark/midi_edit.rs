//! Track 3 — MIDI editing benchmark (Ardour-style sequence semantics).
//!
//! A music sequencer stores MIDI events sorted by `(timestamp, channel, note)`.
//! When a user selects k notes and edits them (drag in time, quantize, paste, etc.),
//! the selected indices are KNOWN exactly — DeltaSort's dirty set is free.
//!
//! Strategies compared:
//!   - BTreeSet:     std::collections::BTreeSet (Ardour's `std::multiset` equivalent).
//!                   Edits: k remove+reinsert. Reads: in-order iter.
//!   - VecFullSort:  sorted Vec + full re-sort after edit. Reads: index/iter.
//!   - VecDeltaSort: sorted Vec + DeltaSort with known dirty set. Reads: index/iter.
//!
//! Edit operations modeled:
//!   - single_drag:        k = 1, retime one note
//!   - selection_drag:     k = 2..200, shift a contiguous time range
//!   - velocity_quantize:  k = 10..500 arbitrary indices, doesn't change sort key
//!                         (but we still treat as updates to measure overhead)
//!   - bulk_paste:         k = 50..1000 newly-appended notes that must be merged
//!                         in (modeled as: append to end with new timestamps, then re-sort)
//!
//! Output: `results/track3_*.csv`

use deltasort::delta_sort_by;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct MidiEvent {
    // f64 stored as fixed-point ticks for Ord; in ticks at 480 PPQ this gives
    // sample-accurate timing up to ~250 hours at 32-bit, ample at 64.
    time_ticks: u64,
    channel: u8,
    note: u8,
    velocity: u8,
    duration_ticks: u32,
    id: u32,
}

fn cmp_event(a: &MidiEvent, b: &MidiEvent) -> Ordering {
    a.time_ticks
        .cmp(&b.time_ticks)
        .then_with(|| a.channel.cmp(&b.channel))
        .then_with(|| a.note.cmp(&b.note))
        .then_with(|| a.id.cmp(&b.id))
}

fn generate_track(n: usize, seed: u64) -> Vec<MidiEvent> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut events: Vec<MidiEvent> = (0..n)
        .map(|i| MidiEvent {
            time_ticks: rng.gen_range(0..(n as u64 * 240)),
            channel: rng.gen_range(0..16),
            note: 24 + rng.gen_range(0..72),
            velocity: 64 + rng.gen_range(0..64),
            duration_ticks: rng.gen_range(60..960),
            id: i as u32,
        })
        .collect();
    events.sort_by(cmp_event);
    events
}

// =============================================================================
// Edit operations
// =============================================================================

#[derive(Clone, Copy, Debug)]
enum Op {
    SingleDrag,
    SelectionDrag,
    VelocityQuantize,
    BulkPaste,
}

impl Op {
    fn name(self) -> &'static str {
        match self {
            Op::SingleDrag => "single_drag",
            Op::SelectionDrag => "selection_drag",
            Op::VelocityQuantize => "velocity_quantize",
            Op::BulkPaste => "bulk_paste",
        }
    }
}

/// Choose k indices and a transformation; return (dirty_indices, mutator).
fn sample_edit(rng: &mut StdRng, n: usize, op: Op) -> Vec<(usize, MidiEvent)> {
    match op {
        Op::SingleDrag => {
            let idx = rng.gen_range(0..n);
            vec![(idx, MidiEvent::default())] // placeholder; mutated later
        }
        Op::SelectionDrag => {
            let k = rng.gen_range(2..200.min(n));
            let start = rng.gen_range(0..(n - k));
            (start..start + k).map(|i| (i, MidiEvent::default())).collect()
        }
        Op::VelocityQuantize => {
            let k = rng.gen_range(10..500.min(n));
            let mut idxs: Vec<usize> = (0..n).collect();
            for i in 0..k {
                let j = rng.gen_range(i..n);
                idxs.swap(i, j);
            }
            idxs.truncate(k);
            idxs.sort_unstable();
            idxs.into_iter().map(|i| (i, MidiEvent::default())).collect()
        }
        Op::BulkPaste => {
            // Bulk paste: model as k indices to OVERWRITE with new events
            // (could equivalently be appended; same sort cost).
            let k = rng.gen_range(50..1000.min(n / 2));
            let mut idxs: Vec<usize> = (0..n).collect();
            for i in 0..k {
                let j = rng.gen_range(i..n);
                idxs.swap(i, j);
            }
            idxs.truncate(k);
            idxs.sort_unstable();
            idxs.into_iter().map(|i| (i, MidiEvent::default())).collect()
        }
    }
}

/// Apply the edit to `arr` (mutating in-place). Returns the dirty index list.
/// The actual new MidiEvent values are derived from the op semantics.
fn apply_edit(arr: &mut [MidiEvent], op: Op, edits: &[(usize, MidiEvent)], rng: &mut StdRng) -> Vec<usize> {
    let mut dirty: Vec<usize> = edits.iter().map(|(i, _)| *i).collect();
    match op {
        Op::SingleDrag => {
            let i = edits[0].0;
            // Shift in time by random delta in [-2000, 2000] ticks
            let delta: i64 = rng.gen_range(-2000..2000);
            arr[i].time_ticks = arr[i].time_ticks.saturating_add_signed(delta);
        }
        Op::SelectionDrag => {
            let delta: i64 = rng.gen_range(-5000..5000);
            for (i, _) in edits {
                arr[*i].time_ticks = arr[*i].time_ticks.saturating_add_signed(delta);
            }
        }
        Op::VelocityQuantize => {
            // Velocity quantize doesn't change sort key, but we report dirty to be honest:
            // it's an in-place value update with NO key change. We still measure to capture
            // the cost of "knowing" k items were touched even if they don't need to move.
            for (i, _) in edits {
                arr[*i].velocity = ((arr[*i].velocity as u32 + 31) / 32 * 32).min(127) as u8;
            }
        }
        Op::BulkPaste => {
            // Replace each chosen position with a new event at a random time
            let max_t = arr.last().map(|e| e.time_ticks).unwrap_or(1_000_000);
            for (i, _) in edits {
                arr[*i].time_ticks = rng.gen_range(0..max_t);
                arr[*i].note = 24 + rng.gen_range(0..72);
                arr[*i].velocity = 64 + rng.gen_range(0..64);
            }
        }
    }
    // For VelocityQuantize, no sort movement is needed in principle (key unchanged),
    // but to model "we don't know which dirty items actually moved" we include all.
    // A smarter strategy would skip unchanged keys; we report this in the analysis.
    dirty.sort_unstable();
    dirty
}

// =============================================================================
// Strategies
// =============================================================================

fn strat_full(arr: &mut Vec<MidiEvent>, _dirty: &[usize]) -> Duration {
    let t0 = Instant::now();
    arr.sort_unstable_by(cmp_event);
    t0.elapsed()
}

fn strat_delta(arr: &mut Vec<MidiEvent>, dirty: &[usize]) -> Duration {
    let mut d = dirty.to_vec();
    let t0 = Instant::now();
    delta_sort_by(arr, &mut d, cmp_event);
    t0.elapsed()
}

/// BTreeSet variant: model Ardour's multiset by removing each old event and
/// reinserting the new one. We need to know the OLD event values to remove,
/// so this strategy takes both pre-edit and post-edit views.
fn strat_btreeset(
    btree: &mut BTreeSet<MidiEvent>,
    old_events: &[MidiEvent],
    new_events: &[MidiEvent],
) -> Duration {
    let t0 = Instant::now();
    for (old, new) in old_events.iter().zip(new_events.iter()) {
        btree.remove(old);
        btree.insert(new.clone());
    }
    t0.elapsed()
}

// =============================================================================
// Benchmark driver
// =============================================================================

#[derive(Clone, Debug)]
struct OpResult {
    op: &'static str,
    n: usize,
    mean_k: f64,
    full_us: f64,
    delta_us: f64,
    btree_us: f64,
}

fn benchmark_op(n: usize, op: Op, samples: usize, seed: u64) -> OpResult {
    let mut rng = StdRng::seed_from_u64(seed);
    let base = generate_track(n, seed);

    let mut full_total = Duration::ZERO;
    let mut delta_total = Duration::ZERO;
    let mut btree_total = Duration::ZERO;
    let mut k_sum: usize = 0;

    for _ in 0..samples {
        let edits = sample_edit(&mut rng, n, op);
        // Capture pre-edit events for BTreeSet remove
        let old_events: Vec<MidiEvent> = edits.iter().map(|(i, _)| base[*i].clone()).collect();

        // Apply mutation to a fresh copy
        let mut arr = base.clone();
        let dirty = apply_edit(&mut arr, op, &edits, &mut rng);
        k_sum += dirty.len();
        let new_events: Vec<MidiEvent> = dirty.iter().map(|&i| arr[i].clone()).collect();

        {
            let mut a = arr.clone();
            full_total += strat_full(&mut a, &dirty);
        }
        {
            let mut a = arr.clone();
            delta_total += strat_delta(&mut a, &dirty);
        }
        {
            let mut btree: BTreeSet<MidiEvent> = base.iter().cloned().collect();
            btree_total += strat_btreeset(&mut btree, &old_events, &new_events);
        }
    }

    let to_us = |d: Duration| d.as_nanos() as f64 / 1000.0 / samples as f64;
    OpResult {
        op: op.name(),
        n,
        mean_k: k_sum as f64 / samples as f64,
        full_us: to_us(full_total),
        delta_us: to_us(delta_total),
        btree_us: to_us(btree_total),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let fast = args.iter().any(|a| a == "--fast");

    fs::create_dir_all("results").ok();

    let n_values: &[usize] = if fast {
        &[1_000, 10_000]
    } else {
        &[1_000, 10_000, 100_000, 1_000_000]
    };
    let ops: &[Op] = &[
        Op::SingleDrag,
        Op::SelectionDrag,
        Op::VelocityQuantize,
        Op::BulkPaste,
    ];

    let samples_for_n = |n: usize| -> usize {
        if fast {
            5
        } else if n >= 1_000_000 {
            10
        } else if n >= 100_000 {
            40
        } else {
            200
        }
    };

    let mut csv = String::from("op,n,samples,mean_k,full_us,delta_us,btree_us\n");
    println!(
        "{:>20} {:>9} {:>9} {:>10} {:>12} {:>12} {:>12}",
        "op", "n", "samples", "mean_k", "full", "delta", "btree"
    );

    for &n in n_values {
        for &op in ops {
            let samples = samples_for_n(n);
            let res = benchmark_op(n, op, samples, 0xFEEDBEEF_u64.wrapping_add(n as u64));
            println!(
                "{:>20} {:>9} {:>9} {:>10.1} {:>12.1} {:>12.1} {:>12.1}",
                res.op, res.n, samples, res.mean_k, res.full_us, res.delta_us, res.btree_us
            );
            csv.push_str(&format!(
                "{},{},{},{:.1},{:.1},{:.1},{:.1}\n",
                res.op, res.n, samples, res.mean_k, res.full_us, res.delta_us, res.btree_us
            ));
        }
    }

    let path = Path::new("results/track3_summary.csv");
    fs::File::create(path)
        .and_then(|mut f| f.write_all(csv.as_bytes()))
        .expect("write summary csv");
    println!("\nWrote {}", path.display());
}
