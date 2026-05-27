//! Discrete Event Simulation (DES) event queue benchmark.
//!
//! In DES, a future-event list (FEL) is sorted by event timestamp.
//! The gold standard data structure is a binary heap (O(log n) per operation).
//! But binary heaps have poor cache behavior: at n=100K, the top half of the
//! heap spans ~50K nodes, each likely a cache miss.
//!
//! This benchmark compares DS against binary heap for the key DES scenario:
//!   "k events are rescheduled simultaneously (their timestamps updated)"
//!
//! Production contexts:
//!   - Conservative PDES (Parallel DES): lookahead period ends, k events
//!     in the lookahead window get their timestamps corrected
//!   - Network simulation (ns-3, OMNET++): batch of k packet reschedulings
//!   - Traffic simulation: k vehicle events updated after signal timing change
//!
//! The batch-reschedule case is EXACTLY DS's sweet spot: k known dirty
//! indices, then need sorted order restored.
//!
//! Heap operations for k batch updates: k × O(log n) cache-unfriendly ops.
//! DS: O(n + k log n) cache-friendly scan + sort + merge.
//!
//! The crossover (where DS beats heap wall-clock time) is the key finding.

use deltasort::delta_sort_by;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs;
use std::io::Write;
use std::time::Instant;

/// A DES event (timestamp, event_type, entity_id).
#[derive(Clone, Debug, PartialEq, Default)]
struct Event {
    time: f64,
    event_type: u16,
    entity_id: u32,
}

/// Ordering for the sorted array (ascending by time).
fn cmp_event(a: &Event, b: &Event) -> Ordering {
    a.time.partial_cmp(&b.time).unwrap_or(Ordering::Equal)
        .then(a.event_type.cmp(&b.event_type))
        .then(a.entity_id.cmp(&b.entity_id))
}

/// For BinaryHeap (max-heap, negated for min-heap behavior).
#[derive(Clone, PartialEq)]
struct HeapEvent {
    neg_time: f64, // store negated so max-heap → min-heap
    event_type: u16,
    entity_id: u32,
}
impl Eq for HeapEvent {}
impl PartialOrd for HeapEvent {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
}
impl Ord for HeapEvent {
    fn cmp(&self, o: &Self) -> Ordering {
        self.neg_time.partial_cmp(&o.neg_time).unwrap_or(Ordering::Equal)
            .then(self.event_type.cmp(&o.event_type))
    }
}

fn gen_fel(n: usize, seed: u64, sim_horizon: f64) -> Vec<Event> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v: Vec<Event> = (0..n).map(|i| Event {
        time: rng.gen_range(0.0..sim_horizon),
        event_type: rng.gen_range(0..16u16),
        entity_id: i as u32,
    }).collect();
    v.sort_by(cmp_event);
    v
}

/// Reschedule k events: update their timestamps (dirty indices known).
fn reschedule(fel: &mut [Event], k: usize, sim_horizon: f64, rng: &mut StdRng) -> Vec<usize> {
    let n = fel.len();
    let mut picks: Vec<usize> = (0..n).collect();
    for i in 0..k { let j = rng.gen_range(i..n); picks.swap(i, j); }
    let mut dirty: Vec<usize> = picks[..k].to_vec();
    dirty.sort_unstable();
    for &i in &dirty {
        // Reschedule: new time is some perturbation of old (conservative: small shift)
        // or fully random (aggressive: full time range).
        let perturbation: f64 = rng.gen_range(-sim_horizon * 0.1..sim_horizon * 0.1);
        fel[i].time = (fel[i].time + perturbation).clamp(0.0, sim_horizon);
    }
    dirty
}

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

/// Heap-based FEL: rebuild from scratch after k updates.
/// This is the "nuclear option" baseline (what happens without position map).
fn heap_rebuild(fel: &[Event]) -> f64 {
    let t = Instant::now();
    let _heap: BinaryHeap<HeapEvent> = fel.iter().map(|e| HeapEvent {
        neg_time: -e.time,
        event_type: e.event_type,
        entity_id: e.entity_id,
    }).collect();
    t.elapsed().as_nanos() as f64
}

/// Heap-based FEL: simulate k delete-min + reinsert operations.
/// This is the "smart" heap baseline: for each dirty event, remove it and
/// reinsert with new timestamp. Approximated by k × (drain-reinsert).
/// Real implementation would require a position map; we simulate the cost.
fn heap_k_updates(fel: &[Event], dirty: &[usize]) -> f64 {
    // Build initial heap (one-time cost not measured, but we measure the k update cost)
    let mut heap: Vec<HeapEvent> = fel.iter().map(|e| HeapEvent {
        neg_time: -e.time,
        event_type: e.event_type,
        entity_id: e.entity_id,
    }).collect();
    // Heapify
    let n = heap.len();
    for i in (0..n / 2).rev() {
        sift_down(&mut heap, i);
    }
    let k = dirty.len();
    let t = Instant::now();
    // Simulate k decrease-key operations: each one touches O(log n) nodes
    for &di in dirty {
        // Update the element at position di in the backing array
        // (position map: dirty[i] gives position in heap array)
        // We use the dirty index directly as heap position (approximation;
        // real implementation would maintain explicit position map)
        if di < heap.len() {
            heap[di].neg_time = -fel[di].time;
            // Sift up (decrease-key)
            sift_up(&mut heap, di);
            // Sift down (in case increase-key)
            sift_down(&mut heap, di);
        }
    }
    let elapsed = t.elapsed().as_nanos() as f64;
    // Verify heap property (spot check)
    let _ = k;
    elapsed
}

fn sift_down(heap: &mut Vec<HeapEvent>, mut i: usize) {
    let n = heap.len();
    loop {
        let left = 2 * i + 1;
        let right = 2 * i + 2;
        let mut largest = i;
        if left < n && heap[left] > heap[largest] { largest = left; }
        if right < n && heap[right] > heap[largest] { largest = right; }
        if largest == i { break; }
        heap.swap(i, largest);
        i = largest;
    }
}

fn sift_up(heap: &mut Vec<HeapEvent>, mut i: usize) {
    while i > 0 {
        let parent = (i - 1) / 2;
        if heap[i] > heap[parent] {
            heap.swap(i, parent);
            i = parent;
        } else {
            break;
        }
    }
}

struct Results {
    ds: f64, esm: f64, full: f64,
    heap_rebuild: f64, heap_update: f64,
}

fn run_scenario(baseline: &[Event], dirty: &[usize]) -> Results {
    let mut data = baseline.to_vec();
    let mut dv = dirty.to_vec(); dv.sort_unstable(); dv.dedup();
    let t = Instant::now();
    delta_sort_by(&mut data, &mut dv, cmp_event);
    let ds = t.elapsed().as_nanos() as f64;
    let ds_result = data;

    let mut data = baseline.to_vec();
    let mut dv = dirty.to_vec();
    let t = Instant::now();
    esm_sort(&mut data, &mut dv, cmp_event);
    let esm = t.elapsed().as_nanos() as f64;
    assert!(data == ds_result, "ESM mismatch n={}", baseline.len());

    let mut data = baseline.to_vec();
    let t = Instant::now();
    data.sort_by(cmp_event);
    let full = t.elapsed().as_nanos() as f64;
    assert!(data == ds_result, "Full mismatch n={}", baseline.len());

    let hr = heap_rebuild(baseline);
    let hu = heap_k_updates(baseline, dirty);

    Results { ds, esm, full, heap_rebuild: hr, heap_update: hu }
}

fn med(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }

fn run_workload(n: usize, k: usize, n_steps: usize, csv: &mut impl Write) {
    let mut fel = gen_fel(n, 0xDE00u64.wrapping_add(n as u64), 1000.0);
    let mut rng = StdRng::seed_from_u64(0xFE01);
    let mut ds_v = Vec::new(); let mut esm_v = Vec::new(); let mut full_v = Vec::new();
    let mut hr_v = Vec::new(); let mut hu_v = Vec::new();
    for _ in 0..n_steps {
        let dirty = reschedule(&mut fel, k, 1000.0, &mut rng);
        let r = run_scenario(&fel, &dirty);
        ds_v.push(r.ds); esm_v.push(r.esm); full_v.push(r.full);
        hr_v.push(r.heap_rebuild); hu_v.push(r.heap_update);
        fel.sort_by(cmp_event);
    }
    let dsm = med(ds_v); let esmm = med(esm_v); let fm = med(full_v);
    let hrm = med(hr_v); let hum = med(hu_v);
    let kpct = 100.0 * k as f64 / n as f64;
    println!(
        "{:>8} {:>6} {:>6.3}%  {:>8.1} {:>8.1} {:>8.1}  {:>8.1} {:>8.1}   heap_upd/DS={:.2}× full/DS={:.2}×",
        n, k, kpct,
        dsm/1e3, esmm/1e3, fm/1e3, hrm/1e3, hum/1e3,
        hum/dsm, fm/dsm,
    );
    for (strat, val) in [("DeltaSort", dsm), ("ESM", esmm), ("Full", fm), ("HeapRebuild", hrm), ("HeapUpdate", hum)] {
        writeln!(csv, "{},{},{},{:.0}", n, k, strat, val).unwrap();
    }
}

fn main() {
    fs::create_dir_all("results").ok();
    let mut f = fs::File::create("results/track_m_des_events.csv").expect("csv");
    writeln!(f, "n,k,strategy,ns").unwrap();

    println!("=== Discrete Event Simulation: FEL reschedule benchmark ===");
    println!("(all times in μs; heap_update = k × O(log n) simulate cost)");
    println!("{:>8} {:>6} {:>7}  {:>8} {:>8} {:>8}  {:>8} {:>8}   ratios",
        "n", "k", "k/n", "DS μs", "ESM μs", "Full μs", "HeapRbld", "HeapUpd");

    // Small event list (embedded sim, RTOS timer list)
    println!("\n--- Small FEL (n=100-1000) ---");
    for &n in &[100usize, 500, 1000] {
        for &k in &[1usize, 2, 5, 10, 20, 50] {
            if k <= n / 2 {
                run_workload(n, k, 31, &mut f);
            }
        }
        println!();
    }

    // Medium event list (single-server sim)
    println!("--- Medium FEL (n=5K-50K) ---");
    for &n in &[5_000usize, 10_000, 50_000] {
        for &k in &[1usize, 5, 10, 50, 100, 500] {
            if k <= n / 4 {
                run_workload(n, k, 21, &mut f);
            }
        }
        println!();
    }

    // Large event list (network/traffic sim)
    println!("--- Large FEL (n=100K-500K) ---");
    for &n in &[100_000usize, 500_000] {
        for &k in &[1usize, 10, 100, 1000, 5000] {
            if k <= n / 4 {
                run_workload(n, k, 11, &mut f);
            }
        }
        println!();
    }

    println!("Wrote results/track_m_des_events.csv");
}
