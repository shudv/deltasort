//! Network packet scheduler (QoS / Weighted Fair Queueing) benchmark.
//!
//! In modern packet processing fast-paths (DPDK, XDP, eBPF, P4 switches),
//! per-flow QoS is implemented as a sorted priority array. The packet
//! scheduler picks the next flow to dequeue based on a virtual-finish-time
//! (WFQ) or priority + deficit counter.
//!
//! Per-interrupt-batch workload:
//!   - n = 64..1024 active flows per port
//!   - k = 4..64 flows have their virtual time updated per batch
//!     (because they just sent a packet, or their EWMA changed)
//!   - Deadline: sub-microsecond per batch (line-rate 10/40/100 GbE)
//!
//! Production scheduling algorithms:
//!   - Linux HFSC (Hierarchical Fair Service Curve): uses red-black tree
//!   - Linux SFQ (Stochastic Fair Queueing): hash-based, not sorted
//!   - DPDK rte_sched: hierarchical token buckets + sorted active set
//!   - Open vSwitch: per-queue token bucket
//!
//! WFQ requires the sorted view: pick min virtual-finish-time, dispatch,
//! and update finish time for that flow (or batch of k flows).
//!
//! Comparators tested:
//!   - cheap: u64 virtual_finish_time
//!   - expensive: composite (finish_time, priority_class, weight, deficit)
//!     — models real DSCP class + WRR weight + DRR deficit counter

use deltasort::delta_sort_by;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::fs;
use std::io::Write;
use std::time::Instant;

/// A flow queue entry in the scheduler's active set.
#[derive(Clone, Default, PartialEq)]
struct FlowQueue {
    /// Virtual finish time (WFQ primary key, ascending)
    vft: u64,
    /// DSCP class (lower = higher priority; secondary)
    dscp_class: u8,
    /// Weighted Round Robin weight
    wrr_weight: u16,
    /// Deficit Round Robin counter
    drr_deficit: i32,
    /// Flow id
    flow_id: u32,
}

impl Eq for FlowQueue {}
impl Ord for FlowQueue {
    fn cmp(&self, o: &Self) -> Ordering {
        // Primary: vft ascending (WFQ order)
        self.vft.cmp(&o.vft)
            .then(self.dscp_class.cmp(&o.dscp_class))
            .then(o.wrr_weight.cmp(&self.wrr_weight))
            .then(o.drr_deficit.cmp(&self.drr_deficit))
            .then(self.flow_id.cmp(&o.flow_id))
    }
}
impl PartialOrd for FlowQueue {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
}

fn cmp_full(a: &FlowQueue, b: &FlowQueue) -> Ordering { a.cmp(b) }

/// Cheap comparator: only primary key (no class/weight/deficit)
fn cmp_cheap(a: &FlowQueue, b: &FlowQueue) -> Ordering {
    a.vft.cmp(&b.vft).then(a.flow_id.cmp(&b.flow_id))
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

// Binary min-heap with sift_up/sift_down for k batch updates
#[allow(dead_code)]
fn heap_k_updates(baseline: &[FlowQueue], dirty: &[usize], expensive: bool) -> f64 {
    let mut heap: Vec<FlowQueue> = baseline.to_vec();
    let n = heap.len();
    let cmp_fn: fn(&FlowQueue, &FlowQueue) -> Ordering = if expensive { cmp_full } else { cmp_cheap };
    // Heapify
    for i in (0..n / 2).rev() {
        let mut cur = i;
        loop {
            let l = 2 * cur + 1; let r = 2 * cur + 2; let mut s = cur;
            if l < n && cmp_fn(&heap[l], &heap[s]) == Ordering::Less { s = l; }
            if r < n && cmp_fn(&heap[r], &heap[s]) == Ordering::Less { s = r; }
            if s == cur { break; }
            heap.swap(cur, s);
            cur = s;
        }
    }
    let t = Instant::now();
    for &di in dirty {
        if di < heap.len() {
            // Without a position map; in real systems this would be O(log n) with map
            let mut cur = di;
            // sift up
            while cur > 0 {
                let parent = (cur - 1) / 2;
                if cmp_fn(&heap[cur], &heap[parent]) == Ordering::Less {
                    heap.swap(cur, parent); cur = parent;
                } else { break; }
            }
            // sift down
            loop {
                let l = 2 * cur + 1; let r = 2 * cur + 2; let mut s = cur;
                if l < n && cmp_fn(&heap[l], &heap[s]) == Ordering::Less { s = l; }
                if r < n && cmp_fn(&heap[r], &heap[s]) == Ordering::Less { s = r; }
                if s == cur { break; }
                heap.swap(cur, s);
                cur = s;
            }
        }
    }
    t.elapsed().as_nanos() as f64
}

fn init_flows(n: usize, seed: u64) -> Vec<FlowQueue> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v: Vec<FlowQueue> = (0..n as u32).map(|i| FlowQueue {
        vft: rng.gen_range(0..1_000_000u64),
        dscp_class: rng.gen_range(0..8u8),
        wrr_weight: rng.gen_range(1..1000u16),
        drr_deficit: rng.gen_range(-1500..1500i32),
        flow_id: i,
    }).collect();
    v.sort();
    v
}

/// Simulate one packet-processing interrupt batch: k flows get their VFT updated.
fn process_batch(flows: &mut [FlowQueue], k: usize, rng: &mut StdRng) -> Vec<usize> {
    let n = flows.len();
    let mut picks: Vec<usize> = (0..n).collect();
    for i in 0..k { let j = rng.gen_range(i..n); picks.swap(i, j); }
    let mut dirty: Vec<usize> = picks[..k].to_vec();
    dirty.sort_unstable();
    for &i in &dirty {
        // VFT advances by (packet_size / weight) — simulate
        let pkt_size: u64 = rng.gen_range(64..1500);
        let weight = flows[i].wrr_weight as u64;
        flows[i].vft = flows[i].vft.saturating_add(pkt_size * 1000 / weight.max(1));
        flows[i].drr_deficit = flows[i].drr_deficit.saturating_sub(pkt_size as i32);
        if flows[i].drr_deficit < 0 {
            flows[i].drr_deficit += 1500; // quantum
        }
    }
    dirty
}

struct Out { ds: f64, esm: f64, full: f64, heap: f64 }

fn run_batch_expensive(baseline: &[FlowQueue], dirty: &[usize]) -> Out {
    let mut data = baseline.to_vec();
    let mut dv = dirty.to_vec(); dv.sort_unstable(); dv.dedup();
    let t = Instant::now();
    delta_sort_by(&mut data, &mut dv, cmp_full);
    let ds = t.elapsed().as_nanos() as f64;
    let ds_result = data;

    let mut data = baseline.to_vec();
    let mut dv = dirty.to_vec();
    let t = Instant::now();
    esm_sort(&mut data, &mut dv, cmp_full);
    let esm = t.elapsed().as_nanos() as f64;
    assert!(data == ds_result, "ESM mismatch");

    let mut data = baseline.to_vec();
    let t = Instant::now();
    data.sort_unstable_by(cmp_full);
    let full = t.elapsed().as_nanos() as f64;
    assert!(data == ds_result, "Full mismatch");

    let heap = heap_k_updates(baseline, dirty, true);

    Out { ds, esm, full, heap }
}

fn med(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }

fn run_workload(n: usize, k: usize, n_batches: usize, csv: &mut impl Write) {
    let mut flows = init_flows(n, 0xF101u64.wrapping_add(n as u64));
    let mut rng = StdRng::seed_from_u64(0xDEAD);
    let mut ds_v = Vec::new(); let mut esm_v = Vec::new(); let mut full_v = Vec::new(); let mut hv = Vec::new();

    for _ in 0..n_batches {
        let dirty = process_batch(&mut flows, k, &mut rng);
        let r = run_batch_expensive(&flows, &dirty);
        ds_v.push(r.ds); esm_v.push(r.esm); full_v.push(r.full); hv.push(r.heap);
        flows.sort();
    }
    let dsm = med(ds_v); let esmm = med(esm_v); let fm = med(full_v); let hm = med(hv);
    let kpct = 100.0 * k as f64 / n as f64;

    // Latency context: at 10 GbE with avg packet 256B, batch rate is ~5 Mpps.
    // Per-batch budget (batch=32 packets): ~6.4μs.
    let budget_ns = 6_400.0f64;
    println!(
        "{:>5} {:>4} {:>6.2}%  {:>7.1} {:>7.1} {:>7.1} {:>7.1}   {:>5.2}× {:>5.2}× heap/DS={:.2}×  DS={:.1}% Full={:.1}% of 6.4μs",
        n, k, kpct,
        dsm, esmm, fm, hm,
        esmm/dsm, fm/dsm, hm/dsm,
        100.0 * dsm / budget_ns, 100.0 * fm / budget_ns,
    );
    for (strat, val) in [("DeltaSort", dsm), ("ESM", esmm), ("Full", fm), ("HeapNoPosMap", hm)] {
        writeln!(csv, "{},{},{},{:.0}", n, k, strat, val).unwrap();
    }
}

fn main() {
    fs::create_dir_all("results").ok();
    let mut f = fs::File::create("results/track_q_nw_qos.csv").expect("csv");
    writeln!(f, "n,k,strategy,ns").unwrap();

    println!("=== Network QoS (WFQ) Scheduler ===");
    println!("(times in ns; budget = 6.4μs per batch at 10 GbE / 5Mpps)");
    println!("{:>5} {:>4} {:>7}  {:>7} {:>7} {:>7} {:>7}", "n", "k", "k/n", "DS ns", "ESM ns", "Full ns", "Heap ns");

    println!("\n--- Small (single port, n=64-256 flows) ---");
    for &n in &[64usize, 128, 256] {
        for &k in &[1usize, 4, 8, 16, 32] {
            if k <= n / 2 {
                run_workload(n, k, 51, &mut f);
            }
        }
        println!();
    }

    println!("--- Medium (per-port, n=512-1024 flows) ---");
    for &n in &[512usize, 1_024] {
        for &k in &[1usize, 4, 8, 16, 32, 64, 128] {
            if k <= n / 2 {
                run_workload(n, k, 51, &mut f);
            }
        }
        println!();
    }

    println!("--- Large (aggregator/switch, n=4K-16K flows) ---");
    for &n in &[4_096usize, 16_384] {
        for &k in &[1usize, 16, 64, 256, 1024] {
            if k <= n / 4 {
                run_workload(n, k, 21, &mut f);
            }
        }
        println!();
    }

    println!("Wrote results/track_q_nw_qos.csv");
}
