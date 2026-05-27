//! Limit Order Book (LOB) price-level maintenance.
//!
//! THE canonical HFT data structure. A LOB is a sorted list of price levels;
//! each price level holds a FIFO queue of orders at that price. The LOB has
//! two sides:
//!   - bids: sorted descending by price (best bid = max)
//!   - asks: sorted ascending by price (best ask = min)
//!
//! Production LOBs (Nasdaq ITCH, CBOE PITCH, EBS, etc.) typically maintain
//! ~50–500 active price levels per side at any moment. During volatile market
//! conditions (open/close, news events, large trades), k price levels are
//! modified per market data tick:
//!   - cancels: remove order from level (level may become empty)
//!   - modifies: order qty changed (no sort change unless level emptied)
//!   - new orders at new price levels (insert at sorted position)
//!   - sweep trades that consume multiple levels (k levels disappear)
//!
//! The sort-maintenance question: when a tick touches k price levels, what is
//! the fastest way to maintain the sorted price-level array?
//!
//! Why a sorted array (not a heap):
//!   - Iterate top-10 levels for "market depth" feed (every tick)
//!   - Iterate from best down to find liquidity for a market order
//!   - Compute imbalance/microprice across top-N levels
//!   - Cache-friendly sequential access
//!
//! Why DS:
//!   - Dirty levels are KNOWN from the incoming order messages
//!   - k is small (typically 1-10) during normal trading
//!   - n is small (50-500)
//!   - Latency budget is sub-microsecond at colocation
//!
//! Why this is distinct from `hft_basket.rs`:
//!   - LOB level priority is just price (1 f64), not composite
//!   - LOB is sorted ascending OR descending (the two sides differ)
//!   - Update pattern is order-book-natural: levels concentrated near top
//!
//! Production comparisons:
//!   - Skiplist (CME iLink, EBS): O(log n) per update, cache-hostile pointer
//!     chasing. Skiplist with k batch updates is k × O(log n) cache-miss-heavy.
//!   - Red-black tree (some venues): same profile as skiplist.
//!   - Sorted vec + linear scan (compact LOBs, top-of-book strategies): O(n)
//!     per cancel, O(n) per insert (worst case). For k batched changes,
//!     this is k × O(n). DS replaces this with O(n + k log n).

use deltasort::delta_sort_by;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::fs;
use std::io::Write;
use std::time::Instant;

/// A single price level in a side of the LOB.
#[derive(Clone, Default, PartialEq)]
struct PriceLevel {
    /// Price in ticks (10^-4 of a dollar typical; scaled to i64 for exact ordering)
    price_ticks: i64,
    /// Aggregate quantity at this price (sum of all orders at this level)
    qty: u64,
    /// Order count at this level (for fair queueing in matching engine)
    n_orders: u32,
    /// Last-update sequence number (for ITCH feed reconstruction)
    last_seq: u64,
}

impl Eq for PriceLevel {}
impl Ord for PriceLevel {
    fn cmp(&self, o: &Self) -> Ordering {
        self.price_ticks.cmp(&o.price_ticks)
            .then(self.last_seq.cmp(&o.last_seq))
    }
}
impl PartialOrd for PriceLevel { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }

fn cmp_asc(a: &PriceLevel, b: &PriceLevel) -> Ordering { a.cmp(b) }
fn cmp_desc(a: &PriceLevel, b: &PriceLevel) -> Ordering { b.cmp(a) }

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

// ============================================================
//  Skiplist baseline (production LOB structure on many venues)
// ============================================================

/// Simplified skiplist node with up to 8 levels (matches typical LOB implementations).
struct SkipNode {
    key: i64,
    value: PriceLevel,
    next: [Option<usize>; 8], // indices into nodes vec
    height: u8,
}

struct Skiplist {
    nodes: Vec<SkipNode>,
    head: usize,
    free: Vec<usize>,
}

impl Skiplist {
    fn new() -> Self {
        let head = SkipNode {
            key: i64::MIN, value: PriceLevel::default(),
            next: [None; 8], height: 8,
        };
        Skiplist { nodes: vec![head], head: 0, free: Vec::new() }
    }

    fn random_height(rng: &mut StdRng) -> u8 {
        let mut h = 1;
        while rng.gen::<f64>() < 0.5 && h < 8 { h += 1; }
        h
    }

    fn insert(&mut self, key: i64, value: PriceLevel, rng: &mut StdRng) {
        let mut update = [self.head; 8];
        let mut cur = self.head;
        for lvl in (0..8).rev() {
            while let Some(nxt) = self.nodes[cur].next[lvl] {
                if self.nodes[nxt].key < key { cur = nxt; } else { break; }
            }
            update[lvl] = cur;
        }
        // Check if exists
        if let Some(nxt) = self.nodes[cur].next[0] {
            if self.nodes[nxt].key == key {
                self.nodes[nxt].value = value;
                return;
            }
        }
        let h = Self::random_height(rng);
        let new_idx = if let Some(idx) = self.free.pop() {
            self.nodes[idx] = SkipNode { key, value, next: [None; 8], height: h };
            idx
        } else {
            self.nodes.push(SkipNode { key, value, next: [None; 8], height: h });
            self.nodes.len() - 1
        };
        for lvl in 0..h as usize {
            self.nodes[new_idx].next[lvl] = self.nodes[update[lvl]].next[lvl];
            self.nodes[update[lvl]].next[lvl] = Some(new_idx);
        }
    }

    fn delete(&mut self, key: i64) {
        let mut update = [self.head; 8];
        let mut cur = self.head;
        for lvl in (0..8).rev() {
            while let Some(nxt) = self.nodes[cur].next[lvl] {
                if self.nodes[nxt].key < key { cur = nxt; } else { break; }
            }
            update[lvl] = cur;
        }
        if let Some(target) = self.nodes[cur].next[0] {
            if self.nodes[target].key == key {
                let h = self.nodes[target].height as usize;
                for lvl in 0..h {
                    if self.nodes[update[lvl]].next[lvl] == Some(target) {
                        self.nodes[update[lvl]].next[lvl] = self.nodes[target].next[lvl];
                    }
                }
                self.free.push(target);
            }
        }
    }
}

// ============================================================
//  Workload generation
// ============================================================

fn init_lob(n: usize, seed: u64, side_desc: bool) -> Vec<PriceLevel> {
    let mut rng = StdRng::seed_from_u64(seed);
    // Realistic price levels concentrated around mid-price
    let mid = 100_0000_i64; // $100.0000
    let tick = 1_i64;
    let mut levels: Vec<PriceLevel> = (0..n).map(|i| {
        let offset = (i as i64) * tick * if side_desc { -1 } else { 1 };
        PriceLevel {
            price_ticks: mid + offset,
            qty: rng.gen_range(100..10_000u64),
            n_orders: rng.gen_range(1..20u32),
            last_seq: i as u64,
        }
    }).collect();
    if side_desc { levels.sort_by(cmp_desc); } else { levels.sort_by(cmp_asc); }
    levels
}

/// Simulate a market data tick: k price levels are touched.
/// Realistic distribution: 70% of touches are at top-of-book (first ~20% of levels).
fn simulate_tick(
    lob: &mut Vec<PriceLevel>, k: usize, rng: &mut StdRng, side_desc: bool, seq: &mut u64,
) -> Vec<usize> {
    let n = lob.len();
    let top_band = (n / 5).max(1);
    let mut dirty = Vec::with_capacity(k);
    let mut used = vec![false; n];
    for _ in 0..k {
        let idx = if rng.gen::<f64>() < 0.7 {
            rng.gen_range(0..top_band)
        } else {
            rng.gen_range(0..n)
        };
        if used[idx] { continue; }
        used[idx] = true;
        dirty.push(idx);
        // Update: qty changes (cancel/modify), occasionally price moves slightly
        if rng.gen::<f64>() < 0.3 {
            // Price moves: this triggers the sort
            let delta: i64 = rng.gen_range(-50..50);
            lob[idx].price_ticks += if side_desc { -delta } else { delta };
        }
        lob[idx].qty = rng.gen_range(100..10_000u64);
        *seq += 1;
        lob[idx].last_seq = *seq;
    }
    dirty.sort_unstable();
    dirty.dedup();
    dirty
}

struct Out { ds: f64, esm: f64, full: f64, skip: f64 }

fn run_tick(
    baseline: &[PriceLevel], dirty: &[usize], side_desc: bool, seq_base: u64,
) -> Out {
    let cmp: fn(&PriceLevel, &PriceLevel) -> Ordering = if side_desc { cmp_desc } else { cmp_asc };

    // DS
    let mut data = baseline.to_vec();
    let mut dv = dirty.to_vec(); dv.sort_unstable(); dv.dedup();
    let t = Instant::now();
    delta_sort_by(&mut data, &mut dv, cmp);
    let ds = t.elapsed().as_nanos() as f64;
    let ds_result = data;

    // ESM
    let mut data = baseline.to_vec();
    let mut dv = dirty.to_vec();
    let t = Instant::now();
    esm_sort(&mut data, &mut dv, cmp);
    let esm = t.elapsed().as_nanos() as f64;
    assert!(data == ds_result, "ESM mismatch");

    // Full sort
    let mut data = baseline.to_vec();
    let t = Instant::now();
    data.sort_by(cmp);
    let full = t.elapsed().as_nanos() as f64;
    assert!(data == ds_result, "Full mismatch");

    // Skiplist: rebuild from baseline first (clean state) then apply k changes
    // We measure the k-changes cost (analogous to running k delete+insert ops).
    // Note: skiplist also needs a separate index to traverse for top-N, which
    // we don't model here. The DS sorted array natively supports top-N.
    let mut sl = Skiplist::new();
    let mut rng = StdRng::seed_from_u64(0x517E);
    for lvl in baseline { sl.insert(lvl.price_ticks, lvl.clone(), &mut rng); }
    // Pre-compute the change-list: for each dirty index, find the new value
    // in the DS result (which contains the resorted final state).
    let mut updates: Vec<(i64, PriceLevel)> = Vec::with_capacity(dirty.len());
    for &di in dirty {
        let old_seq = baseline[di].last_seq;
        // Find the level in the final result that has the corresponding new seq.
        // Each dirty index produces a new seq > all originals.
        let new_lvl = ds_result.iter().find(|l| l.last_seq > seq_base + (dirty.len() as u64) || l.last_seq == old_seq + 1)
            .cloned().unwrap_or_else(|| baseline[di].clone());
        updates.push((baseline[di].price_ticks, new_lvl));
    }
    let t = Instant::now();
    for (old_price, new_lvl) in &updates {
        sl.delete(*old_price);
        sl.insert(new_lvl.price_ticks, new_lvl.clone(), &mut rng);
    }
    let skip = t.elapsed().as_nanos() as f64;

    Out { ds, esm, full, skip }
}

fn med(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }

fn run_workload(n: usize, k: usize, n_ticks: usize, side_desc: bool, csv: &mut impl Write) {
    let mut lob = init_lob(n, 0xB100u64.wrapping_add(n as u64).wrapping_add(if side_desc { 1 } else { 0 }), side_desc);
    let mut rng = StdRng::seed_from_u64(0xC0FE);
    let mut seq = n as u64;
    let mut ds_v = Vec::new(); let mut esm_v = Vec::new(); let mut full_v = Vec::new(); let mut sk_v = Vec::new();

    for _ in 0..n_ticks {
        let dirty = simulate_tick(&mut lob, k, &mut rng, side_desc, &mut seq);
        let r = run_tick(&lob, &dirty, side_desc, seq);
        ds_v.push(r.ds); esm_v.push(r.esm); full_v.push(r.full); sk_v.push(r.skip);
        if side_desc { lob.sort_by(cmp_desc); } else { lob.sort_by(cmp_asc); }
    }

    let dsm = med(ds_v); let esmm = med(esm_v); let fm = med(full_v); let sm = med(sk_v);
    let kpct = 100.0 * k as f64 / n as f64;

    println!(
        "{:>5} {:>4} {:>6.2}% {:>4}  {:>7.1} {:>7.1} {:>7.1} {:>7.1}   {:>5.2}× {:>5.2}× {:>5.2}×",
        n, k, kpct, if side_desc { "bid" } else { "ask" },
        dsm, esmm, fm, sm,
        esmm/dsm, fm/dsm, sm/dsm,
    );
    let side = if side_desc { "bid" } else { "ask" };
    for (strat, val) in [("DeltaSort", dsm), ("ESM", esmm), ("Full", fm), ("Skiplist", sm)] {
        writeln!(csv, "{},{},{},{},{:.0}", side, n, k, strat, val).unwrap();
    }
}

fn main() {
    fs::create_dir_all("results").ok();
    let mut f = fs::File::create("results/track_p_lob.csv").expect("csv");
    writeln!(f, "side,n,k,strategy,ns").unwrap();

    println!("=== Limit Order Book: Price-Level Maintenance ===");
    println!("(all in ns; Skiplist = k batch insert+delete operations)");
    println!("{:>5} {:>4} {:>7} {:>4}  {:>7} {:>7} {:>7} {:>7}   {:>6} {:>6} {:>6}",
        "n", "k", "k/n", "side", "DS ns", "ESM ns", "Full ns", "Skip ns", "ESM/DS", "Full/DS", "Skip/DS");

    println!("\n--- Compact LOB (top-of-book strategies, n=10-50) ---");
    for &n in &[10usize, 20, 50] {
        for &k in &[1usize, 2, 5] {
            if k <= n / 2 {
                run_workload(n, k, 51, false, &mut f);
            }
        }
        println!();
    }

    println!("--- Standard LOB (typical venue, n=100-500) ---");
    for &n in &[100usize, 250, 500] {
        for &k in &[1usize, 2, 5, 10, 25, 50] {
            if k <= n / 2 {
                run_workload(n, k, 51, false, &mut f);
            }
        }
        println!();
    }

    println!("--- Deep LOB (futures/FX, n=1K-5K) ---");
    for &n in &[1_000usize, 5_000] {
        for &k in &[1usize, 5, 10, 25, 50, 100] {
            if k <= n / 4 {
                run_workload(n, k, 51, false, &mut f);
            }
        }
        println!();
    }

    println!("Wrote results/track_p_lob.csv");
}
