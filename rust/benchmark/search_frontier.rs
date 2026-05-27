//! Batch DECREASE-KEY benchmark: DS as a priority-queue updater.
//!
//! In heuristic graph search (A*, Dijkstra, bidirectional search), the OPEN
//! list (frontier) is a priority queue sorted by f-score = g + h.
//! During beam search, anytime-A*, or parallel relaxation, a batch of k
//! frontier nodes have their f-scores decreased simultaneously.
//!
//! Standard approach: binary heap + position map, k × O(log n) DECREASE-KEY.
//! DS approach: maintain OPEN as sorted array + position map.
//!   After k known dirty positions are updated, DS re-sorts.
//!   O(n + k log n) but cache-friendly (one sequential scan).
//!
//! Key question: at what (n, k) does DS's cache-friendly O(n) scan beat
//! the binary heap's O(k log n) cache-unfriendly operations?
//!
//! Applications:
//!   - Anytime A* (ARA*): batch improvement of k frontier nodes each iteration
//!   - Parallel Dijkstra (δ-stepping): batch relaxation of a bucket's nodes
//!   - Neural architecture search: sorted candidate models, k repriced
//!   - Protein folding beam search: k hypotheses re-ranked per iteration
//!   - Monte Carlo Tree Search with batch rollouts: k UCT scores updated

use deltasort::delta_sort_by;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::fs;
use std::io::Write;
use std::time::Instant;

/// A frontier node with f-score (g + h).
#[derive(Clone, Default, PartialEq)]
struct FrontierNode {
    f_score: f64,    // primary sort key (ascending: lower = better)
    g_score: f64,    // cost so far
    h_score: f64,    // heuristic
    node_id: u32,
}

impl Eq for FrontierNode {}
impl Ord for FrontierNode {
    fn cmp(&self, o: &Self) -> Ordering {
        self.f_score.partial_cmp(&o.f_score).unwrap_or(Ordering::Equal)
            .then(self.node_id.cmp(&o.node_id))
    }
}
impl PartialOrd for FrontierNode {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
}
fn cmp_node(a: &FrontierNode, b: &FrontierNode) -> Ordering { a.cmp(b) }

// ============================================================
//  Binary heap with position map (O(log n) DECREASE-KEY)
// ============================================================

struct HeapWithPosMap {
    heap: Vec<FrontierNode>,
    pos: Vec<usize>, // pos[node_id] = index in heap
}

impl HeapWithPosMap {
    fn build(nodes: &[FrontierNode]) -> Self {
        let n = nodes.len();
        let mut heap = nodes.to_vec();
        let mut pos = vec![0usize; n];
        // Sort ascending by f_score to build min-heap via heapify
        for i in (0..n / 2).rev() {
            Self::sift_down_static(&mut heap, &mut pos, i, n);
        }
        for (i, node) in heap.iter().enumerate() {
            pos[node.node_id as usize] = i;
        }
        HeapWithPosMap { heap, pos }
    }

    fn sift_down_static(heap: &mut Vec<FrontierNode>, pos: &mut Vec<usize>, mut i: usize, n: usize) {
        loop {
            let l = 2 * i + 1;
            let r = 2 * i + 2;
            let mut smallest = i;
            if l < n && heap[l] < heap[smallest] { smallest = l; }
            if r < n && heap[r] < heap[smallest] { smallest = r; }
            if smallest == i { break; }
            let id_i = heap[i].node_id as usize;
            let id_s = heap[smallest].node_id as usize;
            pos[id_i] = smallest;
            pos[id_s] = i;
            heap.swap(i, smallest);
            i = smallest;
        }
    }

    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let parent = (i - 1) / 2;
            if self.heap[i] < self.heap[parent] {
                let id_i = self.heap[i].node_id as usize;
                let id_p = self.heap[parent].node_id as usize;
                self.pos[id_i] = parent;
                self.pos[id_p] = i;
                self.heap.swap(i, parent);
                i = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, i: usize) {
        let n = self.heap.len();
        let mut cur = i;
        loop {
            let l = 2 * cur + 1;
            let r = 2 * cur + 2;
            let mut smallest = cur;
            if l < n && self.heap[l] < self.heap[smallest] { smallest = l; }
            if r < n && self.heap[r] < self.heap[smallest] { smallest = r; }
            if smallest == cur { break; }
            let id_c = self.heap[cur].node_id as usize;
            let id_s = self.heap[smallest].node_id as usize;
            self.pos[id_c] = smallest;
            self.pos[id_s] = cur;
            self.heap.swap(cur, smallest);
            cur = smallest;
        }
    }

    /// DECREASE-KEY: update node with given id to new f_score.
    fn decrease_key(&mut self, node_id: u32, new_f: f64, new_g: f64, new_h: f64) {
        let i = self.pos[node_id as usize];
        self.heap[i].f_score = new_f;
        self.heap[i].g_score = new_g;
        self.heap[i].h_score = new_h;
        // Could be decrease or increase
        self.sift_up(i);
        let new_i = self.pos[node_id as usize];
        self.sift_down(new_i);
    }
}

// ============================================================
//  ESM (for completeness)
// ============================================================

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
//  Workload generation
// ============================================================

fn init_frontier(n: usize, seed: u64) -> Vec<FrontierNode> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v: Vec<FrontierNode> = (0..n as u32).map(|i| {
        let g: f64 = rng.gen_range(0.0..100.0);
        let h: f64 = rng.gen_range(0.0..100.0);
        FrontierNode { f_score: g + h, g_score: g, h_score: h, node_id: i }
    }).collect();
    v.sort();
    v
}

/// Batch relaxation: k nodes get improved g-scores (f decreases).
fn batch_relax(frontier: &mut [FrontierNode], k: usize, rng: &mut StdRng) -> Vec<usize> {
    let n = frontier.len();
    let mut picks: Vec<usize> = (0..n).collect();
    for i in 0..k { let j = rng.gen_range(i..n); picks.swap(i, j); }
    let mut dirty: Vec<usize> = picks[..k].to_vec();
    dirty.sort_unstable();
    for &i in &dirty {
        // Better path found: g decreases (f decreases)
        let improvement: f64 = rng.gen_range(0.0..20.0);
        frontier[i].g_score = (frontier[i].g_score - improvement).max(0.0);
        frontier[i].f_score = frontier[i].g_score + frontier[i].h_score;
    }
    dirty
}

fn med(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }

fn run_workload(n: usize, k: usize, n_iters: usize, csv: &mut impl Write) {
    let mut frontier = init_frontier(n, 0x4A5Bu64.wrapping_add(n as u64));
    let mut rng = StdRng::seed_from_u64(0xA5A5);

    let mut ds_v = Vec::new(); let mut esm_v = Vec::new(); let mut full_v = Vec::new();
    let mut heap_v = Vec::new();

    for _ in 0..n_iters {
        let dirty = batch_relax(&mut frontier, k, &mut rng);

        // --- DS (flat array + position map) ---
        let mut data = frontier.clone();
        let mut dv = dirty.clone(); dv.sort_unstable(); dv.dedup();
        let t = Instant::now();
        delta_sort_by(&mut data, &mut dv, cmp_node);
        ds_v.push(t.elapsed().as_nanos() as f64);
        let ds_result = data.clone();

        // --- ESM ---
        let mut data = frontier.clone();
        let mut dv = dirty.clone();
        let t = Instant::now();
        esm_sort(&mut data, &mut dv, cmp_node);
        esm_v.push(t.elapsed().as_nanos() as f64);
        assert!(data == ds_result, "ESM mismatch");

        // --- Full sort ---
        let mut data = frontier.clone();
        let t = Instant::now();
        data.sort_unstable_by(cmp_node);
        full_v.push(t.elapsed().as_nanos() as f64);
        assert!(data == ds_result, "Full mismatch");

        // --- Heap with position map: k DECREASE-KEY operations ---
        let mut heap = HeapWithPosMap::build(&frontier);
        let t = Instant::now();
        for &di in &dirty {
            let node = &frontier[di];
            heap.decrease_key(node.node_id, node.f_score, node.g_score, node.h_score);
        }
        heap_v.push(t.elapsed().as_nanos() as f64);

        // Restore sorted frontier for next iteration
        frontier.sort();
    }

    let dsm = med(ds_v); let esmm = med(esm_v); let fm = med(full_v); let hm = med(heap_v);
    let kpct = 100.0 * k as f64 / n as f64;

    let ds_wins_heap = if hm > dsm { format!("DS {:.1}×", hm/dsm) } else { format!("Heap {:.1}×", dsm/hm) };
    println!(
        "{:>8} {:>6} {:>6.2}%  {:>8.1} {:>8.1} {:>8.1}  {:>8.1}   {}",
        n, k, kpct,
        dsm/1e3, esmm/1e3, fm/1e3, hm/1e3,
        ds_wins_heap,
    );
    for (strat, val) in [("DeltaSort", dsm), ("ESM", esmm), ("Full", fm), ("Heap+PosMap", hm)] {
        writeln!(csv, "{},{},{},{:.0}", n, k, strat, val).unwrap();
    }
}

fn main() {
    fs::create_dir_all("results").ok();
    let mut f = fs::File::create("results/track_o_search_frontier.csv").expect("csv");
    writeln!(f, "n,k,strategy,ns").unwrap();

    println!("=== A* / Beam Search Frontier: Batch DECREASE-KEY ===");
    println!("(all times in μs; Heap = binary heap with position map, k × O(log n))");
    println!("{:>8} {:>6} {:>7}  {:>8} {:>8} {:>8}  {:>8}   winner",
        "n", "k", "k/n", "DS μs", "ESM μs", "Full μs", "Heap μs");

    // Beam search: small frontier
    println!("\n--- Beam search frontier (n=50-500) ---");
    for &n in &[50usize, 100, 200, 500] {
        for &k in &[1usize, 2, 5, 10, 20, 50] {
            if k <= n / 2 {
                run_workload(n, k, 51, &mut f);
            }
        }
        println!();
    }

    // A* / Dijkstra: medium graph frontier
    println!("--- A* frontier (n=1K-20K) ---");
    for &n in &[1_000usize, 5_000, 10_000, 20_000] {
        for &k in &[1usize, 5, 10, 50, 100, 500] {
            if k <= n / 4 {
                run_workload(n, k, 31, &mut f);
            }
        }
        println!();
    }

    // Large graph (road network, protein interaction)
    println!("--- Large frontier (n=50K-500K) ---");
    for &n in &[50_000usize, 100_000, 500_000] {
        for &k in &[1usize, 10, 100, 1000, 5000] {
            if k <= n / 4 {
                run_workload(n, k, 11, &mut f);
            }
        }
        println!();
    }

    println!("Wrote results/track_o_search_frontier.csv");
}
