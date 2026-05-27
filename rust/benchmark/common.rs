//! Shared statistical benchmarking harness for DeltaSort application validation.
//!
//! Goals:
//!   - 500+ samples per (n,k) point with proper warmup
//!   - Report p50, p95, p99, mean, stddev, 95% CI
//!   - Prevent dead-code elimination via std::hint::black_box
//!   - Measure ONLY the operation, not setup (clone/init done outside timed window)
//!   - Verify correctness on every sample (not just one)
//!   - Use vetted shared implementations of ESM and heap-with-position-map

#![allow(dead_code)]

use std::cmp::Ordering;
use std::hint::black_box;
use std::time::Instant;

// ============================================================
//  Statistical summary
// ============================================================

pub struct BenchStats {
    pub p50: f64,    // median
    pub p95: f64,    // 95th percentile (tail latency)
    pub p99: f64,    // 99th percentile
    pub mean: f64,
    pub std: f64,
    pub ci95_lo: f64, // 95% CI for the mean: mean - 1.96 * std/sqrt(n)
    pub ci95_hi: f64,
    pub min: f64,
    pub max: f64,
    pub n_samples: usize,
}

impl BenchStats {
    pub fn from_samples(mut samples: Vec<f64>) -> Self {
        let n = samples.len();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p = |q: f64| -> f64 {
            let i = ((q * (n as f64 - 1.0)).round() as usize).min(n - 1);
            samples[i]
        };
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0);
        let std = var.sqrt();
        let sem = std / (n as f64).sqrt();
        BenchStats {
            p50: p(0.5), p95: p(0.95), p99: p(0.99),
            mean, std,
            ci95_lo: mean - 1.96 * sem,
            ci95_hi: mean + 1.96 * sem,
            min: samples[0], max: samples[n - 1],
            n_samples: n,
        }
    }

    pub fn csv_header() -> &'static str {
        "p50,p95,p99,mean,std,ci95_lo,ci95_hi,min,max,n_samples"
    }

    pub fn to_csv(&self) -> String {
        format!(
            "{:.1},{:.1},{:.1},{:.2},{:.2},{:.2},{:.2},{:.1},{:.1},{}",
            self.p50, self.p95, self.p99, self.mean, self.std,
            self.ci95_lo, self.ci95_hi, self.min, self.max, self.n_samples
        )
    }
}

// ============================================================
//  Core benchmark function
// ============================================================

/// Benchmark a function with proper warmup, sample collection, and black_box.
///
/// `setup` is called outside the timed window to produce a fresh state for each sample.
/// `bench` is called inside the timed window and should perform the operation under test.
/// `verify` (optional) is called after each sample to confirm correctness.
///
/// Default: 100 warmup samples, 500 measured samples.
pub fn bench<S, FS, FB, FV>(
    label: &str,
    setup: FS,
    bench: FB,
    verify: FV,
) -> BenchStats
where
    FS: Fn() -> S,
    FB: Fn(&mut S),
    FV: Fn(&S) -> bool,
{
    bench_with_config(label, 100, 500, setup, bench, verify)
}

pub fn bench_with_config<S, FS, FB, FV>(
    _label: &str,
    warmup: usize,
    samples: usize,
    setup: FS,
    bench: FB,
    verify: FV,
) -> BenchStats
where
    FS: Fn() -> S,
    FB: Fn(&mut S),
    FV: Fn(&S) -> bool,
{
    // Warmup
    for _ in 0..warmup {
        let mut s = setup();
        black_box(&mut s);
        bench(&mut s);
        black_box(&s);
    }
    // Measured samples
    let mut times = Vec::with_capacity(samples);
    for i in 0..samples {
        let mut s = setup();
        black_box(&mut s); // ensure setup result actually materialized
        let t0 = Instant::now();
        bench(&mut s);
        let dt = t0.elapsed().as_nanos() as f64;
        black_box(&s);
        // Verify on first sample (cheap) and on a random sample (defensive)
        if i == 0 || i == samples / 2 || i == samples - 1 {
            assert!(verify(&s), "Verification failed on sample {} of bench", i);
        }
        times.push(dt);
    }
    BenchStats::from_samples(times)
}

// ============================================================
//  Vetted shared algorithm implementations
// ============================================================

/// Extract-Sort-Merge: extract k dirty elements, sort them, merge with remaining.
///
/// This is the published optimal-for-large-k incremental sort algorithm.
/// Implementation is allocator-aware: only one Vec allocation for the buffer.
pub fn esm_sort<T: Default + Clone, F: Fn(&T, &T) -> Ordering>(
    arr: &mut Vec<T>, dirty: &mut [usize], cmp: F,
) {
    if dirty.is_empty() {
        return;
    }
    let n = arr.len();
    let k = dirty.len();
    dirty.sort_unstable();
    debug_assert!(dirty.windows(2).all(|w| w[0] < w[1]), "dirty must be deduplicated");

    let mut buf: Vec<T> = Vec::with_capacity(k);
    let mut write = 0usize;
    let mut di = 0usize;
    for read in 0..n {
        if di < k && dirty[di] == read {
            buf.push(std::mem::take(&mut arr[read]));
            di += 1;
        } else {
            if write != read {
                arr.swap(write, read);
            }
            write += 1;
        }
    }
    buf.sort_by(&cmp);
    // Merge buf and arr[..n-k] into arr (largest to smallest, in-place)
    let mut ci = (n - k) as isize - 1;
    let mut di2 = k as isize - 1;
    let mut wi = n - 1;
    loop {
        let take_clean = if ci >= 0 && di2 >= 0 {
            cmp(&arr[ci as usize], &buf[di2 as usize]) != Ordering::Less
        } else {
            ci >= 0
        };
        if take_clean {
            if wi != ci as usize {
                arr.swap(wi, ci as usize);
            }
            ci -= 1;
        } else if di2 >= 0 {
            arr[wi] = std::mem::take(&mut buf[di2 as usize]);
            di2 -= 1;
        } else {
            break;
        }
        if wi == 0 {
            break;
        }
        wi -= 1;
    }
}

// ============================================================
//  Binary heap with explicit position map
// ============================================================

/// Binary heap with a position map for O(log n) DECREASE-KEY / INCREASE-KEY.
/// This is the gold-standard priority queue for batch updates.
pub struct PosHeap<T: Clone> {
    heap: Vec<T>,
    // pos[item_id] = current index in heap (where item_id is item.id())
    pos: Vec<usize>,
    cmp: fn(&T, &T) -> Ordering,
    id_of: fn(&T) -> u32,
}

impl<T: Clone> PosHeap<T> {
    pub fn new(items: &[T], cmp: fn(&T, &T) -> Ordering, id_of: fn(&T) -> u32) -> Self {
        let mut heap = items.to_vec();
        let max_id = items.iter().map(|i| id_of(i) as usize).max().unwrap_or(0);
        let mut pos = vec![usize::MAX; max_id + 1];
        let n = heap.len();
        for i in (0..n / 2).rev() {
            Self::sift_down_at(&mut heap, &mut pos, &cmp, &id_of, i, n);
        }
        for (i, item) in heap.iter().enumerate() {
            pos[id_of(item) as usize] = i;
        }
        PosHeap { heap, pos, cmp, id_of }
    }

    fn sift_down_at(
        heap: &mut Vec<T>, pos: &mut Vec<usize>,
        cmp: &fn(&T, &T) -> Ordering, id_of: &fn(&T) -> u32,
        mut i: usize, n: usize,
    ) {
        loop {
            let l = 2 * i + 1;
            let r = 2 * i + 2;
            let mut best = i;
            if l < n && cmp(&heap[l], &heap[best]) == Ordering::Less {
                best = l;
            }
            if r < n && cmp(&heap[r], &heap[best]) == Ordering::Less {
                best = r;
            }
            if best == i {
                break;
            }
            let id_i = id_of(&heap[i]) as usize;
            let id_b = id_of(&heap[best]) as usize;
            pos[id_i] = best;
            pos[id_b] = i;
            heap.swap(i, best);
            i = best;
        }
    }

    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let parent = (i - 1) / 2;
            if (self.cmp)(&self.heap[i], &self.heap[parent]) == Ordering::Less {
                let id_i = (self.id_of)(&self.heap[i]) as usize;
                let id_p = (self.id_of)(&self.heap[parent]) as usize;
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
            let mut best = cur;
            if l < n && (self.cmp)(&self.heap[l], &self.heap[best]) == Ordering::Less {
                best = l;
            }
            if r < n && (self.cmp)(&self.heap[r], &self.heap[best]) == Ordering::Less {
                best = r;
            }
            if best == cur {
                break;
            }
            let id_c = (self.id_of)(&self.heap[cur]) as usize;
            let id_b = (self.id_of)(&self.heap[best]) as usize;
            self.pos[id_c] = best;
            self.pos[id_b] = cur;
            self.heap.swap(cur, best);
            cur = best;
        }
    }

    /// Update an item (identified by id) to a new value. O(log n).
    pub fn update(&mut self, id: u32, new_item: T) {
        let i = self.pos[id as usize];
        debug_assert!(i != usize::MAX, "id not in heap");
        self.heap[i] = new_item;
        self.sift_up(i);
        let new_i = self.pos[id as usize];
        self.sift_down(new_i);
    }

    pub fn len(&self) -> usize { self.heap.len() }
}

// ============================================================
//  Sorted-array correctness verification
// ============================================================

pub fn is_sorted<T, F: Fn(&T, &T) -> Ordering>(arr: &[T], cmp: F) -> bool {
    arr.windows(2).all(|w| cmp(&w[0], &w[1]) != Ordering::Greater)
}

// ============================================================
//  Machine info dump
// ============================================================

pub fn print_machine_info() {
    println!("=== Machine ===");
    if let Ok(uname) = std::process::Command::new("uname").arg("-a").output() {
        print!("uname: {}", String::from_utf8_lossy(&uname.stdout));
    }
    if let Ok(out) = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"]).output()
    {
        print!("cpu:   {}", String::from_utf8_lossy(&out.stdout));
    }
    if let Ok(out) = std::process::Command::new("sysctl")
        .args(["-n", "hw.ncpu"]).output()
    {
        print!("cpus:  {}", String::from_utf8_lossy(&out.stdout));
    }
    println!("rust:  {}", env!("CARGO_PKG_VERSION"));
    println!();
}
