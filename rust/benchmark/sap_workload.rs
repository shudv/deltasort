//! SAP (sweep-and-prune) sort-workload benchmark.
//!
//! Realistic scenario: a sorted array of AABB endpoints. Each frame, a subset
//! of objects move; their endpoint values are perturbed slightly. The array
//! must be re-sorted to maintain broadphase invariants.
//!
//! This is the canonical "frame-coherent flat-array incremental sort" use case
//! used by Bullet, PhysX, Box2D, Havok. Standard impl: BIS per moved endpoint.
//!
//! Comparators:
//!   - cheap (just f32) — typical SAP
//!   - composite (priority + f32) — multi-axis tie-break
//!
//! Algorithms compared:
//!   - DeltaSort
//!   - BIS (binary insertion sort over dirty list, sorted)
//!   - ESM (extract-sort-merge)
//!   - sort_unstable_by (full re-sort baseline)
//!   - std::BTreeMap<f32-bits, EndpointId> (indirected baseline for sanity)

use deltasort::delta_sort_by;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::time::Instant;

// Endpoint type for SAP: an axis endpoint with (position, object_id, kind)
#[derive(Clone, Default, PartialEq)]
struct Endpoint {
    pos: f32,
    obj_id: u32,
    kind: u8, // 0=min, 1=max
}
impl Eq for Endpoint {}
impl Ord for Endpoint {
    fn cmp(&self, o: &Self) -> Ordering {
        self.pos.partial_cmp(&o.pos).unwrap_or(Ordering::Equal)
            .then(self.obj_id.cmp(&o.obj_id))
            .then(self.kind.cmp(&o.kind))
    }
}
impl PartialOrd for Endpoint { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }

fn cmp_ep(a: &Endpoint, b: &Endpoint) -> Ordering { a.cmp(b) }

// ----------- Algorithms -----------

fn bis_sort<T: Clone, F: Fn(&T, &T) -> Ordering>(arr: &mut [T], dirty: &[usize], cmp: F) {
    // Standard binary-insertion-sort-on-dirty-list approach (BIS).
    // For each dirty index (sorted), extract and re-insert via binary search
    // into the remaining sorted array.
    if dirty.is_empty() { return; }
    let mut d = dirty.to_vec();
    d.sort_unstable();
    let n = arr.len();
    // Extract dirty values to a side buffer first to avoid double-processing
    let dirty_vals: Vec<T> = d.iter().map(|&i| arr[i].clone()).collect();
    // Mark dirty positions as "removed" by compacting. We use a simple approach:
    // make a Vec of clean elements, then binary-insert each dirty value.
    let mut clean: Vec<T> = Vec::with_capacity(n - d.len());
    let mut di = 0;
    for i in 0..n {
        if di < d.len() && d[di] == i { di += 1; } else { clean.push(arr[i].clone()); }
    }
    // Now binary-insert each dirty value into clean
    for v in dirty_vals {
        let pos = clean.binary_search_by(|x| cmp(x, &v)).unwrap_or_else(|e| e);
        clean.insert(pos, v);
    }
    arr.clone_from_slice(&clean);
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

// ----------- Scenario generator -----------

struct Scene {
    endpoints: Vec<Endpoint>,         // sorted on x-axis
    obj_positions: Vec<(f32, f32)>,   // (min, max) per object
    n_obj: usize,
}

fn build_scene(n_obj: usize, seed: u64) -> Scene {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut obj_positions = Vec::with_capacity(n_obj);
    let mut endpoints = Vec::with_capacity(2 * n_obj);
    let box_size = 1.0f32;
    let world_extent = (n_obj as f32) * box_size * 0.5;
    for i in 0..n_obj {
        let center: f32 = rng.gen_range(0.0..world_extent);
        let mn = center - box_size * 0.5;
        let mx = center + box_size * 0.5;
        obj_positions.push((mn, mx));
        endpoints.push(Endpoint { pos: mn, obj_id: i as u32, kind: 0 });
        endpoints.push(Endpoint { pos: mx, obj_id: i as u32, kind: 1 });
    }
    endpoints.sort();
    Scene { endpoints, obj_positions, n_obj }
}

/// Move `n_moving` random objects by a small delta. Returns dirty endpoint
/// indices (the positions in `endpoints` that changed value).
fn move_objects(scene: &mut Scene, n_moving: usize, rng: &mut StdRng) -> Vec<usize> {
    let mut moved_objs: Vec<u32> = (0..scene.n_obj as u32).collect();
    // Fisher-Yates pick n_moving
    for i in 0..n_moving {
        let j = rng.gen_range(i..scene.n_obj);
        moved_objs.swap(i, j);
    }
    let moved_set: std::collections::HashSet<u32> = moved_objs[..n_moving].iter().copied().collect();

    // Update obj_positions
    for &obj_id in &moved_objs[..n_moving] {
        let delta: f32 = rng.gen_range(-0.3..0.3);
        let (mn, mx) = scene.obj_positions[obj_id as usize];
        scene.obj_positions[obj_id as usize] = (mn + delta, mx + delta);
    }

    // Find dirty endpoint indices and update their pos in place
    let mut dirty: Vec<usize> = Vec::with_capacity(2 * n_moving);
    for (i, ep) in scene.endpoints.iter_mut().enumerate() {
        if moved_set.contains(&ep.obj_id) {
            let (mn, mx) = scene.obj_positions[ep.obj_id as usize];
            ep.pos = if ep.kind == 0 { mn } else { mx };
            dirty.push(i);
        }
    }
    dirty
}

// ----------- Runner -----------

fn run_frame(scene: &Scene, dirty: &[usize]) -> (f64, f64, f64, f64, f64) {
    // Snapshot the endpoints array (already updated in-place); dirty list is sorted ascending
    let baseline = &scene.endpoints;
    let n = baseline.len();
    let k = dirty.len();
    let _ = k; let _ = n;

    // --- DeltaSort ---
    let mut data = baseline.clone();
    let mut dirty_v: Vec<usize> = dirty.to_vec();
    dirty_v.sort_unstable();
    dirty_v.dedup();
    let t = Instant::now();
    delta_sort_by(&mut data, &mut dirty_v, cmp_ep);
    let ds_ns = t.elapsed().as_nanos() as f64;
    let ds_result = data;

    // --- BIS ---
    let mut data = baseline.clone();
    let t = Instant::now();
    bis_sort(&mut data, dirty, cmp_ep);
    let bis_ns = t.elapsed().as_nanos() as f64;
    assert!(data == ds_result, "BIS mismatch");

    // --- ESM ---
    let mut data = baseline.clone();
    let mut dirty_v: Vec<usize> = dirty.to_vec();
    let t = Instant::now();
    esm_sort(&mut data, &mut dirty_v, cmp_ep);
    let esm_ns = t.elapsed().as_nanos() as f64;
    assert!(data == ds_result, "ESM mismatch");

    // --- Full re-sort ---
    let mut data = baseline.clone();
    let t = Instant::now();
    data.sort_unstable_by(cmp_ep);
    let full_ns = t.elapsed().as_nanos() as f64;
    assert!(data == ds_result, "Full mismatch");

    // --- BTreeMap (indirected, for reference only — not flat array) ---
    // We model it as: maintain a BTreeMap<Endpoint, ()> incrementally
    // Per frame: remove old k entries, insert k new entries
    // To keep things fair, we precompute the "before" endpoint values per dirty index
    // For this benchmark, we instead just rebuild from data (worst case for BTreeMap)
    // since we don't track "old" values. We measure just the bulk insert cost.
    let t = Instant::now();
    let map: BTreeMap<Endpoint, ()> = baseline.iter().cloned().map(|e| (e, ())).collect();
    let btmap_full_ns = t.elapsed().as_nanos() as f64;
    let _ = map;

    (ds_ns, bis_ns, esm_ns, full_ns, btmap_full_ns)
}

fn run_workload(n_obj: usize, k_frac: f64, n_frames: usize, csv: &mut impl Write) {
    let n_moving = ((n_obj as f64) * k_frac).ceil() as usize;
    let n_moving = n_moving.max(1);
    let mut scene = build_scene(n_obj, 0xCA75_F00Du64.wrapping_add(n_obj as u64));
    let mut rng = StdRng::seed_from_u64(0xB0F0u64);

    let mut ds_v = Vec::with_capacity(n_frames);
    let mut bis_v = Vec::with_capacity(n_frames);
    let mut esm_v = Vec::with_capacity(n_frames);
    let mut full_v = Vec::with_capacity(n_frames);
    let mut btmap_v = Vec::with_capacity(n_frames);

    for _ in 0..n_frames {
        let dirty = move_objects(&mut scene, n_moving, &mut rng);
        let (ds, bis, esm, full, btmap) = run_frame(&scene, &dirty);
        ds_v.push(ds); bis_v.push(bis); esm_v.push(esm); full_v.push(full); btmap_v.push(btmap);
        // After this frame, scene.endpoints is *not* re-sorted, so the next move starts from
        // a permuted state. But our benchmark always sorts a fresh clone so it's fine
        // to fix it back to sorted to keep the experiment stationary.
        scene.endpoints.sort();
    }

    let med = |mut v: Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
    let dsm = med(ds_v); let bism = med(bis_v); let esmm = med(esm_v); let fullm = med(full_v); let btmapm = med(btmap_v);

    let n_endpoints = 2 * n_obj;
    let k_endpoints = 2 * n_moving;
    let k_pct_ep = 100.0 * (k_endpoints as f64) / (n_endpoints as f64);

    println!(
        "{:>10} {:>10} {:>6.3}%  {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>12.1}   {:>6.2}× {:>6.2}× {:>6.2}×",
        n_obj, k_endpoints, k_pct_ep,
        dsm / 1000.0, bism / 1000.0, esmm / 1000.0, fullm / 1000.0, btmapm / 1000.0,
        bism / dsm, esmm / dsm, fullm / dsm,
    );

    writeln!(csv, "{},{},{},DeltaSort,{:.0}", n_obj, n_endpoints, k_endpoints, dsm).unwrap();
    writeln!(csv, "{},{},{},BIS,{:.0}", n_obj, n_endpoints, k_endpoints, bism).unwrap();
    writeln!(csv, "{},{},{},ESM,{:.0}", n_obj, n_endpoints, k_endpoints, esmm).unwrap();
    writeln!(csv, "{},{},{},FullSort,{:.0}", n_obj, n_endpoints, k_endpoints, fullm).unwrap();
    writeln!(csv, "{},{},{},BTreeMap_rebuild,{:.0}", n_obj, n_endpoints, k_endpoints, btmapm).unwrap();
}

fn main() {
    fs::create_dir_all("results").ok();
    let mut f = fs::File::create("results/track_e_sap.csv").expect("create csv");
    writeln!(f, "n_obj,n_endpoints,k_endpoints,strategy,ns").unwrap();

    println!(
        "{:>10} {:>10} {:>7}  {:>10} {:>10} {:>10} {:>10} {:>12}   {:>7} {:>7} {:>7}",
        "n_obj", "k_ep", "k/n_ep", "DS us", "BIS us", "ESM us", "Full us", "BT_full us",
        "BIS/DS", "ESM/DS", "Full/DS",
    );

    let n_frames = 21;

    // Realistic frame coherence: 0.5% - 10% of objects move per frame
    for &n in &[1_000usize, 5_000, 25_000, 100_000] {
        for &k_frac in &[0.001f64, 0.005, 0.01, 0.05, 0.1, 0.25] {
            run_workload(n, k_frac, n_frames, &mut f);
        }
        println!();
    }

    println!("Wrote results/track_e_sap.csv");
}
