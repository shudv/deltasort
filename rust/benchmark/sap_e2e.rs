//! SAP E2E broadphase benchmark: sort + pair-list maintenance.
//!
//! Validation extensions:
//!   - Pair-list maintenance (overlapping pair add/remove)
//!   - Adversarial clustered-motion workload (worst case for DS)
//!   - Delta-magnitude sweep (how much frame coherence matters)
//!
//! For DS: pair maintenance is a post-pass: for each dirty object, scan
//! rightward in the sorted endpoints list and check overlap with neighbors.
//! For bubble-SAP: pair maintenance happens during swaps (free-ish).
//! For ESM: same post-pass as DS.

use deltasort::delta_sort_by;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::time::Instant;

#[derive(Clone, Default, PartialEq, Debug)]
struct Endpoint { pos: f32, obj_id: u32, kind: u8 }
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

fn bubble_sap_with_pairs(
    arr: &mut [Endpoint],
    pairs: &mut HashSet<(u32, u32)>,
    dirty: &[usize],
    deltas: &std::collections::HashMap<u32, f32>,
) {
    let n = arr.len();
    // For each dirty endpoint, bubble in the direction of motion.
    // Multiple dirty endpoints may need re-bubbling after each other shifts
    // the array, so iterate to a fixed point. (In practice 1-2 iterations.)
    let dirty_set: HashSet<usize> = dirty.iter().copied().collect();
    let _ = dirty_set;
    // Track current location of each "dirty obj_id, kind" pair since indices
    // shift as we bubble.
    let mut cur_idx: std::collections::HashMap<(u32, u8), usize> = std::collections::HashMap::new();
    for &i in dirty {
        cur_idx.insert((arr[i].obj_id, arr[i].kind), i);
    }

    // Process each dirty endpoint, bubbling it to its sorted position
    // Order: process leftward-movers in left-to-right index order,
    // rightward-movers in right-to-left index order, so they don't interfere.
    let mut left_movers: Vec<(usize, u32, u8)> = Vec::new();
    let mut right_movers: Vec<(usize, u32, u8)> = Vec::new();
    for &i in dirty {
        let obj = arr[i].obj_id;
        let delta = *deltas.get(&obj).unwrap_or(&0.0);
        if delta > 0.0 { right_movers.push((i, obj, arr[i].kind)); }
        else if delta < 0.0 { left_movers.push((i, obj, arr[i].kind)); }
    }
    right_movers.sort_by_key(|x| std::cmp::Reverse(x.0));
    left_movers.sort_by_key(|x| x.0);

    for &(_, obj, kind) in &right_movers {
        let mut j = *cur_idx.get(&(obj, kind)).unwrap();
        while j + 1 < n && cmp_ep(&arr[j], &arr[j + 1]) == Ordering::Greater {
            let other_kind = arr[j + 1].kind;
            let other_obj = arr[j + 1].obj_id;
            if obj != other_obj {
                let key = if obj < other_obj { (obj, other_obj) } else { (other_obj, obj) };
                if kind == 1 && other_kind == 0 {
                    pairs.insert(key);
                } else if kind == 0 && other_kind == 1 {
                    pairs.remove(&key);
                }
            }
            // Update cur_idx for the displaced endpoint if it was dirty too
            if let Some(slot) = cur_idx.get_mut(&(other_obj, other_kind)) {
                if *slot == j + 1 { *slot = j; }
            }
            arr.swap(j, j + 1);
            j += 1;
        }
        cur_idx.insert((obj, kind), j);
    }

    for &(_, obj, kind) in &left_movers {
        let mut j = *cur_idx.get(&(obj, kind)).unwrap();
        while j > 0 && cmp_ep(&arr[j - 1], &arr[j]) == Ordering::Greater {
            let other_kind = arr[j - 1].kind;
            let other_obj = arr[j - 1].obj_id;
            if obj != other_obj {
                let key = if obj < other_obj { (obj, other_obj) } else { (other_obj, obj) };
                if kind == 0 && other_kind == 1 {
                    pairs.insert(key);
                } else if kind == 1 && other_kind == 0 {
                    pairs.remove(&key);
                }
            }
            if let Some(slot) = cur_idx.get_mut(&(other_obj, other_kind)) {
                if *slot == j - 1 { *slot = j; }
            }
            arr.swap(j - 1, j);
            j -= 1;
        }
        cur_idx.insert((obj, kind), j);
    }
}

/// Post-pass pair rebuild for DS/ESM: full O(n + p) sweep over sorted axis.
/// This is what a production system would do: re-sort, then walk endpoints
/// maintaining an active set, emitting overlapping pairs.
fn rebuild_pairs_around_dirty(
    arr: &[Endpoint],
    _dirty_obj_ids: &HashSet<u32>,
    pairs: &mut HashSet<(u32, u32)>,
) {
    pairs.clear();
    let mut active: HashSet<u32> = HashSet::with_capacity(64);
    for ep in arr {
        if ep.kind == 0 {
            for &other in &active {
                if other != ep.obj_id {
                    let key = if ep.obj_id < other { (ep.obj_id, other) } else { (other, ep.obj_id) };
                    pairs.insert(key);
                }
            }
            active.insert(ep.obj_id);
        } else {
            active.remove(&ep.obj_id);
        }
    }
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

// =========== Scene management ===========

#[derive(Clone)]
enum Motion { Uniform, Clustered }

struct Scene {
    endpoints: Vec<Endpoint>,
    obj_positions: Vec<(f32, f32)>,
    n_obj: usize,
}

fn build_scene(n_obj: usize, box_size: f32, seed: u64) -> Scene {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut obj_positions = Vec::with_capacity(n_obj);
    let mut endpoints = Vec::with_capacity(2 * n_obj);
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

/// Returns (dirty_endpoint_indices, dirty_obj_ids, dirty_deltas[obj_id->delta]).
fn move_objects(
    scene: &mut Scene,
    n_moving: usize,
    delta_mag: f32,
    motion: Motion,
    rng: &mut StdRng,
) -> (Vec<usize>, HashSet<u32>, std::collections::HashMap<u32, f32>) {
    // Pick which objects move
    let moved_objs: Vec<u32> = match motion {
        Motion::Uniform => {
            let mut all: Vec<u32> = (0..scene.n_obj as u32).collect();
            for i in 0..n_moving { let j = rng.gen_range(i..scene.n_obj); all.swap(i, j); }
            all[..n_moving].to_vec()
        }
        Motion::Clustered => {
            let mut ids_pos: Vec<(u32, f32)> = (0..scene.n_obj as u32)
                .map(|i| (i, scene.obj_positions[i as usize].0))
                .collect();
            ids_pos.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            let start = rng.gen_range(0..(scene.n_obj - n_moving + 1));
            ids_pos[start..start + n_moving].iter().map(|(i, _)| *i).collect()
        }
    };
    let moved_set: HashSet<u32> = moved_objs.iter().copied().collect();

    let direction = match motion { Motion::Clustered => 1.0f32, Motion::Uniform => 0.0 };
    let mut deltas: std::collections::HashMap<u32, f32> = std::collections::HashMap::with_capacity(n_moving);
    for &obj_id in &moved_objs {
        let delta: f32 = if direction != 0.0 {
            direction * delta_mag * (0.5 + rng.gen_range(0.0..1.0))
        } else {
            rng.gen_range(-delta_mag..delta_mag)
        };
        let (mn, mx) = scene.obj_positions[obj_id as usize];
        scene.obj_positions[obj_id as usize] = (mn + delta, mx + delta);
        deltas.insert(obj_id, delta);
    }

    let mut dirty = Vec::with_capacity(2 * n_moving);
    for (i, ep) in scene.endpoints.iter_mut().enumerate() {
        if moved_set.contains(&ep.obj_id) {
            let (mn, mx) = scene.obj_positions[ep.obj_id as usize];
            ep.pos = if ep.kind == 0 { mn } else { mx };
            dirty.push(i);
        }
    }
    (dirty, moved_set, deltas)
}

struct FrameOut {
    ds_sort_ns: f64, ds_pair_ns: f64,
    bub_total_ns: f64,
    esm_sort_ns: f64, esm_pair_ns: f64,
}

fn run_frame(
    baseline: &[Endpoint],
    bub_pairs: &mut HashSet<(u32, u32)>,
    dirty: &[usize],
    dirty_obj_ids: &HashSet<u32>,
    deltas: &std::collections::HashMap<u32, f32>,
) -> FrameOut {
    // DS path: compute pair set from scratch each frame via full sweep
    let mut data = baseline.to_vec();
    let mut dirty_v: Vec<usize> = dirty.to_vec();
    dirty_v.sort_unstable();
    dirty_v.dedup();
    let mut pairs_ds: HashSet<(u32, u32)> = HashSet::new();
    let t = Instant::now();
    delta_sort_by(&mut data, &mut dirty_v, cmp_ep);
    let ds_sort_ns = t.elapsed().as_nanos() as f64;
    let ds_result = data.clone();
    let t = Instant::now();
    rebuild_pairs_around_dirty(&data, dirty_obj_ids, &mut pairs_ds);
    let ds_pair_ns = t.elapsed().as_nanos() as f64;
    let pairs_ds_final = pairs_ds;

    // Bubble path: persistent pair state, incremental update fused with sort
    let mut data = baseline.to_vec();
    let t = Instant::now();
    bubble_sap_with_pairs(&mut data, bub_pairs, dirty, deltas);
    let bub_total_ns = t.elapsed().as_nanos() as f64;
    assert!(data == ds_result, "bubble sort mismatch");
    if *bub_pairs != pairs_ds_final {
        let diff_bub: Vec<_> = bub_pairs.difference(&pairs_ds_final).take(3).collect();
        let diff_ds: Vec<_> = pairs_ds_final.difference(bub_pairs).take(3).collect();
        eprintln!("Bub has but DS doesn't: {:?}", diff_bub);
        eprintln!("DS has but Bub doesn't: {:?}", diff_ds);
        panic!("pair set mismatch: bub {} vs ds {}", bub_pairs.len(), pairs_ds_final.len());
    }

    // ESM path: compute pair set from scratch (same approach as DS)
    let mut data = baseline.to_vec();
    let mut dirty_v: Vec<usize> = dirty.to_vec();
    let mut pairs_esm: HashSet<(u32, u32)> = HashSet::new();
    let t = Instant::now();
    esm_sort(&mut data, &mut dirty_v, cmp_ep);
    let esm_sort_ns = t.elapsed().as_nanos() as f64;
    assert!(data == ds_result, "ESM sort mismatch");
    let t = Instant::now();
    rebuild_pairs_around_dirty(&data, dirty_obj_ids, &mut pairs_esm);
    let esm_pair_ns = t.elapsed().as_nanos() as f64;
    assert!(pairs_esm == pairs_ds_final, "pair ESM mismatch");

    FrameOut { ds_sort_ns, ds_pair_ns, bub_total_ns, esm_sort_ns, esm_pair_ns }
}

/// Compute initial pair list (slow O(n^2), only at scene init)
fn compute_initial_pairs(arr: &[Endpoint]) -> HashSet<(u32, u32)> {
    let n = arr.len();
    let mut pairs = HashSet::new();
    let mut active: HashSet<u32> = HashSet::new();
    for i in 0..n {
        if arr[i].kind == 0 {
            for &other in active.iter() {
                if other != arr[i].obj_id {
                    let key = if arr[i].obj_id < other { (arr[i].obj_id, other) } else { (other, arr[i].obj_id) };
                    pairs.insert(key);
                }
            }
            active.insert(arr[i].obj_id);
        } else {
            active.remove(&arr[i].obj_id);
        }
    }
    pairs
}

fn run_workload(
    label: &str,
    n_obj: usize,
    k_frac: f64,
    delta_mag: f32,
    motion: Motion,
    n_frames: usize,
    csv: &mut impl Write,
) {
    let box_size = 1.0f32;
    let mut scene = build_scene(n_obj, box_size, 0xCA75_F00Du64.wrapping_add(n_obj as u64));
    let mut bub_pairs = compute_initial_pairs(&scene.endpoints);
    let mut rng = StdRng::seed_from_u64(0xB0F0u64);
    let n_moving = ((n_obj as f64) * k_frac).ceil().max(1.0) as usize;

    let mut ds_sort = Vec::new(); let mut ds_pair = Vec::new();
    let mut bub_total = Vec::new();
    let mut esm_sort = Vec::new(); let mut esm_pair = Vec::new();

    for _ in 0..n_frames {
        let (dirty, dirty_objs, deltas) = move_objects(&mut scene, n_moving, delta_mag, motion.clone(), &mut rng);
        let r = run_frame(&scene.endpoints, &mut bub_pairs, &dirty, &dirty_objs, &deltas);
        ds_sort.push(r.ds_sort_ns); ds_pair.push(r.ds_pair_ns);
        bub_total.push(r.bub_total_ns);
        esm_sort.push(r.esm_sort_ns); esm_pair.push(r.esm_pair_ns);
        scene.endpoints.sort();
    }
    let med = |mut v: Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
    let dsm = med(ds_sort) + med(ds_pair);
    let bm = med(bub_total);
    let esmm = med(esm_sort) + med(esm_pair);
    let k_ep = 2 * n_moving;
    let kpct = 100.0 * k_ep as f64 / (2 * n_obj) as f64;
    println!(
        "{:>10} {:>8} {:>7} {:>6.3}% δ={:>5.2}  {:>10.1} {:>10.1} {:>10.1}   {:>7.2}× {:>7.2}×",
        label, n_obj, k_ep, kpct, delta_mag,
        dsm / 1000.0, bm / 1000.0, esmm / 1000.0,
        bm / dsm, esmm / dsm,
    );
    writeln!(csv, "{},{},{},{},DeltaSort_e2e,{:.0}", label, n_obj, k_ep, delta_mag, dsm).unwrap();
    writeln!(csv, "{},{},{},{},BubbleSAP_e2e,{:.0}", label, n_obj, k_ep, delta_mag, bm).unwrap();
    writeln!(csv, "{},{},{},{},ESM_e2e,{:.0}", label, n_obj, k_ep, delta_mag, esmm).unwrap();
}

fn main() {
    fs::create_dir_all("results").ok();
    let mut f = fs::File::create("results/track_h_sap_e2e.csv").expect("csv");
    writeln!(f, "scenario,n_obj,k_ep,delta_mag,strategy,ns").unwrap();
    println!(
        "{:>10} {:>8} {:>7} {:>7} {:>7}  {:>10} {:>10} {:>10}   {:>8} {:>8}",
        "scenario", "n_obj", "k_ep", "k/n_ep", "δ",
        "DS E2E us", "Bub E2E us", "ESM E2E us", "Bub/DS", "ESM/DS",
    );

    let n_frames = 21;

    println!("\n=== 1) Frame-coherent (uniform motion, small delta) — REALISTIC ===");
    for &n in &[10_000usize, 100_000] {
        for &k in &[0.001f64, 0.01, 0.05, 0.1] {
            run_workload("stable", n, k, 0.1, Motion::Uniform, n_frames, &mut f);
        }
    }

    println!("\n=== 2) High-energy uniform (large delta, sparse motion) ===");
    for &n in &[10_000usize, 100_000] {
        for &k in &[0.01f64, 0.05, 0.1] {
            run_workload("explosive", n, k, 5.0, Motion::Uniform, n_frames, &mut f);
        }
    }

    println!("\n=== 3) ADVERSARIAL clustered motion (worst case for DS) ===");
    for &n in &[10_000usize, 100_000] {
        for &k in &[0.01f64, 0.05, 0.1] {
            run_workload("clustered", n, k, 1.0, Motion::Clustered, n_frames, &mut f);
        }
    }

    println!("\n=== 4) Delta sweep (n=100k, k=1%) ===");
    for &d in &[0.01f32, 0.1, 0.5, 1.0, 5.0, 20.0] {
        run_workload("delta-sweep", 100_000, 0.01, d, Motion::Uniform, n_frames, &mut f);
    }

    println!("\nWrote results/track_h_sap_e2e.csv");
}
