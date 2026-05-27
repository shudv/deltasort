//! Particle/sprite depth-sort benchmark.
//!
//! Realistic rendering scenario: per-frame painter's algorithm. Sprites are
//! depth-sorted for alpha blending. Camera and most sprites are stable; only
//! a small subset of sprites have meaningful depth changes per frame:
//!   - Newly spawned particles (e.g., explosions, smoke wisps)
//!   - Dying particles being removed
//!   - Animated sprites whose anchor moves
//!
//! Most game engines today use:
//!   - std::sort / Array.sort (full re-sort)
//!   - Insertion sort (for small N)
//!
//! BTreeMap is not used because GPU draw-call submission requires a flat
//! sorted array (state changes per draw call must be linear).
//!
//! Comparator: cheap f32 depth + tiebreaker.

use deltasort::delta_sort_by;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::fs;
use std::io::Write;
use std::time::Instant;

#[derive(Clone, Default, PartialEq)]
struct Sprite {
    depth: f32,
    sprite_id: u32,
    material: u8, // tiebreaker (batch by material)
}
impl Eq for Sprite {}
impl Ord for Sprite {
    fn cmp(&self, o: &Self) -> Ordering {
        self.depth.partial_cmp(&o.depth).unwrap_or(Ordering::Equal)
            .then(self.material.cmp(&o.material))
            .then(self.sprite_id.cmp(&o.sprite_id))
    }
}
impl PartialOrd for Sprite { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }

fn cmp_sp(a: &Sprite, b: &Sprite) -> Ordering { a.cmp(b) }

/// Directional bubble (Bullet/Box2D style): each dirty endpoint bubbles
/// to its correct position. We need old positions to determine direction.
/// (For depth-sort, the production baseline is std::sort; bubble is what
/// you'd do with cached prev-pos.)
fn directional_bubble<F: Fn(&Sprite, &Sprite) -> Ordering + Copy>(
    arr: &mut [Sprite], dirty: &[usize], old_depths: &[(usize, f32)], cmp: F,
) {
    let n = arr.len();
    let mut right_movers: Vec<usize> = Vec::new();
    let mut left_movers: Vec<usize> = Vec::new();
    for (idx, &(i, old)) in dirty.iter().zip(old_depths.iter()) {
        let _ = idx;
        if arr[i].depth > old { right_movers.push(i); }
        else if arr[i].depth < old { left_movers.push(i); }
    }
    right_movers.sort_by_key(|&x| std::cmp::Reverse(x));
    left_movers.sort_by_key(|&x| x);
    for i in right_movers {
        let mut j = i;
        while j + 1 < n && cmp(&arr[j], &arr[j + 1]) == Ordering::Greater {
            arr.swap(j, j + 1); j += 1;
        }
    }
    for i in left_movers {
        let mut j = i;
        while j > 0 && cmp(&arr[j - 1], &arr[j]) == Ordering::Greater {
            arr.swap(j - 1, j); j -= 1;
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

fn init_scene(n: usize, seed: u64) -> Vec<Sprite> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut sprites: Vec<Sprite> = (0..n).map(|i| Sprite {
        depth: rng.gen_range(-100.0..100.0),
        sprite_id: i as u32,
        material: rng.gen_range(0..8),
    }).collect();
    sprites.sort();
    sprites
}

/// Each frame: a subset of sprites get NEW random depths (newly spawned,
/// or animation-driven). Camera unchanged. Returns (dirty indices, old depths).
fn perturb_frame(sprites: &mut [Sprite], k: usize, rng: &mut StdRng) -> (Vec<usize>, Vec<(usize, f32)>) {
    let n = sprites.len();
    let mut picks: Vec<usize> = (0..n).collect();
    for i in 0..k { let j = rng.gen_range(i..n); picks.swap(i, j); }
    let mut dirty: Vec<usize> = picks[..k].to_vec();
    dirty.sort_unstable();
    let mut old_depths = Vec::with_capacity(k);
    for &i in &dirty {
        old_depths.push((i, sprites[i].depth));
        sprites[i].depth = rng.gen_range(-100.0..100.0);
    }
    (dirty, old_depths)
}

struct Out { ds: f64, esm: f64, full: f64 }

fn run_frame(baseline: &[Sprite], dirty: &[usize], _old_depths: &[(usize, f32)]) -> Out {
    let mut data = baseline.to_vec();
    let mut dirty_v: Vec<usize> = dirty.to_vec();
    dirty_v.sort_unstable();
    dirty_v.dedup();
    let t = Instant::now();
    delta_sort_by(&mut data, &mut dirty_v, cmp_sp);
    let ds = t.elapsed().as_nanos() as f64;
    let ds_result = data;

    let mut data = baseline.to_vec();
    let mut dirty_v: Vec<usize> = dirty.to_vec();
    let t = Instant::now();
    esm_sort(&mut data, &mut dirty_v, cmp_sp);
    let esm = t.elapsed().as_nanos() as f64;
    assert!(data == ds_result, "ESM mismatch");

    let mut data = baseline.to_vec();
    let t = Instant::now();
    data.sort_unstable_by(cmp_sp);
    let full = t.elapsed().as_nanos() as f64;
    assert!(data == ds_result, "full mismatch");

    Out { ds, esm, full }
}

fn run_workload(n: usize, k_frac: f64, n_frames: usize, csv: &mut impl Write) {
    let mut sprites = init_scene(n, 0xDEADu64.wrapping_add(n as u64));
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let k = ((n as f64) * k_frac).ceil().max(1.0) as usize;

    let mut ds_v = Vec::new(); let mut esm_v = Vec::new(); let mut full_v = Vec::new();
    for _ in 0..n_frames {
        let (dirty, old_depths) = perturb_frame(&mut sprites, k, &mut rng);
        let r = run_frame(&sprites, &dirty, &old_depths);
        ds_v.push(r.ds); esm_v.push(r.esm); full_v.push(r.full);
        sprites.sort();
    }
    let med = |mut v: Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
    let dsm = med(ds_v); let esmm = med(esm_v); let fm = med(full_v);
    let kpct = 100.0 * k as f64 / n as f64;
    println!(
        "{:>8} {:>8} {:>6.3}%  {:>10.1} {:>10.1} {:>10.1}   {:>7.2}× {:>7.2}×",
        n, k, kpct,
        dsm / 1000.0, esmm / 1000.0, fm / 1000.0,
        esmm / dsm, fm / dsm,
    );
    writeln!(csv, "{},{},DeltaSort,{:.0}", n, k, dsm).unwrap();
    writeln!(csv, "{},{},ESM,{:.0}", n, k, esmm).unwrap();
    writeln!(csv, "{},{},Full,{:.0}", n, k, fm).unwrap();
}

fn main() {
    fs::create_dir_all("results").ok();
    let mut f = fs::File::create("results/track_i_depth_sort.csv").expect("csv");
    writeln!(f, "n,k,strategy,ns").unwrap();

    println!(
        "{:>8} {:>8} {:>7}  {:>10} {:>10} {:>10}   {:>8} {:>8}",
        "n", "k", "k/n",
        "DS us", "ESM us", "Full us",
        "ESM/DS", "Full/DS",
    );

    for &n in &[1_000usize, 10_000, 100_000] {
        for &kf in &[0.001f64, 0.005, 0.01, 0.05, 0.1, 0.25] {
            run_workload(n, kf, 21, &mut f);
        }
        println!();
    }

    println!("Wrote results/track_i_depth_sort.csv");
}
