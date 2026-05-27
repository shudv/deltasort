//! Track 2 — synthetic HPC particle-sort benchmark.
//!
//! Models the inner loop of an N-body / SPH code: particles are sorted by
//! their Morton (Z-order) key for cache-friendly force computation. Each
//! timestep, the integrator moves every particle a little; a small fraction
//! k cross Morton-cell boundaries. The dirty indices fall out of the
//! integration loop for free (a single compare per particle).
//!
//! Strategies compared:
//!   - FullSort:   `sort_unstable_by(key)` each step (current GADGET-4 baseline)
//!   - DeltaSort:  k-known dirty indices → restore order
//!   - ESM:        extract-sort-merge baseline
//!
//! Two initial conditions:
//!   - Uniform: positions uniform in [0,1]^3
//!   - Plummer: clustered (denser core, sparse halo) — typical late-time cosmology
//!
//! Outputs: `results/track2_*.csv`

mod morton;

use deltasort::delta_sort_by;
use morton::morton_unit;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::cmp::Ordering;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

const MORTON_BITS_DEFAULT: u32 = 0; // 0 means: derive from n

/// Choose Morton resolution so cells_per_axis ≈ n^(1/3), i.e., each cell
/// holds ~1 particle on average. This matches typical HPC cell sizing for
/// neighbor finding / FMM.
fn morton_bits_for(n: usize) -> u32 {
    // bits = ceil(log2(n^(1/3))) clamped to [4, 12]
    let cells_per_axis = (n as f64).powf(1.0 / 3.0).ceil() as u32;
    let bits = (cells_per_axis.next_power_of_two().trailing_zeros()).max(4).min(12);
    bits
}

#[derive(Clone, Debug, Default)]
struct Particle {
    pos: [f64; 3],
    vel: [f64; 3],
    morton: u64,
}

impl Particle {
    fn recompute_morton(&mut self, bits: u32) {
        for c in &mut self.pos {
            *c = c.rem_euclid(1.0);
        }
        self.morton = morton_unit(self.pos, bits);
    }
}

// ---------------- initial conditions ----------------

fn ic_uniform(n: usize, bits: u32, vel_sigma: f64, seed: u64) -> Vec<Particle> {
    let mut rng = StdRng::seed_from_u64(seed);
    let vd = Normal::new(0.0, vel_sigma).unwrap();
    (0..n)
        .map(|_| {
            let pos = [rng.gen(), rng.gen(), rng.gen()];
            let vel = [vd.sample(&mut rng), vd.sample(&mut rng), vd.sample(&mut rng)];
            let mut p = Particle { pos, vel, morton: 0 };
            p.recompute_morton(bits);
            p
        })
        .collect()
}

fn ic_plummer(n: usize, bits: u32, vel_sigma: f64, seed: u64) -> Vec<Particle> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    let scale = 0.1;
    let vd = Normal::new(0.0, vel_sigma).unwrap();
    while out.len() < n {
        let u: f64 = rng.gen();
        let r = scale / (u.powf(-2.0 / 3.0) - 1.0).sqrt();
        if !r.is_finite() {
            continue;
        }
        let cos_th = rng.gen_range(-1.0_f64..1.0);
        let sin_th = (1.0 - cos_th * cos_th).max(0.0).sqrt();
        let phi = rng.gen_range(0.0..std::f64::consts::TAU);
        let pos = [
            0.5 + r * sin_th * phi.cos(),
            0.5 + r * sin_th * phi.sin(),
            0.5 + r * cos_th,
        ];
        if pos.iter().any(|&c| !(0.0..1.0).contains(&c)) {
            continue;
        }
        let vel = [vd.sample(&mut rng), vd.sample(&mut rng), vd.sample(&mut rng)];
        let mut p = Particle { pos, vel, morton: 0 };
        p.recompute_morton(bits);
        out.push(p);
    }
    out
}

// ---------------- integration ----------------

/// Advance positions by velocity*dt + a thermal kick. Returns the dirty set.
fn integrate_step(
    particles: &mut [Particle],
    dt: f64,
    kick_sigma: f64,
    bits: u32,
    rng: &mut StdRng,
) -> Vec<usize> {
    let kick = Normal::new(0.0, kick_sigma).unwrap();
    let mut dirty = Vec::new();
    for (i, p) in particles.iter_mut().enumerate() {
        let old_key = p.morton;
        for c in 0..3 {
            p.pos[c] += p.vel[c] * dt + kick.sample(rng) * dt;
        }
        p.recompute_morton(bits);
        if p.morton != old_key {
            dirty.push(i);
        }
    }
    dirty
}

// ---------------- sort strategies ----------------

fn cmp_morton(a: &Particle, b: &Particle) -> Ordering {
    a.morton.cmp(&b.morton)
}

fn strat_full(arr: &mut [Particle], _dirty: &[usize]) -> Duration {
    let t0 = Instant::now();
    arr.sort_unstable_by(cmp_morton);
    t0.elapsed()
}

fn strat_delta(arr: &mut [Particle], dirty: &[usize]) -> Duration {
    let mut d = dirty.to_vec();
    let t0 = Instant::now();
    delta_sort_by(arr, &mut d, cmp_morton);
    t0.elapsed()
}

fn strat_esm(arr: &mut [Particle], dirty: &[usize]) -> Duration {
    // Reuse the User-specialised extract_sort_merge would be ideal, but for
    // Particle we inline an equivalent generic version here. Same algorithm.
    let mut d = dirty.to_vec();
    let t0 = Instant::now();
    d.sort_unstable();
    let n = arr.len();
    let k = d.len();
    let mut dirty_buf: Vec<Particle> = Vec::with_capacity(k);
    let mut write = 0;
    let mut di = 0;
    for read in 0..n {
        if di < k && d[di] == read {
            dirty_buf.push(std::mem::replace(
                &mut arr[read],
                Particle {
                    pos: [0.0; 3],
                    vel: [0.0; 3],
                    morton: 0,
                },
            ));
            di += 1;
        } else {
            if write != read {
                arr.swap(write, read);
            }
            write += 1;
        }
    }
    dirty_buf.sort_unstable_by(cmp_morton);
    let mut ci = (n - k) as isize - 1;
    let mut dj = k as isize - 1;
    let mut wi = n - 1;
    loop {
        let take_clean = if ci >= 0 && dj >= 0 {
            cmp_morton(&arr[ci as usize], &dirty_buf[dj as usize]) != Ordering::Less
        } else {
            ci >= 0
        };
        if take_clean {
            if wi != ci as usize {
                arr.swap(wi, ci as usize);
            }
            ci -= 1;
        } else if dj >= 0 {
            arr[wi] = std::mem::replace(
                &mut dirty_buf[dj as usize],
                Particle {
                    pos: [0.0; 3],
                    vel: [0.0; 3],
                    morton: 0,
                },
            );
            dj -= 1;
        }
        if wi == 0 {
            break;
        }
        wi -= 1;
    }
    t0.elapsed()
}

// ---------------- simulation driver ----------------

#[derive(Clone, Copy, Debug)]
enum InitialCondition {
    Uniform,
    Plummer,
}

#[derive(Clone, Debug)]
struct StepResult {
    step: usize,
    k: usize,
    n: usize,
    full_us: f64,
    delta_us: f64,
    esm_us: f64,
}

fn run_simulation(
    n: usize,
    steps: usize,
    ic: InitialCondition,
    motion_factor: f64,
    seed: u64,
) -> Vec<StepResult> {
    let bits = morton_bits_for(n);
    let cell_width = 1.0 / (1u32 << bits) as f64;
    // Scale velocity & thermal kick so per-step displacement = motion_factor * cell_width.
    // Drift via velocity gives ~half of the displacement; kick the rest.
    let vel_sigma = motion_factor * cell_width * 0.5;
    let kick_sigma = motion_factor * cell_width * 0.5;

    let mut particles = match ic {
        InitialCondition::Uniform => ic_uniform(n, bits, vel_sigma, seed),
        InitialCondition::Plummer => ic_plummer(n, bits, vel_sigma, seed.wrapping_add(1)),
    };
    particles.sort_unstable_by(cmp_morton);

    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(2));
    let dt = 1.0;
    let mut results = Vec::with_capacity(steps);

    for step in 0..steps {
        let dirty = integrate_step(&mut particles, dt, kick_sigma, bits, &mut rng);
        let k = dirty.len();
        let full_us = {
            let mut a = particles.clone();
            let d = strat_full(&mut a, &dirty);
            d.as_nanos() as f64 / 1000.0
        };
        let delta_us = {
            let mut a = particles.clone();
            let d = strat_delta(&mut a, &dirty);
            d.as_nanos() as f64 / 1000.0
        };
        let esm_us = {
            let mut a = particles.clone();
            let d = strat_esm(&mut a, &dirty);
            d.as_nanos() as f64 / 1000.0
        };

        let mut d_apply = dirty.clone();
        delta_sort_by(&mut particles, &mut d_apply, cmp_morton);

        results.push(StepResult {
            step,
            k,
            n,
            full_us,
            delta_us,
            esm_us,
        });
    }

    results
}

// ---------------- main ----------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let fast = args.iter().any(|a| a == "--fast");

    fs::create_dir_all("results").ok();

    let n_values: &[usize] = if fast {
        &[10_000, 100_000]
    } else {
        &[10_000, 100_000, 1_000_000]
    };
    let steps = if fast { 50 } else { 200 };

    // Three motion regimes (per-step displacement as fraction of cell width):
    //   cold:   tiny motion (late-time cold DM regime)
    //   warm:   moderate motion
    //   hot:    large motion (early-time / very dynamic regime)
    let regimes: &[(&str, f64)] = &[("cold", 0.02), ("warm", 0.1), ("hot", 0.3)];
    let ics: &[(InitialCondition, &str)] = &[
        (InitialCondition::Uniform, "uniform"),
        (InitialCondition::Plummer, "plummer"),
    ];

    let mut csv = String::from("n,ic,regime,step,k,full_us,delta_us,esm_us\n");
    let mut summary_csv =
        String::from("n,ic,regime,steps,mean_k,median_k,mean_full_us,mean_delta_us,mean_esm_us,total_full_us,total_delta_us,total_esm_us\n");

    println!(
        "{:>9} {:>10} {:>8} {:>7} {:>10} {:>12} {:>12} {:>12} {:>9}",
        "n", "ic", "regime", "steps", "mean_k", "mean_full", "mean_delta", "mean_esm", "k%"
    );

    for &n in n_values {
        for &(ic, ic_name) in ics {
            for &(regime_name, motion_factor) in regimes {
                let res =
                    run_simulation(n, steps, ic, motion_factor, 0xDEAD_u64.wrapping_add(n as u64));
                let mean_k = res.iter().map(|s| s.k as f64).sum::<f64>() / res.len() as f64;
                let mut ks: Vec<usize> = res.iter().map(|s| s.k).collect();
                ks.sort_unstable();
                let median_k = ks[ks.len() / 2];
                let mean_full = res.iter().map(|s| s.full_us).sum::<f64>() / res.len() as f64;
                let mean_delta = res.iter().map(|s| s.delta_us).sum::<f64>() / res.len() as f64;
                let mean_esm = res.iter().map(|s| s.esm_us).sum::<f64>() / res.len() as f64;
                let total_full: f64 = res.iter().map(|s| s.full_us).sum();
                let total_delta: f64 = res.iter().map(|s| s.delta_us).sum();
                let total_esm: f64 = res.iter().map(|s| s.esm_us).sum();

                println!(
                    "{:>9} {:>10} {:>8} {:>7} {:>10.1} {:>12.1} {:>12.1} {:>12.1} {:>9.2}",
                    n,
                    ic_name,
                    regime_name,
                    steps,
                    mean_k,
                    mean_full,
                    mean_delta,
                    mean_esm,
                    100.0 * mean_k / n as f64,
                );

                summary_csv.push_str(&format!(
                    "{},{},{},{},{:.1},{},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}\n",
                    n,
                    ic_name,
                    regime_name,
                    steps,
                    mean_k,
                    median_k,
                    mean_full,
                    mean_delta,
                    mean_esm,
                    total_full,
                    total_delta,
                    total_esm,
                ));

                for s in &res {
                    csv.push_str(&format!(
                        "{},{},{},{},{},{:.1},{:.1},{:.1}\n",
                        n, ic_name, regime_name, s.step, s.k, s.full_us, s.delta_us, s.esm_us
                    ));
                }
            }
        }
    }

    let trace_path = Path::new("results/track2_trace.csv");
    fs::File::create(trace_path)
        .and_then(|mut f| f.write_all(csv.as_bytes()))
        .expect("write trace csv");
    let summary_path = Path::new("results/track2_summary.csv");
    fs::File::create(summary_path)
        .and_then(|mut f| f.write_all(summary_csv.as_bytes()))
        .expect("write summary csv");

    println!("\nWrote {}", trace_path.display());
    println!("Wrote {}", summary_path.display());
}
