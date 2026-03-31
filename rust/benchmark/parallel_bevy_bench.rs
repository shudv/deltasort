//! Three-way benchmark: FullSort vs Serial DeltaSort vs Parallel DeltaSort
//! on Bevy-style transparent render phase workload.

use deltasort::delta_sort_by;
use deltasort::parallel::parallel_delta_sort_by;
use rand::Rng;
use std::collections::HashSet;
use std::time::Instant;

#[derive(Clone, Debug)]
struct PhaseItem {
    distance: f32,
    _pad: [u64; 3], // simulate realistic phase item size
}

fn dist_cmp(a: &PhaseItem, b: &PhaseItem) -> std::cmp::Ordering {
    a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
}

fn main() {
    let threads = rayon::current_num_threads();
    println!("Three-way: FullSort vs Serial DeltaSort vs Parallel DeltaSort");
    println!("Threads: {threads}\n");

    let iters = 30;
    let warmup = 5;

    for &n in &[100_000usize, 500_000, 1_000_000] {
        println!("n = {n}");
        println!("{:<10} {:>12} {:>12} {:>12} {:>8} {:>8}",
            "k", "FullSort", "Serial DS", "Parallel DS", "S vs F", "P vs F");
        println!("{}", "─".repeat(72));

        let k_values: Vec<usize> = vec![0, 1, 10, 50, 100, 500, 1000, 5000];
        let k_values: Vec<usize> = k_values.into_iter().filter(|&k| k <= n).collect();

        for &k in &k_values {
            let mut rng = rand::thread_rng();

            // Build sorted base
            let base: Vec<PhaseItem> = (0..n).map(|i| PhaseItem {
                distance: i as f32 * 0.01,
                _pad: [0; 3],
            }).collect();

            // --- FullSort ---
            let mut t_full = Vec::new();
            let mut arr = base.clone();
            for iter in 0..(warmup + iters) {
                let indices: Vec<usize> = (0..k).map(|_| rng.gen_range(0..n)).collect();
                for &i in &indices { arr[i].distance = rng.gen_range(0.0..n as f32 * 0.01); }
                let s = Instant::now();
                arr.sort_by(dist_cmp);
                let e = s.elapsed().as_secs_f64() * 1e6;
                if iter >= warmup { t_full.push(e); }
            }

            // --- Serial DeltaSort ---
            let mut t_serial = Vec::new();
            let mut arr = base.clone();
            for iter in 0..(warmup + iters) {
                let indices: Vec<usize> = (0..k).map(|_| rng.gen_range(0..n)).collect();
                let dirty: HashSet<usize> = indices.iter().copied().collect();
                for &i in &indices { arr[i].distance = rng.gen_range(0.0..n as f32 * 0.01); }
                let s = Instant::now();
                if !dirty.is_empty() {
                    delta_sort_by(&mut arr, &dirty, dist_cmp);
                }
                let e = s.elapsed().as_secs_f64() * 1e6;
                if iter >= warmup { t_serial.push(e); }
            }

            // --- Parallel DeltaSort ---
            let mut t_par = Vec::new();
            let mut arr = base.clone();
            for iter in 0..(warmup + iters) {
                let indices: Vec<usize> = (0..k).map(|_| rng.gen_range(0..n)).collect();
                let dirty: HashSet<usize> = indices.iter().copied().collect();
                for &i in &indices { arr[i].distance = rng.gen_range(0.0..n as f32 * 0.01); }
                let s = Instant::now();
                if !dirty.is_empty() {
                    parallel_delta_sort_by(&mut arr, &dirty, dist_cmp);
                }
                let e = s.elapsed().as_secs_f64() * 1e6;
                if iter >= warmup { t_par.push(e); }
            }

            let fm = t_full.iter().sum::<f64>() / t_full.len() as f64;
            let sm = t_serial.iter().sum::<f64>() / t_serial.len() as f64;
            let pm = t_par.iter().sum::<f64>() / t_par.len() as f64;
            let s_vs_f = if sm > 0.0 { fm / sm } else { f64::INFINITY };
            let p_vs_f = if pm > 0.0 { fm / pm } else { f64::INFINITY };

            println!("{:<10} {:>9.1}µs {:>9.1}µs {:>9.1}µs {:>7.1}x {:>7.1}x",
                k, fm, sm, pm, s_vs_f, p_vs_f);
        }
        println!();
    }
}
