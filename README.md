# DeltaSort

An incremental repair algorithm for sorted arrays. When you know _which_ values changed, DeltaSort restores order multi-fold faster than a full re-sort.

📄 **[Read the pre-print (OpenReview)](https://openreview.net/attachment?id=AlMKtDfJvy&name=pdf)**

## The Problem

You have a sorted array. A few values get updated. How do you re-sort efficiently?

| Approach      | Time                     | When to use            |
| ------------- | ------------------------ | ---------------------- |
| Full re-sort  | O(n log n)               | Many values changed    |
| **DeltaSort** | O(k log k + k log n + M) | **Few values changed** |

_k = number of changed values, M = total movement (empirically mesure to be much smaller than O(n\*k)). The exact crossover threshold varies by environment — see benchmarks from a sample run below._

## Quick Start

### Rust

```bash
cd rust
cargo benchmark   # Run performance benchmarks
cargo test        # Run correctness tests
```

### JavaScript

```bash
cd js
pnpm install && pnpm test
```

## Sample Benchmark Run (n = 50K, Rust)

| #Updated (k) | DeltaSort | NativeSort | Speedup   |
| ------------ | --------- | ---------- | --------- |
| 100 (0.2%)   | 151.5 µs  | 1879.6 µs  | **12×**   |
| 1K (2%)      | 694.8 µs  | 4202.3 µs  | **6×**    |
| 5K (10%)     | 2387.1 µs | 4320.2 µs  | **1.4×**  |
| 10K (20%)    | 3550.8 µs | 4421.6 µs  | **1.8×**  |
| 20K (40%)    | 6343.8 µs | 5233.9 µs  | **0.8x**  |

_Results from Rust implementation on M3 Pro. The crossover threshold lies in range 20-40%. Speedup numbers are specific to this environment — results will vary in other runtimes (e.g., JavaScript on v8 has a much lower crossover threshold because of highly optimized native sort)._

## How It Works

1. **Phase 1:** Extract dirty values, sort them, write back to original indices
2. **Phase 2:** Repair each "segment" independently using stack-based processing

The key insight: pre-sorting dirty values creates _segments_ that can be repaired independently. See the paper for formal proofs.

## Repository Structure

```
paper/   — LaTeX source for the paper
rust/    — Rust implementation + benchmarks
js/      — JavaScript implementation
```

## Feedback Welcome

This is early-stage. If you:

- Find bugs or edge cases
- Have suggestions for the paper
- Want to discuss applications

Please [open an issue](https://github.com/shudv/deltasort/issues) or reach out!

## License

MIT
