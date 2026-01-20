# DeltaSort

<!-- DOI badge removed for anonymous review -->

<!-- CI badge removed for anonymous review -->

An incremental soting algorithm for arrays. When you know _which_ values changed, DeltaSort restores order multi-fold faster than a full re-sort.

<!-- Pre-print link removed for anonymous review -->

## Quick Start

### Rust

```bash
cd rust
cargo test               # Run correctness tests
cargo benchmark          # Run benchmarks
cargo benchmark-export   # Run benchmarks and export data to diagrams
```

### JavaScript

```bash
cd js
pnpm install
pnpm test
pnpm benchmark
pnpm benchmark:export
```

## Benchmark (n = 100K, Rust)

| k             | FullSort (µs) | BIS (µs)        | ESM (µs)        | **DeltaSort** (µs)    |
| ------------- | ------------- | --------------- | --------------- | --------------------- |
| 1 (0.001%)    | 1215.0 ±0.3%  | 113.4 ±1.5% 🪶  | 797.8 ±0.4%     | **15.7 ±4.3%** ⚡     |
| 10 (0.01%)    | 2012.6 ±0.5%  | 1127.8 ±1.1% 🪶 | 1006.8 ±0.6%    | **98.2 ±3.0%** ⚡     |
| 100 (0.1%)    | 4559.8 ±0.5%  | 🐢              | 1195.7 ±0.5%    | **395.5 ±4.4%** ⚡🪶  |
| 1000 (1%)     | 12065.1 ±0.4% | 🐢              | 1426.2 ±1.0%    | **1375.7 ±5.9%** ⚡🪶 |
| 10000 (10%)   | 12577.8 ±0.2% | 🐢              | 3023.4 ±0.9% ⚡ | **6053.9 ±2.7%** 🪶   |
| 20000 (20%)   | 13507.6 ±0.3% | 🐢              | 4801.4 ±0.4% ⚡ | **10024.5 ±2.0%** 🪶  |
| 50000 (50%)   | 15421.2 ±0.6% | 🐢              | 10516.2 ±0.2%   | 🐢                    |
| 100000 (100%) | 16711.4 ±0.2% | 🐢              | 🐢              | 🐢                    |

⚡ = Fastest &nbsp;&nbsp; 🪶 = Uses least memory &nbsp;&nbsp; 🐢 = too slow, FullSort is faster

_Rust on Apple M-series. Results are environment-specific — JavaScript on V8 has a [much lower crossover threshold](paper/figures/js)._

## How It Works

1. **Phase 1:** Extract updated values, sort them, write back to original indices.
2. **Phase 2:** Fix each violation using binary insertion on a constrained range.

The key insight: pre-sorting dirty values creates _segments_ that can be fixed _locally_ and _independently_. See the paper for formal proofs.

## Repository Structure

```
paper/   — LaTeX source for the paper
rust/    — Rust implementation + benchmarks
js/      — JavaScript implementation + benchmarks
```

## Feedback Welcome

This is early-stage. If you:

- Find bugs or edge cases
- Have suggestions for the paper
- Want to discuss applications

Please reach out via the review process.

## License

MIT
