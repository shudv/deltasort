# DeltaSort

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18132074.svg)](https://doi.org/10.5281/zenodo.18132074)
[![CI](https://github.com/shudv/deltasort/actions/workflows/ci.yml/badge.svg)](https://github.com/shudv/deltasort/actions)

An incremental soting algorithm for arrays. When you know _which_ values changed, DeltaSort restores order multi-fold faster than a full re-sort.

📄 **[Read the pre-print](https://doi.org/10.5281/zenodo.18132074)**

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

| k             | FullSort (µs) | BIS (µs)        | ESM (µs)      | DeltaSort (µs)      |
| ------------- | ------------- | --------------- | ------------- | ------------------- |
| 1 (0.001%)    | 1039.9 ±1.2%  | 101.2 ±3.2%     | 667.3 ±0.8%   | **11.8 ±10.0%** ⚡  |
| 10 (0.01%)    | 1720.7 ±0.7%  | 962.2 ±1.0%     | 849.7 ±0.6%   | **80.7 ±3.1%** ⚡   |
| 100 (0.1%)    | 3929.4 ±0.5%  | 9570.8 ±0.7%    | 986.8 ±0.4%   | **296.9 ±5.0%** ⚡  |
| 1000 (1%)     | 10310.5 ±0.3% | 97226.1 ±1.3%   | 1246.5 ±1.1%  | **1068.5 ±5.2%** ⚡ |
| 10000 (10%)   | 10695.9 ±0.3% | 901773.9 ±0.3%  | 2528.3 ±0.7%  | **4982.7 ±2.7%** 🪶 |
| 20000 (20%)   | 11609.1 ±0.5% | 1698725.8 ±0.3% | 4085.1 ±0.6%  | **8938.4 ±2.1%** 🪶 |
| 50000 (50%)   | 13002.7 ±0.8% | 3389991.9 ±0.5% | 10335.7 ±3.4% | 18364.3 ±1.1%       |
| 100000 (100%) | 13844.5 ±0.2% | 3911565.0 ±0.8% | 16238.7 ±0.6% | 32921.1 ±2.7%       |

⚡ = DeltaSort is fastest &nbsp;&nbsp; 🪶 = DeltaSort beats FullSort, slower than ESM but uses much less memory

_Rust on Apple M-series. Results are environment-specific — JavaScript on V8 has a [much lower crossover threshold](paper/figures/js) due to V8's highly optimized native sort._

## How It Works

1. **Phase 1:** Extract updated values, sort them, write back to original indices.
2. **Phase 2:** Fix each violation using binary insertion on a constrained range.

The key insight: pre-sorting dirty values creates _segments_ that can be fixed _locally_ and _independently_. See the paper for formal proofs.

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
