# DeltaSort

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18132074.svg)](https://doi.org/10.5281/zenodo.18132074)
[![CI](https://github.com/shudv/deltasort/actions/workflows/ci.yml/badge.svg)](https://github.com/shudv/deltasort/actions)

An incremental soting algorithm for arrays. When you know _which_ values changed, DeltaSort restores order multi-fold faster than a full re-sort.

📄 **[Read the pre-print](https://doi.org/10.5281/zenodo.18132074)**

## Quick Start

## Building the paper

```bash
cd paper
make          # Builds both article (paper/article.pdf) and SEA/LIPIcs version (paper/sea.pdf)
make clean    # Remove build artifacts
```

> Requires a TeX Live installation with `pdflatex` and `bibtex`.

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

| k             | FullSort (µs) | BIS (µs)       | **DeltaSort** (µs) | ESM (µs)        |
| ------------- | ------------- | -------------- | ------------------ | --------------- |
| 1 (0.001%)    | 931.2 ±0.4%   | 242.7 ±0.3% 🪶 | **12.2 ±3.3%** ⚡  | 345.3 ±0.6%     |
| 10 (0.01%)    | 1574.1 ±0.6%  | 684.2 ±0.8% 🪶 | **74.8 ±3.2%** ⚡  | 566.4 ±0.4%     |
| 100 (0.1%)    | 3527.8 ±0.3%  | 🐢             | **277.3 ±3.8%** ⚡ | 702.2 ±0.1%     |
| 1000 (1%)     | 9181.7 ±0.1%  | 🐢             | **818.9 ±3.8%** ⚡ | 903.1 ±0.2%     |
| 10000 (10%)   | 9401.7 ±0.2%  | 🐢             | 3617.7 ±3.5%       | 2126.8 ±0.2% ⚡ |
| 20000 (20%)   | 10187.1 ±0.3% | 🐢             | 5939.7 ±2.5%       | 3498.4 ±0.3% ⚡ |
| 50000 (50%)   | 11619.3 ±0.4% | 🐢             | 🐢                 | 7777.3 ±1.0% ⚡ |
| 100000 (100%) | 12567.5 ±0.5% | 🐢             | 🐢                 | 🐢              |

⚡ = Fastest &nbsp;&nbsp; 🪶 = Least memory (O(1) auxiliary) &nbsp;&nbsp; 🐢 = too slow, FullSort is faster

_Rust on Apple M3 Pro. Results are environment-specific — JavaScript on V8 has a [much lower crossover threshold](paper/figures/js)._

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

Please [open an issue](https://github.com/shudv/deltasort/issues) or reach out!

## License

MIT
