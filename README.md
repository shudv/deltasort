# DeltaSort

An incremental repair algorithm for sorted arrays. When you know _which_ elements changed, DeltaSort restores order **5–20× faster** than a full re-sort.

📄 **[Read the paper](paper/out/main.pdf)** — formal algorithm, proofs, and benchmarks

## The Problem

You have a sorted array. A few elements get updated. How do you re-sort efficiently?

| Approach         | Time                     | When to use                |
| ---------------- | ------------------------ | -------------------------- |
| Full re-sort     | O(n log n)               | >25% elements changed      |
| Binary Insertion | O(k log n + kn)          | Very few updates (k < 5)   |
| **DeltaSort**    | O(k log k + k log n + M) | **1–25% elements changed** |

_k = number of changed elements, M = total movement (empirically small)_

## Quick Start

### Rust

```bash
cd rust
cargo run --release --bin benchmark   # Run performance benchmarks
cargo test                            # Run correctness tests
```

### JavaScript

```bash
cd js
pnpm install && pnpm test
```

## Key Results (n = 50,000)

| Updated (k) | DeltaSort | Native Sort | Speedup   |
| ----------- | --------- | ----------- | --------- |
| 100         | 145 µs    | 1743 µs     | **12×**   |
| 1,000       | 759 µs    | 3803 µs     | **5×**    |
| 5,000       | 2073 µs   | 3972 µs     | **1.9×**  |
| 15,000      | 4612 µs   | 4569 µs     | crossover |

## How It Works

1. **Phase 1:** Extract dirty values, sort them, write back to original indices
2. **Phase 2:** Repair each "segment" independently using stack-based processing

The key insight: pre-sorting dirty values creates _directional segments_ that can be repaired without interfering with each other. See the paper for formal proofs.

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
