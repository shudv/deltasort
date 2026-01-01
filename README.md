# DeltaSort

A coordinated incremental repair algorithm for sorted arrays. When a small number of elements change in an already-sorted array, DeltaSort restores order in **O(k log n)** comparisons—optimal for comparison-based algorithms.

📄 **[Read the paper](paper/out/main.pdf)**

## Repository Structure

| Directory | Contents |
|-----------|----------|
| `paper/`  | LaTeX source and compiled PDF |
| `rust/`   | Authoritative Rust implementation + benchmarks |
| `js/`     | TypeScript implementation + tests |

## Reproducing Results

### Rust Benchmarks (Table 1 & Figure 2)

```bash
cd rust
cargo run --release --bin benchmark
```

### Crossover Analysis (Table 2 & Figure 1)

```bash
cd rust
cargo run --release --bin crossover
```

### TypeScript Tests

```bash
cd js
pnpm install && pnpm test
```

## Key Finding

DeltaSort achieves **5–17× speedup** over native sort when fewer than ~25% of elements are dirty. Above this threshold, native sort wins.

## License

MIT
