# DeltaSort

A fast incremental repair algorithm for sorted arrays. When a fraction (< ~25%) of elements change in an already-sorted array, DeltaSort restores order much faster than a full sort.

📄 **[Read the paper](paper/out/main.pdf)**

## Repository Structure

| Directory | Contents                                 |
| --------- | ---------------------------------------- |
| `paper/`  | LaTeX source and compiled PDF            |
| `rust/`   | Rust implementation + tests + benchmarks |
| `js/`     | JavaScript implementation + tests        |

## Reproducing Results

### Rust Benchmarks

```bash
cd rust
cargo run --release --bin benchmark
```

### Crossover Analysis

```bash
cd rust
cargo run --release --bin crossover
```

### JavaScript Tests

```bash
cd js
pnpm install && pnpm test
```

## Key Finding

DeltaSort achieves **5–17× speedup** over native sort when fewer than ~25% of elements are dirty. Above this threshold, native sort wins.

## License

MIT
