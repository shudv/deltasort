# DeltaSort (Rust)

Efficient incremental repair of sorted arrays via coordinated element movement.

## Overview

When a small number of elements in a sorted array change, DeltaSort restores sorted order more efficiently than a full re-sort by exploiting knowledge of which indices changed.

## Usage

```rust
use deltasort::deltasort;
use std::collections::HashSet;

let mut arr = vec![1, 3, 5, 7, 9];

// Modify some elements
arr[1] = 8;
arr[3] = 2;

// Track which indices changed
let dirty: HashSet<usize> = [1, 3].into_iter().collect();

// Restore sorted order efficiently
deltasort(&mut arr, &dirty, |a, b| a.cmp(b));

assert_eq!(arr, vec![1, 2, 5, 8, 9]);
```

## Building

```bash
cargo build --release
```

## Testing

```bash
cargo test
```

## Benchmarks

```bash
cargo bench
```

Benchmark results are written to `target/criterion/`.

## Algorithm

DeltaSort operates in three phases:

1. **Phase 1 (Pre-sort)**: Extract dirty values, sort them, write back to dirty indices in index order. This establishes monotonicity among dirty values.

2. **Phase 2 (Left-to-right scan)**: Process dirty indices, immediately fixing LEFT violations while deferring RIGHT violations to a stack.

3. **Phase 3 (Flush)**: Process deferred RIGHT violations in LIFO order.

The pre-sorting phase enables:

- **Progressive search narrowing**: Binary search ranges shrink as elements are processed
- **Movement cancellation**: When dirty values would "cross" (one moving left, another right), pre-sorting assigns them to minimize displacement

## Complexity

- **Comparisons**: O(k log n) - optimal for comparison-based algorithms
- **Movement**: O(kn) worst case, but often much better due to movement cancellation
- **Space**: O(k) auxiliary space

## License

MIT
