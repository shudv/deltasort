# DeltaSort: Real-World Applicability Analysis

> **DeltaSort's model**: You have a sorted array of _n_ elements, _k_ of them have changed values (and you know WHICH indices changed), and you need to re-sort. Achieves $O(n\sqrt{k})$ time with $O(1)$ auxiliary space.

> **Hard preconditions**: (1) contiguous array, (2) previously sorted, (3) caller tracks dirty indices, (4) values change in-place at known positions.

---

## A. Pandas DataFrame (`sort_values`)

**Source examined**: `pandas/core/frame.py` — `sort_values()` method

### Data Structure

Column-oriented storage via `BlockManager`. Each column is a NumPy `ndarray`. Row order is tracked via an internal `RangeIndex` or `Int64Index`. `sort_values()` computes a full permutation indexer via `nargsort()` (wrapping NumPy's `argsort` — default quicksort) or `lexsort_indexer()` for multi-column sorts, then applies `self._mgr.take(indexer)` which physically reorders all blocks.

### Update Mechanism

Setting values via `df.loc[idx, col] = new_val` goes through `_set_value` → `_iset_item_mgr`. This is a **fire-and-forget** write — no metadata about what changed is retained for sorting purposes. Every `sort_values()` call is a complete re-sort from scratch.

### Dirty Index Tracking

**None.** Pandas has no concept of "which rows changed since last sort." The DataFrame doesn't even remember whether it was sorted. There is no persistent sorted state — `sort_values()` returns a new DataFrame (or sorts in-place, but discards any order metadata).

### Current Sort Algorithm

`nargsort()` → NumPy's `argsort()` with `kind='quicksort'` (default, actually introsort in modern NumPy), `'mergesort'`/`'stable'` (timsort), or `'heapsort'`. Returns a full index permutation of size _n_.

### Array Sizes

Typical: thousands to millions of rows. DataFrames with 10M+ rows are common in data science workflows.

### Model Match: 25/100

**Mismatch analysis:**

- ✅ Underlying storage IS contiguous arrays (NumPy ndarrays)
- ❌ No dirty index tracking exists anywhere in Pandas
- ❌ `sort_values()` returns a permutation indexer — it doesn't sort the array in-place, it creates a new order. DeltaSort sorts the array in-place.
- ❌ Pandas sorts across columns (lexicographic), not within a single array
- ❌ The `sort_values` API doesn't accept "changed indices" and would require deep surgery to add this

**Integration path**: Would require (1) adding a change-tracking layer to DataFrame (recording which row indices had values set since last sort), (2) modifying `sort_values` to accept a "dirty set" parameter, (3) bypassing the take-based permutation approach in favor of in-place sorting. This is a **fundamental API change** — Pandas maintainers would likely reject it. The opportunity exists conceptually (users DO modify cells, then re-sort), but the architectural fit is poor.

---

## B. DuckDB (ORDER BY / Physical Sort)

**Source examined**: `src/execution/operator/order/physical_order.cpp`

### Data Structure

DuckDB uses a **vectorized push-based pipeline**. `PhysicalOrder` is a Sink/Source operator. Data arrives in `DataChunk`s (~2048 tuples each). The `Sort` object (from `duckdb/common/sorting/sort.hpp`) accumulates chunks during the sink phase, then produces sorted output during the source phase. Internally, DuckDB uses a **radix sort** + **merge sort** hybrid on binary-comparable sort keys, with data stored in row-oriented sort blocks.

### Update Mechanism

SQL is **declarative** — `SELECT ... ORDER BY` operates on a query result, not a mutable sorted array. There is no "previously sorted state" to update. Each query execution produces a fresh sort from scratch. DuckDB has no persistent sorted materialized view with incremental maintenance.

### Dirty Index Tracking

**N/A.** This is a query engine, not a mutable sorted container. The concept doesn't apply — each ORDER BY is a one-shot operation on a new result set.

### Current Sort Algorithm

Radix sort on normalized binary-comparable keys + merge sort for combining sorted runs. Parallelized across threads. External sort (spill to disk) for data exceeding memory.

### Array Sizes

Rows: hundreds to billions. DuckDB is designed for OLAP workloads on large datasets.

### Model Match: 5/100

**Mismatch analysis:**

- ❌ No mutable sorted array — this is a one-shot query operator
- ❌ No "previously sorted" state to exploit
- ❌ No dirty index tracking (nor any way to add it — queries don't have persistent state)
- ❌ Data arrives in chunks, not as a single contiguous array
- ❌ Radix sort approach is fundamentally different from comparison-based sorting

**Integration path**: Near-impossible. DeltaSort's model requires a persistent sorted array with tracked mutations. DuckDB rebuilds sort state per-query. The only conceivable path would be sorted materialized views with incremental maintenance, but that's an entirely different problem (and DuckDB doesn't have materialized views).

---

## C. Redis Sorted Sets (`ZSET`)

**Source examined**: `src/t_zset.c` — ZADD, `zslUpdateScore`, skip list implementation

### Data Structure

ZSET uses a **dual structure**: skip list (`zskiplist`) + hash table (`dict`). The skip list maintains elements in score-sorted order with O(log n) operations. For small sets (≤128 elements, all values ≤64 bytes), a **ziplist/listpack** encoding is used instead. Neither is a contiguous sorted array.

### Update Mechanism

`ZADD` processes elements one at a time via `zsetAdd()`. When updating an existing element's score, `zslUpdateScore()` has a **fast path**: if the new score keeps the element between its predecessor and successor, it updates in-place (O(1)). Otherwise, it unlinks the node and reinserts at the correct position (O(log n)). This is already highly optimized for single-element updates.

### Dirty Index Tracking

**Not needed.** Each score update is handled immediately and atomically. There is no deferred batch sorting. The data structure is always in sorted order after every operation.

### Current Sort Algorithm

No "sort" step exists — the skip list maintains sorted order via insertion. For bulk operations (`ZUNIONSTORE`, `ZINTERSTORE`), elements are inserted one at a time into a new skip list.

### Array Sizes

Typical: hundreds to millions of members per ZSET. Redis is designed for fast O(log n) per-operation access patterns.

### Model Match: 5/100

**Mismatch analysis:**

- ❌ **Not an array** — skip list is a linked/pointer-based structure with O(log n) random access
- ❌ No batch sorting needed — skip list is always sorted
- ❌ Updates are processed one-at-a-time, not batched
- ❌ `zslUpdateScore()` already has a fast path for no-movement updates
- ❌ The ziplist encoding for small sets IS a contiguous structure, but it's always kept sorted via insertion

**Integration path**: Fundamentally incompatible. Redis ZSET's entire design is about maintaining sorted order continuously via a pointer-based structure. DeltaSort solves a completely different problem (batched re-sorting of arrays). To use DeltaSort, Redis would need to abandon skip lists for flat arrays — which would sacrifice O(log n) insertion/deletion for O(n) operations on every single command. This would be a catastrophic performance regression for Redis's primary use case.

---

## D. Apache Lucene / Elasticsearch (Doc-Value Sorting)

**Knowledge-based analysis** (Lucene segment architecture)

### Data Structure

Lucene stores data in **immutable segments**. Each segment contains doc-values (column-oriented, sorted numeric/binary data) stored as compressed arrays. When searching with a sort clause, Lucene uses a `FieldComparator` to maintain a priority queue of top-N results across segments. For index-time sorting, segments are sorted at flush time using `IndexSorter`, which applies TimSort to a doc ID permutation array.

### Update Mechanism

Lucene segments are **immutable** — documents are never updated in place. An "update" deletes the old document and inserts a new one (potentially into a new segment). Segment merging combines multiple segments into a new sorted segment. This is a merge of pre-sorted runs, not a re-sort of a mutated array.

### Dirty Index Tracking

**N/A.** Immutability means no in-place mutations exist. Deleted documents are tracked via a bitset (`.liv` file), but there's no concept of "changed values at known positions."

### Current Sort Algorithm

Index-time: TimSort on doc IDs during segment flush. Query-time: priority queue (top-K) across segments. Segment merge: k-way merge of pre-sorted segments.

### Array Sizes

Segments: thousands to millions of documents. Typical index: millions to billions of documents total.

### Model Match: 3/100

**Mismatch analysis:**

- ❌ **Immutable segments** — values never change in place
- ❌ No "previously sorted array with mutations" — updates create new documents
- ❌ Sorting is either at flush time (one-shot) or at query time (priority queue, not array sort)
- ❌ Segment merge is a merge of sorted runs, not fixing violations in a single array
- ❌ The entire architecture is designed around immutability and append-only operations

**Integration path**: There is no path. Lucene's immutable segment architecture is fundamentally incompatible with DeltaSort's mutable-array model. The closest analogy would be NRT (near-real-time) sorted segment updates, but Lucene handles this via segment merging, not in-place re-sorting.

---

## E. Linux Kernel (`lib/sort.c`)

**Source examined**: `lib/sort.c` — kernel generic array sort

### Data Structure

Generic contiguous array (`void *base`, element count, element size). Used for sorting arrays of structs in various kernel subsystems (device lists, page tables, extent trees, etc.).

### Update Mechanism

The kernel's `sort()` function is called as a one-shot operation whenever a subsystem needs a sorted array. There is no persistent "sorted array that gets mutated" pattern — callers build an array, sort it, use it. The function signature is: `void sort(void *base, size_t num, size_t size, cmp_func_t cmp, swap_func_t swap)`.

### Dirty Index Tracking

**None.** The API takes a raw array and sorts it completely. No mechanism for specifying which elements changed.

### Current Sort Algorithm

**Heapsort** (bottom-up, with optimization for aligned 4/8-byte elements using word-sized swaps). Chosen specifically for O(1) auxiliary space and worst-case O(n log n) — important in kernel context where stack space is severely limited and worst-case guarantees matter.

### Array Sizes

Varies wildly: tens to thousands of elements typically. Kernel arrays being sorted are usually small (device lists, IRQ tables, extent maps).

### Model Match: 30/100

**Mismatch analysis:**

- ✅ **Contiguous array** — yes, this is exactly the data structure
- ✅ **In-place sort** — heapsort is O(1) space, matching DeltaSort's design philosophy
- ❌ **No dirty tracking** — callers don't track what changed. The API would need a new `delta_sort()` variant
- ❌ Most kernel sort use cases are **one-shot** (build array, sort once, traverse) — not incremental
- ⚠️ Some subsystems (extent trees, inode tables) do maintain sorted arrays that get occasional updates, but they typically use binary search + insertion for single updates, not batch re-sorting

**Integration path**: Would require (1) a new `delta_sort()` kernel API accepting a dirty set, (2) kernel callers that maintain sorted arrays with tracked mutations to opt in. The main barriers are: kernel coding style strongly favors simplicity, the existing heapsort is well-audited and trusted, and most sort call sites don't match the incremental pattern. However, for specific subsystems (e.g., block I/O elevator queue reordering, memory compaction page lists), there could be a narrow fit if someone does the kernel archaeology to find suitable call sites.

---

## F. Chromium (Browser Tab / Bookmark Sorting)

**Knowledge-based analysis** (Chromium bookmark manager, tab strip model)

### Data Structure

**Bookmarks**: tree structure (`BookmarkNode` with children `std::vector<std::unique_ptr<BookmarkNode>>`). Each folder's children are stored in a vector, ordered by user-defined position (manual drag-and-drop) or sorted by name/date on demand.

**Tabs**: `TabStripModel` stores tabs in a `std::vector<std::unique_ptr<Tab>>`. Tab order is user-managed (drag, pin, group). Sorting tabs (e.g., alphabetically) is a rarely-used feature.

### Update Mechanism

Bookmarks: user adds/removes/renames/moves bookmarks. Each operation mutates the tree and triggers a sync to disk. Tab strip: tabs open, close, move, update title/URL as pages load.

### Dirty Index Tracking

**None for sorting purposes.** Bookmark sync tracks changes for cloud sync (via `BookmarkModel::Observer`), but not for maintaining sorted order. Tab strip notifies observers of changes (`TabStripModelObserver`) but for UI updates, not sort optimization.

### Current Sort Algorithm

Bookmarks: `std::sort` with `BookmarkNodeTitleComparator` when user clicks "Sort by name." This is a one-shot operation. Tabs: manual ordering (no automatic sort). When "Sort tabs" extensions sort, they use `std::sort` or equivalent.

### Array Sizes

Bookmarks: tens to thousands per folder. Tabs: typically 5–100, power users up to 500+.

### Model Match: 45/100

**Mismatch analysis:**

- ✅ **Contiguous array** — children vectors are contiguous
- ✅ **Previously sorted** — yes, if user has sorted bookmarks and then renames one
- ✅ **Small k** — user typically modifies 1-2 bookmarks between re-sorts
- ⚠️ **Dirty tracking would need to be added** — not currently tracked for sort purposes
- ❌ **Arrays are tiny** — for <100 elements, DeltaSort's overhead (computing directions, binary search) exceeds just calling `std::sort` (which is highly optimized introsort for small arrays with insertion sort fallback)
- ❌ **Sort is rare** — users sort bookmarks once, not continuously

**Integration path**: Technically feasible but **not worthwhile**. The arrays are too small for DeltaSort to beat optimized `std::sort`. At n=100, k=2, a full sort takes ~700 comparisons (n log n), while DeltaSort would still need a full Phase 2 scan of all n elements. The constant-factor overhead of DeltaSort's binary searches and rotations would likely make it slower for these sizes. This is a "correct model, wrong scale" situation.

---

## G. Godot Engine (Render List Sorting)

**Source examined**: `servers/rendering/renderer_rd/forward_clustered/render_forward_clustered.cpp`

### Data Structure

`RenderList` contains a `LocalVector<GeometryInstanceSurfaceDataCache*>` of element pointers plus `LocalVector<RenderElementInfo>` metadata. Each element has `sort.sort_key1` and `sort.sort_key2` (packed uint64 fields encoding material ID, shader ID, geometry ID, depth layer, priority, etc.).

### Update Mechanism

**The render list is completely rebuilt every frame.** `_fill_render_list()` clears the list (`rl->clear()`) then iterates all visible instances, computing depth values, LOD, GI flags, and adding surfaces via `rl->add_element(surf)`. After filling:

```cpp
render_list[RENDER_LIST_OPAQUE].sort_by_key();
render_list[RENDER_LIST_MOTION].sort_by_key();
render_list[RENDER_LIST_ALPHA].sort_by_reverse_depth_and_priority();
```

The sort keys are computed fresh each frame (depth depends on camera position).

### Dirty Index Tracking

**N/A.** The list doesn't persist between frames — it's rebuilt from scratch every frame. `_update_dirty_geometry_instances()` handles geometry instance invalidation (mesh changes, material changes), not render list order.

### Current Sort Algorithm

`sort_by_key()` uses a radix-like or std::sort on the packed sort keys. Applied to the freshly built list each frame.

### Array Sizes

Depends on scene complexity. Typically hundreds to tens of thousands of render elements per frame. AAA scenes: 10K–100K draw calls.

### Model Match: 15/100

**Mismatch analysis:**

- ✅ Contiguous array of pointers/elements
- ❌ **List rebuilt from scratch every frame** — no "previously sorted array" persists
- ❌ Sort keys change every frame (depth depends on camera position/orientation)
- ❌ No concept of "which elements changed" — ALL elements have new depth values each frame
- ❌ The clear→fill→sort pipeline is the standard game engine pattern

**Integration path**: Would require **temporal coherence exploitation** — keeping the previous frame's sorted list and tracking which elements had their sort keys change. In practice, when the camera moves, ALL depth values change, making k ≈ n, which eliminates DeltaSort's advantage. DeltaSort would only help for static camera with a few objects moving, which is a rare game scenario. Engines that do exploit temporal coherence (like insertion sort on nearly-sorted lists) already exist, but they use simpler approaches than DeltaSort.

---

## H. Apache Arrow (Columnar Compute)

**Knowledge-based analysis** (Arrow compute kernel architecture)

### Data Structure

Arrow arrays are **immutable contiguous buffers** with a specific memory layout (validity bitmap + data buffer). `SortIndices()` compute kernel produces an `Array<UInt64>` of sorted indices. `SortToIndices()` and `Take()` are used to produce sorted output. The original array is never mutated.

### Update Mechanism

Arrow arrays are **immutable by design**. You cannot change a value at index i. Instead, you create a new array. This is fundamental to Arrow's zero-copy and shared-memory architecture. `MutableBuffer` exists for building arrays, but once wrapped in an `Array`, it becomes immutable.

### Dirty Index Tracking

**Impossible by design.** Immutable arrays cannot have "changed values." Any mutation requires constructing a new array.

### Current Sort Algorithm

`SortIndices()` uses a merge sort variant (stable) or in-place sort on the indices array. For chunked arrays, it sorts within chunks and merges. Arrow also has `partition_nth_indices` for partial sorting.

### Array Sizes

Thousands to billions of elements. Arrow is designed for large-scale columnar analytics.

### Model Match: 5/100

**Mismatch analysis:**

- ❌ **Immutable arrays** — the fundamental data model prohibits in-place mutation
- ❌ Cannot track "changed indices" because values never change
- ❌ Sort produces a new index array, not an in-place sort
- ❌ Zero-copy sharing means mutation would violate safety guarantees for other readers

**Integration path**: Impossible without violating Arrow's core invariants. Arrow would need to abandon immutability, which would break its entire ecosystem (IPC, Flight, shared memory, zero-copy). The closest possibility: an `IncrementalSortIndices(old_indices, changed_positions, new_array)` compute kernel, but this would be operating on a mutable "sorted view" on top of an immutable array — a fundamentally different abstraction than DeltaSort.

---

## I. SQLite (`vdbesort.c` — ORDER BY Sorter)

**Knowledge-based analysis** (SQLite VDBE sorting subsystem)

### Data Structure

SQLite's sorter (`VdbeSorter`) uses an **external merge sort** for ORDER BY. Records are accumulated in-memory in PMAs (Pre-sorted Merge Areas), sorted when memory fills up, written to temp files, then multi-way merged to produce final sorted output. The in-memory sort uses a single-linked list or array of sort keys.

### Update Mechanism

The sorter is a **one-shot pipeline operator**. The VDBE program generates `OP_SorterInsert` opcodes to feed records in, then `OP_SorterSort` triggers the sort, and `OP_SorterNext` iterates results. There is no "previously sorted result that gets updated." Each query execution builds a new sorter from scratch.

### Dirty Index Tracking

**N/A.** One-shot query execution. No persistent sorted state.

### Current Sort Algorithm

Internal sort: merge sort on in-memory PMAs. External sort: multi-way merge of sorted runs from temp files. Optimized for minimal memory usage and streaming output.

### Array Sizes

Depends on query and available memory. SQLite's default sorter cache is 250 pages (~1MB). Can handle millions of rows via external sort.

### Model Match: 3/100

**Mismatch analysis:**

- ❌ **Not a persistent mutable sorted array** — one-shot query operator
- ❌ External merge sort architecture is fundamentally different from in-place sorting
- ❌ Data flows through as a stream (insert→sort→iterate), not a mutable buffer
- ❌ No concept of "previously sorted state" — each ORDER BY creates a new sorter
- ❌ SQLite values simplicity and minimal code — adding incremental sort would add significant complexity for unclear benefit

**Integration path**: The only conceivable path would be sorted virtual tables or cached ORDER BY results that are incrementally maintained when base tables change. But this is a materialized view problem, not a sorting algorithm problem. SQLite intentionally avoids materialized view complexity.

---

## J. VLC / Music Player Daemon (Playlist Sorting)

**Knowledge-based analysis** (VLC playlist system, MPD sorted database views)

### Data Structure

**VLC**: Playlist is a tree structure (`playlist_item_t` with children array). The media library uses a SQL database (medialibrary). Sort is applied via `qsort` on the items array when the user requests it.

**MPD**: Song database is an in-memory collection. Sorted views are maintained as arrays of pointers to song structs. When the user requests sort by artist/title/etc., the pointer array is sorted via `qsort`/`std::sort`.

### Update Mechanism

**VLC**: Media scanner discovers/updates files. Playlist items update when metadata changes (tag editing). Queue modifications add/remove items.

**MPD**: Database updates happen when music files are added/modified/removed. The `update` command rescans the music directory. Individual song metadata updates are tracked per-file.

### Dirty Index Tracking

**VLC**: No tracking of which playlist items changed for sorting. Metadata changes trigger UI refresh but not incremental sort.

**MPD**: Database updates track which songs were modified (via inode/mtime), but this is for the scanner, not for sort order maintenance. Sorted views are rebuilt from scratch.

### Current Sort Algorithm

`qsort()` (C stdlib) or `std::sort()`. Applied as a one-shot operation when user changes sort criteria or explicitly requests re-sort.

### Array Sizes

Typical music libraries: hundreds to tens of thousands of songs. Large collections: 50K–200K.

### Model Match: 55/100 ⭐

**Mismatch analysis:**

- ✅ **Contiguous array** of pointers to items — yes
- ✅ **Previously sorted** — the playlist/library view maintains sorted order between user actions
- ✅ **Small k** — user edits tags on a few tracks, library scanner updates a handful of files
- ✅ **Dirty tracking is feasible** — MPD already tracks which songs were modified during `update`
- ✅ **Array sizes are in the sweet spot** — 10K–100K where DeltaSort's O(n√k) is better than O(n log n)
- ⚠️ Would need to add index tracking (map modified songs back to positions in sorted view)
- ⚠️ Sort is relatively rare (only when user changes sort order or metadata changes) — not a hot path

**Integration path**: **Most feasible of all 10 systems.** For MPD specifically:

1. Maintain a sorted view (pointer array) of the song database
2. When `update` scans and finds modified songs, record their positions in the sorted view
3. Call DeltaSort with the dirty set instead of full `qsort`
4. For n=50K, k=10 modified songs: DeltaSort ≈ O(50K × √10) ≈ 158K operations vs. full sort O(50K × 17) ≈ 850K comparisons

The practical challenge: sort operations are infrequent enough that the latency savings (maybe 2ms → 0.5ms) are unlikely to be user-perceptible. The engineering effort to integrate DeltaSort would likely not be justified by the performance gain in this specific application.

---

## Summary Table

| System              | Array?            | Sorted persists?   | Dirty tracking? | k ≪ n pattern? | Match   |
| ------------------- | ----------------- | ------------------ | --------------- | -------------- | ------- |
| **A. Pandas**       | ✅ (NumPy)        | ❌                 | ❌              | ✅             | 25%     |
| **B. DuckDB**       | ❌ (chunks)       | ❌                 | ❌              | ❌             | 5%      |
| **C. Redis ZSET**   | ❌ (skip list)    | ✅ (always sorted) | N/A             | N/A            | 5%      |
| **D. Lucene**       | ❌ (immutable)    | ❌                 | ❌              | ❌             | 3%      |
| **E. Linux kernel** | ✅                | ❌ (one-shot)      | ❌              | ⚠️ (rare)      | 30%     |
| **F. Chromium**     | ✅ (vector)       | ✅                 | ❌              | ✅             | 45%     |
| **G. Godot**        | ✅ (vector)       | ❌ (rebuilt)       | ❌              | ❌             | 15%     |
| **H. Arrow**        | ✅ (immutable)    | ❌                 | ❌              | ❌             | 5%      |
| **I. SQLite**       | ❌ (PMA/external) | ❌                 | ❌              | ❌             | 3%      |
| **J. VLC/MPD**      | ✅                | ✅                 | ⚠️ (feasible)   | ✅             | **55%** |

---

## Brutal Honesty: The Core Problem

**DeltaSort's model is theoretically clean but practically rare.** The four preconditions — (1) contiguous array, (2) persistent sorted state, (3) dirty index tracking, (4) batch re-sort pattern — rarely co-occur in real systems:

- **Database systems** (DuckDB, SQLite, Lucene) never have persistent mutable sorted arrays. They use one-shot sort operators, immutable segments, or index structures.
- **Always-sorted structures** (Redis ZSET) don't need batch re-sorting — they maintain order continuously.
- **Immutable data** (Arrow, Lucene) can't have "changed values" by definition.
- **Frame-rebuilt lists** (Godot) destroy and recreate state too frequently.
- **Small arrays** (Chromium bookmarks) don't benefit because optimized `std::sort`/introsort with insertion sort fallback already handles small inputs efficiently.

The **most natural fit** is application-level sorted views of mutable collections (playlist managers, task lists, dashboards) where:

- Data lives in a contiguous array
- Users make sporadic small edits
- The sorted view needs to be refreshed
- The array is large enough (n > 1K) that O(n√k) meaningfully outperforms O(n log n)

This aligns exactly with the paper's introduction: _"a modern client application that needs to display a large dynamic sorted list of items (e.g., task management applications or interactive dashboards)."_ The paper's motivating use case is honest — this IS the sweet spot, and the surveyed real-world systems confirm that DeltaSort targets a specific niche rather than a broadly applicable primitive.

### Where DeltaSort Actually Fits Best (not in the 10 systems above)

1. **React/UI framework sorted lists** — the paper's motivating example. Large sorted tables in web apps with `useMemo`-style memoization.
2. **Spreadsheet engines** — Google Sheets, Excel Online. User edits cells, sorted view updates.
3. **In-memory OLTP sorted indexes** — systems like MemSQL/SingleStore, VoltDB with sorted secondary indexes.
4. **Real-time leaderboard systems** — game servers with batch score updates and sorted display.
5. **Time-series dashboards** — sorted sensor/metric displays with streaming updates.

---

---

# Second Sweep: 10 Less-Obvious Candidate Domains

The first sweep (above) examined obvious candidates (pandas, Redis, DuckDB, Lucene, Linux kernel) and found poor fits. This second sweep investigates 10 less-obvious domains, with emphasis on N-body simulation (#2), event simulation (#6), particle depth sorting (#7), and financial order books (#8).

---

## Candidate 1: CGAL Sweep Line Algorithm

**Domain:** Computational geometry — sweep line algorithms for segment intersection, Voronoi diagrams, etc.

### Source Code Analysis

- Could not fetch specific file (404 on raw GitHub URL); analysis based on CGAL architecture knowledge.

### Assessment

| Criterion                 | Finding                                                                                                                                                                                                               |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Contiguous sorted array?  | **No.** CGAL's sweep line status structure uses a balanced BST (`std::set` or equivalent AVL/red-black tree), not a contiguous sorted array. The event queue is typically a `std::priority_queue` or `std::multiset`. |
| Tracked changed indices?  | **No.** Events are inserted/removed one at a time from tree structures.                                                                                                                                               |
| Batch update pattern?     | **No.** Events are processed one by one in sweep order.                                                                                                                                                               |
| Current sorting approach? | Self-balancing BST — O(log n) insert/delete.                                                                                                                                                                          |
| Practical sizes?          | Event queues: O(n + k) where n = segments, k = intersections. Hundreds to millions.                                                                                                                                   |

**Confidence: 5/100**

CGAL's sweep line fundamentally relies on tree-based data structures for O(log n) insert/delete of individual elements. The sweep paradigm processes events sequentially, not in batches. DeltaSort's batch re-sort model is architecturally incompatible.

---

## Candidate 2: SWIFT N-Body Simulation (Particle Sorting) ⭐

**Domain:** Astrophysical N-body/SPH simulation — sorting particles within spatial cells for neighbor-finding.

### Source Code Analysis

- **Files examined:** `space.c` (~5000+ lines), `runner_sort.c` (complete), `sort_part.h` (complete)
- **Repository:** [SWIFTSIM/SWIFT](https://github.com/SWIFTSIM/SWIFT)

### Key Findings

**Data structure:** Contiguous `sort_entry` arrays:

```c
struct sort_entry {
    float d;  // Distance on axis
    int i;    // Particle index
};
```

**Sorting pattern:** Each cell maintains up to **13 sorted arrays** (one per cardinal direction pair). At leaf cells, the array is filled from particle positions and sorted with an in-place quicksort (`runner_do_sort_ascending`). At non-leaf cells, children's sorted arrays are 8-way merge-sorted.

**Sort triggers:** Cells track displacement via `dx_max_sort` and per-particle `x_diff_sort[3]`. When particles drift beyond threshold (`space_maxreldx * cell_width`), cells are flagged for re-sorting via bitmask flags (`cell_flag_do_hydro_sub_sort`, `cell_flag_do_stars_resort`).

**Current re-sort behavior:** Despite displacement tracking, re-sorting **rebuilds the sort array from scratch**:

```c
// Fill the sort array from particle positions
for (int k = 0; k < count; k++) {
    entries[k].i = k;
    entries[k].d = px[0] * runner_shift[j][0] + px[1] * runner_shift[j][1] + px[2] * runner_shift[j][2];
}
// Full quicksort
runner_do_sort_ascending(entries, count);
```

**Star resort:** After star formation events (new stars created), a dedicated `runner_do_stars_resort` task re-sorts all star particles in affected cells.

### Assessment

| Criterion                 | Finding                                                                                                                                                                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Contiguous sorted array?  | **✅ Yes.** `sort_entry` arrays are contiguous flat memory.                                                                                                                                                                     |
| Tracked changed indices?  | **Partially.** Per-particle displacement tracking exists (`x_diff_sort`), and per-cell sort validity flags. But SWIFT does NOT currently track _which specific sort entries_ changed order — it rebuilds the entire sort array. |
| Batch update pattern?     | **✅ Yes.** Particles are "drifted" (positions updated) in bulk between timesteps, then cells are re-sorted. Many particles move only slightly (preserving sort order), while a subset move enough to change relative order.    |
| Current sorting approach? | In-place quicksort (custom implementation with manual stack), O(n log n) per cell per direction.                                                                                                                                |
| Practical sizes?          | 100–1000 particles per leaf cell (max ≈ 2^10 = 1024). 13 directions × many cells.                                                                                                                                               |

**Confidence: 35/100**

SWIFT is the **most promising** candidate across both sweeps. The infrastructure is close to DeltaSort's model:

- Contiguous sorted arrays ✅
- Repeated re-sorting after position updates ✅
- Per-particle displacement tracking already exists ✅
- In-place sorting ✅

**However**, the gap is significant: SWIFT does not identify which sort entries changed rank — it rebuilds from scratch. Adapting would require:

1. After drifting, scan each entry to check if `sort[k].d` ≤ `sort[k+1].d` still holds, collecting violations
2. Pass those indices to DeltaSort instead of quicksort

For a cell of 500 particles where ~50 changed sort order (k/n = 10%): DeltaSort's O(500 × √50) ≈ O(3500) vs quicksort's O(500 × 9) ≈ O(4500) — a marginal win. The win grows for larger cells or lower k/n ratios. HPC profiling of real SWIFT simulations would be needed to assess practical impact.

---

## Candidate 3: ClickHouse MergeTree

**Domain:** Column-oriented OLAP database — merging sorted data parts.

### Source Code Analysis

- **File examined:** `MergeTreeDataMergerMutator.cpp` (complete)
- **Repository:** [ClickHouse/ClickHouse](https://github.com/ClickHouse/ClickHouse)

### Key Findings

ClickHouse stores data in immutable sorted "parts" (SSTable-like files on disk). The `MergeTreeDataMergerMutator` selects ranges of parts to merge, then `MergeTask` performs a streaming k-way merge of sorted runs into a new part. Parts are immutable; mutations create new parts.

### Assessment

| Criterion                 | Finding                                                                                      |
| ------------------------- | -------------------------------------------------------------------------------------------- |
| Contiguous sorted array?  | **No.** Data in immutable file-based parts.                                                  |
| Tracked changed indices?  | **No.** Parts are immutable; mutations create new parts.                                     |
| Batch update pattern?     | **No.** Merging combines multiple sorted parts (k-way merge), not re-sorting a single array. |
| Current sorting approach? | K-way merge of sorted runs (standard LSM-tree pattern).                                      |
| Practical sizes?          | Parts: millions to billions of rows, but merge is streaming, not in-place.                   |

**Confidence: 5/100**

ClickHouse's MergeTree is an LSM-tree variant. Merging is a streaming k-way merge of immutable sorted parts — entirely different from in-place re-sorting.

---

## Candidate 4: Open vSwitch Flow Classifier

**Domain:** Software-defined networking — classifying packets against flow rules by priority.

### Source Code Analysis

- **Files examined:** `classifier.c` (~2000+ lines), `classifier.h` (complete)
- **Repository:** [openvswitch/ovs](https://github.com/openvswitch/ovs)

### Key Findings

The classifier uses:

- `cmap` (concurrent hash map) for rule storage within subtables
- `pvector` (priority-sorted vector) for subtable ordering by max priority
- Prefix **tries** for IP address prefix matching optimization
- Rules with the same mask go into the same "subtable"; lookup hashes in each subtable and tracks highest-priority match

No sorted arrays anywhere in the hot path. The `pvector` of subtables is sorted by priority but has <100 entries and changes rarely.

### Assessment

| Criterion                 | Finding                                                             |
| ------------------------- | ------------------------------------------------------------------- |
| Contiguous sorted array?  | **No.** Hash maps, tries, priority vectors (small).                 |
| Tracked changed indices?  | **No.** Rules inserted/removed via hash operations.                 |
| Batch update pattern?     | **Partially.** "Deferred mode" batches RCU visibility, not sorting. |
| Current sorting approach? | Hash-based lookup, not sort-based.                                  |
| Practical sizes?          | Subtable pvector: ~10–100. Rule cmap: thousands per subtable.       |

**Confidence: 5/100**

OVS's classifier is entirely hash-based with trie acceleration. No contiguous sorted array pattern exists.

---

## Candidate 5: samtools BAM File Sorting

**Domain:** Bioinformatics — sorting genomic alignment records by coordinate or read name.

### Source Code Analysis

- **File examined:** `bam_sort.c` (~3000+ lines, complete)
- **Repository:** [samtools/samtools](https://github.com/samtools/samtools)

### Key Findings

One-shot external sort:

1. Read BAM records into memory buffer (`bam1_tag` array)
2. Sort using **radix sort** for coordinate ordering (`ks_radixsort` — LSD radix sort on packed tid/pos/strand), or **merge sort** (`ks_mergesort`) for name/tag ordering
3. If buffer exceeds memory limit, write sorted temp file, reset buffer
4. K-way heap merge (`ks_heapmake`/`ks_heapadjust`) of all temp files + in-memory buffer into final output

Multi-threaded sorting via `sort_blocks()` — buffer is split into chunks, each sorted independently by a thread, then merged.

### Assessment

| Criterion                 | Finding                                                     |
| ------------------------- | ----------------------------------------------------------- |
| Contiguous sorted array?  | **Yes** (in-memory buffer), but single-use, not maintained. |
| Tracked changed indices?  | **No.** Records read sequentially from files.               |
| Batch update pattern?     | **No.** One-pass sort-and-write.                            |
| Current sorting approach? | Radix sort + k-way heap merge.                              |
| Practical sizes?          | Buffer: 768MB default, millions of records.                 |

**Confidence: 5/100**

Classic external sort — read once, sort once, write once. No re-sort scenario exists.

---

## Candidate 6: ns-3 Discrete Event Simulation Scheduler

**Domain:** Network simulation — priority queue of future events sorted by timestamp.

### Source Code Analysis

- **Files examined:** `scheduler.h`, `calendar-scheduler.cc`, `calendar-scheduler.h`
- **Repository:** [nsnam/ns-3-dev](https://github.com/nsnam/ns-3-dev-git)

### Key Findings

Five scheduler implementations, none using sorted arrays:

- **CalendarScheduler:** Array of `std::list<Scheduler::Event>` buckets (calendar queue). O(1) amortized insert/remove. Auto-resizes bucket count.
- **HeapScheduler:** Binary heap on `std::vector`
- **MapScheduler:** `std::map<EventKey, Event>` (red-black tree)
- **ListScheduler:** `std::list<Event>` (sorted linked list)
- **PriorityQueueScheduler:** `std::priority_queue`

### Assessment

| Criterion                 | Finding                                                   |
| ------------------------- | --------------------------------------------------------- |
| Contiguous sorted array?  | **No.** Lists, trees, heaps, bucket arrays.               |
| Tracked changed indices?  | **No.** Events inserted/removed one at a time.            |
| Batch update pattern?     | **No.** Events processed individually in timestamp order. |
| Current sorting approach? | Calendar queue O(1), heap O(log n), or tree O(log n).     |
| Practical sizes?          | Thousands to millions of pending events.                  |

**Confidence: 5/100**

Event-driven simulators process single events, the opposite of DeltaSort's batch model.

---

## Candidate 7: Godot Engine Particle Depth Sorting

**Domain:** Game engine — sorting particles by camera distance for correct alpha blending.

### Source Code Analysis

- **Files examined:** `particles_storage.cpp` (~1500+ lines), `particles_storage.h` (~500 lines)
- **Repository:** [godotengine/godot](https://github.com/godotengine/godot)

### Key Findings

Sorting is entirely **GPU-based**:

```cpp
sort_effects->sort_buffer(p_uniform_set, p_particles->amount);
```

Pipeline: `COPY_MODE_FILL_SORT_BUFFER` populates GPU buffer with (depth, index) pairs → `SortEffects::sort_buffer()` runs GPU bitonic/radix sort → `COPY_MODE_FILL_INSTANCES_WITH_SORT_BUFFER` reorders instances.

Draw order modes: `VIEW_DEPTH` (camera distance), `LIFETIME`, `INDEX`, `REVERSE_LIFETIME`.

### Assessment

| Criterion                 | Finding                                                            |
| ------------------------- | ------------------------------------------------------------------ |
| Contiguous sorted array?  | **Yes** (GPU buffer), but on GPU, not CPU.                         |
| Tracked changed indices?  | **No.** All particles recompute depth each frame.                  |
| Batch update pattern?     | **Yes** technically, but k ≈ n (every particle moves every frame). |
| Current sorting approach? | GPU bitonic/radix sort via compute shaders.                        |
| Practical sizes?          | Hundreds to tens of thousands of particles per emitter.            |

**Confidence: 5/100**

Two fundamental problems: (1) sorting runs on GPU, not CPU — DeltaSort would need a GPU compute shader port; (2) every frame every particle moves, so k ≈ n and DeltaSort has no advantage over a full sort.

---

## Candidate 8: Financial Limit Order Book

**Domain:** Trading systems — maintaining sorted bid/ask price levels for order matching.

### Source Code Analysis

- Could not fetch specific open-source implementation (404). Analysis based on well-known order book architectures.

### Key Findings

**Typical implementations:**

- **Red-black tree / `std::map`**: Price level → order queue. Most common in production. O(log P).
- **Fixed array by price tick**: Array indexed by `(price - min_price) / tick_size`. O(1) lookup. Used when price range is bounded.
- **Sorted vector of active price levels**: Rare but exists in some simpler implementations.

**Update pattern:** Orders arrive **one at a time**. Each order either adds a new price level, removes one (last order filled), or modifies quantity. Batch updates exist in market data feeds (e.g., Nasdaq ITCH) but are typically 1–10 changes per message.

### Assessment

| Criterion                 | Finding                                                   |
| ------------------------- | --------------------------------------------------------- |
| Contiguous sorted array?  | **Rarely.** Production systems use trees or fixed arrays. |
| Tracked changed indices?  | **Yes** (the changed price level is known).               |
| Batch update pattern?     | **Mostly k=1.** Single-order updates dominate.            |
| Current sorting approach? | Tree O(log n), or direct-indexed array O(1).              |
| Practical sizes?          | 5–500 active price levels per side.                       |

**Confidence: 10/100**

While the model occasionally aligns (sorted array of price levels, known changes), two factors kill it: (1) k=1 dominates, where binary insertion sort is already O(n) — optimal; (2) production systems use trees or direct-indexed arrays specifically to avoid the O(n) shift cost.

---

## Candidate 9: MANET Sorted Neighbor Tables

**Domain:** Mobile Ad-hoc Networks — routing protocols maintaining neighbor tables sorted by link metrics.

### Source Code Analysis

- No specific implementation fetched; analysis based on OLSR, AODV, BATMAN protocol knowledge.

### Assessment

| Criterion                 | Finding                                                                  |
| ------------------------- | ------------------------------------------------------------------------ |
| Contiguous sorted array?  | **Rarely.** Implementations use linked lists or hash tables.             |
| Tracked changed indices?  | **Partially.** Hello messages identify which neighbors changed RSSI/ETX. |
| Batch update pattern?     | **Partially.** Hello messages can report multiple metric changes.        |
| Current sorting approach? | List insertion sort, or hash table + periodic sort.                      |
| Practical sizes?          | Very small: 5–50 neighbors typically.                                    |

**Confidence: 10/100**

The theory has some alignment (neighbors change metrics simultaneously) but practice kills it: (1) tables are tiny (n < 50), making any algorithm fast; (2) linked lists, not arrays; (3) DeltaSort's overhead dominates at these sizes.

---

## Candidate 10: Incremental Compilation Sorted Symbol Tables

**Domain:** Compilers/IDEs — maintaining sorted symbol tables across incremental edits.

### Source Code Analysis

- No specific implementation fetched; analysis based on rust-analyzer, TypeScript compiler, LLVM architecture.

### Assessment

| Criterion                 | Finding                                                                      |
| ------------------------- | ---------------------------------------------------------------------------- |
| Contiguous sorted array?  | **No.** Compilers use hash tables (`HashMap`, `DenseMap`) for symbol lookup. |
| Tracked changed indices?  | **Yes** (incrementally — compilers know which symbols changed).              |
| Batch update pattern?     | **Yes** (a file edit can change multiple symbols).                           |
| Current sorting approach? | Hash tables for lookup. Sorted output built from scratch when needed.        |
| Practical sizes?          | Thousands to millions of symbols.                                            |

**Confidence: 8/100**

Conceptually appealing: many symbols, few change per edit, compiler knows which ones. But compilers don't use sorted arrays — they use hash tables for O(1) lookup. Sorted output (debug info, module exports) is built from scratch. Switching would require fundamental architectural change.

---

## Second Sweep Summary Table

| #   | Candidate               |    Array?    | Tracked? | Batch? |     Sizes     | Confidence |
| --- | ----------------------- | :----------: | :------: | :----: | :-----------: | :--------: |
| 1   | CGAL Sweep Line         |    ❌ BST    |    ❌    |   ❌   |    100–1M     |     5%     |
| 2   | **SWIFT N-Body** ⭐     |    **✅**    |  **~**   | **✅** | 100–1000/cell |  **35%**   |
| 3   | ClickHouse MergeTree    |   ❌ files   |    ❌    |   ❌   |   millions    |     5%     |
| 4   | OVS Flow Classifier     |   ❌ hash    |    ❌    |   ❌   |    10–100     |     5%     |
| 5   | samtools BAM Sort       | ✅ one-shot  |    ❌    |   ❌   |   millions    |     5%     |
| 6   | ns-3 Event Scheduler    | ❌ list/heap |    ❌    |   ❌   |   thousands   |     5%     |
| 7   | Godot Particle Sort     |    ✅ GPU    |    ❌    |  k≈n   |    100–10K    |     5%     |
| 8   | Order Book              |    rarely    |    ✅    |  k=1   |     5–500     |    10%     |
| 9   | MANET Neighbors         |    rarely    |    ~     |   ~    |     5–50      |    10%     |
| 10  | Incremental Compilation |   ❌ hash    |    ✅    |   ✅   |   thousands   |     8%     |

---

## Cross-Sweep Conclusions

### The Recurring Mismatch Patterns

Across all 15 candidates examined (5 first sweep + 10 second sweep), the same four failure modes recur:

1. **Wrong data structure** (11/15 candidates): Most real-world systems use hash tables, trees, heaps, or linked lists instead of contiguous sorted arrays. This is the dominant mismatch.

2. **Wrong update granularity** (4/15): Systems optimized for k=1 single-element updates (event schedulers, order books, sweep line) use O(log n) structures that beat DeltaSort's O(n√k) = O(n) for k=1.

3. **Wrong paradigm** (3/15): One-shot sorts (samtools), streaming merges (ClickHouse), or GPU sorts (Godot) don't maintain a sorted array across updates.

4. **Too small** (2/15): Some candidates (MANET, order books) have arrays too small for DeltaSort's asymptotic advantage to materialize.

### Best Candidate: SWIFT N-Body (35%)

SWIFT N-body simulation is the strongest candidate found across both sweeps:

- ✅ Contiguous `sort_entry` arrays, sorted per cell per direction
- ✅ Repeated re-sort after drift steps
- ✅ Per-particle displacement tracking already exists
- ✅ HPC context where constant factors matter
- ⚠️ Requires adding changed-index detection (scan for out-of-order neighbors post-drift)
- ⚠️ Cell sizes (100–1000) may be too small for meaningful speedup

### Refined "Ideal Application" Profile

An ideal DeltaSort application would have:

- A **large contiguous sorted array** (n > 10,000) maintained across iterations
- **Moderate batch changes** (k in the range √n to n/10) at **known indices**
- **Memory constraints** favoring O(1) auxiliary space
- **Repeated re-sorting** amortizing the cost of tracking infrastructure

Domains worth exploring next:

- **In-memory database sorted indexes** (single-node OLTP with batch transactions modifying sorted secondary indexes)
- **Scientific computing sorted meshes** (finite element methods with adaptive mesh refinement maintaining sorted node/element arrays)
- **Real-time leaderboard systems** (batch score updates → re-sort for display)
- **Spreadsheet engines** (Google Sheets, Excel Online — user edits cells, sorted view updates)
- **Sensor array telemetry** (sorted measurement arrays updated with new readings from a known subset of sensors)

---

---

# Third Sweep: Application-Layer Scenarios

The first two sweeps focused on systems-level infrastructure and found poor fits. This third sweep investigates **application-layer scenarios** where the key insight holds: an application maintains a sorted view, processes updates to known items, and re-sorts. This sweep also includes a critical analysis of TimSort's behavior on nearly-sorted data (Section G), which directly informs DeltaSort's competitive position.

---

## A. PostgreSQL Materialized View Maintenance

**Domain:** `REFRESH MATERIALIZED VIEW CONCURRENTLY` — incremental update of a pre-computed sorted query result.

### Source Code Analysis

- **File examined:** `src/backend/commands/matview.c` (complete, 600+ lines)
- **Repository:** [postgres/postgres](https://github.com/postgres/postgres)

### Key Findings

PostgreSQL's `REFRESH MATERIALIZED VIEW CONCURRENTLY` works via a **diff-based approach**, not incremental re-sorting:

1. **Full re-execution:** The entire query is re-executed into a temporary table (`refresh_matview_datafill`)
2. **Diff computation:** A `FULL OUTER JOIN` between old and new data computes the diff:
    ```sql
    SELECT mv.ctid AS tid, newdata.*::type AS newdata
    FROM matview mv FULL JOIN tempdata newdata ON (equality_columns)
    WHERE newdata.* IS NULL OR mv.* IS NULL  -- changed rows only
    ORDER BY tid
    ```
3. **Set-based mutation:** Applies `DELETE` for removed rows and `INSERT` for new rows
4. **Non-concurrent path:** `refresh_by_heap_swap` literally swaps the heap file — no incremental anything

### Assessment

| Criterion                 | Finding                                                                                                                                               |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Contiguous sorted array?  | **No.** PostgreSQL heap pages are unordered. The "sorted" view comes from index scans or `ORDER BY` at query time.                                    |
| Tracked changed indices?  | **Partially.** The diff table identifies changed rows by `ctid`, but these are page-level tuple identifiers, not array indices in a sorted structure. |
| Batch update pattern?     | **Yes** (the diff is a batch of changes), but applied as SQL DML, not as array mutations.                                                             |
| Current sorting approach? | The `ORDER BY tid` in the diff query uses standard query execution (sort or index scan). No persistent sorted array exists.                           |
| Practical sizes?          | Thousands to millions of rows.                                                                                                                        |

**Confidence: 10/100**

The fundamental disconnect: PostgreSQL doesn't maintain a sorted array. Materialized views are stored as unordered heap tables with optional indexes. `CONCURRENTLY` mode computes a diff and applies set-based mutations — it's an entirely SQL-based approach. The sorted order exists only at query-time (via `ORDER BY` using indexes or explicit sorts). There's no contiguous sorted array to delta-sort.

The only conceivable integration point would be if PostgreSQL added a "sorted materialized view" that physically stores rows in sort order and incrementally maintains that order when base tables change. This would require:

1. A new storage format (clustered/sorted heap)
2. Identifying which rows in the sorted structure changed
3. Re-sorting in-place with DeltaSort instead of doing the full FULL JOIN + DELETE/INSERT dance

This is a multi-year PostgreSQL kernel project, and the PG community has historically rejected such complexity in favor of the existing clean-separation approach (heap + separate indexes).

---

## B. ag-Grid Delta Sort ⭐⭐⭐ (ALREADY EXISTS)

**Domain:** JavaScript data grid — sorting a large table after cell edits or row transactions.

### Source Code Analysis

- **Files examined:** `sortStage.ts`, `deltaSort.ts`, `changedRowNodes.ts` (all complete)
- **Repository:** [ag-grid/ag-grid](https://github.com/ag-grid/ag-grid)
- **Path:** `packages/ag-grid-community/src/clientSideRowModel/`

### Critical Discovery

**ag-Grid already implements a delta sort algorithm with the EXACT same conceptual model as DeltaSort.** The function `doDeltaSort()` in `deltaSort.ts` does the following:

```typescript
// From ag-Grid's deltaSort.ts (actual production code)
/**
 * Time complexity: O(t log t + n) where t = touched rows, n = total rows
 * This is faster than full sort O(n log n) when t << n
 */
export const doDeltaSort = (
    rowNodeSorter: RowNodeSorter,
    rowNode: RowNode,
    changedRowNodes: ChangedRowNodes,
    changedPath: ChangedPath,
    sortOptions: SortOption[]
): RowNode[] => { ... }
```

**Algorithm outline:**

1. Classify rows as "touched" (updated, added, or in changed path) vs "untouched"
2. If no rows are touched, return previous sorted array
3. Sort only touched rows: `touchedRows.sort(comparator)` — O(t log t)
4. If all rows are touched, return sorted array directly
5. Merge sorted touched rows with untouched rows from previous sort order using a **two-pointer merge** — O(n)

**Dirty tracking infrastructure:**

```typescript
export class ChangedRowNodes<TData = any> {
    public reordered = false;
    public readonly removals: RowNode<TData>[] = [];
    public readonly updates = new Set<RowNode<TData>>();
    public readonly adds = new Set<RowNode<TData>>();
}
```

**Integration in the sort pipeline (`sortStage.ts`):**

```typescript
const useDeltaSort =
    sortOptions.length > 0 &&
    !!changedRowNodes &&
    this.gos.get('deltaSort');  // user-facing option!

if (useDeltaSort && changedRowNodes) {
    newChildrenAfterSort = doDeltaSort(rowNodeSorter!, rowNode,
        changedRowNodes, changedPath!, sortOptions);
} else {
    newChildrenAfterSort = rowNodeSorter!.doFullSortInPlace(...);
}
```

### Key Differences: ag-Grid's Delta Sort vs. DeltaSort (the paper's algorithm)

| Aspect                         | ag-Grid `doDeltaSort`                                          | DeltaSort (paper)                                                   |
| ------------------------------ | -------------------------------------------------------------- | ------------------------------------------------------------------- |
| **Data structure**             | Array of `RowNode` references                                  | Generic array `T[]`                                                 |
| **Changed item tracking**      | `ChangedRowNodes` with `updates`/`adds` Sets                   | `Set<number>` of indices                                            |
| **Algorithm**                  | Sort-touched + merge with untouched                            | Extract-sort-replace + directional scan with binary search rotation |
| **Time complexity**            | O(t log t + n)                                                 | O(n√k)                                                              |
| **Space complexity**           | O(n) — allocates new result array + Map + Sets                 | O(1) auxiliary (in-place)                                           |
| **Stability**                  | Stable (preserves original index as tie-breaker)               | Not inherently stable                                               |
| **Minimum threshold**          | `MIN_DELTA_SORT_ROWS = 4` — falls back to full sort below this | No minimum                                                          |
| **Handles removals/additions** | Yes — filters removed, appends new                             | Only handles value changes at known indices                         |

### Assessment

| Criterion                 | Finding                                                                                                                 |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Contiguous sorted array?  | **Yes.** `childrenAfterSort` is a `RowNode[]` array.                                                                    |
| Tracked changed indices?  | **Yes.** `ChangedRowNodes` tracks exactly which rows were updated/added/removed via transactions.                       |
| Batch update pattern?     | **Yes.** ag-Grid uses "transactions" to batch row updates before re-sorting.                                            |
| Current sorting approach? | Full sort (`Array.sort` via `doFullSortInPlace`) when `deltaSort` option is off; `doDeltaSort` merge algorithm when on. |
| Practical sizes?          | ag-Grid supports 1M+ rows. Typical enterprise grids: 10K–500K rows.                                                     |

**Confidence: 95/100** ⭐⭐⭐

This is the single strongest validation of DeltaSort's model found across all three sweeps. ag-Grid:

- **Maintains a sorted array** (`childrenAfterSort`) that persists between transactions
- **Tracks exactly which rows changed** (`ChangedRowNodes.updates`, `.adds`)
- **Implements an incremental sort** that only sorts changed rows and merges
- **Exposes it as a user-facing feature** (`deltaSort: true` grid option)
- **Handles real-world scale** (100K+ rows in production)

**Where DeltaSort (the paper's algorithm) could improve on ag-Grid's approach:**

1. **O(1) space vs O(n) space**: ag-Grid allocates a new result array, Map, and Sets. DeltaSort works in-place.
2. **Asymptotic advantage in specific regimes**: For moderate k, DeltaSort's O(n√k) could beat ag-Grid's O(t log t + n) when t log t is the bottleneck, though ag-Grid's O(n) merge dominates for small t.
3. **No new array allocation**: DeltaSort modifies the array in-place, which is better for memory pressure in large grids.

**Practical integration challenge**: ag-Grid's TypeScript codebase and the `RowNode` abstraction would make it straightforward to swap in DeltaSort. However, ag-Grid's delta sort also handles additions and removals, which DeltaSort doesn't — DeltaSort only handles value changes at known indices. A drop-in replacement would need to handle the superset of operations.

**This is a direct competitor/validation that proves the problem space is real and commercially important.**

---

## C. Apache Spark Sorted RDD Partitions

**Domain:** After `sortByKey()`, partitions are sorted. After narrow transformations, can we re-sort incrementally?

### Source Code Analysis

- **File examined:** `OrderedRDDFunctions.scala` (complete)
- **Repository:** [apache/spark](https://github.com/apache/spark)

### Key Findings

`sortByKey()` creates a `ShuffledRDD` with a `RangePartitioner` and key ordering. This triggers a full shuffle + sort. Once sorted, the RDD is immutable.

The `repartitionAndSortWithinPartitions()` method sorts within each partition using `ExternalSorter` — a full merge sort with optional spill to disk.

**Critical observation:** Spark RDDs are **immutable.** After `sortByKey()`, you cannot mutate elements. Any transformation (`map`, `filter`, `mapValues`) creates a **new RDD**, which must be sorted from scratch if sort order is needed. There is no concept of "a few items changed" because nothing changes — you get a new dataset.

Even `mapValues` (which preserves partitioning) doesn't preserve sort order within partitions if the values are part of the sort key. And if values aren't part of the sort key, sort order is already preserved without re-sorting.

### Assessment

| Criterion                 | Finding                                                                                                              |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Contiguous sorted array?  | **Partially.** Within a partition, data can be in a contiguous array if it fits in memory (otherwise external sort). |
| Tracked changed indices?  | **No.** RDDs are immutable — nothing changes.                                                                        |
| Batch update pattern?     | **No.** Transformations create new RDDs, not mutations.                                                              |
| Current sorting approach? | `ExternalSorter` (merge sort with spill), or `TimSort` for in-memory arrays.                                         |
| Practical sizes?          | Partitions: thousands to millions of records.                                                                        |

**Confidence: 5/100**

Spark's immutable RDD model is fundamentally incompatible with DeltaSort's mutable-array model. The only conceivable path would be Spark's structured streaming with sorted state, but even there, state updates go through a key-value store (RocksDB), not sorted arrays.

---

## D. Prometheus Sorted Postings Lists

**Domain:** Time-series database — maintaining sorted postings lists (series ID lists per label pair).

### Source Code Analysis

- **File examined:** `tsdb/index/postings.go` (complete, 1000+ lines)
- **Repository:** [prometheus/prometheus](https://github.com/prometheus/prometheus)

### Key Findings

`MemPostings` is the core in-memory index structure. It maps `label_name -> label_value -> []SeriesRef` (sorted slices of series IDs).

**Sorting mechanism:**

- `NewUnorderedMemPostings()`: During bulk loading (WAL replay at startup), postings are appended out of order for speed
- `EnsureOrder()`: Called once after bulk load — sorts ALL postings lists in parallel using `slices.Sort(l)`:
    ```go
    for i := 0; i < concurrency; i++ {
        go func() {
            for job := range workc {
                for _, l := range *job {
                    slices.Sort(l)  // Full sort of each list
                }
            }
        }()
    }
    ```
- `addFor()` (during normal operation): Inserts with **repair sort** — appends the new ID and bubbles it backward to maintain order:
    ```go
    for i := len(list) - 1; i >= 1; i-- {
        if list[i] >= list[i-1] { break }
        list[i], list[i-1] = list[i-1], list[i]
    }
    ```

**Key insight:** During normal operation, series IDs are mostly-increasing (assigned sequentially), so the repair sort in `addFor()` is O(1) amortized. The full `EnsureOrder()` is only called during startup replay.

**Deletion:** `Delete()` iterates affected postings lists and filters out deleted IDs by creating a new slice without them — O(n) per list.

### Assessment

| Criterion                 | Finding                                                                                                                |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Contiguous sorted array?  | **Yes.** `[]SeriesRef` is a Go slice (contiguous array).                                                               |
| Tracked changed indices?  | **Partially.** `Delete()` knows which labels are affected. But `addFor()` handles insertions one at a time, not batch. |
| Batch update pattern?     | **Only at startup** (`EnsureOrder`). During operation, insertions are one-at-a-time with inline repair.                |
| Current sorting approach? | Startup: `slices.Sort` (pdqsort in Go). Runtime: insertion-repair (bubble the last element back).                      |
| Practical sizes?          | Postings lists: hundreds to millions of series IDs per label pair.                                                     |

**Confidence: 15/100**

While the data structure is correct (sorted slices), the access pattern doesn't match:

- **Normal operation:** IDs are added one at a time with O(1) amortized repair — already optimal for k=1
- **Startup:** `EnsureOrder()` is a one-time bulk sort from completely unordered data (k ≈ n), where DeltaSort has no advantage
- **Deletion:** Creates new filtered slices rather than re-sorting

The hypothetical integration point would be **batch compaction** where multiple series are deleted/added simultaneously, changing the sorted postings lists. But Prometheus doesn't currently batch these; it processes them sequentially with individual repairs.

---

## E. Varnish Cache Binary Heap for Expiry

**Domain:** CDN/HTTP reverse proxy — maintaining sorted expiration order for cache objects.

### Source Code Analysis

- **File examined:** `bin/varnishd/cache/cache_expire.c` (complete)
- **Repository:** [varnishcache/varnish-cache](https://github.com/varnishcache/varnish-cache)

### Key Findings

Varnish does **NOT** use a sorted array or sorted list for cache expiry. It uses a **binary heap** (`VBH` — Varnish Binary Heap):

```c
ep->heap = VBH_new(NULL, object_cmp, object_update);
```

Operations:

- **Insert:** `VBH_insert(ep->heap, oc)` — O(log n)
- **Delete:** `VBH_delete(ep->heap, oc->timer_idx)` — O(log n)
- **Update (TTL change):** `VBH_reorder(ep->heap, oc->timer_idx)` — O(log n)
- **Expire (poll root):** `VBH_root(ep->heap)` — O(1)

The comparator compares `timer_when` (expiration timestamps). Objects track their heap index via `oc->timer_idx` for O(1) lookup during reorder/delete.

Communication uses a **mailbox pattern**: the main thread posts objects to an inbox (`VSTAILQ`), and the expiry thread processes them one at a time.

### Assessment

| Criterion                 | Finding                                                    |
| ------------------------- | ---------------------------------------------------------- |
| Contiguous sorted array?  | **No.** Binary heap — partially ordered, not fully sorted. |
| Tracked changed indices?  | **Yes** (heap index tracked per object via `timer_idx`).   |
| Batch update pattern?     | **No.** Objects processed one at a time from inbox queue.  |
| Current sorting approach? | Binary heap with O(log n) insert/delete/reorder.           |
| Practical sizes?          | Thousands to millions of cached objects.                   |

**Confidence: 3/100**

Binary heap is the textbook-correct data structure for this problem: O(log n) per operation, O(1) to peek at the minimum. A contiguous sorted array would be worse: O(n) for insertion/deletion due to shifting. DeltaSort cannot improve on a binary heap for single-element expiration polling.

---

## F. Email Clients: Sorted Mailbox Views

**Domain:** Thunderbird/email clients maintaining sorted views of large mailboxes.

### Source Code Analysis

- Could not fetch Thunderbird's `threadPane.js` (404). Analysis based on Thunderbird architecture knowledge (XPCOM `nsIMsgDBView` and `nsMsgDBView.cpp`).

### Architecture Overview

Thunderbird uses a layered architecture:

1. **Backend (C++):** `nsMsgDBView` maintains a sorted view of messages. Messages are stored in `.msf` summary files (Mork database format, later moved to a custom format). The view holds an array of `nsMsgKey` values (message IDs) in display order.
2. **Sort mechanism:** `nsMsgDBView::Sort()` performs a full sort using a custom comparison function based on the active sort column (date, sender, subject, size, etc.). Uses `std::sort` on the `nsMsgKey` array.
3. **Threading:** Threaded view groups messages by conversation thread, then sorts threads. `nsMsgThreadedDBView` extends `nsMsgDBView`.

### Update Pattern

When new messages arrive:

- Messages are appended to the database
- `nsMsgDBView::OnNewHeader()` or `NoteChange()` is called
- The view is typically **rebuilt from scratch** or the new message is inserted at the correct position via binary search (`FindInsertPos`)
- Flag changes (read/unread, starred) don't change sort order unless sorting by flag column

### Assessment

| Criterion                 | Finding                                                                                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Contiguous sorted array?  | **Yes.** `nsTArray<nsMsgKey>` (contiguous array of message keys in sorted display order).                                                                     |
| Tracked changed indices?  | **Partially.** `NoteChange()` knows which message changed, but for sorting purposes it's a single-message update.                                             |
| Batch update pattern?     | **Partially.** Fetching new mail can deliver dozens of messages at once. `nsMsgDBView::OnNewHeader()` is called per message, but batch insertion is possible. |
| Current sorting approach? | `std::sort` for full re-sort. Binary search + insertion for single-message adds.                                                                              |
| Practical sizes?          | Average folder: hundreds of messages. Large folders: 10K–100K+ messages. Gmail imports can have 500K+.                                                        |

**Confidence: 40/100**

Reasonable model match:

- Contiguous sorted array of message keys
- Persistent sorted state between operations
- Messages arrive in batches (fetch cycle)
- Most operations are single-message (k=1), where binary insertion is already O(n)
- Full re-sorts are rare (user changes sort column, which is infrequent)
- The interesting case: bulk IMAP sync adds dozens of messages to a sorted mailbox

**Best-case scenario:** User has 100K messages sorted by date. IMAP sync downloads 50 new messages. Current approach: 50 x binary-search-insert = O(50 x 100K) = O(5M) shifts. DeltaSort approach: O(100K x sqrt(50)) ~ O(700K) — a 7x improvement. This is actually meaningful for large mailboxes.

**Integration challenge:** Thunderbird's `nsMsgDBView` is ancient C++ code (XPCOM-era). Adding DeltaSort would require tracking the set of newly-arrived message positions in the sorted array, which the existing `OnNewHeader()` callback doesn't provide. The messages arrive one at a time and are inserted incrementally, so you'd need to batch them first.

---

## G. TimSort Behavior Analysis on Nearly-Sorted Data ⭐⭐

**Domain:** Understanding DeltaSort's primary competitor for nearly-sorted arrays.

### Source Analysis

- **File examined:** Tim Peters' `listsort.txt` from CPython (complete, ~1000 lines of detailed analysis)
- **TimSort Wikipedia article** (algorithm overview)

### How TimSort Handles Nearly-Sorted Data

TimSort's core strategy is **run detection + intelligent merging**:

1. **Run detection:** `count_run()` scans left-to-right, finding ascending (non-decreasing) or descending (non-increasing) runs. Short runs are extended to `minrun` length via binary insertion sort.

2. **Merge strategy:** The "powersort" merge pattern (since Python 3.11) creates near-optimal binary merge trees. Runs are merged pairwise when their "power" (tree depth) indicates it's efficient.

3. **Galloping mode:** When one run consistently wins the merge comparison (elements come from the same side), it switches to exponential search (galloping) to find the endpoint efficiently.

### What Happens with k Perturbations on a Sorted Array

Consider a sorted array of n elements where k elements have their values changed (at known positions). After the changes, the array has certain "disorder".

**TimSort's behavior depends critically on how the k changed elements distribute:**

**Case 1: Randomly scattered k changes (changes at random positions)**

- The sorted runs between changed elements have average length n/k
- TimSort detects ~2k runs (original sorted segments between perturbations, plus the perturbations themselves)
- Each short run (single changed element) gets boosted to `minrun` length via binary insertion sort
- Merge phase: ~2k runs merged in O(n) total via galloping (since most runs are already correctly positioned relative to each other)
- **Total: O(n + k x minrun x log(minrun)) ~ O(n + k log k)** if minrun is O(log k), but actually minrun is fixed at ~32, so it's **O(n + 32k) ~ O(n)** for k << n

**Detailed analysis for k scattered perturbations:**

After k values change at random positions in a sorted array of n, the array looks like:

```
[sorted_run_1] [CHANGED] [sorted_run_2] [CHANGED] ... [sorted_run_k] [CHANGED] [sorted_run_k+1]
```

Each `CHANGED` element may or may not break the local sort order. If the new value is smaller than its left neighbor or larger than its right neighbor, it creates a "break" in the run.

In the worst case, each changed element creates a break, yielding ~k+1 runs of average length n/(k+1). These runs are long (>> minrun for moderate k), so no binary insertion sort extension is needed.

The merge phase must merge k+1 sorted runs. TimSort's powersort merge creates a balanced merge tree of depth O(log k). Each merge at each level processes O(n) total elements. But TimSort's **galloping optimization** is crucial here:

When merging two long sorted runs where most elements in run A are smaller than most elements in run B (which is true for adjacent segments of an originally-sorted array), galloping finds boundaries in O(log(n/k)) time per merge. Total across all merges:

**O(n + k log(n/k)) ~ O(n log(n/(n-k)))** ~ **O(n)** when k is small relative to n.

**In practice, Tim Peters' own benchmarks confirm this:**

From `listsort.txt` — the `3sort` test case (ascending data with 3 random exchanges):

```
n=1048576: TimSort comparisons = 1,048,958 (vs n-1 = 1,048,575 for already sorted)
```

That's only 383 extra comparisons for 3 random exchanges on 1M elements! This is essentially O(n).

The `%sort` test case (ascending, 1% of elements randomly replaced):

```
n=1048576: TimSort comparisons = 1,694,896 (vs lg(n!)=19,458,756 for random)
```

That's 1.6n comparisons for k = 10,486 changes — far better than O(n log n) = 20n.

### Complexity Comparison: TimSort vs DeltaSort

| Scenario                    | TimSort                       | DeltaSort                             | Winner                                      |
| --------------------------- | ----------------------------- | ------------------------------------- | ------------------------------------------- |
| k scattered changes, k << n | ~O(n) comparisons, O(n) space | O(n\*sqrt(k)) comparisons, O(1) space | **TimSort** in time, **DeltaSort** in space |
| k clustered changes         | O(n)                          | O(n\*sqrt(k))                         | **TimSort**                                 |
| k changes, values move far  | O(n + k log(n/k))             | O(n\*sqrt(k))                         | **Depends on k**                            |
| k = sqrt(n)                 | O(n)                          | O(n^(3/4))                            | **DeltaSort** (fewer ops)                   |
| k = n/10                    | O(n)                          | O(n \* sqrt(n/10)) ~ O(n^(3/2)/3)     | **TimSort**                                 |
| Any k, space-constrained    | O(n/2) extra space            | O(1) extra space                      | **DeltaSort**                               |

### Critical Insight

**TimSort is already near-optimal on nearly-sorted data in terms of comparisons.** Its run-detection + galloping strategy naturally exploits existing order, achieving ~O(n) comparisons for k << n perturbations.

**DeltaSort's advantages over TimSort narrow to:**

1. **O(1) auxiliary space vs O(n/2) auxiliary space**: TimSort needs up to n/2 temporary storage for merging. DeltaSort is in-place. This matters in **memory-constrained environments** (embedded, kernel, very large arrays).

2. **Knowledge of WHICH indices changed**: DeltaSort can skip Phase 2's scan for unchanged regions, while TimSort must linearly scan the entire array to detect runs. However, TimSort's linear scan is extremely cache-friendly and cheap — it's essentially a `memcmp` of consecutive elements.

3. **Specific k ranges**: For k ~ sqrt(n), DeltaSort achieves O(n^(3/4)) which beats TimSort's O(n). But this is a narrow sweet spot.

4. **Comparison-expensive scenarios**: If comparisons are very expensive (complex objects, network round-trips), DeltaSort's O(n\*sqrt(k)) comparisons vs TimSort's ~O(n) comparisons both dominate, but DeltaSort can potentially skip more comparisons by knowing the dirty set.

**The honest assessment:** TimSort (~Powersort) is DeltaSort's most formidable competitor. On nearly-sorted data, TimSort achieves near-optimal performance **without** needing to know which elements changed. DeltaSort's value proposition is primarily:

- **O(1) space** (vs O(n) for TimSort)
- **The ability to skip unchanged regions entirely** when the dirty set is very small
- **A cleaner theoretical bound** parameterized by k

For most application-layer scenarios, TimSort (already the default in Python, Java, JavaScript V8, Rust's stable sort) would handle nearly-sorted data well without any code changes.

**Confidence in this analysis: 90/100**

---

## H. htop Process List Sorting

**Domain:** System process monitor — sorting the process list by CPU%, memory, PID, etc. between refresh cycles.

### Source Code Analysis

- **Files examined:** `Table.c` (complete, ~400 lines), `Process.c` (complete, ~900 lines)
- **Repository:** [htop-dev/htop](https://github.com/htop-dev/htop)

### Key Findings

**Data structure:**

- `Table.rows`: `Vector` of `Row*` (pointers to `Process` structs) — the master list
- `Table.displayList`: `Vector` of `Row*` — the filtered/sorted view for rendering
- `Table.table`: `Hashtable` of id->Row for O(1) lookup
- `needsSort`: boolean flag indicating whether re-sort is needed

**Sorting mechanism:**

```c
void Table_updateDisplayList(Table* this) {
    if (settings->ss->treeView) {
        if (this->needsSort)
            Table_buildTree(this);   // Complex tree sort for tree view
    } else {
        if (this->needsSort)
            Vector_insertionSort(this->rows);  // <- INSERTION SORT!
        // Then copy to displayList
    }
    this->needsSort = false;
}
```

**htop uses INSERTION SORT** for the flat (non-tree) view! This is likely because:

1. The list is ~99% sorted between refreshes (only a few processes change CPU%)
2. Insertion sort is O(n) on nearly-sorted data
3. The list is small enough (typically <1000 processes) that insertion sort's simplicity wins

**Comparison function (`Process_compare`):**

```c
int Process_compare(const void* v1, const void* v2) {
    ProcessField key = ScreenSettings_getActiveSortKey(ss);
    int result = Process_compareByKey(p1, p2, key);
    if (!result) return SPACESHIP_NUMBER(Process_getPid(p1), Process_getPid(p2));
    return (direction == 1) ? result : -result;
}
```

**Update pattern:**

- `Table_prepareEntries()`: Marks all rows as `updated = false`
- Platform-specific scan (e.g., `/proc` on Linux) updates Process fields
- `Table_cleanupEntries()`: Removes dead processes, compacts vector
- `Table_rebuildPanel()` -> `Table_updateDisplayList()` -> insertion sort
- `needsSort = true` is set when processes are added/removed or sort key changes

### Assessment

| Criterion                 | Finding                                                                                             |
| ------------------------- | --------------------------------------------------------------------------------------------------- |
| Contiguous sorted array?  | **Yes.** `Vector` is a contiguous array of pointers.                                                |
| Tracked changed indices?  | **No.** htop doesn't track which processes changed sort-key values. It just re-does insertion sort. |
| Batch update pattern?     | **Yes.** All processes are refreshed in one scan cycle, then the list is re-sorted.                 |
| Current sorting approach? | **Insertion sort** — already exploits nearly-sorted order! O(n + d) where d = number of inversions. |
| Practical sizes?          | 100–5000 processes typically. Busy servers: up to ~30K.                                             |

**Confidence: 20/100**

**htop already exploits the nearly-sorted property with insertion sort.** For its use case (small array, few changes per refresh), insertion sort is essentially optimal:

- n ~ 500 processes
- Between refreshes, ~10 processes change their CPU% ranking significantly
- Insertion sort: O(n + d) ~ O(500 + 50) = O(550) — essentially free
- DeltaSort would need dirty tracking overhead that exceeds the sorting cost

**DeltaSort cannot improve on this.** The arrays are too small, insertion sort already achieves O(n) for nearly-sorted data, and the overhead of DeltaSort's Phase 1 (extract, sort updated, replace) and Phase 2 (binary search, rotations) would be worse than insertion sort's simple swap-backward loop for n < 5000.

**Interesting parallel:** htop's choice of insertion sort validates the same insight that motivates DeltaSort — nearly-sorted data should be handled differently than random data. But htop chose the simpler O(n+d) algorithm that doesn't need dirty tracking, which is the right call at this scale.

---

## Third Sweep Summary Table

| #     | Candidate                     |     Array?      |    Tracked?    |  Batch?  |   Sizes    | Confidence |
| ----- | ----------------------------- | :-------------: | :------------: | :------: | :--------: | :--------: |
| A     | PostgreSQL MatView            | No (heap pages) | Partial (diff) |   Yes    |   1K–1M    |    10%     |
| **B** | **ag-Grid Delta Sort** ⭐⭐⭐ |     **Yes**     |    **Yes**     | **Yes**  | **10K–1M** |  **95%**   |
| C     | Apache Spark                  | No (immutable)  |       No       |    No    |  millions  |     5%     |
| D     | Prometheus Postings           |  Yes (slices)   |    Partial     | No (k=1) |  100–10M   |    15%     |
| E     | Varnish Cache                 |    No (heap)    |   Yes (idx)    |    No    | thousands  |     3%     |
| F     | Email Clients                 |       Yes       |    Partial     | Partial  |  1K–500K   |    40%     |
| G     | **TimSort Analysis** ⭐⭐     |        —        |       —        |    —     |     —      |    n/a     |
| H     | htop Process List             |       Yes       |       No       |   Yes    |   100–5K   |    20%     |

---

## Cross-Sweep Conclusions (All Three Sweeps Combined)

### The Definitive Finding: ag-Grid

Across **25 candidates** in three sweeps, **ag-Grid is the only system that has independently implemented the exact same algorithm class.** This is powerful evidence that:

1. **The problem is real.** A commercially important open-source project (ag-Grid is used by thousands of enterprises) identified the exact same problem and built a solution.
2. **The model matches.** ag-Grid has all four preconditions: contiguous array, persistent sorted state, dirty tracking, batch re-sort.
3. **DeltaSort could be a drop-in improvement.** ag-Grid's `doDeltaSort` uses O(n) space; DeltaSort achieves O(1) space with complementary time complexity.

### Revised "Best Fit" Rankings (All 25 Systems)

| Rank | System                    | Confidence | Why                                                                       |
| ---- | ------------------------- | :--------: | ------------------------------------------------------------------------- |
| 1    | **ag-Grid** `doDeltaSort` |  **95%**   | Already implements the same concept. DeltaSort could improve space usage. |
| 2    | VLC/MPD Playlists         |    55%     | Correct model, feasible integration, but sort is rare.                    |
| 3    | Chromium Bookmarks        |    45%     | Correct model but arrays too small.                                       |
| 4    | Email Clients             |    40%     | Large mailboxes with batch message arrival.                               |
| 5    | SWIFT N-Body              |    35%     | HPC, contiguous arrays, but small per-cell.                               |

### TimSort: The Elephant in the Room

Section G's analysis reveals that **TimSort/Powersort already handles nearly-sorted data in ~O(n) time** without any knowledge of which elements changed. This means:

**DeltaSort's true value proposition is:**

1. **O(1) auxiliary space** — TimSort needs O(n/2). In memory-constrained environments (embedded, very large arrays, kernel), this matters.
2. **The knowledge of which indices changed** — allows skipping the run-detection scan entirely, which is valuable when k is very small relative to n.
3. **A clean theoretical characterization** — O(n\*sqrt(k)) parameterized by the number of changes, useful for formal analysis and worst-case guarantees.

**DeltaSort's value proposition is NOT:**

- "We sort nearly-sorted data faster than existing algorithms" — TimSort already does this well.
- "We achieve better time complexity" — TimSort's galloping achieves ~O(n) for k << n, competitive with or better than O(n\*sqrt(k)).

**The honest framing for the paper's practical significance:**

> DeltaSort fills a niche for **in-place incremental re-sorting with known dirty indices**. When the application already tracks which elements changed (as ag-Grid demonstrates is practical), and memory is constrained (ruling out TimSort's O(n) temporary buffer), DeltaSort provides an optimal solution with clean theoretical bounds. The ag-Grid case study proves this problem space is commercially relevant.

### Final Assessment Matrix

| Property            |       DeltaSort        |        TimSort        |      ag-Grid Delta      |     Insertion Sort      |
| ------------------- | :--------------------: | :-------------------: | :---------------------: | :---------------------: |
| Time (k << n)       |     O(n\*sqrt(k))      |         ~O(n)         |     O(k log k + n)      |    O(n + inversions)    |
| Space               |        **O(1)**        |        O(n/2)         |          O(n)           |        **O(1)**         |
| Needs dirty set?    |          Yes           |          No           |           Yes           |           No            |
| Stable?             |           No           |        **Yes**        |         **Yes**         |         **Yes**         |
| Handles add/remove? |           No           |        **Yes**        |         **Yes**         |         **Yes**         |
| Best for...         | In-place + known dirty | General nearly-sorted | App-level with tracking | Small arrays, few swaps |

---

## Sweep 4: Data Grid & Dashboard Ecosystem Survey

> **Motivation**: ag-Grid's `doDeltaSort` is the only known production implementation of the ESM (Extract-Sort-Merge) pattern. This sweep surveys the broader ecosystem of JavaScript data grids, spreadsheet UIs, and dashboard tools to determine whether the pattern is emerging elsewhere or remains unique to ag-Grid.

---

### ag-Grid `doDeltaSort` — Deep Dive (Reference Implementation)

**Source**: `packages/ag-grid-community/src/clientSideRowModel/deltaSort.ts`

ag-Grid's implementation is a textbook ESM algorithm. The source code documents it clearly:

```
Algorithm outline:
1. Handle edge cases: empty input or single element — return early
2. Fall back to full sort if no previous sorted result or too few rows
3. Classify rows as "touched" (updated, added, or in changed path) vs "untouched"
4. If no rows are touched, return previous sorted array (filtering removed nodes if needed)
5. Sort only the touched rows using a stable sort with original index as tie-breaker
6. If all rows are touched, return the sorted touched rows directly
7. Merge the sorted touched rows with untouched rows from previous sort order
   using a two-pointer merge algorithm (similar to merge sort's merge step)

Time complexity: O(t log t + n) where t = touched rows, n = total rows
```

Key implementation details:

- **Minimum threshold**: `MIN_DELTA_SORT_ROWS = 4` — below this, full sort is used (lower overhead)
- **Touched classification**: Uses a `Map<RowNode, number>` with sign encoding: bitwise NOT (`~i`) for touched indices, non-negative for untouched
- **Merge**: A two-pointer merge between sorted touched array and untouched-nodes-in-old-order, with comparator tie-breaking on original index
- **Space**: O(n) — allocates `indexByNode` map, `touchedRows` array, and `result` array
- **Activation**: Gated behind `deltaSort` grid option (opt-in) and requires transaction-based updates via `applyTransaction()`

The `sortStage.ts` orchestrator calls `doDeltaSort` only when:

1. Sort options are active (`sortOptions.length > 0`)
2. Changed row nodes are tracked (`changedRowNodes` exists — from transactions)
3. The `deltaSort` grid option is enabled
4. Otherwise falls back to `rowNodeSorter.doFullSortInPlace()` (standard full sort)

**ag-Grid's documentation explicitly states** the performance trade-offs:

- Delta sort is beneficial when data has many rows per group and comparatively small transactions
- May be slower when the dataset is small relative to transaction size
- May be slower with many groups containing few rows each

---

### S4-A. TanStack Table (formerly React Table)

**Source examined**: `packages/table-core/src/features/RowSorting.ts`

#### Sorting Approach

TanStack Table is a **headless** UI table library — it provides sorting state management and delegates the actual sort to `getSortedRowModel()`. The sorting implementation:

1. **State**: Sorting state is a simple `SortingState = ColumnSort[]` array of `{ id, desc }` pairs
2. **Sort execution**: `getSortedRowModel()` returns a lazily-computed `RowModel`. When sorting state changes, the entire row model is recomputed
3. **Comparators**: Auto-detected per column type (`basic`, `text`, `datetime`, `alphanumeric`) or user-provided
4. **No caching**: No memoization of previous sort results. Every `getSortedRowModel()` call that detects a state change recomputes from scratch

#### Incremental Sort?

**No.** There is no concept of "changed rows" or "dirty tracking" in TanStack Table. The `RowSorting` feature has no awareness of which rows were modified between renders. When data changes (via React state), the entire sorted row model is recomputed. There is no `deltaSort` equivalent, no transaction API, and no changed-path tracking.

The library's philosophy is **declarative and stateless**: provide data + sort config → get sorted output. There is no persistent sorted state to incrementally maintain.

#### Would DeltaSort/ESM Help?

**Yes, significantly.** TanStack Table is used with large datasets (thousands of rows) in interactive applications where users edit cells and expect the sort order to update. Currently, every cell edit triggers a full O(n log n) re-sort. An ESM-style optimization could reduce this to O(t log t + n) ≈ O(n) for single-cell edits. However, TanStack's headless/declarative architecture would need transaction-like change tracking — a significant architectural addition.

---

### S4-B. Handsontable

**Source examined**: `handsontable/src/plugins/columnSorting/columnSorting.js`, `sortFunction/default.js`

#### Sorting Approach

Handsontable's column sorting plugin performs a **full re-sort** on every sort operation:

1. `sortByPresetSortStates(sortConfigs)` gathers all row data for sorted columns
2. Calls `sort()` from `sortService.js` which runs a comparator-based sort on the full index array
3. The sort operates on physical-to-visual index mappings, not on the data directly
4. Uses `indexesSequenceCache` to store the original unsorted order for reset

The comparator (`compareFunctionFactory` in `default.js`) is a standard pairwise comparison function with locale-aware string comparison, empty-cell handling, and type coercion. Returns `FIRST_BEFORE_SECOND`, `FIRST_AFTER_SECOND`, or `DO_NOT_SWAP`.

#### Incremental Sort?

**No.** Every call to `sort()` or `sortByPresetSortStates()`:

1. Resets to the cached original index sequence
2. Gathers ALL row data for sorted columns
3. Performs a full sort on the entire dataset
4. Updates the row index mapper

There is no tracking of which cells were edited between sort operations. No delta sort, no changed-row tracking, no incremental merge.

#### Would DeltaSort/ESM Help?

**Yes.** Handsontable is a spreadsheet-like widget where users actively edit cells. After each edit, if auto-sort is enabled, the entire dataset is re-sorted. For large spreadsheets (10K+ rows) with occasional cell edits, an ESM approach would be a major performance improvement. Handsontable already has row change hooks (`afterChange`) that could be leveraged for dirty tracking.

---

### S4-C. SlickGrid

**Source examined**: `src/slick.dataview.ts`

#### Sorting Approach

SlickGrid's `DataView` has a standard `sort(comparer, ascending)` method that calls `this.items.sort(comparer)` — a full JavaScript `Array.sort()` on the entire items array. Additionally:

- `fastSort(field)` — a deprecated IE-era optimization using `Object.prototype.toString` override
- `reSort()` — re-applies the last sort (full sort again)

#### Incremental Sort? — **Partial!**

SlickGrid has two noteworthy methods:

1. **`sortedAddItem(item)`**: Uses `sortedIndex(item)` (binary search) to find the correct insertion position, then calls `insertItem()`. This is an O(n) operation (binary search is O(log n), but array splice is O(n)). **This is incremental for additions.**

2. **`sortedUpdateItem(id, item)`**: Checks if the sort comparator gives the same result for old and new values. If so, does a regular in-place update. If NOT, **deletes the old item and does a sorted-add** (binary search insertion). This is O(n) per update.

However, **these are single-item operations**, not batch operations. There is no batch delta sort. For batch updates:

- `updateItems(ids, newItems)` calls `updateSingleItem()` in a loop, then `refresh()`, which calls `recalc()`, which does NOT re-sort — it only recalculates filtered/grouped views
- Sorting is only triggered by explicit `sort()` or `reSort()` calls, which do full sorts

SlickGrid also has a sophisticated **refresh hints** system (`setRefreshHints()`) for optimizing filter recalculation (narrowing vs. expanding filters, with caching), but this optimization does NOT extend to sorting.

#### Would DeltaSort/ESM Help?

**Moderate.** SlickGrid already handles single-item updates efficiently via binary insertion. The gap is in **batch updates** — if multiple rows change, there's no way to sort only the changed rows. An ESM approach would help when many rows are updated simultaneously (e.g., real-time data feeds updating prices). SlickGrid's `updated` map (tracking which IDs were changed) could serve as the dirty set.

---

### S4-D. DataTables.js

**Source examined**: `js/core/core.sort.js`

#### Sorting Approach

DataTables uses a classic full-sort approach:

1. `_fnSort()` computes sort keys for all rows via `_fnSortData()` (pre-computing formatted values into `_aSortData` cache)
2. `displayMaster.sort()` — standard `Array.sort()` with a multi-column comparator
3. Stable sort via tie-breaking on original row indices (`aiOrig`)
4. After sort, `_fnSortDisplay()` re-orders the display array to match master order

The sort data is **cached per cell** (`row._aSortData[colIdx]`) with pre-formatting via type-specific formatters (`extSort[type + "-pre"]`). This avoids repeated value extraction but doesn't help with incremental sorting.

#### Incremental Sort?

**No.** Every user sort interaction triggers `_fnSort()` which sorts the entire `displayMaster` array. DataTables' architecture is event-driven: click header → `_fnSortAdd()` → `_fnSort()` → `_fnSortDisplay()` → `_fnReDraw()`. There is no concept of changed rows or delta sorting.

DataTables does have a `rows().invalidate()` API for marking rows as needing data refresh, but this only triggers re-reading of DOM/data and doesn't feed into any incremental sort mechanism.

#### Would DeltaSort/ESM Help?

**Moderate.** DataTables is heavily used with server-side data where sorting is often delegated to the server. For client-side mode with large datasets and real-time row updates (via `row().data()` API), an ESM approach would help. However, DataTables' primary market is document-ready tables, not real-time data grids — the use case for incremental re-sorting is less common than in ag-Grid or SlickGrid.

---

### S4-E. LibreOffice Calc

**Architecture**: LibreOffice Calc stores data in `ScColumn` objects (one per column) within `ScTable`. Sorting is handled by `ScTable::Sort()` in `table3.cxx`.

#### Sorting Approach

When a user sorts a range:

1. `ScTable::Sort()` creates a `ScSortInfoArray` — an array of sort keys extracted from the specified columns
2. Performs a full sort using `std::sort` with a multi-key comparator
3. Applies the resulting permutation to all columns in the range (physical row reordering)

For **auto-recalculation after cell edits**: LibreOffice does NOT automatically re-sort after cell edits. The user must explicitly re-apply the sort (Data → Sort). There is no persistent "this range is sorted" metadata that triggers incremental re-sorting.

#### Incremental Sort?

**No.** Sorting is always a user-initiated, full-range operation. There is no concept of "dirty cells since last sort" or incremental re-sorting. Each sort is a one-shot operation on the full range.

#### Would DeltaSort/ESM Help?

**Low value.** Since sorting is user-initiated (not automatic after each edit), the performance of the sort itself is rarely the bottleneck. Users typically edit many cells, then sort once. The sort takes milliseconds for typical spreadsheet sizes (< 1M rows). For very large spreadsheets with auto-sort add-ons, there could be value, but this is a niche use case.

---

### S4-F. Google Sheets

**Architecture**: Cloud-based, closed source. Sorting happens server-side.

#### Sorting Approach

Google Sheets supports "Sort Range" (one-shot) and "Create a filter" with sort (persistent sort view). When using filter views with sort, editing a cell does NOT automatically re-sort the view — the user must re-apply the filter/sort.

#### Incremental Sort?

**Unknown (closed source).** Given that Google Sheets doesn't auto-re-sort after edits even in filter views, incremental sorting is likely not implemented. The server-side sort is presumably a standard full sort on the range.

---

### S4-G. Apache Superset / Metabase (Analytics Dashboards)

#### Sorting Approach

Both Superset and Metabase are **analytics dashboard tools** that display query results. Sorting is handled at two levels:

1. **Server-side**: SQL `ORDER BY` — the database sorts the data
2. **Client-side**: For already-fetched result sets, JavaScript `Array.sort()` on the table component

Neither tool has a concept of "incrementally updating" a sorted result. Dashboards fetch new query results periodically (or on user action) and re-sort the entire result set.

#### Incremental Sort?

**No.** The data pipeline is: query DB → receive results → sort client-side → render. There is no persistent sorted state that gets incrementally updated. Each data refresh is a full replace.

#### Would DeltaSort/ESM Help?

**Very low.** The bottleneck in analytics dashboards is the database query and network transfer, not client-side sorting. Result sets are typically small enough (< 10K rows displayed) that full sorts are instantaneous.

---

### S4-H. VS Code (File Explorer, Extension List)

#### Sorting Approach

VS Code's file explorer sorts file/folder lists using `Array.sort()` with comparators for name, type, modified date, etc. The tree view is re-sorted when:

- Files are created/deleted/renamed (filesystem watcher events)
- User changes sort order

#### Incremental Sort?

**No.** When a file system change is detected, VS Code refreshes the affected tree node and re-sorts its children completely. There is no delta sort for file lists. Given typical directory sizes (tens to hundreds of files), full sort is effectively instantaneous.

#### Would DeltaSort/ESM Help?

**No practical benefit.** Directory listings are too small (typically < 1K entries) for incremental sorting to matter. The overhead of maintaining dirty tracking would exceed the cost of just re-sorting.

---

### S4-I. Notion (Productivity Tool)

#### Sorting Approach

Notion is **closed source**. Database views support sorting by properties. From observable behavior:

- Sort is applied server-side and results are streamed to the client
- Editing a property in a sorted view triggers a re-fetch/re-render (the row animates to its new position)
- The animation suggests the client may do some local sort computation, but the architecture is opaque

#### Incremental Sort?

**Unknown (closed source).** The smooth re-positioning animation when editing a sorted field could indicate local incremental sort logic, or simply CSS transition on the new server-provided order.

---

### Sweep 4 Summary: Data Grid Ecosystem

| System            |            Incremental Sort?            | Sorting Approach                               |        Would Benefit from ESM?         |
| ----------------- | :-------------------------------------: | ---------------------------------------------- | :------------------------------------: |
| **ag-Grid**       |           **YES — full ESM**            | Extract-Sort-Merge, O(t log t + n), O(n) space |             Already has it             |
| TanStack Table    |                   No                    | Full re-sort via `getSortedRowModel()`         |  **High** — large interactive tables   |
| Handsontable      |                   No                    | Full re-sort via `sortByPresetSortStates()`    | **High** — spreadsheet with cell edits |
| SlickGrid         | **Partial** (single-item binary insert) | Full sort + per-item `sortedUpdateItem()`      |   **Moderate** — batch updates only    |
| DataTables.js     |                   No                    | Full `displayMaster.sort()`                    |    Moderate — less common use case     |
| LibreOffice Calc  |                   No                    | Full range sort via `ScTable::Sort()`          |  Low — user-initiated, not automatic   |
| Google Sheets     |            Unknown (closed)             | Server-side, no auto-re-sort                   |                  Low                   |
| Superset/Metabase |                   No                    | SQL ORDER BY + client `Array.sort()`           |     Very low — query is bottleneck     |
| VS Code           |                   No                    | Full `Array.sort()` on file lists              |         No — arrays too small          |
| Notion            |            Unknown (closed)             | Server-side, opaque client logic               |                Unknown                 |

### Key Finding: ag-Grid Is Alone

**ag-Grid is the ONLY data grid in the JavaScript ecosystem that has implemented incremental/delta sorting.** Every other data grid — including major players like TanStack Table, Handsontable, SlickGrid, and DataTables — performs a full O(n log n) sort on every sort operation, with no awareness of which rows changed.

This is remarkable because:

1. **The problem is universal**: All these grids handle the same use case — users editing cells in a sorted table and expecting the order to update efficiently.

2. **The infrastructure exists**: Most grids already track row changes for rendering purposes (e.g., SlickGrid's `updated` map, Handsontable's `afterChange` hook, DataTables' `rows().invalidate()`). They simply don't feed this information into the sort pipeline.

3. **ag-Grid proved it's worth doing**: ag-Grid's `deltaSort` option is specifically documented as a performance feature for "large amounts of rows in each row group and comparatively small transactions" — exactly the ESM sweet spot.

4. **SlickGrid came closest**: Its `sortedUpdateItem()` and `sortedAddItem()` methods show awareness of the problem (don't re-sort everything for single-item updates), but the approach is per-item (O(n) per update due to array splice) rather than batched. It's the only other grid with ANY incremental sort logic.

### The Emerging Pattern

The pattern of **"extract changed, sort changed, merge with unchanged"** is NOT emerging elsewhere. ag-Grid's implementation appears to be a unique, in-house optimization. The likely reasons other grids haven't adopted it:

1. **JavaScript's `Array.sort()` is fast**: V8's TimSort handles nearly-sorted data well, reducing the perceived need for optimization
2. **Complexity cost**: ESM requires transaction tracking infrastructure that most grids don't have
3. **"Good enough" performance**: For typical grid sizes (< 10K rows), full sorts complete in < 5ms, below the perceptual threshold
4. **Server-side sorting**: Many grid users sort server-side, making client-side sort optimization moot

### Implications for DeltaSort Paper

This survey strengthens the paper's position:

- ag-Grid's lone implementation confirms the **problem is real but under-addressed**
- The gap in TanStack Table and Handsontable (both high-traffic open-source projects) suggests an **opportunity for algorithmic contribution**
- ag-Grid's O(n) space usage (vs. DeltaSort's O(1) space) shows there is room for improvement even in the one existing implementation
- The `MIN_DELTA_SORT_ROWS = 4` threshold in ag-Grid's code validates DeltaSort's own crossover analysis — both algorithms recognize that below a certain change count, full sort wins

---

---

# Fifth Sweep: In-Memory Computing Systems That Maintain Sorted Data

> **Motivation**: The first four sweeps focused on application-level systems, UI grids, and database query operators. This sweep systematically examines **in-memory data structures and computing frameworks** that maintain sorted data as a core primitive — sorted containers, in-memory caches/stores, query engines, and streaming aggregation systems. The goal is to find systems where sorted arrays are maintained across mutations, where DeltaSort's $O(n\sqrt{k})$ in-place re-sort or ESM's $O(n + k \log k)$ extract-sort-merge could replace existing approaches.

---

## S5-1. Python `sortedcontainers` — `SortedList`

**Source examined**: [`grantjenks/python-sortedcontainers`](https://github.com/grantjenks/python-sortedcontainers) — `sortedcontainers/sortedlist.py`

### Data Structure

`SortedList` uses a **list of sorted sublists** — a B-tree-like structure constrained to exactly two levels. Three internal arrays maintain the structure:

- `_lists`: list of sublists, each internally sorted
- `_maxes`: maximum value in each sublist (for bisect-based sublist selection)
- `_index`: positional index tree (for O(log n) indexing by position)

The `DEFAULT_LOAD_FACTOR = 1000` controls sublist capacity. Sublists are maintained between `load/2` and `load*2` in size. When a sublist exceeds `2 * load`, it is split (`_expand()`). When adjacent sublists are both below `load/2`, they are merged.

### Sorting Mechanism

**Single-element `add()` (line ~192):**

1. Bisect `_maxes` to find the correct sublist — O(log(n/load))
2. `insort` within the sublist — O(load) via binary search + list insert
3. `_expand()` if sublist exceeds 2 \* load

**Batch `update()` (line ~261) — KEY FINDING:**

```python
def update(self, iterable):
    values = sorted(iterable)
    if len(values) * 4 >= self._len:
        # Concatenate ALL sublists, append new values, FULL SORT
        self._lists.append(values)
        values = reduce(iadd, self._lists, [])
        values.sort()  # ← full sort of everything
        self._clear()
        # Re-chunk into sublists
    else:
        # Insert elements one at a time via add()
        _add = self.add
        for val in values:
            _add(val)
```

**The threshold**: When the batch size ≥ 25% of the current list size (`len(values) * 4 >= self._len`), `update()` concatenates everything and does a **full sort** via Python's Timsort, then re-chunks into sublists.

### Assessment

| Criterion                | Finding                                                                                              |
| ------------------------ | ---------------------------------------------------------------------------------------------------- |
| Contiguous sorted array? | **Partially.** Each sublist is a Python list (contiguous). The full `SortedList` is a list of lists. |
| Tracked changed indices? | **No.** `update()` receives new values, not positions of changed values.                             |
| Batch update pattern?    | **Yes.** `update()` is designed for bulk insertion.                                                  |
| Current sorting approach | Full Timsort (`values.sort()`) when batch ≥ 25%; individual `insort` otherwise.                      |
| Practical sizes          | Hundreds to millions of elements.                                                                    |

**Confidence: 30/100**

**DeltaSort opportunity**: The `update()` path that triggers a full sort when batch ≥ 25% is the most concrete opportunity. After the concatenation, the combined list is _mostly sorted_ (the original elements are in order; only the new values are out of place). Timsort will exploit this run structure efficiently (~O(n) with galloping), so DeltaSort's $O(n\sqrt{k})$ would need to beat Timsort's adaptive behavior. Given that Timsort achieves ~O(n) on this input structure (one long sorted run + scattered insertions), **DeltaSort would likely not improve time complexity here** but could win on **space** — Timsort uses O(n/2) auxiliary memory for merging, while DeltaSort uses O(1).

**The deeper mismatch**: `SortedList` doesn't track "which indices changed" — it handles insertions (new elements), not value mutations at known positions. DeltaSort's model assumes values change at known array positions, which isn't the `SortedList` API pattern.

---

## S5-2. Dragonfly — Redis-Compatible In-Memory Store

**Source examined**: [`dragonflydb/dragonfly`](https://github.com/dragonflydb/dragonfly) — `src/server/zset_family.cc`, `src/server/detail/sorted_map.h`

### Data Structure

Dragonfly implements Redis-compatible sorted sets (ZSET) using the **same dual-encoding** as Redis:

1. **Small sets**: `OBJ_ENCODING_LISTPACK` — a compact, sequentially-encoded byte array with (element, score) pairs. Elements are stored in score-sorted order within the listpack.

2. **Large sets**: `OBJ_ENCODING_SKIPLIST` — a skip list (`SortedMap`) + hash table for O(log n) operations.

**Transition trigger**: When `zl_len >= ZSET_MAX_LISTPACK_ENTRIES` (default: 128) or any element exceeds `ZSET_MAX_LISTPACK_VALUE` bytes (default: 64), the encoding converts from listpack to skiplist via `SortedMap::FromListPack()`.

### Sorting Mechanism

**Listpack insertion (`ZzlInsert()`):**
Iterates the listpack sequentially to find the correct score-ordered position, then inserts. This is O(n) per insertion due to the linear scan + byte shifting.

**Batch ZADD optimization:**
When adding multiple members (`num_members > 2`) to a listpack encoding, Dragonfly pre-sorts members by score before inserting, reducing the number of listpack traversals.

**Skiplist insertion:**
Standard skip list insert — O(log n) per element. Score updates via `Update()` remove and reinsert.

### Assessment

| Criterion                | Finding                                                                                                       |
| ------------------------ | ------------------------------------------------------------------------------------------------------------- |
| Contiguous sorted array? | **Yes (listpack only).** Listpack is a contiguous byte buffer with sorted entries. Skiplist is pointer-based. |
| Tracked changed indices? | **No.** Each ZADD is processed individually.                                                                  |
| Batch update pattern?    | **Yes.** Multi-member ZADD pre-sorts before insertion.                                                        |
| Current sorting approach | Listpack: linear scan insertion O(n). Skiplist: O(log n) insertion.                                           |
| Practical sizes          | Listpack: ≤128 elements. Skiplist: unlimited (millions).                                                      |

**Confidence: 15/100**

**DeltaSort opportunity**: Limited. The listpack encoding is the only contiguous sorted structure, and it's capped at 128 elements — far too small for DeltaSort's asymptotic advantage to materialize. Past 128 elements, the skiplist takes over and maintains sorted order continuously via O(log n) operations. The batch ZADD optimization (pre-sorting members) is already an effective approach for the listpack encoding.

**The small→large transition** (listpack → skiplist at 128 elements) is interesting architecturally but doesn't create an opportunity for DeltaSort — the listpack is too small, and the skiplist doesn't need array-style re-sorting.

---

## S5-3. Microsoft Garnet — .NET In-Memory Store

**Source examined**: [`microsoft/garnet`](https://github.com/microsoft/garnet) — `libs/server/Objects/SortedSet/SortedSetObject.cs`

### Data Structure

Garnet's `SortedSetObject` uses C#'s standard library collections:

```csharp
private readonly SortedSet<(double Score, byte[] Element)> sortedSet;
private readonly Dictionary<byte[], double> sortedSetDict;
```

- `SortedSet<T>`: .NET's **red-black tree** implementation — a self-balancing BST, NOT a contiguous array.
- `Dictionary<byte[], double>`: hash table for O(1) score lookup by element.

**Unlike Redis/Dragonfly**: Garnet does NOT use a listpack encoding for small sets. There is no compact-array → tree transition. Every sorted set, regardless of size, uses the red-black tree + dictionary dual structure from creation.

### Sorting Mechanism

`SortedSet<T>` maintains order via red-black tree rebalancing on every insert/remove — O(log n) per operation. The comparator (`SortedSetComparer.Instance`) orders by score first, then by element bytes.

**Add operation:**

```csharp
public void Add(byte[] item, double score)
{
    sortedSetDict.Add(item, score);
    sortedSet.Add((score, item));
    this.UpdateSize(item);
}
```

Single-element insertion into the tree. No batch add optimization exists.

### Assessment

| Criterion                | Finding                                                    |
| ------------------------ | ---------------------------------------------------------- |
| Contiguous sorted array? | **No.** Red-black tree (pointer-based).                    |
| Tracked changed indices? | **No.** Each operation modifies the tree individually.     |
| Batch update pattern?    | **No batch optimize.** Multi-ZADD calls `Add()` in a loop. |
| Current sorting approach | Red-black tree O(log n) insert/delete. Always sorted.      |
| Practical sizes          | Hundreds to millions per sorted set.                       |

**Confidence: 3/100**

**DeltaSort opportunity**: None. Garnet chose a fundamentally different architecture from Redis/Dragonfly — no contiguous encoding at any size. The red-black tree maintains sorted order continuously with O(log n) per operation. To use DeltaSort, Garnet would need to replace the red-black tree with a flat sorted array, sacrificing O(log n) single-element operations for O(n) shifts — a regression for Garnet's primary Redis-compatible workload.

---

## S5-4. Apache DataFusion — Query Engine Sorted Merge & TDigest

**Source examined**: [`apache/datafusion`](https://github.com/apache/datafusion) — `sorts/merge.rs`, `sorts/sort.rs`, `functions-aggregate-common/src/tdigest.rs`

### Data Structures

**A. SortPreservingMergeExec (K-way merge):**

`SortPreservingMergeStream` performs K-way merge of pre-sorted input partitions using a **loser tree** stored in a vector. Each partition provides sorted batches; the loser tree selects the minimum across partitions in O(log K) per element. This is streaming — no array is maintained or re-sorted.

**B. SortExec (Full sort):**

`SortExec` accumulates all input records, sorts them using the Arrow compute `sort` kernel (typically a merge sort or radix sort variant), then emits sorted output. One-shot operation per query.

**C. TDigest (Streaming percentiles, lines 192–265):**

`merge_unsorted_f64()` collects unsorted float values, sorts them with `sort_f64()`, then calls `merge_sorted_f64()` to update the TDigest. The TDigest maintains an array of centroids sorted by mean value:

```rust
pub struct TDigest {
    centroids: Vec<Centroid>,  // sorted by mean
    max: f64,
    min: f64,
    count: usize,
    // ...
}
```

Centroids are merged when the digest exceeds a compression threshold.

### Assessment

| Criterion                | Finding                                                                                          |
| ------------------------ | ------------------------------------------------------------------------------------------------ |
| Contiguous sorted array? | **TDigest centroids: Yes.** `Vec<Centroid>` sorted by mean. SortPreservingMerge: No (streaming). |
| Tracked changed indices? | **No.** New values are bulk-merged, not tracked by position.                                     |
| Batch update pattern?    | **TDigest: Yes.** `merge_unsorted_f64()` batches values then sorts.                              |
| Current sorting approach | TDigest: full sort of batch → merge into digest. Query: one-shot sort or K-way streaming merge.  |
| Practical sizes          | TDigest: ~100–300 centroids (compression-limited). Query sort: millions.                         |

**Confidence: 8/100**

**DeltaSort opportunity**: Minimal. The TDigest centroid array is small (~100–300 centroids at typical compression factors) — too small for DeltaSort's advantage over Timsort. The K-way merge in `SortPreservingMergeExec` is a streaming operation on immutable sorted runs, not an array re-sort. `SortExec` is a one-shot sort with no persistent sorted state.

---

## S5-5. DuckDB — `TemplatedMergePartition` (Re-Sort Instead of Merge)

**Source examined**: [`duckdb/duckdb`](https://github.com/duckdb/duckdb) — `src/common/sorting/sorted_run_merger.cpp`, `src/storage/table/update_segment.cpp`

### Key Finding: Explicit Re-Sort Over K-Way Merge

**`TemplatedMergePartition`** (sorted_run_merger.cpp, lines 656–674) contains an extraordinary comment:

> _"Seems counter-intuitive to re-sort instead of merging, but modern sorting algorithms detect and merge..."_

DuckDB's merge partition code uses **`pdqsort_branchless`** (pattern-defeating quicksort) to re-sort the concatenated data from sorted runs rather than performing a traditional K-way merge. The rationale: pdqsort is adaptive — it detects existing runs in the data and merges them efficiently, achieving near-O(n) on nearly-sorted input while avoiding the overhead of maintaining a merge data structure.

**Update segments** (`update_segment.cpp`): DuckDB maintains sorted update chains for MVCC. `MergeLoop` and `MergeUpdateLoopInternal` perform sorted merges of update chains — these are small sorted arrays (typically ≤ STANDARD_VECTOR_SIZE = 2048 elements) merged when transactions commit.

**Merge-sort tree** (`merge_sort_tree.hpp`): A generic data structure with fanout F=32, cascading C=32, used for window function evaluation. Maintains sorted data in a tree structure.

### Assessment

| Criterion                | Finding                                                                                            |
| ------------------------ | -------------------------------------------------------------------------------------------------- |
| Contiguous sorted array? | **Yes** (post-concatenation). Sorted runs are concatenated into a flat buffer, then pdqsorted.     |
| Tracked changed indices? | **No.** The "changes" are additional sorted runs to incorporate, not mutations at known positions. |
| Batch update pattern?    | **Yes.** Multiple sorted runs are merged at once.                                                  |
| Current sorting approach | `pdqsort_branchless` on concatenated nearly-sorted data (instead of K-way merge).                  |
| Practical sizes          | Partitions: thousands to millions of rows. Update segments: ≤ 2048.                                |

**Confidence: 20/100**

**DeltaSort opportunity**: The `TemplatedMergePartition` code is architecturally interesting — DuckDB already chose re-sort over K-way merge because adaptive sorts handle nearly-sorted data well. DeltaSort could potentially replace `pdqsort_branchless` here _if_ the positions where new data was inserted were tracked. However:

1. pdqsort is already highly adaptive on nearly-sorted data (~O(n) with detected runs)
2. The concatenated data has a specific structure (interleaved sorted runs), not "k random changes"
3. DuckDB doesn't track which positions correspond to which original run

The update segment merges are too small (≤ 2048 elements) for DeltaSort's advantage.

---

## S5-6. ClickHouse MergeTree — In-Memory Sort During Merge

**Source examined**: [`ClickHouse/ClickHouse`](https://github.com/ClickHouse/ClickHouse) — `MergeTreeDataWriter`, `MergeSortingTransform`, `SortedBlocksWriter`, architecture docs

### Architecture

ClickHouse MergeTree is explicitly **not an LSM tree** (from `docs/architecture.md`): _"MergeTree is not an LSM tree because it does not contain MEMTABLE and LOG: inserted data is written directly to the filesystem."_

Each INSERT creates a new **immutable sorted part** (sorted by primary key). Background threads select and merge parts using k-way merge algorithms (`IMergingAlgorithm` with strategies: `ReplacingSorted`, `AggregatingSorted`, `SummingSorted`, etc.).

### In-Memory Re-Sort: `MergeSortingTransform::remerge()`

When `MergeSortingTransform` accumulates blocks in memory and hits the memory limit, it calls `remerge()` to reduce memory usage. This operation:

1. Collects all accumulated sorted blocks
2. Merges them via `MergingSortedTransform` (K-way merge using a loser tree)
3. Splits the merged result back into blocks

The `remerge()` operates on data that is already partially sorted (each block is sorted, but blocks may overlap). This is a K-way merge, not an array re-sort.

### `SortedBlocksWriter`

Maintains sorted blocks and merges them using `MergingSortedTransform`. The lab simulation (`MergeTree.js`) has a TODO: _"optimize it by maintaining this array dynamically with merges and inserts"_ — suggesting the developers recognize the inefficiency of rebuilding sorted state.

### `MergedBlockOutputStream::writeWithPermutation()`

> _"If the data is not sorted, but we have previously calculated the permutation... This method is used to save RAM."_

This writes data using a pre-computed permutation rather than physically sorting — an optimization that avoids creating a sorted copy when the permutation is already known.

### Assessment

| Criterion                | Finding                                                                 |
| ------------------------ | ----------------------------------------------------------------------- |
| Contiguous sorted array? | **No.** Immutable file-based parts. In-memory: blocks, not flat arrays. |
| Tracked changed indices? | **No.** Parts are immutable; merges combine whole parts.                |
| Batch update pattern?    | **No.** K-way merge of immutable sorted parts.                          |
| Current sorting approach | K-way merge via loser tree / `MergingSortedTransform`.                  |
| Practical sizes          | Parts: millions to billions of rows, but merge is streaming.            |

**Confidence: 5/100**

**DeltaSort opportunity**: Very limited. ClickHouse operates on immutable sorted parts merged via streaming K-way merge. The only in-memory sorting happens in `MergeSortingTransform` when accumulating blocks, and even there it's a merge of sorted runs, not a re-sort of a nearly-sorted array.

---

## S5-7. Apache Kafka — TimeIndex & OffsetIndex

**Source examined**: [`apache/kafka`](https://github.com/apache/kafka) — `storage/src/main/java/org/apache/kafka/storage/internals/log/TimeIndex.java`, `OffsetIndex.java`, `AbstractIndex.java`

### Data Structure

Kafka maintains **two sorted on-disk indexes** per log segment, memory-mapped for efficient access:

1. **OffsetIndex**: Maps message offsets to physical file positions. Entry size: 8 bytes (4-byte relative offset + 4-byte position). Sorted by offset.

2. **TimeIndex**: Maps timestamps to message offsets. Entry size: 12 bytes (8-byte timestamp + 4-byte relative offset). Sorted by timestamp.

Both are stored in pre-allocated memory-mapped files (`MappedByteBuffer`) and accessed via binary search (`AbstractIndex.binarySearch()`).

### Sorting Mechanism

**These indexes are APPEND-ONLY and MONOTONICALLY INCREASING:**

```java
// TimeIndex.maybeAppend():
if (entries() != 0 && offset < lastEntry.offset())
    throw new InvalidOffsetException("Attempt to append an offset ("
        + offset + ") no larger than the last offset appended ("
        + lastEntry.offset() + ")");
```

```java
// OffsetIndex.append():
// appendOutOfOrder test confirms: throws InvalidOffsetException
```

Entries can only be appended in sorted order. Violations throw exceptions. There is no "re-sort" operation. Truncation (`truncateTo()`) removes entries beyond a threshold but maintains order. The indexes are rebuilt from scratch during recovery if corrupted.

**Kafka Streams segments**: In Kafka Streams, `AbstractSegments` uses a `TreeMap<Long, S>` (Java red-black tree) to track segments by segment ID — a tree, not an array.

### Assessment

| Criterion                | Finding                                                             |
| ------------------------ | ------------------------------------------------------------------- |
| Contiguous sorted array? | **Yes.** Memory-mapped flat byte arrays with fixed-size entries.    |
| Tracked changed indices? | **N/A.** Append-only — no in-place mutations.                       |
| Batch update pattern?    | **No.** Single-entry appends only.                                  |
| Current sorting approach | No sorting needed — append-only order is enforced at the API level. |
| Practical sizes          | Index entries: thousands to millions per segment.                   |

**Confidence: 2/100**

**DeltaSort opportunity**: None. Kafka indexes are append-only with monotonicity enforced by the API — entries cannot be modified or inserted out of order. The sorted order is maintained by construction, not by sorting. There is nothing to re-sort.

---

## S5-8. Prometheus — Histogram Quantile Bucket Sorting

**Source examined**: [`prometheus/prometheus`](https://github.com/prometheus/prometheus) — `promql/quantile.go`, `model/histogram/generic.go`, `storage/merge.go`

### Data Structure

**Classic histogram quantile calculation** (`BucketQuantile()` in `quantile.go`):

```go
func BucketQuantile(q float64, buckets Buckets) (...) {
    slices.SortFunc(buckets, func(a, b Bucket) int {
        if a.UpperBound < b.UpperBound { return -1 }
        if a.UpperBound > b.UpperBound { return +1 }
        return 0
    })
    // ... coalesceBuckets, ensureMonotonic, interpolation
}
```

**Every quantile query sorts the bucket array.** `Buckets` is a `[]Bucket` (Go slice — contiguous array) sorted by `UpperBound`. The sort is performed on every call to `BucketQuantile()` and `BucketFraction()`.

**Native histograms** use a different path: `HistogramQuantile()` iterates pre-sorted buckets via `AllBucketIterator()` without sorting — bucket order is maintained by the histogram's span structure.

**Histogram merging** (`FloatHistogram.Add()`): When adding two histograms with different schemas, `reduceResolution()` maps buckets from a finer schema to a coarser one — a merge-like operation on sorted span/bucket arrays. This is O(n) where n is the number of buckets.

**Time-series merge** (`storage/merge.go`): `mergeGenericQuerier` merges sorted series from multiple queriers using a heap-based priority queue.

### Assessment

| Criterion                | Finding                                                               |
| ------------------------ | --------------------------------------------------------------------- |
| Contiguous sorted array? | **Yes.** `[]Bucket` is a Go slice.                                    |
| Tracked changed indices? | **No.** Buckets are recomputed from label matchers each query.        |
| Batch update pattern?    | **No.** Full sort performed per query evaluation.                     |
| Current sorting approach | `slices.SortFunc` (pdqsort in Go ≥1.21) on every quantile query.      |
| Practical sizes          | Classic histograms: 10–50 buckets. Native histograms: 10–160 buckets. |

**Confidence: 3/100**

**DeltaSort opportunity**: None. The bucket arrays are tiny (typically 10–50 elements) and recomputed from scratch for each query evaluation. There is no persistent sorted state to incrementally maintain. Even if there were, the arrays are too small for DeltaSort to beat simple sorting algorithms.

---

## S5-9. Rust `BTreeMap` & `rustc_data_structures::SortedMap`

**Source examined**: [`rust-lang/rust`](https://github.com/rust-lang/rust) — `library/alloc/src/collections/btree/node.rs`, `compiler/rustc_data_structures/src/sorted_map.rs`

### A. `BTreeMap` — Standard Library

**Node sizing:**

```rust
const B: usize = 6;
pub const CAPACITY: usize = 2 * B - 1;  // = 11
const MIN_LEN_AFTER_SPLIT: usize = B - 1;  // = 5
```

`LeafNode<K, V>`: parent pointer + parent_idx + len (u16) + keys `[MaybeUninit<K>; 11]` + vals `[MaybeUninit<V>; 11]`. On x86_64 with `(i64, i64)`: ~192 bytes per leaf node.

`InternalNode<K, V>`: Contains a `LeafNode<K, V>` plus `edges: [MaybeUninit<BoxedNode<K, V>>; 2 * B]` — 12 child pointers. ~288 bytes.

**Insertion mechanism:**

- `insert_fit()`: Uses `slice_insert()` — shifts elements right via `ptr::copy` (memmove). O(B) = O(11) — effectively O(1) within a node.
- If node is full: `splitpoint()` determines split position, `split_leaf_data()` uses `move_to_slice()` (`ptr::copy_nonoverlapping` / memcpy) to move half to a new node.
- `insert_recursing()`: Recursive split propagation up to root.

**Within-node search**: From the BTreeMap documentation:

> _"Currently, our implementation simply performs naive linear search... This provides excellent performance on small nodes of elements which are cheap to compare."_

**Assessment for BTreeMap: Confidence 2/100.** BTreeMap is a pointer-based B-tree with 11-element nodes. No contiguous sorted array exists at any level. The within-node arrays are CAPACITY=11, far too small for DeltaSort.

### B. `rustc_data_structures::SortedMap` — Sorted Vec Wrapper ⭐

**BONUS FIND** — The Rust compiler's internal data structures include a `SortedMap`:

```rust
// compiler/rustc_data_structures/src/sorted_map.rs
/// "SortedMap is a sorted Vec.
/// Lookup is O(log(n)), insertion and removal are O(n).
/// Can be faster than BTreeMap for small sizes (<50).
/// Supports accessing contiguous ranges as a slice, and slices
/// of already sorted elements can be inserted efficiently."
```

This is a **sorted `Vec<(K, V)>`** — exactly a contiguous sorted array. It uses:

- Binary search for lookup: O(log n)
- `Vec::insert()` / `Vec::remove()` for mutations: O(n) due to shifting
- Optimized batch insert of pre-sorted slices

### Assessment for `SortedMap`

| Criterion                | Finding                                                                        |
| ------------------------ | ------------------------------------------------------------------------------ |
| Contiguous sorted array? | **✅ YES.** `Vec<(K, V)>` — textbook contiguous sorted array.                  |
| Tracked changed indices? | **No explicit tracking.** But the compiler knows which symbols it's modifying. |
| Batch update pattern?    | **Yes.** "Slices of already sorted elements can be inserted efficiently."      |
| Current sorting approach | Binary search + Vec::insert (O(n) per element due to shifting).                |
| Practical sizes          | Typically < 50 elements (documented sweet spot).                               |

**Confidence: 15/100**

**DeltaSort opportunity**: `SortedMap` is architecturally a perfect match — it IS a sorted array that gets mutated. However:

1. **It's designed for small sizes** (< 50 elements) where BTreeMap's overhead dominates. At n < 50, DeltaSort's bookkeeping overhead exceeds the cost of simple shifting.
2. **Insertions, not value changes**: The API pattern is insert/remove, not "change value at known position." DeltaSort addresses value mutations, not structural changes.
3. **Batch pre-sorted insert** already exists — inserting a sorted slice of k elements into a sorted array of n can be done in O(n + k) via merge, which is better than DeltaSort's $O(n\sqrt{k})$ for this specific operation.

If Rust compiler internals ever needed a larger `SortedMap` (n > 1000) with frequent value updates at known positions, DeltaSort would be directly applicable. But the current use case (small compiler metadata maps) doesn't justify it.

---

## S5-10. Streaming Percentiles: TDigest & Running Median

**Source examined**: DataFusion's `tdigest.rs`, Prometheus's `quantile.go`, general streaming percentile algorithms

### TDigest (Dunning, 2019)

TDigest maintains a sorted array of centroids `(mean, weight)` for approximate quantile estimation. The algorithm:

1. **Accumulate**: Buffer incoming values
2. **Sort**: Sort the buffer — full sort via comparison-based algorithm
3. **Merge**: Walk through sorted buffer and existing digest centroids simultaneously, merging centroids that fall within compression bounds
4. **Compress**: If centroid count exceeds compression threshold (~100–300), re-merge to reduce

**DataFusion's implementation** (`tdigest.rs` lines 192–265):

```rust
pub fn merge_unsorted_f64(&self, unsorted_values: &[f64]) -> TDigest {
    let mut sorted = sort_f64(unsorted_values);  // full sort of batch
    self.merge_sorted_f64(&sorted)
}
```

**Running median** algorithms (e.g., dual-heap approach):
Use two heaps (max-heap for lower half, min-heap for upper half). O(log n) per insertion, O(1) median query. No sorted array involved.

**Exponential histogram** (used in OpenTelemetry):
Maintains buckets at exponentially-distributed boundaries. Bucket counts are stored in arrays indexed by bucket number — no sorting required; bucket order is implicit in the indexing.

### Assessment

| Criterion                | Finding                                                                 |
| ------------------------ | ----------------------------------------------------------------------- |
| Contiguous sorted array? | **TDigest centroids: Yes** (small). Dual-heap: No. Exponential: No.     |
| Tracked changed indices? | **No.** New values are accumulated in batches, not tracked by position. |
| Batch update pattern?    | **TDigest: Yes** (buffer → sort → merge).                               |
| Current sorting approach | Full sort of batch buffer; merge walk with existing centroids.          |
| Practical sizes          | TDigest centroids: 100–300. Batch buffers: hundreds to thousands.       |

**Confidence: 5/100**

**DeltaSort opportunity**: None practical. TDigest centroids are small (~100–300 elements) and the sorted array is rebuilt each merge cycle rather than modified in-place. The batch buffer sort is on completely unsorted new data (k = n), where DeltaSort has no advantage. Running median and exponential histogram approaches don't use sorted arrays at all.

---

## Fifth Sweep Summary Table

| #   | System                            |         Array?         | Tracked? |      Batch?       |        Sizes         | Confidence |
| --- | --------------------------------- | :--------------------: | :------: | :---------------: | :------------------: | :--------: |
| 1   | **sortedcontainers** `SortedList` |  Partially (sublists)  |    ❌    |    ✅ (update)    |        100–1M        |    30%     |
| 2   | **Dragonfly** ZSET                |   ✅ (listpack ≤128)   |    ❌    |  ✅ (multi-ZADD)  |   ≤128 → skiplist    |    15%     |
| 3   | **Garnet** SortedSet              |  ❌ (red-black tree)   |    ❌    |        ❌         |      unlimited       |     3%     |
| 4   | **DataFusion** Sort/TDigest       | ✅ (TDigest centroids) |    ❌    |    ✅ (merge)     |  100–300 centroids   |     8%     |
| 5   | **DuckDB** MergePartition         |    ✅ (post-concat)    |    ❌    |     ✅ (runs)     |      1K–1M rows      |    20%     |
| 6   | **ClickHouse** MergeTree          |  ❌ (immutable parts)  |    ❌    |        ❌         | millions (streaming) |     5%     |
| 7   | **Kafka** TimeIndex/OffsetIndex   |       ✅ (mmap)        |   N/A    | ❌ (append-only)  |    1K–1M entries     |     2%     |
| 8   | **Prometheus** Histogram          |     ✅ ([]Bucket)      |    ❌    |        ❌         |    10–50 buckets     |     3%     |
| 9   | **Rust BTreeMap** / SortedMap     |     ✅ (SortedMap)     |    ❌    | ✅ (batch insert) |   < 50 (SortedMap)   |    15%     |
| 10  | **TDigest** / Streaming %         |     ✅ (centroids)     |    ❌    |        ✅         |       100–300        |     5%     |

---

## Fifth Sweep Conclusions

### The Recurring Pattern: Sorted ≠ Sortable

Across all 10 in-memory systems, a consistent pattern emerges: **systems that maintain sorted data do NOT use sorted arrays that get re-sorted.** Instead, they use one of three strategies:

1. **Always-sorted structures** (Garnet's red-black tree, Dragonfly's skiplist, Kafka's append-only index, Rust's BTreeMap): Sorted order is maintained continuously via O(log n) insertion. No re-sort ever occurs.

2. **One-shot sort + stream** (DataFusion's SortExec, ClickHouse's merge, Prometheus's bucket sort): Data is sorted once, consumed, and discarded. No persistent sorted state exists to incrementally maintain.

3. **Batch rebuild** (sortedcontainers' `update()`, DuckDB's `TemplatedMergePartition`, TDigest's merge): When enough changes accumulate, the entire structure is rebuilt from scratch — concatenate + full sort. These systems rely on adaptive sort algorithms (Timsort, pdqsort) to exploit the nearly-sorted structure of the concatenated data.

### The DeltaSort Model Gap

DeltaSort's model requires all four preconditions: (1) contiguous array, (2) previously sorted, (3) values change at known positions, (4) batch re-sort. **Precondition (3) — values change at known positions — is the one that almost never occurs in these systems.** In-memory data structures handle:

- **Insertions** (new elements added) — not "value at position i changed"
- **Deletions** (elements removed) — not "value at position i changed"
- **Score/priority updates** (which are handled as remove+reinsert in tree structures)

The only system that truly mutates values in a contiguous sorted array is `rustc_data_structures::SortedMap`, and it's designed for < 50 elements.

### Where Does This Leave DeltaSort?

The five sweeps (35+ systems) consistently point to the same conclusion: **DeltaSort's natural habitat is application-layer sorted views**, not infrastructure-level sorted data structures. The confirmed sweet spot remains:

1. **ag-Grid** (Sweep 4): Already implements ESM. DeltaSort could improve space complexity.
2. **UI data grids** (TanStack Table, Handsontable): Full re-sort on every change. ESM would help.
3. **Email clients** (Sweep 3): Large sorted mailbox views with batch arrivals.
4. **Playlist managers** (Sweep 1): Sorted libraries with metadata edits.

Infrastructure systems that maintain sorted data have evolved specialized structures (skip lists, B-trees, heaps, append-only indexes) that make DeltaSort's "re-sort a mutated array" model inapplicable. The lesson: **sorted data maintenance at the infrastructure level has been thoroughly solved by self-balancing trees and append-only ordered writes — the "sort an array" primitive is an application-layer concern.**
