//! LeafBTree: a B-tree variant that uses DeltaSort to compact leaves.
//!
//! Standard B-trees keep leaves small (typically 16-256 entries) because per-insert
//! cost is O(L). With DeltaSort, we can buffer up to k unsorted inserts at the leaf
//! tail and compact in O(L * sqrt(k)) when the buffer fills. The amortized per-insert
//! cost is O(L / sqrt(k)), so for L=1000, k=100 the per-insert cost is ~3.2x lower
//! than sorted-insert. This shifts the design point toward larger leaves.
//!
//! This is a research prototype, not a production data structure. Key simplifications:
//!   - Internal nodes are stored as Box<Node>; no arena allocation
//!   - Single-threaded; no concurrency
//!   - Generic over K: Ord + Clone + Default; V: Clone + Default
//!   - No persistence; in-memory only
//!   - Inserts overwrite on duplicate key (like BTreeMap insert)
//!   - No delete (insert+lookup+range only)
//!
//! Design:
//!   Leaf = { sorted: Vec<(K,V)>, buffer: Vec<(K,V)> }
//!     - `sorted` is maintained in K-order
//!     - `buffer` holds unsorted recent inserts up to `buffer_capacity`
//!     - On compaction: append buffer to sorted, mark new positions dirty,
//!       call DeltaSort, then clear buffer.
//!     - On split: triggered when sorted.len() > leaf_capacity. Compact first.
//!
//! Internal = { keys: Vec<K>, children: Vec<Box<Node<K,V>>> }
//!     - Standard B-tree internal node, fanout `internal_capacity`
//!     - keys[i] is the separator: all keys in children[i] are < keys[i],
//!       and all keys in children[i+1] are >= keys[i].

use crate::delta_sort_by;

pub struct LeafBTree<K, V> {
    root: Box<Node<K, V>>,
    leaf_capacity: usize,
    buffer_capacity: usize,
    internal_capacity: usize,
    len: usize,
    compactor: Compactor,
}

/// Choice of leaf compaction algorithm. Used to isolate the contribution of DeltaSort
/// from the "large-leaf design" itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compactor {
    /// DeltaSort over only the dirty (buffered) tail.
    DeltaSort,
    /// std::sort_unstable_by on the whole leaf (compares against "naive large-leaf B+tree").
    FullSort,
}

enum Node<K, V> {
    Leaf(Leaf<K, V>),
    Internal(Internal<K, V>),
}

struct Leaf<K, V> {
    sorted: Vec<(K, V)>,
    buffer: Vec<(K, V)>,
}

struct Internal<K, V> {
    keys: Vec<K>,
    children: Vec<Box<Node<K, V>>>,
}

impl<K, V> LeafBTree<K, V>
where
    K: Ord + Clone + Default,
    V: Clone + Default,
{
    /// Create a new LeafBTree with the given configuration.
    ///   - `leaf_capacity`: max entries in a leaf's sorted region before split
    ///   - `buffer_capacity`: max entries in the leaf's insert buffer before compaction
    ///   - `internal_capacity`: max children per internal node
    pub fn with_config(leaf_capacity: usize, buffer_capacity: usize, internal_capacity: usize) -> Self {
        Self::with_config_and_compactor(leaf_capacity, buffer_capacity, internal_capacity, Compactor::DeltaSort)
    }

    pub fn with_config_and_compactor(
        leaf_capacity: usize,
        buffer_capacity: usize,
        internal_capacity: usize,
        compactor: Compactor,
    ) -> Self {
        assert!(leaf_capacity >= 4);
        assert!(buffer_capacity >= 1);
        assert!(internal_capacity >= 4);
        Self {
            root: Box::new(Node::Leaf(Leaf {
                sorted: Vec::with_capacity(leaf_capacity + buffer_capacity),
                buffer: Vec::with_capacity(buffer_capacity),
            })),
            leaf_capacity,
            buffer_capacity,
            internal_capacity,
            len: 0,
            compactor,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Insert a key-value pair. If the key already exists, the value is updated
    /// (overwrites previous). Returns the previous value if present.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let leaf_cap = self.leaf_capacity;
        let buf_cap = self.buffer_capacity;
        let int_cap = self.internal_capacity;
        let compactor = self.compactor;

        let (prev, split) = Self::insert_into(&mut self.root, key, value, leaf_cap, buf_cap, int_cap, compactor);
        if prev.is_none() {
            self.len += 1;
        }
        if let Some((sep_key, new_right)) = split {
            // Root split: create new root
            let old_root = std::mem::replace(
                &mut self.root,
                Box::new(Node::Leaf(Leaf {
                    sorted: Vec::new(),
                    buffer: Vec::new(),
                })),
            );
            self.root = Box::new(Node::Internal(Internal {
                keys: vec![sep_key],
                children: vec![old_root, new_right],
            }));
        }
        prev
    }

    /// Look up a key. Returns Some(&V) if present.
    pub fn get(&self, key: &K) -> Option<&V> {
        let mut node = &*self.root;
        loop {
            match node {
                Node::Leaf(leaf) => return leaf_get(leaf, key),
                Node::Internal(int) => {
                    let i = int.keys.partition_point(|k| k <= key);
                    node = &*int.children[i];
                }
            }
        }
    }

    /// Number of elements in the tree.
    pub fn iter(&self) -> Iter<'_, K, V> {
        let mut stack = Vec::new();
        stack.push(NodeFrame { node: &*self.root, idx: 0 });
        Iter { stack, leaf_cursor: None, start_key: None }
    }

    /// Iterate from the first key >= `start` in O(log n + scan).
    pub fn range_from<'a>(&'a self, start: &'a K) -> Iter<'a, K, V> {
        // Descend to the leaf that should contain `start`, building the stack so that
        // when the leaf is exhausted, iteration continues to subsequent leaves in order.
        let mut stack: Vec<NodeFrame<'a, K, V>> = Vec::new();
        let mut node: &'a Node<K, V> = &*self.root;
        loop {
            match node {
                Node::Leaf(_) => {
                    stack.push(NodeFrame { node, idx: 0 });
                    break;
                }
                Node::Internal(int) => {
                    let i = int.keys.partition_point(|k| k <= start);
                    // After we descend into child[i], next time we resume this frame
                    // we should advance to child[i+1]. So store idx = i+1.
                    stack.push(NodeFrame { node, idx: i + 1 });
                    node = &*int.children[i];
                }
            }
        }
        Iter { stack, leaf_cursor: None, start_key: Some(start) }
    }

    /// Force compaction of all leaves (for benchmarking).
    pub fn compact_all(&mut self) {
        compact_node(&mut self.root, self.compactor);
    }

    fn insert_into(
        node: &mut Box<Node<K, V>>,
        key: K,
        value: V,
        leaf_cap: usize,
        buf_cap: usize,
        int_cap: usize,
        compactor: Compactor,
    ) -> (Option<V>, Option<(K, Box<Node<K, V>>)>) {
        match &mut **node {
            Node::Leaf(leaf) => {
                if let Ok(idx) = leaf.sorted.binary_search_by(|(k, _)| k.cmp(&key)) {
                    let old = std::mem::replace(&mut leaf.sorted[idx].1, value);
                    return (Some(old), None);
                }
                for (k, v) in leaf.buffer.iter_mut() {
                    if k == &key {
                        let old = std::mem::replace(v, value);
                        return (Some(old), None);
                    }
                }
                leaf.buffer.push((key, value));

                if leaf.buffer.len() >= buf_cap {
                    compact_leaf(leaf, compactor);
                }

                if leaf.sorted.len() > leaf_cap {
                    let new_right = split_leaf(leaf, compactor);
                    let sep_key = match &*new_right {
                        Node::Leaf(r) => r.sorted[0].0.clone(),
                        _ => unreachable!(),
                    };
                    return (None, Some((sep_key, new_right)));
                }
                (None, None)
            }
            Node::Internal(int) => {
                let i = int.keys.partition_point(|k| k <= &key);
                let (prev, child_split) = Self::insert_into(&mut int.children[i], key, value, leaf_cap, buf_cap, int_cap, compactor);
                if let Some((sep_key, new_right)) = child_split {
                    int.keys.insert(i, sep_key);
                    int.children.insert(i + 1, new_right);
                    if int.children.len() > int_cap {
                        let mid_child = int.children.len() / 2;
                        let promoted = int.keys.remove(mid_child - 1);
                        let right_children: Vec<_> = int.children.drain(mid_child..).collect();
                        let right_keys: Vec<_> = int.keys.drain(mid_child - 1..).collect();
                        let right_node = Box::new(Node::Internal(Internal {
                            keys: right_keys,
                            children: right_children,
                        }));
                        return (prev, Some((promoted, right_node)));
                    }
                }
                (prev, None)
            }
        }
    }
}

// (Removed unused helpers split_internal and promote_internal_split — split logic is inlined.)

fn split_leaf<K, V>(leaf: &mut Leaf<K, V>, compactor: Compactor) -> Box<Node<K, V>>
where
    K: Ord + Clone + Default,
    V: Clone + Default,
{
    compact_leaf(leaf, compactor);
    let mid = leaf.sorted.len() / 2;
    let right_sorted: Vec<_> = leaf.sorted.drain(mid..).collect();
    Box::new(Node::Leaf(Leaf {
        sorted: right_sorted,
        buffer: Vec::new(),
    }))
}

fn compact_leaf<K, V>(leaf: &mut Leaf<K, V>, compactor: Compactor)
where
    K: Ord + Clone + Default,
    V: Clone + Default,
{
    if leaf.buffer.is_empty() {
        return;
    }
    let buf_size = leaf.buffer.len();
    let sorted_size = leaf.sorted.len();
    match compactor {
        Compactor::DeltaSort => {
            let mut dirty: Vec<usize> = (sorted_size..sorted_size + buf_size).collect();
            leaf.sorted.extend(leaf.buffer.drain(..));
            delta_sort_by(&mut leaf.sorted, &mut dirty, |a, b| a.0.cmp(&b.0));
        }
        Compactor::FullSort => {
            leaf.sorted.extend(leaf.buffer.drain(..));
            leaf.sorted.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        }
    }
}

fn compact_node<K, V>(node: &mut Box<Node<K, V>>, compactor: Compactor)
where
    K: Ord + Clone + Default,
    V: Clone + Default,
{
    match &mut **node {
        Node::Leaf(leaf) => compact_leaf(leaf, compactor),
        Node::Internal(int) => {
            for child in int.children.iter_mut() {
                compact_node(child, compactor);
            }
        }
    }
}

fn leaf_get<'a, K, V>(leaf: &'a Leaf<K, V>, key: &K) -> Option<&'a V>
where
    K: Ord,
{
    // Buffer is unsorted and recent: scan it first (overrides sorted on duplicate).
    for (k, v) in leaf.buffer.iter().rev() {
        if k == key {
            return Some(v);
        }
    }
    if let Ok(idx) = leaf.sorted.binary_search_by(|(k, _)| k.cmp(key)) {
        return Some(&leaf.sorted[idx].1);
    }
    None
}

// ============================================================================
// Iterator (in-order)
// ============================================================================

pub struct Iter<'a, K, V> {
    stack: Vec<NodeFrame<'a, K, V>>,
    leaf_cursor: Option<LeafCursor<'a, K, V>>,
    start_key: Option<&'a K>,
}

struct NodeFrame<'a, K, V> {
    node: &'a Node<K, V>,
    idx: usize,
}

struct LeafCursor<'a, K, V> {
    // Merged view: we need to produce sorted+buffer entries in key order.
    // For simplicity (and to keep buffer scan honest in benchmark), we materialize merged on-demand.
    merged: Vec<&'a (K, V)>,
    pos: usize,
}

impl<'a, K, V> Iterator for Iter<'a, K, V>
where
    K: Ord,
{
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(cursor) = &mut self.leaf_cursor {
                if cursor.pos < cursor.merged.len() {
                    let (k, v) = cursor.merged[cursor.pos];
                    cursor.pos += 1;
                    return Some((k, v));
                }
                self.leaf_cursor = None;
            }
            let frame = self.stack.last_mut()?;
            match frame.node {
                Node::Leaf(leaf) => {
                    let idx = frame.idx;
                    if idx == 0 {
                        // Initialize cursor by merging buffer into a sorted view.
                        let mut merged: Vec<&(K, V)> = leaf.sorted.iter().collect();
                        for entry in leaf.buffer.iter() {
                            let pos = merged.partition_point(|x| x.0 < entry.0);
                            merged.insert(pos, entry);
                        }
                        let pos = if let Some(start) = self.start_key.take() {
                            merged.partition_point(|x| &x.0 < start)
                        } else { 0 };
                        self.leaf_cursor = Some(LeafCursor { merged, pos });
                        frame.idx = 1;
                    } else {
                        self.stack.pop();
                    }
                }
                Node::Internal(int) => {
                    if frame.idx < int.children.len() {
                        let i = frame.idx;
                        frame.idx += 1;
                        let child = &*int.children[i];
                        self.stack.push(NodeFrame { node: child, idx: 0 });
                    } else {
                        self.stack.pop();
                    }
                }
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::BTreeMap;

    #[test]
    fn empty_tree() {
        let t = LeafBTree::<u64, u64>::with_config(256, 32, 16);
        assert!(t.is_empty());
        assert_eq!(t.get(&5), None);
    }

    #[test]
    fn single_insert_and_lookup() {
        let mut t = LeafBTree::<u64, u64>::with_config(256, 32, 16);
        t.insert(5, 50);
        assert_eq!(t.get(&5), Some(&50));
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn many_inserts_against_btreemap() {
        let mut t = LeafBTree::<u64, u64>::with_config(256, 32, 16);
        let mut ref_ = BTreeMap::<u64, u64>::new();
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..5000 {
            let k: u64 = rng.gen_range(0..10000);
            let v: u64 = rng.gen();
            let prev_t = t.insert(k, v);
            let prev_r = ref_.insert(k, v);
            assert_eq!(prev_t, prev_r, "insert prev mismatch at k={k}");
        }
        assert_eq!(t.len(), ref_.len());
        for (k, v) in ref_.iter() {
            assert_eq!(t.get(k), Some(v), "lookup mismatch at k={k}");
        }
    }

    #[test]
    fn iter_in_order() {
        let mut t = LeafBTree::<u64, u64>::with_config(64, 16, 8);
        let mut ref_ = BTreeMap::<u64, u64>::new();
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..2000 {
            let k: u64 = rng.gen_range(0..100000);
            let v: u64 = rng.gen();
            t.insert(k, v);
            ref_.insert(k, v);
        }
        let got: Vec<(u64, u64)> = t.iter().map(|(k, v)| (*k, *v)).collect();
        let want: Vec<(u64, u64)> = ref_.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(got.len(), want.len());
        assert_eq!(got, want);
    }

    #[test]
    fn deep_tree_random() {
        let mut t = LeafBTree::<u32, u32>::with_config(32, 8, 8);
        let mut ref_ = BTreeMap::<u32, u32>::new();
        let mut rng = StdRng::seed_from_u64(99);
        for _ in 0..20000 {
            let k: u32 = rng.gen_range(0..50000);
            let v: u32 = rng.gen();
            let p1 = t.insert(k, v);
            let p2 = ref_.insert(k, v);
            assert_eq!(p1, p2);
        }
        assert_eq!(t.len(), ref_.len());
        let got: Vec<(u32, u32)> = t.iter().map(|(k, v)| (*k, *v)).collect();
        let want: Vec<(u32, u32)> = ref_.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(got, want);
    }
}
