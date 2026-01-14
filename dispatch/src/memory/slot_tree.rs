/// A segment tree that tracks contiguous runs of free slots,
/// enabling O(log n) allocation and deallocation of contiguous ranges.
///
/// Stored as an implicit binary tree in a flat array (1-indexed).
/// For 4096 slots, the tree is ~8K nodes × 8 bytes = 64KB.

#[derive(Clone, Copy, Debug)]
struct Node {
    /// Longest contiguous free run anywhere in this subtree.
    max_run: u16,
    /// Longest contiguous free run anchored to the left edge.
    prefix_free: u16,
    /// Longest contiguous free run anchored to the right edge.
    suffix_free: u16,
    /// Total number of slots covered by this node.
    total: u16,
}

impl Node {
    fn new_free(count: u16) -> Self {
        Self {
            max_run: count,
            prefix_free: count,
            suffix_free: count,
            total: count,
        }
    }

    fn new_used() -> Self {
        Self {
            max_run: 0,
            prefix_free: 0,
            suffix_free: 0,
            total: 1,
        }
    }

    fn merge(left: &Node, right: &Node) -> Node {
        Node {
            max_run: left
                .max_run
                .max(right.max_run)
                .max(left.suffix_free + right.prefix_free),
            prefix_free: if left.prefix_free == left.total {
                left.total + right.prefix_free
            } else {
                left.prefix_free
            },
            suffix_free: if right.suffix_free == right.total {
                right.total + left.suffix_free
            } else {
                right.suffix_free
            },
            total: left.total + right.total,
        }
    }
}

pub struct SlotTree {
    tree: Vec<Node>,
    num_slots: usize,
}

impl SlotTree {
    /// Create a new tree where all `num_slots` slots start as free.
    pub fn new(num_slots: usize) -> Self {
        assert!(num_slots > 0);
        // We use 1-indexed implicit binary tree, so we need 4 * num_slots to be safe.
        let size = 4 * num_slots;
        let mut st = Self {
            tree: vec![Node::new_free(0); size],
            num_slots,
        };
        st.build(1, 0, num_slots - 1);
        st
    }

    fn build(&mut self, node: usize, lo: usize, hi: usize) {
        if lo == hi {
            self.tree[node] = Node::new_free(1);
            return;
        }
        let mid = lo + (hi - lo) / 2;
        self.build(2 * node, lo, mid);
        self.build(2 * node + 1, mid + 1, hi);
        self.tree[node] = Node::merge(&self.tree[2 * node], &self.tree[2 * node + 1]);
    }

    /// Returns the maximum contiguous free run available.
    pub fn max_contiguous_free(&self) -> usize {
        self.tree[1].max_run as usize
    }

    /// Try to allocate `count` contiguous free slots.
    /// Returns `Some(start_slot)` on success (first-fit, left-biased).
    pub fn alloc(&mut self, count: usize) -> Option<usize> {
        if count == 0 || self.tree[1].max_run < count as u16 {
            return None;
        }
        let start = self.find_first_fit(1, 0, self.num_slots - 1, count as u16);
        // Mark the range as used.
        for slot in start..start + count {
            self.set_slot(slot, false);
        }
        Some(start)
    }

    /// Find the leftmost position where `count` contiguous free slots exist.
    fn find_first_fit(&self, node: usize, lo: usize, hi: usize, count: u16) -> usize {
        if lo == hi {
            // Must be a single free slot and count == 1.
            return lo;
        }

        let mid = lo + (hi - lo) / 2;
        let left = &self.tree[2 * node];
        let right = &self.tree[2 * node + 1];

        // 1) Try entirely within the left child.
        if left.max_run >= count {
            return self.find_first_fit(2 * node, lo, mid, count);
        }

        // 2) Try spanning the boundary: left.suffix + right.prefix.
        if left.suffix_free + right.prefix_free >= count {
            // The run starts at (mid + 1 - left.suffix_free).
            return (mid + 1 - left.suffix_free as usize);
        }

        // 3) Must be entirely within the right child.
        self.find_first_fit(2 * node + 1, mid + 1, hi, count)
    }

    /// Free a previously allocated range of slots.
    pub fn free(&mut self, start: usize, count: usize) {
        for slot in start..start + count {
            self.set_slot(slot, true);
        }
    }

    /// Mark a single slot as free (true) or used (false).
    pub fn set_slot(&mut self, slot: usize, free: bool) {
        assert!(slot < self.num_slots);
        self.update(1, 0, self.num_slots - 1, slot, free);
    }

    fn update(&mut self, node: usize, lo: usize, hi: usize, slot: usize, free: bool) {
        if lo == hi {
            self.tree[node] = if free {
                Node::new_free(1)
            } else {
                Node::new_used()
            };
            return;
        }
        let mid = lo + (hi - lo) / 2;
        if slot <= mid {
            self.update(2 * node, lo, mid, slot, free);
        } else {
            self.update(2 * node + 1, mid + 1, hi, slot, free);
        }
        self.tree[node] = Node::merge(&self.tree[2 * node], &self.tree[2 * node + 1]);
    }

    /// Check if a specific slot is free.
    pub fn is_free(&self, slot: usize) -> bool {
        assert!(slot < self.num_slots);
        self.query_slot(1, 0, self.num_slots - 1, slot)
    }

    fn query_slot(&self, node: usize, lo: usize, hi: usize, slot: usize) -> bool {
        if lo == hi {
            return self.tree[node].max_run == 1;
        }
        let mid = lo + (hi - lo) / 2;
        if slot <= mid {
            self.query_slot(2 * node, lo, mid, slot)
        } else {
            self.query_slot(2 * node + 1, mid + 1, hi, slot)
        }
    }

    /// Debug: print the root node stats.
    pub fn stats(&self) {
        let root = &self.tree[1];
        println!(
            "slots: {}, max_run: {}, prefix_free: {}, suffix_free: {}",
            root.total, root.max_run, root.prefix_free, root.suffix_free
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fresh_tree() {
        let tree = SlotTree::new(4096);
        assert_eq!(tree.max_contiguous_free(), 4096);
    }

    #[test]
    fn test_alloc_and_free() {
        let mut tree = SlotTree::new(4096);

        // Allocate 20 slots (80MB with 4MB slots).
        let start = tree.alloc(20).unwrap();
        assert_eq!(start, 0); // First-fit, left-biased.
        assert_eq!(tree.max_contiguous_free(), 4096 - 20);

        // Allocate another 20.
        let start2 = tree.alloc(20).unwrap();
        assert_eq!(start2, 20);
        assert_eq!(tree.max_contiguous_free(), 4096 - 40);

        // Free the first allocation → creates a gap.
        tree.free(0, 20);
        // Max run should be 4096 - 40 + 20 ... but there's a gap.
        // Slots 0..20 free, 20..40 used, 40..4096 free.
        // Max run = 4096 - 40 = 4056.
        assert_eq!(tree.max_contiguous_free(), 4056);

        // Free the second → everything coalesces.
        tree.free(20, 20);
        assert_eq!(tree.max_contiguous_free(), 4096);
    }

    #[test]
    fn test_fragmentation() {
        let mut tree = SlotTree::new(100);

        // Allocate every other slot to maximally fragment.
        for i in (0..100).step_by(2) {
            tree.set_slot(i, false);
        }
        // Max contiguous free run should be 1 (single slots between used ones).
        assert_eq!(tree.max_contiguous_free(), 1);

        // Can't allocate 2 contiguous.
        assert!(tree.alloc(2).is_none());

        // Free slot 2 → slots 1, 2, 3 are free (run of 3 if slot 3 is free).
        // Actually: slot 0 used, 1 free, 2 now free, 3 free, 4 used.
        tree.set_slot(2, true);
        assert_eq!(tree.max_contiguous_free(), 3);
        assert!(tree.alloc(3).is_some());
    }

    #[test]
    fn test_alloc_too_large() {
        let mut tree = SlotTree::new(64);
        tree.alloc(64).unwrap();
        assert!(tree.alloc(1).is_none());
    }

    #[test]
    fn test_boundary_spanning() {
        let mut tree = SlotTree::new(64);

        // Use slots 0..10 and 30..64, leaving 10..30 free (20 slots).
        for i in 0..10 {
            tree.set_slot(i, false);
        }
        for i in 30..64 {
            tree.set_slot(i, false);
        }
        assert_eq!(tree.max_contiguous_free(), 20);

        let start = tree.alloc(20).unwrap();
        assert_eq!(start, 10);
    }

    #[test]
    fn test_single_slot() {
        let mut tree = SlotTree::new(1);
        assert_eq!(tree.max_contiguous_free(), 1);

        let start = tree.alloc(1).unwrap();
        assert_eq!(start, 0);
        assert_eq!(tree.max_contiguous_free(), 0);

        tree.free(0, 1);
        assert_eq!(tree.max_contiguous_free(), 1);
    }
}