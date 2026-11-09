//! The shape shared by everything that keeps per-partition-group state while
//! spilling: a binary tree whose 64 leaves are the partition groups.
//!
//! Partitions are gathered into groups so that a group's partitions can share
//! buffers instead of each pinning a whole ring slot. Early in a query every
//! group shares one buffer at the root. Each time a node's buffer is exhausted
//! the groups pointing at it are split between its two children, so buffers
//! are shared by fewer and fewer partitions as the data grows, until each leaf
//! group has buffers of its own.

use std::ops::{ControlFlow, Range};

/// How many partitions the data is hashed into.
pub const PARTITIONS: usize = 4096;
/// How many groups the partitions are gathered into, one per leaf.
pub const GROUPS: usize = 64;
pub const PARTITIONS_PER_GROUP: usize = PARTITIONS / GROUPS;
/// Nodes of a complete binary tree with `GROUPS` leaves.
const NODES: usize = 2 * GROUPS - 1;
/// Nodes from here on are leaves; leaf `FIRST_LEAF + g` is group `g`.
const FIRST_LEAF: usize = GROUPS - 1;

/// A complete binary tree of 127 nodes, each holding an `N`, with a record
/// of which node each partition group currently points at. Nodes are indexed
/// so that the children of node `i` are `2i + 1` and `2i + 2`, the root is
/// `0`, and the leaves are the last 64 nodes in group order.
pub struct PartitionTree<N> {
    nodes: Box<[N]>,
    /// The node each group currently points at. Always a node whose subtree
    /// contains the group's leaf.
    group_node: [u8; GROUPS],
}

pub fn group_of_partition(partition: usize) -> usize {
    partition / PARTITIONS_PER_GROUP
}

pub fn is_leaf(node: usize) -> bool {
    node >= FIRST_LEAF
}

/// The groups whose leaves are under `node`, itself included if it is one.
pub fn groups_under(node: usize) -> Range<usize> {
    let depth = (node + 1).ilog2() as usize;
    let index_in_level = node - ((1 << depth) - 1);
    let span = GROUPS >> depth;
    index_in_level * span..(index_in_level + 1) * span
}

impl<N> PartitionTree<N> {
    /// A tree whose nodes are each made by `make_node`, with every group
    /// pointing at the root.
    pub fn new(make_node: impl FnMut() -> N) -> Self {
        Self {
            nodes: std::iter::repeat_with(make_node).take(NODES).collect(),
            group_node: [0; GROUPS],
        }
    }

    /// The node `group` currently points at.
    pub fn node_of_group(&self, group: usize) -> usize {
        self.group_node[group] as usize
    }

    pub fn node_mut(&mut self, node: usize) -> &mut N {
        &mut self.nodes[node]
    }

    /// Every node, in index order.
    pub fn nodes(&self) -> &[N] {
        &self.nodes
    }

    /// Take every node, in index order.
    pub fn into_nodes(self) -> Vec<N> {
        self.nodes.into_vec()
    }

    /// Point the groups at `node` to its children instead: the first half of
    /// them to the left child, the rest to the right.
    pub fn split(&mut self, node: usize) {
        assert!(!is_leaf(node), "a leaf cannot be split");
        let groups = groups_under(node);
        let middle = (groups.start + groups.end) / 2;
        for group in groups {
            debug_assert_eq!(self.node_of_group(group), node);
            let child = if group < middle {
                2 * node + 1
            } else {
                2 * node + 2
            };
            self.group_node[group] = child as u8;
        }
    }

    /// Visit every node in the order buffers should be spilled and evicted:
    /// a node's right subtree, then its left subtree, then the node itself.
    /// So the last group's leaf comes first, an inner node comes only after
    /// every leaf under it, and the root comes last. A buffer at an inner node
    /// holds slabs of every group under it, so spilling it touches all of
    /// those partitions; spilling the leaves first keeps each spill's effect
    /// to one group, and by the time an inner node is reached its partitions
    /// have already been touched. Visiting the last groups first matches the
    /// order the output drains them, so what is spilled is needed latest.
    pub fn walk_spill_order<T>(
        &mut self,
        mut visit: impl FnMut(usize, &mut N) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        self.walk_right_left_node(0, &mut visit)
    }

    fn walk_right_left_node<T>(
        &mut self,
        node: usize,
        visit: &mut impl FnMut(usize, &mut N) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        if !is_leaf(node) {
            self.walk_right_left_node(2 * node + 2, visit)?;
            self.walk_right_left_node(2 * node + 1, visit)?;
        }
        visit(node, &mut self.nodes[node])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_group_starts_at_the_root() {
        let tree = PartitionTree::new(|| ());

        let nodes: Vec<usize> = (0..GROUPS).map(|g| tree.node_of_group(g)).collect();

        assert!(nodes.iter().all(|&node| node == 0));
    }

    #[test]
    fn groups_under_a_node_are_the_leaves_of_its_subtree() {
        assert_eq!(groups_under(0), 0..64);
        assert_eq!(groups_under(2), 32..64);
        assert_eq!(groups_under(3), 0..16);
        assert_eq!(groups_under(FIRST_LEAF + 5), 5..6);
    }

    #[test]
    fn splitting_sends_each_half_of_the_groups_to_a_child() {
        let mut tree = PartitionTree::new(|| ());

        tree.split(0);
        tree.split(2);

        assert_eq!(tree.node_of_group(0), 1);
        assert_eq!(tree.node_of_group(31), 1);
        assert_eq!(tree.node_of_group(32), 5);
        assert_eq!(tree.node_of_group(63), 6);
    }

    #[test]
    fn spill_order_visits_the_last_leaves_first_and_a_node_after_its_subtree() {
        let mut tree = PartitionTree::new(|| ());
        let mut order = Vec::new();

        let _: ControlFlow<()> = tree.walk_spill_order(|node, ()| {
            order.push(node);
            ControlFlow::Continue(())
        });

        assert_eq!(order.len(), NODES);
        assert_eq!(&order[..7], &[126, 125, 62, 124, 123, 61, 30]);
        assert_eq!(order[NODES - 1], 0);
        assert_eq!(order[NODES - 2], 1);
    }

    #[test]
    fn spill_order_stops_where_the_visitor_breaks() {
        let mut tree = PartitionTree::new(|| 0usize);

        let found = tree.walk_spill_order(|node, _| {
            if node == 62 {
                ControlFlow::Break(node)
            } else {
                ControlFlow::Continue(())
            }
        });

        assert_eq!(found, ControlFlow::Break(62));
    }
}
