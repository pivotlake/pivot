//! How the global ring's slots map onto NUMA nodes and their workers.
//!
//! The ring is one contiguous mmap shared by every worker, split into one
//! contiguous *region* of slots per NUMA node. A slot's memory lives on its
//! region's node (each node's workers fault their own region at startup, and
//! first-touch pins the pages there), so keeping placement node-local reduces
//! to one rule: **a worker only ever acquires slots from its own region**.
//! Reads may cross regions freely; a remote read of cached data is far cheaper
//! than re-reading it from disk into a local slot.
//!
//! ```text
//!  slots:   [ node 0 region | node 1 region | ... ]
//!  acquire: node-local only (free pools, eviction sweeps)
//!  read:    any worker, any slot
//!  release: routed back to the slot's home worker, from any thread
//! ```
//!
//! [`RingLayout`] is the one place that mapping is defined; the free pools use
//! it to route released slots home, and each worker's prefault/evict paths use
//! it to bound themselves to their region.

use crate::numa::Topology;
use std::ops::Range;

#[derive(Clone, Copy, Debug)]
pub struct RingLayout {
    topology: Topology,
    slots_per_node: usize,
}

impl RingLayout {
    pub fn new(topology: Topology, slots_per_node: usize) -> Self {
        Self {
            topology,
            slots_per_node,
        }
    }

    /// A layout with every slot on one node, the shape of a single-socket machine.
    pub fn single_node(workers: usize, slots: usize) -> Self {
        Self::new(Topology::single_node(workers), slots)
    }

    pub fn topology(&self) -> Topology {
        self.topology
    }

    pub fn total_slots(&self) -> usize {
        self.slots_per_node * self.topology.node_count
    }

    pub fn total_workers(&self) -> usize {
        self.topology.total_workers()
    }

    /// The NUMA node whose region contains `slot`.
    pub fn node_of_slot(&self, slot: usize) -> usize {
        slot / self.slots_per_node
    }

    /// The slot range owned by `node`.
    pub fn node_slots(&self, node: usize) -> Range<usize> {
        node * self.slots_per_node..(node + 1) * self.slots_per_node
    }

    /// The global index of the worker a released `slot` is routed back to.
    /// Always a worker on the slot's own node, and consistent for a given slot,
    /// so every region's slots spread evenly over that node's free pools.
    pub fn home_worker(&self, slot: usize) -> usize {
        self.node_of_slot(slot) * self.topology.workers_per_node
            + slot % self.topology.workers_per_node
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_node_layout() -> RingLayout {
        RingLayout::new(
            Topology {
                workers_per_node: 3,
                node_count: 2,
            },
            6,
        )
    }

    #[test]
    fn slots_split_into_contiguous_node_regions() {
        let layout = two_node_layout();

        assert_eq!(layout.node_slots(0), 0..6);
        assert_eq!(layout.node_slots(1), 6..12);
        assert_eq!(layout.node_of_slot(5), 0);
        assert_eq!(layout.node_of_slot(6), 1);
    }

    #[test]
    fn a_slots_home_worker_is_on_its_own_node() {
        let layout = two_node_layout();

        for slot in 0..layout.total_slots() {
            let home = layout.home_worker(slot);
            assert_eq!(
                layout.topology().node_of_worker(home),
                layout.node_of_slot(slot),
            );
        }
    }

    #[test]
    fn home_workers_cover_every_worker_of_a_node() {
        let layout = two_node_layout();

        let homes: std::collections::HashSet<usize> = layout
            .node_slots(1)
            .map(|s| layout.home_worker(s))
            .collect();

        assert_eq!(homes, (3..6).collect());
    }
}
