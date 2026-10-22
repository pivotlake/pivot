//! Keeping spilled buffers until their ring slots are actually needed.
//!
//! Spilling writes a buffer's rows to disk; it does not free the buffer. The
//! buffer stays resident, and its rows readable, in the [`PartitionEvictor`],
//! laid out in the same [`PartitionTree`] its slabs were allocated from. The
//! memory context reaches the evictor through the [`Evict`] handle an operator
//! registers at build time, and takes a slot only when it has none to give:
//! the first buffer, in spill order, whose writes have all landed.

use crate::io::FileRange;
use crate::memory::evict::Evict;
use crate::memory::spill::SpilledBuffer;
use crate::memory::spill::partition_tree::PartitionTree;
use crate::memory::{WriteBuffer, memory_ctx};
use ahash::HashMap;
use std::cell::RefCell;
use std::ops::ControlFlow;
use std::rc::Rc;
use std::sync::Arc;

/// Spilled buffers awaiting eviction, each under the partition group slot it
/// was allocated from.
pub struct PartitionEvictor {
    tree: PartitionTree<Vec<SpilledBuffer>>,
    /// For each partition that lost rows to eviction, where in its spill file
    /// those rows are. Rows still resident are not listed: they are read from
    /// memory.
    rows_on_disk_only: HashMap<usize, Vec<FileRange>>,
}

/// What the evictor holds when an operator is done consuming: the spilled
/// buffers still resident, whose rows are readable in memory, and for each
/// partition the file ranges whose rows exist only on disk.
pub struct TakenSpills {
    pub resident: Vec<SpilledBuffer>,
    pub on_disk_by_partition: HashMap<usize, Vec<FileRange>>,
}

/// The evictor as the operator and the memory context share it. The two never
/// hold it at the same time: the operator pushes into it between allocations,
/// and the memory context evicts from it inside an allocation, when the
/// operator holds no borrow.
pub type SharedPartitionEvictor = Rc<RefCell<PartitionEvictor>>;

impl PartitionEvictor {
    pub fn new() -> Self {
        Self {
            tree: PartitionTree::new(Vec::new),
            rows_on_disk_only: HashMap::default(),
        }
    }

    /// Take everything, leaving nothing to evict: the spilled buffers still
    /// held, in tree order, and the file ranges of the rows that were evicted.
    /// The buffers stay resident for as long as the caller holds their slabs,
    /// as ordinary memory rather than spilled: the memory context stops
    /// counting them as reclaimable.
    pub fn take_all(&mut self) -> TakenSpills {
        let mut resident = Vec::new();
        let _: ControlFlow<()> = self.tree.walk_spill_order(|_, spilled_buffers| {
            resident.append(spilled_buffers);
            ControlFlow::Continue(())
        });
        memory_ctx().record_spilled_buffers_released(resident.len());
        TakenSpills {
            resident,
            on_disk_by_partition: std::mem::take(&mut self.rows_on_disk_only),
        }
    }

    /// Keep `spilled` until it is evicted.
    pub fn push(&mut self, spilled: SpilledBuffer) {
        self.tree
            .node_mut(spilled.partition_group_slot)
            .push(spilled);
    }

    /// Give up the first buffer, in spill order, whose rows are all on disk,
    /// or `None` when every spilled buffer still has writes in flight.
    pub fn evict(&mut self) -> Option<WriteBuffer> {
        let on_disk = self.tree.walk_spill_order(|_, spilled_buffers| {
            match spilled_buffers.iter().position(SpilledBuffer::is_on_disk) {
                Some(idx) => ControlFlow::Break(spilled_buffers.remove(idx)),
                None => ControlFlow::Continue(()),
            }
        });
        let ControlFlow::Break(SpilledBuffer { buffer, slabs, .. }) = on_disk else {
            return None;
        };
        // From here on these rows can only be read from the partition files.
        for slab in &slabs {
            self.rows_on_disk_only
                .entry(slab.spill_partition_id)
                .or_default()
                .push(FileRange::new(slab.file_offset as usize, slab.slab.size()));
        }
        // The writes let go of their handles when they landed, so once these
        // slabs are gone the buffer has no holder left but this one.
        drop(slabs);
        let write_buffer = Arc::into_inner(buffer)
            .expect("a spilled buffer whose writes have landed is held only by its slabs");
        memory_ctx().record_spilled_buffers_released(1);
        Some(write_buffer)
    }
}

impl Default for PartitionEvictor {
    fn default() -> Self {
        Self::new()
    }
}

impl Evict for SharedPartitionEvictor {
    fn evict(&self) -> Option<WriteBuffer> {
        self.borrow_mut().evict()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::spill::SpilledSlab;
    use crate::memory::{Slab, init_test_free_pool};
    use std::sync::atomic::AtomicUsize;

    /// A spilled buffer with one slab, filed under `partition_group_slot`, with
    /// `writes_in_flight` of its writes still to land. Counted as spilled with
    /// the memory context, as the spiller would have.
    fn spilled(partition_group_slot: usize, writes_in_flight: usize) -> SpilledBuffer {
        memory_ctx().record_buffers_spilled(1);
        let buffer = Arc::new(memory_ctx().get_write_buffer(false));
        let slab = Slab::new(buffer.ptr, 1024, buffer.clone()).freeze();
        SpilledBuffer {
            writes_in_flight: Arc::new(AtomicUsize::new(writes_in_flight)),
            buffer,
            slabs: vec![SpilledSlab {
                spill_partition_id: 0,
                file_offset: 0,
                slab,
            }],
            partition_group_slot,
        }
    }

    #[test]
    fn nothing_is_evicted_while_a_buffers_writes_are_in_flight() {
        init_test_free_pool(1);
        let mut evictor = PartitionEvictor::new();
        evictor.push(spilled(0, 1));

        let evicted = evictor.evict();

        assert!(evicted.is_none());
    }

    #[test]
    fn evicting_remembers_where_the_partitions_rows_are_on_disk() {
        init_test_free_pool(2);
        let mut evictor = PartitionEvictor::new();
        evictor.push(spilled(0, 0));
        evictor.push(spilled(0, 1));

        evictor.evict().unwrap();
        let taken = evictor.take_all();

        assert_eq!(taken.resident.len(), 1);
        assert_eq!(
            taken.on_disk_by_partition.get(&0).map(Vec::as_slice),
            Some(&[FileRange::new(0, 1024)][..])
        );
    }

    #[test]
    fn a_buffer_whose_writes_landed_is_given_up_whole() {
        init_test_free_pool(1);
        let mut evictor = PartitionEvictor::new();
        let on_disk = spilled(0, 0);
        let slot_idx = on_disk.buffer.slot_idx;
        evictor.push(on_disk);

        let evicted = evictor.evict();
        let again = evictor.evict();

        assert_eq!(evicted.map(|buffer| buffer.slot_idx), Some(slot_idx));
        assert!(again.is_none());
    }

    #[test]
    fn eviction_takes_a_deeper_slot_before_the_root() {
        init_test_free_pool(2);
        let mut evictor = PartitionEvictor::new();
        let at_root = spilled(0, 0);
        let at_right_child = spilled(2, 0);
        let (root_slot, child_slot) = (at_root.buffer.slot_idx, at_right_child.buffer.slot_idx);
        evictor.push(at_root);
        evictor.push(at_right_child);

        let first = evictor.evict().unwrap();
        let second = evictor.evict().unwrap();

        assert_eq!(first.slot_idx, child_slot);
        assert_eq!(second.slot_idx, root_slot);
    }

    #[test]
    fn the_shared_handle_evicts_through_the_evict_trait() {
        init_test_free_pool(1);
        let shared: SharedPartitionEvictor = Rc::new(RefCell::new(PartitionEvictor::new()));
        shared.borrow_mut().push(spilled(5, 0));
        let handle: Box<dyn Evict> = Box::new(shared.clone());

        let evicted = handle.evict();

        assert!(evicted.is_some());
    }
}
