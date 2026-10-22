//! Per-partition storage for rows that may have to be spilled.
//!
//! Each partition appends rows into a slab of its own; a [`Layout`] says how
//! big a row is and how many rows a slab holds. Slabs are allocated out of
//! 2MB ring buffers that the partitions of a group share,
//! laid out as a [`PartitionTree`]: every group starts at the root buffer, and
//! a node whose buffer is full hands its groups to its children, so buffers
//! come to be shared by fewer partitions as the data grows.
//!
//! A buffer is sealed, and can be spilled, once no partition is still writing
//! into any slab of it. That is read off the buffer's reference count: every slab holds its
//! buffer, and the allocator records every slab it has closed. A partition's
//! open slab is the only other holder, so a buffer referenced by exactly its
//! closed slabs and the allocator's own record has no open slab left.

use crate::io::{IORequesterError, OpenFile};
use crate::memory::spill::layout::Layout;
use crate::memory::spill::partition_tree::{
    PARTITIONS, PartitionTree, group_of_partition, is_leaf,
};
use crate::memory::spill::{SpillWrite, SpilledBuffer, SpilledSlab, Spiller};
use crate::memory::{BUFFER_SIZE, FileBytes, FrozenSlab, Slab, WriteBuffer, memory_ctx};
use std::ops::ControlFlow;
use std::sync::Arc;

/// The slab a partition is appending into, and where it came from, so that
/// once full it can be recorded against its buffer.
struct OpenSlab {
    slab: Slab,
    /// Rows written so far.
    len: usize,
    /// Rows the slab holds when full.
    rows: usize,
    /// The tree node of the partition group slot the slab was allocated from.
    partition_group_slot: usize,
    /// The ring slot of that buffer, which identifies it within the node.
    buffer_slot_idx: usize,
}

struct Partition {
    open_slab: Option<OpenSlab>,
    /// Slabs this partition has filled, which the layout sizes the next by.
    slabs_filled: usize,
    /// The file this partition's spilled rows are appended to, once it has
    /// spilled for the first time.
    spill_file: Option<OpenFile>,
    /// Bytes appended to the spill file so far, which is where the next
    /// spilled slab's rows will begin in it.
    bytes_spilled: u64,
}

/// A full slab, recorded against its buffer until the buffer is spilled. Full
/// means no one writes it again, so it is frozen.
struct ClosedSlab {
    partition: usize,
    slab: FrozenSlab,
}

/// A buffer the partitions under one tree node share, and the full slabs taken
/// from it so far.
struct PartitionGroupBuffer {
    buffer: Arc<WriteBuffer>,
    closed_slabs: Vec<ClosedSlab>,
}

impl PartitionGroupBuffer {
    fn new(buffer: Arc<WriteBuffer>) -> Self {
        Self {
            buffer,
            closed_slabs: Vec::new(),
        }
    }

    /// Whether no partition is still writing into this buffer. Each closed
    /// slab holds one reference and this record holds one; any further
    /// reference is a partition's open slab, or the node still allocating
    /// from the buffer.
    fn is_sealed(&self) -> bool {
        Arc::strong_count(&self.buffer) == self.closed_slabs.len() + 1
    }
}

/// One tree node: the buffers of the partition groups that point at it, and
/// whether slabs are still being allocated from it.
enum PartitionGroupSlot {
    /// No partition under this node has needed a slab yet.
    Unused,
    /// Slabs are allocated from `current`, which is also the last of
    /// `buffers`, from `next_allocation_offset` onwards.
    Allocating {
        current: Arc<WriteBuffer>,
        next_allocation_offset: usize,
        buffers: Vec<PartitionGroupBuffer>,
    },
    /// An inner node whose buffer filled up, so its groups moved to its
    /// children. It keeps its buffers until they are spilled and evicted.
    Split { buffers: Vec<PartitionGroupBuffer> },
}

/// A slab and the ring slot of the buffer it was allocated from.
struct AllocatedSlab {
    slab: Slab,
    buffer_slot_idx: usize,
}

/// What a partition group slot answered when asked for a slab.
enum SlabAllocation {
    Allocated(AllocatedSlab),
    /// The slot's current buffer has no room for another slab.
    BufferFull,
}

impl PartitionGroupSlot {
    fn buffers_mut(&mut self) -> &mut Vec<PartitionGroupBuffer> {
        match self {
            PartitionGroupSlot::Allocating { buffers, .. }
            | PartitionGroupSlot::Split { buffers } => buffers,
            PartitionGroupSlot::Unused => {
                unreachable!("a slab was allocated from an unused partition group slot")
            }
        }
    }

    /// Allocate `size` bytes aligned to `align` from the current buffer,
    /// taking a first buffer if the slot has none yet.
    fn allocate_slab(&mut self, size: usize, align: usize) -> SlabAllocation {
        if let PartitionGroupSlot::Unused = self {
            self.start_new_buffer();
        }
        let PartitionGroupSlot::Allocating {
            current,
            next_allocation_offset,
            ..
        } = self
        else {
            unreachable!("groups only point at partition group slots that are allocating")
        };
        let start = next_allocation_offset.next_multiple_of(align);
        let end = start + size;
        if end > BUFFER_SIZE {
            return SlabAllocation::BufferFull;
        }
        *next_allocation_offset = end;
        // `start..end` lies inside the 2MB buffer, checked just above.
        let ptr = unsafe { current.ptr.add(start) };
        SlabAllocation::Allocated(AllocatedSlab {
            slab: Slab::new(ptr, size, current.clone()),
            buffer_slot_idx: current.slot_idx,
        })
    }

    /// Take a fresh buffer and allocate `size` bytes aligned to `align` from
    /// it. For a slot whose current buffer is full, or has none yet.
    fn allocate_slab_from_new_buffer(&mut self, size: usize, align: usize) -> AllocatedSlab {
        self.start_new_buffer();
        match self.allocate_slab(size, align) {
            SlabAllocation::Allocated(allocated) => allocated,
            SlabAllocation::BufferFull => {
                unreachable!("a slab that fits in a buffer fits in a fresh one")
            }
        }
    }

    /// Start allocating from a fresh buffer, as the slot's first or in place of
    /// a full current one.
    fn start_new_buffer(&mut self) {
        let fresh = Arc::new(memory_ctx().get_write_buffer(false));
        match self {
            PartitionGroupSlot::Unused => {
                *self = PartitionGroupSlot::Allocating {
                    buffers: vec![PartitionGroupBuffer::new(fresh.clone())],
                    current: fresh,
                    next_allocation_offset: 0,
                };
            }
            PartitionGroupSlot::Allocating {
                current,
                next_allocation_offset,
                buffers,
            } => {
                buffers.push(PartitionGroupBuffer::new(fresh.clone()));
                *current = fresh;
                *next_allocation_offset = 0;
            }
            PartitionGroupSlot::Split { .. } => {
                unreachable!("a split partition group slot never allocates again")
            }
        }
    }

    /// Stop allocating here for good, letting go of the current buffer so it
    /// can become sealed. For an inner node whose buffer is full, before
    /// its groups move to its children.
    fn stop_allocating(&mut self) {
        let PartitionGroupSlot::Allocating { buffers, .. } = self else {
            unreachable!("only a partition group slot that is allocating can stop")
        };
        *self = PartitionGroupSlot::Split {
            buffers: std::mem::take(buffers),
        };
    }
}

/// A slab of one partition's rows, as handed out when the allocator is taken
/// apart for reading.
pub struct PartitionSlab {
    pub slab: FrozenSlab,
    /// Rows written into the slab: all it holds for a closed slab, fewer for
    /// the one a partition was still filling.
    pub rows: usize,
}

/// One partition's rows, as handed out when the allocator is taken apart:
/// the slabs still in the allocator, and the file its spilled rows went to,
/// if it ever spilled.
pub struct PartitionContents {
    pub slabs: Vec<PartitionSlab>,
    pub spill_file: Option<OpenFile>,
}

/// Stores rows by partition, in slabs allocated from buffers the partitions
/// of a group share. See the module documentation.
pub struct PartitionAllocator<L: Layout> {
    layout: L,
    tree: PartitionTree<PartitionGroupSlot>,
    partitions: Box<[Partition]>,
}

impl<L: Layout> PartitionAllocator<L> {
    pub fn new(layout: L) -> Self {
        Self {
            layout,
            tree: PartitionTree::new(|| PartitionGroupSlot::Unused),
            partitions: std::iter::repeat_with(|| Partition {
                open_slab: None,
                slabs_filled: 0,
                spill_file: None,
                bytes_spilled: 0,
            })
            .take(PARTITIONS)
            .collect(),
        }
    }

    /// Reserve the next row of `partition` for the caller to fill in place.
    /// The slab the row is in is closed as soon as this row fills it, so the
    /// buffer is let go of at once rather than when the partition next
    /// receives a row; the row itself stays valid, the ring memory does not
    /// move.
    pub fn reserve(&mut self, partition: usize) -> &mut L::Row {
        if self.partitions[partition].open_slab.is_none() {
            let slab = self.allocate_slab_for(partition);
            self.partitions[partition].open_slab = Some(slab);
        }
        let open = self.partitions[partition]
            .open_slab
            .as_mut()
            .expect("a partition has an open slab right after one was allocated for it");
        // The slab holds `rows` rows of the layout's size, aligned for them.
        let row = unsafe {
            open.slab
                .as_mut_slice()
                .as_mut_ptr()
                .add(open.len * self.layout.row_size())
        };
        open.len += 1;
        if open.len == open.rows {
            let full = self.partitions[partition]
                .open_slab
                .take()
                .expect("the slab the row was just reserved in is still open");
            self.close(partition, full);
        }
        unsafe { self.layout.row_mut(row) }
    }

    /// Take the allocator apart into every partition's contents, indexed by
    /// partition, for reading once no more rows will be written. The slabs
    /// here were never spilled; spilled slabs are the evictor's.
    pub fn into_partitions(self) -> Vec<PartitionContents> {
        let row_size = self.layout.row_size();
        let mut slabs_by_partition: Vec<Vec<PartitionSlab>> =
            (0..PARTITIONS).map(|_| Vec::new()).collect();
        for partition_group_slot in self.tree.into_nodes() {
            let buffers = match partition_group_slot {
                PartitionGroupSlot::Allocating { buffers, .. }
                | PartitionGroupSlot::Split { buffers } => buffers,
                PartitionGroupSlot::Unused => continue,
            };
            for closed_slab in buffers.into_iter().flat_map(|buffer| buffer.closed_slabs) {
                slabs_by_partition[closed_slab.partition].push(PartitionSlab {
                    rows: closed_slab.slab.size() / row_size,
                    slab: closed_slab.slab,
                });
            }
        }
        self.partitions
            .into_vec()
            .into_iter()
            .zip(slabs_by_partition)
            .map(|(partition, mut slabs)| {
                if let Some(open_slab) = partition.open_slab
                    && open_slab.len > 0
                {
                    slabs.push(PartitionSlab {
                        slab: open_slab.slab.freeze(),
                        rows: open_slab.len,
                    });
                }
                PartitionContents {
                    slabs,
                    spill_file: partition.spill_file,
                }
            })
            .collect()
    }

    /// Spill the next sealed buffer through `spiller`, one write per partition
    /// with rows in it, and return it for the caller to keep until it is
    /// evicted: its rows stay readable in memory until then. `None` when no
    /// buffer is sealed. Buffers are taken in the tree's spill order (see
    /// [`PartitionTree::walk_spill_order`]), oldest first within a node. A
    /// node's current buffer is never sealed, so it is never taken.
    ///
    /// A partition spilling for the first time gets its file opened through
    /// `spiller` here. Should that fail, the buffer stays where it was.
    pub fn spill(
        &mut self,
        spiller: &mut Spiller,
    ) -> Result<Option<SpilledBuffer>, IORequesterError> {
        // Where the sealed buffer is: its partition group slot in the tree, and
        // its position among that slot's buffers.
        let sealed_buffer_location =
            self.tree
                .walk_spill_order(|partition_group_slot_idx, partition_group_slot| {
                    let slot_buffers = match partition_group_slot {
                        PartitionGroupSlot::Allocating { buffers, .. }
                        | PartitionGroupSlot::Split { buffers } => buffers,
                        PartitionGroupSlot::Unused => return ControlFlow::Continue(()),
                    };
                    match slot_buffers
                        .iter()
                        .position(PartitionGroupBuffer::is_sealed)
                    {
                        Some(sealed_buffer_idx) => {
                            ControlFlow::Break((partition_group_slot_idx, sealed_buffer_idx))
                        }
                        None => ControlFlow::Continue(()),
                    }
                });
        let ControlFlow::Break((partition_group_slot_idx, sealed_buffer_idx)) =
            sealed_buffer_location
        else {
            return Ok(None);
        };
        let slot_buffers = self.tree.node_mut(partition_group_slot_idx).buffers_mut();
        for closed_slab in &slot_buffers[sealed_buffer_idx].closed_slabs {
            let partition = &mut self.partitions[closed_slab.partition];
            if partition.spill_file.is_none() {
                partition.spill_file = Some(spiller.open_spill_file(closed_slab.partition)?);
            }
        }
        let PartitionGroupBuffer {
            buffer,
            mut closed_slabs,
        } = slot_buffers.remove(sealed_buffer_idx);

        // Each partition's rows go out as one write, so bring its slabs
        // together. Sorting is stable, so a partition's slabs keep the order
        // they were filled in, which is the order they land in its file.
        closed_slabs.sort_by_key(|closed_slab| closed_slab.partition);
        let mut writes: Vec<SpillWrite> = Vec::new();
        let mut spilled_slabs: Vec<SpilledSlab> = Vec::with_capacity(closed_slabs.len());
        let mut partition_being_gathered = None;
        for ClosedSlab { partition, slab } in closed_slabs {
            if partition_being_gathered != Some(partition) {
                partition_being_gathered = Some(partition);
                writes.push(SpillWrite {
                    file: self.partitions[partition]
                        .spill_file
                        .clone()
                        .expect("every partition in the buffer had its file opened just above"),
                    rows: FileBytes::new(),
                });
            }
            // The write gets its own handle to the rows; the buffer keeps this one.
            writes
                .last_mut()
                .expect("a write was started for this partition just above")
                .rows
                .push_frozen_slab(slab.clone());
            let file_offset = self.partitions[partition].bytes_spilled;
            self.partitions[partition].bytes_spilled += slab.size() as u64;
            spilled_slabs.push(SpilledSlab {
                spill_partition_id: partition,
                file_offset,
                slab,
            });
        }
        let writes_in_flight = spiller.spill(writes)?;
        Ok(Some(SpilledBuffer {
            writes_in_flight,
            buffer,
            slabs: spilled_slabs,
            partition_group_slot: partition_group_slot_idx,
        }))
    }

    /// Close a full slab, recording it against the buffer it was allocated from.
    fn close(&mut self, partition: usize, full: OpenSlab) {
        assert_eq!(full.len, full.rows, "only full slabs are closed");
        self.partitions[partition].slabs_filled += 1;
        let group_buffer = self
            .tree
            .node_mut(full.partition_group_slot)
            .buffers_mut()
            .iter_mut()
            .rev()
            .find(|group_buffer| group_buffer.buffer.slot_idx == full.buffer_slot_idx)
            .expect(
                "a slab's buffer stays recorded at its partition group slot until it is spilled",
            );
        group_buffer.closed_slabs.push(ClosedSlab {
            partition,
            slab: full.slab.freeze(),
        });
    }

    /// Allocate a fresh slab for `partition` from its group's slot. When the
    /// slot's current buffer is full, a leaf takes a new buffer, and an inner
    /// node hands its groups to its children and the slab comes from the
    /// child's first buffer. A slab fits in one buffer, so a fresh buffer
    /// always has room for it.
    fn allocate_slab_for(&mut self, partition: usize) -> OpenSlab {
        let group = group_of_partition(partition);
        let rows = self
            .layout
            .slab_rows(self.partitions[partition].slabs_filled);
        let size = rows * self.layout.row_size();
        let align = self.layout.row_align();
        assert!(
            size <= BUFFER_SIZE,
            "a slab of {rows} rows must fit in one buffer"
        );
        let node = self.tree.node_of_group(group);
        let (node, allocated) = match self.tree.node_mut(node).allocate_slab(size, align) {
            SlabAllocation::Allocated(allocated) => (node, allocated),
            SlabAllocation::BufferFull if is_leaf(node) => {
                let allocated = self
                    .tree
                    .node_mut(node)
                    .allocate_slab_from_new_buffer(size, align);
                (node, allocated)
            }
            SlabAllocation::BufferFull => {
                self.tree.node_mut(node).stop_allocating();
                self.tree.split(node);
                let child = self.tree.node_of_group(group);
                let allocated = self
                    .tree
                    .node_mut(child)
                    .allocate_slab_from_new_buffer(size, align);
                (child, allocated)
            }
        };
        OpenSlab {
            slab: allocated.slab,
            len: 0,
            rows,
            partition_group_slot: node,
            buffer_slot_idx: allocated.buffer_slot_idx,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WriteRequestId;
    use crate::io::{Completion, IORequester, OperatorIO};
    use crate::memory::init_test_free_pool;
    use crate::memory::spill::directory;
    use crate::memory::spill::layout::{Dynamic, Fixed};
    use crate::memory::spill::partition_tree::PARTITIONS_PER_GROUP;
    use crate::stats::StatsCollector;
    use crate::worker::WORKER_IDX;
    use ahash::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// What a `Spiller` borrows, under a dataflow id of the test's own so its
    /// spill files clash with no other test's.
    struct SpillerParts {
        dataflow: usize,
        requester: IORequester,
        stats: StatsCollector,
        next_request_id: usize,
        writes_in_flight: HashMap<WriteRequestId, Arc<AtomicUsize>>,
    }

    impl SpillerParts {
        fn new(dataflow: usize) -> Self {
            Self {
                dataflow,
                requester: IORequester::default(),
                stats: StatsCollector::disabled(),
                next_request_id: 0,
                writes_in_flight: HashMap::default(),
            }
        }

        fn spiller(&mut self) -> Spiller<'_> {
            let io = OperatorIO::new(
                &mut self.requester,
                self.dataflow,
                0,
                &mut self.stats,
                &mut self.next_request_id,
            );
            Spiller::new(io, &mut self.writes_in_flight)
        }

        /// Complete every spill write issued so far, counting each off its
        /// buffer as the worker does.
        fn finish_writes(&mut self) {
            while !self.writes_in_flight.is_empty() {
                self.requester.wait().unwrap();
                for completion in self.requester.completions().unwrap() {
                    if let Ok(Completion::FsWrite(write)) = completion {
                        let count = self.writes_in_flight.remove(&write.request.id).unwrap();
                        count.fetch_sub(1, Ordering::Release);
                    }
                }
            }
        }

        fn file_len(&self, partition: usize) -> u64 {
            std::fs::metadata(directory::partition_file_path(
                self.dataflow,
                WORKER_IDX.get(),
                0,
                partition,
            ))
            .unwrap()
            .len()
        }
    }

    impl Drop for SpillerParts {
        fn drop(&mut self) {
            self.finish_writes();
            directory::remove_worker_directory(self.dataflow, WORKER_IDX.get()).unwrap();
        }
    }

    const ROWS_PER_SLAB: usize = 64;

    /// `u64` rows in slabs of `ROWS_PER_SLAB`.
    fn small_rows() -> PartitionAllocator<Fixed<u64>> {
        PartitionAllocator::new(Fixed::new(ROWS_PER_SLAB * 8))
    }

    /// A row large enough that a buffer holds only four slabs of
    /// `ROWS_PER_SLAB`, so a few reserves fill it. Only its size and the
    /// bytes it leaves in memory matter.
    #[derive(Clone, Copy)]
    struct LargeRow {
        _bytes: [u8; BUFFER_SIZE / (4 * ROWS_PER_SLAB)],
    }

    fn large_row(fill: u8) -> LargeRow {
        LargeRow {
            _bytes: [fill; BUFFER_SIZE / (4 * ROWS_PER_SLAB)],
        }
    }

    fn large_rows() -> PartitionAllocator<Fixed<LargeRow>> {
        PartitionAllocator::new(Fixed::new(BUFFER_SIZE / 4))
    }

    fn buffers_of_slot(
        allocator: &mut PartitionAllocator<impl Layout>,
        node: usize,
    ) -> &Vec<PartitionGroupBuffer> {
        allocator.tree.node_mut(node).buffers_mut()
    }

    /// Reserve a row in the first partition of each of five groups: four
    /// slabs fill the root buffer, and the fifth finds it full and splits the
    /// root.
    fn fill_and_split_root(allocator: &mut PartitionAllocator<Fixed<LargeRow>>) {
        for group in 0..5 {
            allocator
                .reserve(group * PARTITIONS_PER_GROUP)
                .write(large_row(1));
        }
    }

    /// Fill the four slabs in the root buffer, so it becomes sealed.
    fn fill_root_slabs(allocator: &mut PartitionAllocator<Fixed<LargeRow>>) {
        for group in 0..4 {
            for _ in 1..ROWS_PER_SLAB {
                allocator
                    .reserve(group * PARTITIONS_PER_GROUP)
                    .write(large_row(1));
            }
        }
    }

    #[test]
    fn a_partition_fills_one_slab_before_starting_the_next() {
        init_test_free_pool(1);
        let mut allocator = small_rows();

        for value in 0..ROWS_PER_SLAB as u64 + 1 {
            allocator.reserve(3).write(value);
        }

        let closed = &buffers_of_slot(&mut allocator, 0)[0].closed_slabs;
        assert_eq!(closed.len(), 1);
        assert_eq!(closed[0].partition, 3);
        let values: Vec<u64> = closed[0]
            .slab
            .as_slice()
            .chunks_exact(8)
            .map(|bytes| u64::from_ne_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(values, (0..ROWS_PER_SLAB as u64).collect::<Vec<_>>());
        let open = allocator.partitions[3].open_slab.as_ref().unwrap();
        assert_eq!(open.len, 1);
    }

    #[test]
    fn a_full_slab_is_closed_on_the_reserve_that_fills_it() {
        init_test_free_pool(1);
        let mut allocator = small_rows();

        for value in 0..ROWS_PER_SLAB as u64 {
            allocator.reserve(0).write(value);
        }

        assert!(allocator.partitions[0].open_slab.is_none());
        assert_eq!(allocator.partitions[0].slabs_filled, 1);
        assert_eq!(buffers_of_slot(&mut allocator, 0)[0].closed_slabs.len(), 1);
    }

    #[test]
    fn a_dynamic_layout_hands_out_byte_rows_in_growing_slabs() {
        init_test_free_pool(1);
        let layout = Dynamic::new(24, 8, 2 * 24, 8 * 24);
        let mut allocator = PartitionAllocator::new(layout);

        for row in 0..6u8 {
            allocator.reserve(0).fill(row);
        }

        let closed = &buffers_of_slot(&mut allocator, 0)[0].closed_slabs;
        let slab_sizes: Vec<usize> = closed.iter().map(|s| s.slab.size()).collect();
        assert_eq!(slab_sizes, [2 * 24, 4 * 24]);
        assert_eq!(closed[1].slab.as_slice()[0..24], [2u8; 24]);
        assert_eq!(closed[1].slab.as_slice()[72..96], [5u8; 24]);
        assert!(allocator.partitions[0].open_slab.is_none());
    }

    #[test]
    fn partitions_of_every_group_share_the_root_buffer_at_first() {
        init_test_free_pool(1);
        let mut allocator = small_rows();

        allocator.reserve(0).write(1);
        allocator.reserve(PARTITIONS - 1).write(2);

        assert_eq!(buffers_of_slot(&mut allocator, 0).len(), 1);
        let first = allocator.partitions[0].open_slab.as_ref().unwrap();
        let last = allocator.partitions[PARTITIONS - 1]
            .open_slab
            .as_ref()
            .unwrap();
        assert_eq!(first.buffer_slot_idx, last.buffer_slot_idx);
    }

    #[test]
    fn an_inner_node_whose_buffer_is_full_splits_its_groups_between_its_children() {
        init_test_free_pool(3);
        let mut allocator = large_rows();
        fill_and_split_root(&mut allocator);

        allocator.reserve(PARTITIONS - 1).write(large_row(2));

        assert_eq!(allocator.tree.node_of_group(0), 1);
        assert_eq!(allocator.tree.node_of_group(63), 2);
        assert!(matches!(
            allocator.tree.node_mut(0),
            PartitionGroupSlot::Split { .. }
        ));
        let fifth_group = allocator.partitions[4 * PARTITIONS_PER_GROUP]
            .open_slab
            .as_ref()
            .unwrap();
        let last_group = allocator.partitions[PARTITIONS - 1]
            .open_slab
            .as_ref()
            .unwrap();
        assert_eq!(fifth_group.partition_group_slot, 1);
        assert_eq!(last_group.partition_group_slot, 2);
        assert_ne!(fifth_group.buffer_slot_idx, last_group.buffer_slot_idx);
    }

    #[test]
    fn a_leaf_whose_buffer_is_full_moves_to_a_new_buffer() {
        init_test_free_pool(16);
        let mut allocator = large_rows();
        // Partition 0 alone: four slabs per buffer, one buffer per inner node
        // on the way down to leaf 63, then two buffers at the leaf.
        let inner_nodes_on_path = 6;
        let slabs = 4 * inner_nodes_on_path + 5;

        for _ in 0..slabs * ROWS_PER_SLAB {
            allocator.reserve(0).write(large_row(3));
        }

        assert_eq!(allocator.tree.node_of_group(0), 63);
        assert_eq!(buffers_of_slot(&mut allocator, 63).len(), 2);
    }

    #[test]
    fn a_buffer_is_sealed_only_once_every_slab_in_it_is_full() {
        init_test_free_pool(3);
        let mut allocator = large_rows();
        fill_and_split_root(&mut allocator);
        let before = buffers_of_slot(&mut allocator, 0)[0].is_sealed();

        fill_root_slabs(&mut allocator);

        let after = buffers_of_slot(&mut allocator, 0)[0].is_sealed();
        assert!(!before);
        assert!(after);
    }

    #[test]
    fn a_buffer_still_being_allocated_from_is_never_sealed() {
        init_test_free_pool(1);
        let mut allocator = small_rows();

        for value in 0..ROWS_PER_SLAB as u64 {
            allocator.reserve(0).write(value);
        }

        assert!(!buffers_of_slot(&mut allocator, 0)[0].is_sealed());
    }

    #[test]
    fn spill_writes_a_sealed_buffer_once_with_one_write_per_partition() {
        init_test_free_pool(3);
        let mut parts = SpillerParts::new(9_000_010);
        let mut allocator = large_rows();
        fill_and_split_root(&mut allocator);
        fill_root_slabs(&mut allocator);

        let spilled = allocator.spill(&mut parts.spiller()).unwrap().unwrap();
        let again = allocator.spill(&mut parts.spiller()).unwrap();
        let writes_issued = spilled.writes_in_flight.load(Ordering::Acquire);
        parts.finish_writes();

        assert_eq!(writes_issued, 4);
        assert_eq!(spilled.writes_in_flight.load(Ordering::Acquire), 0);
        let partitions: Vec<usize> = spilled.slabs.iter().map(|s| s.spill_partition_id).collect();
        assert_eq!(partitions, [0, 64, 128, 192]);
        assert!(spilled.slabs.iter().all(|s| s.file_offset == 0));
        assert!(spilled.slabs.iter().all(|s| s.slab.as_slice()[0] == 1));
        // The writes have landed and let go; the buffer and its slabs remain.
        assert_eq!(Arc::strong_count(&spilled.buffer), 1 + spilled.slabs.len());
        assert!(again.is_none());
        assert!(buffers_of_slot(&mut allocator, 0).is_empty());
        for group in 0..4 {
            assert_eq!(
                parts.file_len(group * PARTITIONS_PER_GROUP),
                (BUFFER_SIZE / 4) as u64
            );
        }
    }

    #[test]
    fn spill_opens_a_partitions_file_the_first_time_and_reuses_it_after() {
        init_test_free_pool(8);
        let mut parts = SpillerParts::new(9_000_011);
        let mut allocator = large_rows();
        fill_and_split_root(&mut allocator);
        fill_root_slabs(&mut allocator);
        allocator.spill(&mut parts.spiller()).unwrap().unwrap();
        let file_after_first = allocator.partitions[0].spill_file.clone().unwrap();
        // Partition 0's group moved to the root's left child, whose buffer
        // already holds another partition's open slab, so three more slabs
        // fill it. The four after those fill the next node's buffer by
        // themselves, and one more row moves on, leaving that buffer sealed.
        for _ in 0..(3 + 4) * ROWS_PER_SLAB + 1 {
            allocator.reserve(0).write(large_row(2));
        }

        let second = allocator.spill(&mut parts.spiller()).unwrap().unwrap();
        parts.finish_writes();

        assert_eq!(second.writes_in_flight.load(Ordering::Acquire), 0);
        assert_eq!(
            allocator.partitions[0].spill_file.clone().unwrap(),
            file_after_first
        );
        // The second spill's slabs follow the first spill's one slab in the file.
        let slab = (BUFFER_SIZE / 4) as u64;
        let offsets: Vec<u64> = second.slabs.iter().map(|s| s.file_offset).collect();
        assert_eq!(offsets, [slab, 2 * slab, 3 * slab, 4 * slab]);
        // The file was opened for appending, so on Linux the second spill's
        // four slabs land after the first spill's one. macOS honors the offset
        // the write names instead, so there the second write overwrites the
        // first and only its four slabs remain.
        #[cfg(target_os = "linux")]
        assert_eq!(parts.file_len(0), (5 * BUFFER_SIZE / 4) as u64);
        #[cfg(not(target_os = "linux"))]
        assert_eq!(parts.file_len(0), BUFFER_SIZE as u64);
        assert!(allocator.partitions[64].spill_file.is_some());
        assert!(allocator.partitions[1].spill_file.is_none());
    }

    #[test]
    fn spill_takes_a_deeper_buffer_before_the_root() {
        init_test_free_pool(8);
        let mut parts = SpillerParts::new(9_000_012);
        let mut allocator = large_rows();
        fill_and_split_root(&mut allocator);
        fill_root_slabs(&mut allocator);
        // Do the same to the root's right child, which groups 32..64 moved to.
        for group in 32..37 {
            allocator
                .reserve(group * PARTITIONS_PER_GROUP)
                .write(large_row(2));
        }
        for group in 32..36 {
            for _ in 1..ROWS_PER_SLAB {
                allocator
                    .reserve(group * PARTITIONS_PER_GROUP)
                    .write(large_row(2));
            }
        }

        allocator.spill(&mut parts.spiller()).unwrap().unwrap();
        let after_first: Vec<bool> = [32 * PARTITIONS_PER_GROUP, 0]
            .iter()
            .map(|&partition| allocator.partitions[partition].spill_file.is_some())
            .collect();
        allocator.spill(&mut parts.spiller()).unwrap().unwrap();
        let after_second = allocator.partitions[0].spill_file.is_some();

        assert_eq!(after_first, [true, false]);
        assert!(after_second);
    }
}
