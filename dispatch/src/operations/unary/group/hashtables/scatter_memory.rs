//! The memory the radix scatter writes into, shared so that it can be spilled and evicted
//! without going through the group-by operator.
//!
//! The aggregated table fills the rows. The worker, between operator steps, spills a full
//! buffer when its node is over the spill threshold. The ring, from inside an allocation,
//! takes a spilled buffer back when it has nothing else left to evict. The last of these can
//! happen in the middle of a scatter, so what it touches is kept apart from what the scatter
//! borrows (see [`crate::memory::Allocator`]).

use super::hash_table::PersistedKey;
use super::scatter::StridedScatterRows;
use crate::memory::{Allocator, PartitionAllocator, SpillBuffer, Spiller, WriteBuffer};
use crate::operations::unary::group::values::AggregationValue;
use std::cell::{Cell, RefCell, RefMut};
use std::collections::VecDeque;
use std::io;
use std::rc::Rc;
use std::sync::Arc;

/// The rows being filled and the allocator their chunks come from.
pub struct PartitionRows<KP: PersistedKey, V: AggregationValue + ?Sized> {
    pub allocator: PartitionAllocator,
    /// One rows buffer per partition, from the radix switch until the flush hands them to
    /// the merge.
    pub rows: Option<Vec<StridedScatterRows<KP, V>>>,
}

/// Everything the radix scatter writes into. See the module documentation.
pub struct ScatterMemory<KP: PersistedKey, V: AggregationValue + ?Sized> {
    /// Borrowed by the scatter while it fills rows, and by a spill to finish the chunks
    /// still being filled in the buffer it writes out. Never borrowed by an eviction.
    filling: RefCell<PartitionRows<KP, V>>,
    /// Buffers handed to the spiller and not yet evicted, in the order they were spilled.
    spilled: RefCell<VecDeque<Arc<SpillBuffer>>>,
    /// How many buffers were evicted, so the output knows rows are missing from memory.
    evicted_count: Cell<usize>,
}

impl<KP: PersistedKey, V: AggregationValue + ?Sized> ScatterMemory<KP, V> {
    pub fn new(partition_count: usize) -> Rc<Self> {
        Rc::new(Self {
            filling: RefCell::new(PartitionRows {
                allocator: PartitionAllocator::new(partition_count),
                rows: None,
            }),
            spilled: RefCell::new(VecDeque::new()),
            evicted_count: Cell::new(0),
        })
    }

    /// The rows, for the scatter to fill.
    pub fn partition_rows(&self) -> RefMut<'_, PartitionRows<KP, V>> {
        self.filling.borrow_mut()
    }

    /// Hands the rows to the merge, with no chunk left being filled.
    pub fn take_rows(&self) -> Option<Vec<StridedScatterRows<KP, V>>> {
        let mut rows = self.filling.borrow_mut().rows.take()?;
        for partition in &mut rows {
            partition.finish_current_chunk();
        }
        Some(rows)
    }

    /// Whether any buffer's bytes were taken out of memory, leaving rows on disk only.
    pub fn has_evicted(&self) -> bool {
        self.evicted_count.get() > 0
    }
}

impl<KP: PersistedKey, V: AggregationValue + ?Sized> Allocator for ScatterMemory<KP, V> {
    fn spill(&self, spiller: &mut dyn Spiller) -> io::Result<bool> {
        let mut filling = self.filling.borrow_mut();
        let Some(full) = filling.allocator.take_full_buffer() else {
            return Ok(false);
        };
        // After the flush the rows belong to the merge, with every chunk already finished.
        if let Some(rows) = &mut filling.rows {
            for partition in &full.partitions {
                rows[usize::from(*partition)].finish_chunk_in(&full.buffer);
            }
        }
        spiller.push(full.buffer.clone())?;
        self.spilled.borrow_mut().push_back(full.buffer);
        Ok(true)
    }

    fn evict(&self) -> Option<WriteBuffer> {
        let mut spilled = self.spilled.borrow_mut();
        for i in 0..spilled.len() {
            if let Some(buffer) = spilled[i].evict() {
                spilled.remove(i);
                self.evicted_count.set(self.evicted_count.get() + 1);
                return Some(buffer);
            }
        }
        None
    }
}
