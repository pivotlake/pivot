//! Puts the rows of one column of a merged batch into output order and
//! splits them into the leaves Parquet stores.

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef};
use arrow_data::ArrayData;
use dispatch::arrays::{concat_chunks, take_chunked};
use dispatch::memory::SlabAllocator;
use dispatch::{BuildContext, Built, BuiltUnary, OperatorIO, Sender, Unary, UnaryFactory};

use crate::writing::compacting::memory_budget::{MemoryBudget, MemoryReservation};
use crate::writing::compacting::read::output::MergedBatchColumn;
use crate::writing::error::WriteResult;
use crate::writing::leaves::{self, Leaf};
use crate::writing::shredding::{self, FileShredding};

/// The leaves of one column of one merged batch, with their rows in output
/// order.
pub(crate) struct GatheredLeaves {
    /// The output row group the merged batch is in.
    pub(crate) row_group: usize,
    /// The merged batch's place among the merged batches of its row group.
    pub(crate) batch: usize,
    /// How many merged batches that row group consists of.
    pub(crate) row_group_batch_count: usize,
    /// The table column the leaves are of.
    pub(crate) column: usize,
    /// The column's leaves, depth-first. They share nothing with the values
    /// the rows were decoded as.
    pub(crate) leaves: Vec<Leaf>,
    /// What the memory budget counts for the leaves.
    pub(crate) memory: Arc<MemoryReservation>,
}

impl GatheredLeaves {
    /// How many bytes `leaves` hold: the buffers of their values, and their
    /// definition levels. Leaves under the same required field share their
    /// levels, which are counted once.
    fn count_bytes(leaves: &[Leaf]) -> usize {
        let mut counted_levels = HashSet::new();
        let mut bytes = 0;
        for leaf in leaves {
            bytes += Self::count_array_bytes(&leaf.values.to_data());
            if let Some(levels) = &leaf.def_levels
                && counted_levels.insert(levels.as_ptr())
            {
                bytes += std::mem::size_of_val(&levels[..]);
            }
        }
        bytes
    }

    /// How many bytes `array` holds in its buffers, its validity and its
    /// children.
    pub(crate) fn count_array_bytes(array: &ArrayData) -> usize {
        let buffers = array.buffers().iter();
        let buffer_bytes: usize = buffers.map(|buffer| buffer.capacity()).sum();
        let validity_bytes = array.nulls().map_or(0, |nulls| nulls.buffer().capacity());
        let child_bytes: usize = array.child_data().iter().map(Self::count_array_bytes).sum();
        buffer_bytes + validity_bytes + child_bytes
    }
}

/// Turns each [`MergedBatchColumn`] into the leaves of its column, with the
/// rows in output order ([`GatheredLeaves`]). Runs on every worker but the
/// compaction worker, a batch per turn.
///
/// The rows are copied out of the values they were decoded as, so those
/// values, and the pages their strings point into, are given back as soon
/// as the batch is gathered.
///
/// A variant column arrives the way each input file shredded it. Unless
/// that is already the way the output is shredded, the rows are folded back
/// to plain variants, gathered, and shredded anew.
pub(crate) struct LeafGatherer {
    /// The schema the output is written under, and how its variant columns
    /// are shredded.
    shredding: Arc<FileShredding>,
    memory_budget: Arc<MemoryBudget>,
    /// Whether this worker's gatherer takes batches (see
    /// [`list_workers_that_gather_and_encode`](super::list_workers_that_gather_and_encode)).
    takes_batches: bool,
    /// Backs the gathered values. Created on first use, on the worker.
    allocator: Option<SlabAllocator>,
}

impl LeafGatherer {
    /// One gatherer per worker, for output written as `shredding` says.
    /// `takes_batches_per_worker` says for each worker whether its gatherer
    /// takes batches.
    pub(crate) fn create_for_workers(
        shredding: &Arc<FileShredding>,
        memory_budget: &Arc<MemoryBudget>,
        takes_batches_per_worker: &[bool],
    ) -> Vec<Self> {
        takes_batches_per_worker
            .iter()
            .map(|&takes_batches| LeafGatherer {
                shredding: shredding.clone(),
                memory_budget: memory_budget.clone(),
                takes_batches,
                allocator: None,
            })
            .collect()
    }

    /// The rows of `batch` in output order, copied out of `sources`, which
    /// hold them file by file.
    fn gather_into_output_order(
        allocator: &mut SlabAllocator,
        sources: &[ArrayRef],
        batch: &MergedBatchColumn,
    ) -> WriteResult<ArrayRef> {
        Ok(match sources {
            [_] => concat_chunks(allocator, sources)?,
            _ => take_chunked(allocator, sources, &batch.list_rows())?,
        })
    }
}

impl UnaryFactory<MergedBatchColumn, GatheredLeaves> for LeafGatherer {
    type Unary = Self;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self> {
        Built::new(self)
    }
}

impl Unary<MergedBatchColumn, GatheredLeaves> for LeafGatherer {
    fn consume(
        &mut self,
        batch: MergedBatchColumn,
        sender: &mut dyn Sender<GatheredLeaves>,
        _io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        let allocator = self
            .allocator
            .get_or_insert_with(|| SlabAllocator::new(false));
        let field = self.shredding.schema.field(batch.column);
        let sources = batch.sources.iter().map(|source| source.values.clone());
        let is_shredded_as_output = batch
            .sources
            .iter()
            .all(|source| source.values.data_type() == field.data_type());
        let values = if crate::is_variant_field(field) && !is_shredded_as_output {
            let plain_sources = sources
                .map(|source| shredding::unshred_column(&source, allocator))
                .collect::<WriteResult<Vec<_>>>()?;
            let plain_values = Self::gather_into_output_order(allocator, &plain_sources, &batch)?;
            match &self.shredding.column_shredding[batch.column] {
                Some(column_shredding) => {
                    shredding::shred_column(&plain_values, column_shredding, allocator)?
                }
                None => plain_values,
            }
        } else {
            let sources: Vec<ArrayRef> = sources.collect();
            Self::gather_into_output_order(allocator, &sources, &batch)?
        };
        let leaves = leaves::to_parquet_leaves(field, &values)?;
        drop(values);
        // The leaves are counted before the decoded values are given back,
        // which is when `batch` is dropped.
        let memory = self
            .memory_budget
            .reserve(GatheredLeaves::count_bytes(&leaves));
        sender.send(GatheredLeaves {
            row_group: batch.row_group,
            batch: batch.batch,
            row_group_batch_count: batch.row_group_batch_count,
            column: batch.column,
            leaves,
            memory: Arc::new(memory),
        })?;
        Ok(())
    }

    fn ready_for_more_work(&mut self) -> bool {
        self.takes_batches
    }
}
