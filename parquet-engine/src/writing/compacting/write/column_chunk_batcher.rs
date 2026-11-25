//! Puts the encoded leaves of the output in the order of the output file and
//! passes them on a batch of bytes at a time.

use std::collections::BTreeMap;
use std::sync::Arc;

use dispatch::{BuildContext, Built, BuiltUnary, OperatorIO, Sender, Unary, UnaryFactory};

use super::leaf_encoder::EncodedRowGroupLeaf;
use crate::writing::assembler::ColumnChunkRun;
use crate::writing::compacting::memory_budget::{MemoryBudget, MemoryReservation};
use crate::writing::leaves;
use crate::writing::shredding::FileShredding;

/// The next column chunks of a compaction's output, encoded: what a writer
/// appends to an output file. They follow one another in one output row
/// group, and the row groups follow one another.
pub struct EncodedColumnChunks {
    pub chunks: ColumnChunkRun,
    /// Whether these are the last column chunks of their row group.
    pub ends_row_group: bool,
    /// What the compaction's memory budget counts for the column chunks'
    /// bytes. The writer drops it when the bytes are stored, which lets the
    /// compaction read further.
    pub memory: MemoryReservation,
}

/// Takes the [`EncodedRowGroupLeaf`]s, which arrive in any order, and sends
/// them on in the order of the output file: row group after row group, and
/// in a row group leaf after leaf. They go as [`EncodedColumnChunks`], a
/// batch whenever
/// [`write_batch_bytes`](crate::writing::compacting::MergeSortSettings::write_batch_bytes)
/// of them are in order and
/// whenever a row group ends, so a row group's first columns are written
/// while its last are still being read. The pages are moved, not copied.
/// Runs on the write worker, where all encoded leaves are routed.
pub(super) struct ColumnChunkBatcher {
    /// How many leaves the output's schema has: the leaves of a row group.
    leaf_count: usize,
    /// How many bytes of leaves in order make a batch.
    batch_bytes: usize,
    memory_budget: Arc<MemoryBudget>,
    /// The encoded leaves that arrived before one that goes into the file
    /// before them, by output row group and place among its leaves.
    waiting_leaves: BTreeMap<(usize, usize), EncodedRowGroupLeaf>,
    /// The output row group, and the place among its leaves, of the leaf
    /// that goes into the file next.
    next_leaf: (usize, usize),
    /// The leaves that are in order and not sent on yet. They are of one
    /// row group.
    batch: Vec<EncodedRowGroupLeaf>,
}

impl ColumnChunkBatcher {
    /// One batcher per worker, for output written under the schema of
    /// `shredding`, in batches of `batch_bytes` bytes.
    pub(super) fn create_for_workers(
        shredding: &FileShredding,
        batch_bytes: usize,
        memory_budget: &Arc<MemoryBudget>,
        worker_count: usize,
    ) -> Vec<Self> {
        let fields = shredding.schema.fields().iter();
        let leaf_count = fields.map(|field| leaves::count_leaves(field)).sum();
        (0..worker_count)
            .map(|_| ColumnChunkBatcher {
                leaf_count,
                batch_bytes,
                memory_budget: memory_budget.clone(),
                waiting_leaves: BTreeMap::new(),
                next_leaf: (0, 0),
                batch: Vec::new(),
            })
            .collect()
    }
}

impl UnaryFactory<EncodedRowGroupLeaf, EncodedColumnChunks> for ColumnChunkBatcher {
    type Unary = Self;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self> {
        Built::new(self)
    }
}

impl Unary<EncodedRowGroupLeaf, EncodedColumnChunks> for ColumnChunkBatcher {
    fn consume(
        &mut self,
        leaf: EncodedRowGroupLeaf,
        sender: &mut dyn Sender<EncodedColumnChunks>,
        _io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        self.waiting_leaves
            .insert((leaf.row_group, leaf.leaf_index), leaf);
        while let Some(next) = self.waiting_leaves.remove(&self.next_leaf) {
            self.batch.push(next);
            let (row_group, leaf_index) = self.next_leaf;
            let ends_row_group = leaf_index + 1 == self.leaf_count;
            self.next_leaf = match ends_row_group {
                true => (row_group + 1, 0),
                false => (row_group, leaf_index + 1),
            };
            let bytes_in_order: usize = self.batch.iter().map(|leaf| leaf.page_bytes).sum();
            if !ends_row_group && bytes_in_order < self.batch_bytes {
                continue;
            }
            let mut leaf_reservations = Vec::with_capacity(self.batch.len());
            let encoded_leaves = self.batch.drain(..).map(|leaf| {
                leaf_reservations.push(leaf.memory);
                leaf.leaf
            });
            let chunks = ColumnChunkRun::new(row_group as u64, encoded_leaves)?;
            let memory = self
                .memory_budget
                .reserve_encoded_column_chunks(leaf_reservations, ends_row_group);
            sender.send(EncodedColumnChunks {
                chunks,
                ends_row_group,
                memory,
            })?;
        }
        Ok(())
    }
}
