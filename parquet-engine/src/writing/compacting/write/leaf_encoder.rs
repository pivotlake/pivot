//! Encodes one leaf of one output row group as Parquet pages.

use std::sync::Arc;

use dispatch::arrays::concat_chunks;
use dispatch::memory::SlabAllocator;
use dispatch::{BuildContext, Built, BuiltUnary, OperatorIO, Sender, Unary, UnaryFactory};

use super::leaf_collector::RowGroupLeaf;
use crate::writing::compacting::memory_budget::{MemoryBudget, MemoryReservation};
use crate::writing::encoder;
use crate::writing::error::WriteError;
use crate::writing::leaves::Leaf;
use crate::writing::types::EncodedLeaf;

/// The encoded pages of one leaf of one output row group.
pub(crate) struct EncodedRowGroupLeaf {
    /// The output row group the leaf is of.
    pub(crate) row_group: usize,
    /// The leaf's place among the leaves of the output's schema.
    pub(crate) leaf_index: usize,
    pub(crate) leaf: EncodedLeaf,
    /// How many bytes the pages take.
    pub(crate) page_bytes: usize,
    /// What the memory budget counts for the pages.
    pub(crate) memory: MemoryReservation,
}

/// Encodes each [`RowGroupLeaf`] with the encoder every write uses
/// ([`encoder::encode_leaf`]). Runs on every worker but the compaction
/// worker, a leaf per turn.
///
/// The encoder takes a leaf's values as one array, so the leaf's chunks are
/// copied into one first. The copy is counted before it is made and until
/// the leaf is encoded, and the chunks are let go of once it is made.
pub(crate) struct RowGroupLeafEncoder {
    memory_budget: Arc<MemoryBudget>,
    /// Whether this worker's encoder takes leaves (see
    /// [`list_workers_that_gather_and_encode`](super::list_workers_that_gather_and_encode)).
    takes_leaves: bool,
    /// Backs the copy of a leaf's values. Created on first use, on the
    /// worker.
    value_allocator: Option<SlabAllocator>,
    /// Backs the encoded pages. They have blocks of their own: a page lives
    /// until its row group is stored, and in a block it shared with values
    /// it would keep the whole block in memory long after the values are
    /// gone.
    page_allocator: Option<SlabAllocator>,
}

impl RowGroupLeafEncoder {
    /// One encoder per worker. `takes_leaves_per_worker` says for each
    /// worker whether its encoder takes leaves.
    pub(crate) fn create_for_workers(
        memory_budget: &Arc<MemoryBudget>,
        takes_leaves_per_worker: &[bool],
    ) -> Vec<Self> {
        takes_leaves_per_worker
            .iter()
            .map(|&takes_leaves| RowGroupLeafEncoder {
                memory_budget: memory_budget.clone(),
                takes_leaves,
                value_allocator: None,
                page_allocator: None,
            })
            .collect()
    }
}

impl UnaryFactory<RowGroupLeaf, EncodedRowGroupLeaf> for RowGroupLeafEncoder {
    type Unary = Self;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self> {
        Built::new(self)
    }
}

impl Unary<RowGroupLeaf, EncodedRowGroupLeaf> for RowGroupLeafEncoder {
    fn consume(
        &mut self,
        leaf: RowGroupLeaf,
        sender: &mut dyn Sender<EncodedRowGroupLeaf>,
        _io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        let value_allocator = self
            .value_allocator
            .get_or_insert_with(|| SlabAllocator::new(false));
        let page_allocator = self
            .page_allocator
            .get_or_insert_with(|| SlabAllocator::new(false));
        let copy_memory = self.memory_budget.reserve(leaf.count_bytes());
        let values =
            concat_chunks(value_allocator, &leaf.value_chunks).map_err(WriteError::from)?;
        let def_levels: Option<Arc<[i16]>> = (leaf.max_def_level > 0).then(|| {
            let levels = leaf.def_level_chunks.iter();
            levels.flat_map(|chunk| chunk.iter().copied()).collect()
        });
        let RowGroupLeaf {
            row_group,
            leaf_index,
            path,
            max_def_level,
            value_chunks,
            def_level_chunks,
            memory: chunks_memory,
        } = leaf;
        drop((value_chunks, def_level_chunks, chunks_memory));

        let single_array_leaf = Leaf {
            path,
            values,
            def_levels,
            max_def_level,
        };
        let encoded = encoder::encode_leaf(single_array_leaf, page_allocator)?;
        let pages = encoded.dictionary_page.iter().chain(&encoded.data_pages);
        let page_bytes: usize = pages.map(|page| page.bytes.size()).sum();
        let memory = self.memory_budget.reserve(page_bytes);
        drop(copy_memory);
        sender.send(EncodedRowGroupLeaf {
            row_group,
            leaf_index,
            leaf: encoded,
            page_bytes,
            memory,
        })?;
        Ok(())
    }

    fn ready_for_more_work(&mut self) -> bool {
        self.takes_leaves
    }
}
