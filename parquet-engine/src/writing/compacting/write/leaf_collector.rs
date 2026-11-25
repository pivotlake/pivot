//! Joins the merged batches of one column of an output row group into the
//! column's leaves, each ready to encode.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::ArrayRef;
use dispatch::{BuildContext, Built, BuiltUnary, OperatorIO, Sender, Unary, UnaryFactory};

use super::leaf_gatherer::GatheredLeaves;
use crate::writing::compacting::memory_budget::MemoryReservation;
use crate::writing::leaves::{self, Leaf};
use crate::writing::shredding::FileShredding;

/// One leaf of one output row group: its values and definition levels, a
/// chunk per merged batch of the row group, in row order.
pub(crate) struct RowGroupLeaf {
    /// The output row group the leaf is of.
    pub(crate) row_group: usize,
    /// The leaf's place among the leaves of the output's schema, depth-first,
    /// which is the order a row group's column chunks are laid out in.
    pub(crate) leaf_index: usize,
    /// The leaf's path in the schema, from the top-level column down.
    pub(crate) path: Vec<String>,
    /// The values of the rows that hold one, in row order.
    pub(crate) value_chunks: Vec<ArrayRef>,
    /// One level per row. Empty when nothing on the path is nullable.
    pub(crate) def_level_chunks: Vec<Arc<[i16]>>,
    /// The level that means a row holds a value.
    pub(crate) max_def_level: i16,
    /// What the memory budget counts for the merged batches the chunks are
    /// from. The leaves of one column share it, so the bytes are given back
    /// when the last of them lets go.
    pub(crate) memory: Vec<Arc<MemoryReservation>>,
}

impl RowGroupLeaf {
    /// How many bytes the leaf's values and levels hold.
    pub(crate) fn count_bytes(&self) -> usize {
        let values = self.value_chunks.iter();
        let value_bytes: usize = values
            .map(|chunk| GatheredLeaves::count_array_bytes(&chunk.to_data()))
            .sum();
        let levels = self.def_level_chunks.iter();
        let level_bytes: usize = levels.map(|chunk| std::mem::size_of_val(&chunk[..])).sum();
        value_bytes + level_bytes
    }
}

/// Collects the [`GatheredLeaves`] of each column of each output row group,
/// which arrive in any order, and sends the column's leaves on as one
/// [`RowGroupLeaf`] each once every merged batch of the column has arrived.
/// Nothing is copied here. Runs on the write worker, where all gathered
/// leaves are routed.
pub(crate) struct LeafCollector {
    /// For each table column, the place of its first leaf among the leaves
    /// of the output's schema.
    first_leaf_per_column: Arc<[usize]>,
    /// The merged batches that arrived of the columns that are not complete,
    /// by output row group and column.
    arrived_batches: HashMap<(usize, usize), Vec<GatheredLeaves>>,
}

impl LeafCollector {
    /// One collector per worker, for output written under the schema of
    /// `shredding`.
    pub(crate) fn create_for_workers(shredding: &FileShredding, worker_count: usize) -> Vec<Self> {
        let mut leaf_count = 0;
        let first_leaf_per_column: Arc<[usize]> = shredding
            .schema
            .fields()
            .iter()
            .map(|field| {
                let first_leaf = leaf_count;
                leaf_count += leaves::count_leaves(field);
                first_leaf
            })
            .collect();
        (0..worker_count)
            .map(|_| LeafCollector {
                first_leaf_per_column: first_leaf_per_column.clone(),
                arrived_batches: HashMap::new(),
            })
            .collect()
    }
}

impl UnaryFactory<GatheredLeaves, RowGroupLeaf> for LeafCollector {
    type Unary = Self;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self> {
        Built::new(self)
    }
}

impl Unary<GatheredLeaves, RowGroupLeaf> for LeafCollector {
    fn consume(
        &mut self,
        gathered: GatheredLeaves,
        sender: &mut dyn Sender<RowGroupLeaf>,
        _io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        let (row_group, column) = (gathered.row_group, gathered.column);
        let batch_count = gathered.row_group_batch_count;
        let arrived = self.arrived_batches.entry((row_group, column)).or_default();
        arrived.push(gathered);
        if arrived.len() < batch_count {
            return Ok(());
        }
        let mut batches = self
            .arrived_batches
            .remove(&(row_group, column))
            .expect("the column's batches were just counted");
        batches.sort_by_key(|batch| batch.batch);

        let memory: Vec<Arc<MemoryReservation>> =
            batches.iter().map(|batch| batch.memory.clone()).collect();
        let first_leaf = self.first_leaf_per_column[column];
        let mut column_leaves: Vec<RowGroupLeaf> = Vec::new();
        for batch in batches {
            for (leaf_of_column, leaf) in batch.leaves.into_iter().enumerate() {
                let Leaf {
                    path,
                    values,
                    def_levels,
                    max_def_level,
                } = leaf;
                if leaf_of_column == column_leaves.len() {
                    column_leaves.push(RowGroupLeaf {
                        row_group,
                        leaf_index: first_leaf + leaf_of_column,
                        path,
                        value_chunks: Vec::with_capacity(batch_count),
                        def_level_chunks: Vec::new(),
                        max_def_level,
                        memory: memory.clone(),
                    });
                }
                let column_leaf = &mut column_leaves[leaf_of_column];
                column_leaf.value_chunks.push(values);
                column_leaf.def_level_chunks.extend(def_levels);
            }
        }
        for column_leaf in column_leaves {
            sender.send(column_leaf)?;
        }
        Ok(())
    }
}
