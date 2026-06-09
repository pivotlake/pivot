//! Materializes further projections later on in the dataflow.
//!
//! After upstream operators evaluate predicates on lightweight index columns,
//! the surviving rows arrive here as `RecordBatch`es that still carry metadata
//! columns (global row-group ID and per-row index). The [`Materializer`]
//! accumulates those indices, groups them by row group, and on `finish` emits
//! one [`RowGroupRequest`] per row group with sorted indices so that the
//! downstream Parquet reader can fetch only the rows that passed a filter,
//! reading them sequentially within each row group for optimal IO.

use crate::parquet::record_batch_metadata::{global_row_group, row_index};
use crate::parquet::types::metadata::QueryRowGroupMetadata;
use crate::parquet::types::projection::Projection;
use crate::parquet::{ParquetTable, RowGroupRequest};
use ahash::HashMap;
use arrow_array::{Array, RecordBatch, UInt32Array};
use dispatch::Sender;
use dispatch::{Unary, UnaryFactory};
use std::mem;
use std::sync::Arc;

/// Factory for creating [`Materializer`] instances within the unary pipeline.
///
/// Captures the columns to materialize (`projection`) and the [`ParquetTable`]
/// handle that the materializer will later reference when building
/// [`RowGroupRequest`]s. Consumed once via [`UnaryFactory::build_unary`].
pub struct MaterializerFactory {
    projection: Projection,
    table: Arc<ParquetTable>,
}

impl MaterializerFactory {
    /// Creates a new factory.
    ///
    /// * `projection` – the set of columns that downstream IO should read back
    ///   from disk (i.e. the columns the query actually needs).
    /// * `table` – shared handle to the Parquet table metadata.
    pub fn new(projection: Projection, table: Arc<ParquetTable>) -> Self {
        Self { projection, table }
    }
}

impl UnaryFactory<RecordBatch, RowGroupRequest> for MaterializerFactory {
    type Unary = Materializer;

    fn build_unary(self) -> Materializer {
        Materializer::new(self.projection, self.table)
    }
}

/// Collects filtered row indices from upstream `RecordBatch`es and emits
/// [`RowGroupRequest`]s to fetch the matching rows from disk.
///
/// During `consume`, incoming batches carry metadata columns (global row group
/// ID and per-row index). The materializer groups row indices by their row
/// group. On `finish`, it emits one `RowGroupRequest` per row group with sorted
/// indices, allowing downstream IO to read only the needed rows.
pub struct Materializer {
    projection_to_materialize: Projection,
    /// Row group ID -> collected row indices within that group.
    pending_row_groups: HashMap<u32, Vec<u32>>,
    table: Arc<ParquetTable>,
}

impl Materializer {
    /// Creates a new materializer.
    ///
    /// * `projection_to_materialize` – columns the downstream reader should
    ///   fetch from each row group.
    /// * `table` – shared Parquet table metadata used to build row group
    ///   requests.
    pub fn new(projection_to_materialize: Projection, table: Arc<ParquetTable>) -> Self {
        Self {
            projection_to_materialize,
            pending_row_groups: HashMap::default(),
            table,
        }
    }
}

impl Unary<RecordBatch, RowGroupRequest> for Materializer {
    /// Extracts row group IDs and row indices from the batch metadata columns,
    /// accumulating them into `pending_row_groups` keyed by row group.
    fn consume<S: Sender<RowGroupRequest>>(
        &mut self,
        batch: RecordBatch,
        _sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        // The row group column is a RunArray: consecutive rows in the same group
        // share a single run, so we iterate over runs rather than individual rows.
        let groups = global_row_group(&batch);
        let run_ends = groups.run_ends().values();
        let group_values = groups
            .values()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let row_indices = row_index(&batch);

        let mut logical_start = 0usize;
        for (run_idx, &run_end) in run_ends.iter().enumerate() {
            let logical_end = run_end as usize;
            let group = group_values.value(run_idx);
            let entries = self.pending_row_groups.entry(group).or_default();
            // Collect every row index that falls within this run.
            for i in logical_start..logical_end {
                entries.push(row_indices.value(i));
            }
            logical_start = logical_end;
        }
        Ok(())
    }

    /// Drains accumulated row groups and sends one [`RowGroupRequest`] per group.
    /// Indices are sorted so downstream IO can read them sequentially.
    fn finish<S: Sender<RowGroupRequest>>(
        &mut self,
        sender: &mut S,
    ) -> dispatch::UnaryResult<bool> {
        let pending_row_groups = mem::take(&mut self.pending_row_groups);
        for (group, mut indices) in pending_row_groups {
            indices.sort_unstable();
            sender.send(RowGroupRequest::from(
                QueryRowGroupMetadata::new(&self.table, group as usize, Some(indices)),
                &self.projection_to_materialize,
            ))?;
        }
        Ok(true)
    }
}
