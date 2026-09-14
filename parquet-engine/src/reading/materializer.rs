//! Materializes further projections later on in the dataflow.
//!
//! After upstream operators evaluate predicates on lightweight index columns,
//! the surviving rows arrive here as `RecordBatch`es that still carry metadata
//! columns (global row-group ID and per-row index). The [`Materializer`]
//! accumulates those indices, groups them by row group, and on `finish` emits
//! one [`RowGroupRequest`] per row group with sorted indices so that the
//! downstream Parquet reader can fetch only the rows that passed a filter,
//! reading them sequentially within each row group for optimal IO.
//!
//! A materializer can also keep the order the rows arrived in: fed from one
//! worker below a sort, it records every row\'s `(row group, row)` in that
//! order and publishes the list on `finish`, for a consumer that lays the
//! fetched rows out again in it. Such a consumer needs every row, so the
//! materializer then requests each touched row group whole.

use crate::reading::record_batch_metadata::row_index;
use crate::types::metadata::{QueryRowGroupMetadata, RowSelection};
use crate::types::projection::Projection;
use crate::{ParquetTable, RowGroupRequest};
use ahash::HashMap;
use arrow_array::types::Int32Type;
use arrow_array::{Array, RecordBatch, RunArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use dispatch::Sender;
use dispatch::{Unary, UnaryFactory};
use std::mem;
use std::sync::{Arc, OnceLock};

/// The `(row group, row)` of every row of a file, in the order the rows
/// reached the materializer, set once on `finish` for a consumer to read.
pub type RowOrder = Arc<OnceLock<Vec<(u32, u32)>>>;

/// Replaces a batch\'s run-encoded row-group column with a plain one, so a
/// sort above the materializer can gather it as a fixed-width column.
pub fn plain_row_group_column(batch: RecordBatch) -> RecordBatch {
    let mut plain = Vec::with_capacity(batch.num_rows());
    for (group, run_end) in row_group_runs(&batch) {
        plain.resize(run_end, group);
    }
    let group_column = batch.num_columns() - 2;
    let (schema, mut columns, row_count) = batch.into_parts();
    columns[group_column] = Arc::new(UInt32Array::from(plain));
    let mut fields = schema.fields().to_vec();
    fields[group_column] = Arc::new(Field::new(
        schema.field(group_column).name(),
        DataType::UInt32,
        false,
    ));
    RecordBatch::try_new_with_options(
        Arc::new(Schema::new(fields)),
        columns,
        &arrow_array::RecordBatchOptions::new().with_row_count(Some(row_count)),
    )
    .expect("the plain column has the batch\'s rows")
}

/// The batch\'s rows as runs of one row group: `(group, logical end)` pairs,
/// whether the row-group column is still run-encoded or was made plain.
fn row_group_runs(batch: &RecordBatch) -> Vec<(u32, usize)> {
    let column = batch.column(batch.num_columns() - 2);
    if let Some(groups) = column.as_any().downcast_ref::<RunArray<Int32Type>>() {
        // The batch may be a logical slice (e.g. a `LIMIT` above the scan),
        // so walk the *logical* runs via `RunEndBuffer::sliced_values()`, run
        // ends already adjusted by the slice offset and capped at the slice
        // length, and map each run to its physical group value from
        // `get_start_physical_index()`.
        let run_ends = groups.run_ends();
        let physical_start = run_ends.get_start_physical_index();
        let group_values = groups
            .values()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("row group ids are unsigned");
        return run_ends
            .sliced_values()
            .enumerate()
            .map(|(run_offset, logical_end)| {
                (
                    group_values.value(physical_start + run_offset),
                    logical_end as usize,
                )
            })
            .collect();
    }
    let groups = column
        .as_any()
        .downcast_ref::<UInt32Array>()
        .expect("a row group column is run-encoded or plain unsigned");
    let mut runs: Vec<(u32, usize)> = Vec::new();
    for row in 0..groups.len() {
        let group = groups.value(row);
        match runs.last_mut() {
            Some((run_group, run_end)) if *run_group == group => *run_end = row + 1,
            _ => runs.push((group, row + 1)),
        }
    }
    runs
}

/// Factory for creating [`Materializer`] instances within the unary pipeline.
///
/// Captures the columns to materialize (`projection`) and the [`ParquetTable`]
/// handle that the materializer will later reference when building
/// [`RowGroupRequest`]s. Consumed once via [`UnaryFactory::build_unary`].
pub struct MaterializerFactory {
    projection: Projection,
    table: Arc<ParquetTable>,
    order: Option<RowOrder>,
}

impl MaterializerFactory {
    /// Creates a new factory.
    ///
    /// * `projection` – the set of columns that downstream IO should read back
    ///   from disk (i.e. the columns the query actually needs).
    /// * `table` – shared handle to the Parquet table metadata.
    pub fn new(projection: Projection, table: Arc<ParquetTable>) -> Self {
        Self {
            projection,
            table,
            order: None,
        }
    }

    /// Record the rows\' order into `order` and request their row groups whole
    /// (see the module docs).
    pub fn recording_order(mut self, order: RowOrder) -> Self {
        self.order = Some(order);
        self
    }
}

impl UnaryFactory<RecordBatch, RowGroupRequest> for MaterializerFactory {
    type Unary = Materializer;

    fn build_unary(self) -> Materializer {
        let mut materializer = Materializer::new(self.projection, self.table);
        materializer.order = self.order;
        materializer
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
    /// Where to publish the rows\' order, when it is kept; `rows` collects it.
    order: Option<RowOrder>,
    rows: Vec<(u32, u32)>,
    /// Row groups in the order the rows first name them, so the requests go
    /// out in the order a consumer of the rows will need them.
    groups_by_first_row: Vec<u32>,
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
            order: None,
            rows: Vec::new(),
            groups_by_first_row: Vec::new(),
        }
    }
}

impl Unary<RecordBatch, RowGroupRequest> for Materializer {
    /// Extracts row group IDs and row indices from the batch metadata columns,
    /// accumulating them into `pending_row_groups` keyed by row group.
    fn consume(
        &mut self,
        batch: RecordBatch,
        _sender: &mut dyn Sender<RowGroupRequest>,
        _io: &mut dispatch::OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        let row_indices = row_index(&batch);
        let mut logical = 0usize;
        for (group, logical_end) in row_group_runs(&batch) {
            if self.order.is_some() && !self.pending_row_groups.contains_key(&group) {
                self.groups_by_first_row.push(group);
            }
            let entries = self.pending_row_groups.entry(group).or_default();
            // `row_indices` is logically indexed, so `value(logical)` accounts
            // for any slice offset. A kept order needs no per-group indices:
            // the group is read whole.
            while logical < logical_end {
                let row = row_indices.value(logical);
                if self.order.is_some() {
                    self.rows.push((group, row));
                } else {
                    entries.push(row);
                }
                logical += 1;
            }
        }
        Ok(())
    }

    /// Drains accumulated row groups and sends one [`RowGroupRequest`] per group.
    /// Indices are sorted so downstream IO can read them sequentially.
    fn finish(&mut self, sender: &mut dyn Sender<RowGroupRequest>) -> dispatch::UnaryResult<bool> {
        if let Some(order) = &self.order
            && !self.rows.is_empty()
        {
            order
                .set(mem::take(&mut self.rows))
                .unwrap_or_else(|_| panic!("a file\'s row order is published once"));
        }
        let mut pending_row_groups = mem::take(&mut self.pending_row_groups);
        let groups: Vec<u32> = if self.order.is_some() {
            mem::take(&mut self.groups_by_first_row)
        } else {
            pending_row_groups.keys().copied().collect()
        };
        for group in groups {
            let mut indices = pending_row_groups.remove(&group).unwrap_or_default();
            let selection = if self.order.is_some() {
                RowSelection::All
            } else {
                indices.sort_unstable();
                RowSelection::Indices(indices.into())
            };
            sender.send(RowGroupRequest::from(
                QueryRowGroupMetadata::new(&self.table, group as usize, selection),
                &self.projection_to_materialize,
            ))?;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::Materializer;
    use crate::reading::record_batch_metadata::with_row_group_metadata;
    use crate::{ParquetTable, RowGroupRequest};
    use ahash::HashMap;
    use arrow::compute::concat_batches;
    use arrow_array::{RecordBatch, RecordBatchOptions};
    use arrow_schema::Schema;
    use dispatch::Projection;
    use dispatch::test_utils::CollectSender;
    use std::sync::Arc;

    /// A metadata-only batch (no data columns) tagging `rows` rows with row group
    /// `group` and row indices `0..rows` — the shape a metadata scan emits.
    fn meta_batch(group: usize, rows: usize) -> RecordBatch {
        let empty = RecordBatch::try_new_with_options(
            Arc::new(Schema::empty()),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )
        .unwrap();
        with_row_group_metadata(empty, group, 0)
    }

    /// Run `batch` through a `Materializer`'s `consume` and return the row indices
    /// it grouped by row group. The table is unused by `consume` (only `finish`
    /// reads it), so a dummy empty one suffices.
    fn consume(batch: RecordBatch) -> HashMap<u32, Vec<u32>> {
        let mut materializer = Materializer::new(
            Projection::columns([0]),
            Arc::new(ParquetTable::new(vec![])),
        );
        let mut sink = CollectSender::<RowGroupRequest>::default();
        dispatch::Unary::consume(
            &mut materializer,
            batch,
            &mut sink,
            &mut dispatch::TestOperatorIO::default().io(),
        )
        .unwrap();
        materializer.pending_row_groups
    }

    // Two row groups concatenated → a 2-run RunArray; the materializer recovers
    // every (group, row_idx) pair.
    #[test]
    fn collects_indices_grouped_by_row_group() {
        let a = meta_batch(5, 6);
        let batch = concat_batches(&a.schema(), &[a.clone(), meta_batch(8, 6)]).unwrap();

        let out = consume(batch);

        assert_eq!(out[&5], vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(out[&8], vec![0, 1, 2, 3, 4, 5]);
    }

    // Regression: a logically *sliced* metadata batch (what a LIMIT above the scan
    // produces) must be decoded by logical run bounds, not the unchanged physical
    // run_ends — the latter overran the shorter sliced row_idx. The slice spans
    // into the second run, so it also exercises `get_start_physical_index() > 0`.
    #[test]
    fn collects_indices_from_sliced_batch() {
        let a = meta_batch(5, 6);
        let batch = concat_batches(&a.schema(), &[a.clone(), meta_batch(8, 6)]).unwrap();

        let sliced = batch.slice(8, 3); // logical rows 8,9,10 → group 8, row_idx 2,3,4
        let out = consume(sliced);

        assert_eq!(out.len(), 1);
        assert_eq!(out[&8], vec![2, 3, 4]);
    }
}
