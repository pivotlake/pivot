//! Scan for a projection with no data columns.
//!
//! The page-driven read pipeline is column-driven: with zero projected columns
//! it fetches nothing, produces no pages, and emits nothing. This source instead
//! emits one batch per row group straight from each row group's `num_rows` — no
//! IO, no decode — chunked to [`RECORD_BATCH_SIZE`] so a downstream `LIMIT` can
//! reject most of them after the first chunk.
//!
//! Row-group/row-index metadata columns are appended only when requested: a
//! late-materialized plain `LIMIT` (whose narrow scan has no sort/filter key)
//! needs them for the downstream `Materialize`; any other zero-column scan (a
//! row count) gets plain empty-schema rows.

use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchOptions};
use arrow_schema::Schema;
use crossbeam_deque::{Injector, Steal};
use dispatch::{
    DataFlowDispatcher, Nullary, NullaryFactory, NullaryResult, RECORD_BATCH_SIZE,
    RecordBatchOperatorSpec, Sender, WorkStatus,
};

use crate::parquet::RowGroupFilter;
use crate::parquet::reading::record_batch_metadata::with_row_group_metadata;
use crate::parquet::types::metadata::QueryRowGroupMetadata;
use crate::parquet::types::table::ParquetTable;

/// Build a scan that emits the rows of an empty (zero-data-column) projection.
pub(crate) fn empty_projection_scan(
    dispatcher: &DataFlowDispatcher,
    table: &Arc<ParquetTable>,
    filter: Option<RowGroupFilter>,
    add_row_group_metadata: bool,
) -> RecordBatchOperatorSpec {
    let n = dispatcher.worker_count();
    let row_groups = Arc::new(Injector::new());
    for idx in 0..table.row_groups.len() {
        row_groups.push(QueryRowGroupMetadata::new(table, idx, None));
    }
    let factories = (0..n).map(|_| EmptyProjectionScanFactory {
        row_groups: row_groups.clone(),
        filter: filter.clone(),
        add_row_group_metadata,
    });
    RecordBatchOperatorSpec::from_nullary(dispatcher, factories)
}

/// Per-worker factory for [`EmptyProjectionScan`]. Workers share one
/// work-stealing queue of row groups, so each row group is emitted exactly once.
struct EmptyProjectionScanFactory {
    row_groups: Arc<Injector<QueryRowGroupMetadata>>,
    filter: Option<RowGroupFilter>,
    add_row_group_metadata: bool,
}

impl NullaryFactory<RecordBatch> for EmptyProjectionScanFactory {
    type Nullary = EmptyProjectionScan;
    fn build_nullary(self) -> EmptyProjectionScan {
        EmptyProjectionScan {
            row_groups: self.row_groups,
            filter: self.filter,
            add_row_group_metadata: self.add_row_group_metadata,
        }
    }
}

struct EmptyProjectionScan {
    row_groups: Arc<Injector<QueryRowGroupMetadata>>,
    filter: Option<RowGroupFilter>,
    add_row_group_metadata: bool,
}

impl Nullary<RecordBatch> for EmptyProjectionScan {
    fn run(
        &mut self,
        sender: &mut dyn Sender<RecordBatch>,
        _io: &mut dispatch::io::OperatorIO,
    ) -> NullaryResult<WorkStatus> {
        let rg = match self.row_groups.steal() {
            Steal::Success(rg) => rg,
            Steal::Empty | Steal::Retry => return Ok(WorkStatus::Pending),
        };
        // A pruned row group contributes no rows; skip without emitting.
        if let Some(filter) = &self.filter
            && !filter(rg.get_metadata())
        {
            return Ok(WorkStatus::Ran);
        }
        let total = rg.num_rows() as usize;
        let mut offset = 0;
        while offset < total {
            let chunk = (total - offset).min(RECORD_BATCH_SIZE);
            // Zero data columns + an explicit row count; never fails.
            let empty = RecordBatch::try_new_with_options(
                Arc::new(Schema::empty()),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(chunk)),
            )
            .expect("empty batch");
            let batch = if self.add_row_group_metadata {
                with_row_group_metadata(empty, rg.index(), offset)
            } else {
                empty
            };
            sender.send(batch)?;
            offset += chunk;
        }
        Ok(WorkStatus::Ran)
    }

    fn finish(&mut self, _sender: &mut dyn Sender<RecordBatch>) -> NullaryResult<bool> {
        Ok(self.row_groups.is_empty())
    }
}
