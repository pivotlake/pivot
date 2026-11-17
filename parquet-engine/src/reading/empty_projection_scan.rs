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
//!
//! A scan with a [`RowSelector`] emits only the rows the selector names, so a
//! row count over a selection reads no data page either:
//!
//! ```text
//!   files ─▶ RowSelectionStage ─▶ SelectedRowGroupRows ─▶ batches of no columns
//! ```

use dispatch::{Built, BuiltNullary, BuiltUnary};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use arrow_array::{RecordBatch, RecordBatchOptions, UInt32Array};
use arrow_schema::Schema;
use crossbeam_deque::{Injector, Steal};
use dispatch::{
    BuildContext, DataFlowDispatcher, Nullary, NullaryFactory, NullaryResult, OperatorSpec,
    RECORD_BATCH_SIZE, RecordBatchOperatorSpec, RootUnaryOperatorFactory, Sender, Unary,
    UnaryFactory, UnaryOperatorFactory, UnaryResult, WorkStatus, stealable,
};

use crate::RowGroupFilter;
use crate::reading::record_batch_metadata::{with_row_group_metadata, with_row_group_positions};
use crate::reading::row_selection::{RowSelectionStageFactory, RowSelector, ScannedFiles};
use crate::types::metadata::{QueryRowGroupMetadata, RowSelection};
use crate::types::projection::Projection;
use crate::types::table::ParquetTable;

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
        row_groups.push(QueryRowGroupMetadata::new(table, idx, RowSelection::All));
    }
    let factories = (0..n).map(|_| EmptyProjectionScanFactory {
        row_groups: row_groups.clone(),
        filter: filter.clone(),
        add_row_group_metadata,
    });
    RecordBatchOperatorSpec::from_nullary(dispatcher, factories)
}

/// Build a scan that emits, of an empty (zero-data-column) projection, the
/// rows `row_selector` names.
pub(crate) fn empty_projection_scan_of_selected_rows<Selector>(
    dispatcher: &DataFlowDispatcher,
    table: &Arc<ParquetTable>,
    filter: Option<RowGroupFilter>,
    add_row_group_metadata: bool,
    row_selector: Selector,
) -> RecordBatchOperatorSpec
where
    Selector: RowSelector + Clone + Send + 'static,
{
    let n = dispatcher.worker_count();
    let no_columns = Arc::new(Projection::columns([]));
    let files = ScannedFiles::of_table(table, &no_columns, &row_selector);
    let read_medium = table.get_read_medium();
    let siblings_selection = Arc::new(AtomicUsize::new(n));
    let siblings_rows = Arc::new(AtomicUsize::new(n));
    let factories: Vec<_> = stealable::<QueryRowGroupMetadata>(dispatcher.topology())
        .into_iter()
        .map(|selected_row_groups_channel| {
            UnaryOperatorFactory::new(
                RootUnaryOperatorFactory::new(
                    RowSelectionStageFactory::new(
                        row_selector.clone(),
                        files.clone(),
                        filter.clone(),
                        no_columns.clone(),
                        read_medium,
                        None,
                    ),
                    files.clone(),
                    siblings_selection.clone(),
                ),
                SelectedRowGroupRowsFactory {
                    add_row_group_metadata,
                },
                selected_row_groups_channel,
                siblings_rows.clone(),
            )
        })
        .collect();
    RecordBatchOperatorSpec::from_spec(OperatorSpec::new(dispatcher.clone(), factories))
}

/// The batches a row group stands for under a projection of no columns: no
/// column, just a count of rows, with the row-group and row-index metadata
/// columns when they are asked for.
struct EmptyProjectionRows {
    add_row_group_metadata: bool,
}

impl EmptyProjectionRows {
    /// Hand `send` the batches of the rows of `row_group` that are read:
    /// every row the table format left live, or the rows its selection names.
    fn send_rows_of<E>(
        &self,
        row_group: &QueryRowGroupMetadata,
        mut send: impl FnMut(RecordBatch) -> Result<(), E>,
    ) -> Result<(), E> {
        // Zero data columns + an explicit row count; never fails.
        let empty_rows = |rows: usize| {
            RecordBatch::try_new_with_options(
                Arc::new(Schema::empty()),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(rows)),
            )
            .expect("empty batch")
        };
        let metadata = row_group.get_metadata();
        // The rows by position, where they are not simply the row group's
        // rows in a row: the rows a selection names, or, when positions are
        // asked for, the ones deletions left, which have gaps between them.
        let positions: Option<Arc<[u32]>> = match (row_group.selection(), &metadata.deleted_rows) {
            (RowSelection::Indices(positions), _) => Some(positions.clone()),
            (RowSelection::All, Some(_)) if self.add_row_group_metadata => Some(
                (0..metadata.num_rows as u32)
                    .filter(|row| !metadata.is_deleted(*row))
                    .collect(),
            ),
            (RowSelection::All, _) => None,
        };
        match positions {
            Some(positions) => {
                for chunk in positions.chunks(RECORD_BATCH_SIZE) {
                    let empty = empty_rows(chunk.len());
                    send(if self.add_row_group_metadata {
                        with_row_group_positions(
                            empty,
                            row_group.index(),
                            UInt32Array::from(chunk.to_vec()),
                        )
                    } else {
                        empty
                    })?;
                }
            }
            None => {
                let total = metadata.live_rows() as usize;
                let mut offset = 0;
                while offset < total {
                    let chunk = (total - offset).min(RECORD_BATCH_SIZE);
                    let empty = empty_rows(chunk);
                    send(if self.add_row_group_metadata {
                        with_row_group_metadata(empty, row_group.index(), offset)
                    } else {
                        empty
                    })?;
                    offset += chunk;
                }
            }
        }
        Ok(())
    }
}

/// Builds one worker's [`SelectedRowGroupRows`].
struct SelectedRowGroupRowsFactory {
    add_row_group_metadata: bool,
}

impl UnaryFactory<QueryRowGroupMetadata, RecordBatch> for SelectedRowGroupRowsFactory {
    type Unary = SelectedRowGroupRows;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<SelectedRowGroupRows> {
        Built::new(SelectedRowGroupRows {
            rows: EmptyProjectionRows {
                add_row_group_metadata: self.add_row_group_metadata,
            },
        })
    }
}

/// Emits the rows of each row group a selection stage kept, for a projection
/// of no columns.
struct SelectedRowGroupRows {
    rows: EmptyProjectionRows,
}

impl Unary<QueryRowGroupMetadata, RecordBatch> for SelectedRowGroupRows {
    fn consume(
        &mut self,
        row_group: QueryRowGroupMetadata,
        sender: &mut dyn Sender<RecordBatch>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        self.rows
            .send_rows_of(&row_group, |batch| sender.send(batch))?;
        Ok(())
    }
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
    fn build_nullary(self) -> BuiltNullary<EmptyProjectionScan> {
        Built::new(EmptyProjectionScan {
            row_groups: self.row_groups,
            filter: self.filter,
            rows: EmptyProjectionRows {
                add_row_group_metadata: self.add_row_group_metadata,
            },
        })
    }
}

struct EmptyProjectionScan {
    row_groups: Arc<Injector<QueryRowGroupMetadata>>,
    filter: Option<RowGroupFilter>,
    rows: EmptyProjectionRows,
}

impl Nullary<RecordBatch> for EmptyProjectionScan {
    fn run(
        &mut self,
        sender: &mut dyn Sender<RecordBatch>,
        _io: &mut dispatch::OperatorIO,
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
        self.rows.send_rows_of(&rg, |batch| sender.send(batch))?;
        Ok(WorkStatus::Ran)
    }

    fn finish(&mut self, _sender: &mut dyn Sender<RecordBatch>) -> NullaryResult<bool> {
        Ok(self.row_groups.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reading::record_batch_metadata::row_index;
    use crate::test_utils::{deleted_rows, dummy_row_group};
    use crate::types::metadata::RowGroupMetadata;
    use dispatch::Dispatch;

    /// A table of row groups holding `rows` rows each, with the `deleted`
    /// rows of each marked deleted.
    fn table_of_row_groups(rows: &[i64], deleted: &[&[u32]]) -> Arc<ParquetTable> {
        let dummy = dummy_row_group();
        let row_groups = rows
            .iter()
            .zip(deleted)
            .enumerate()
            .map(|(index, (rows, deleted))| {
                Arc::new(RowGroupMetadata {
                    num_rows: *rows,
                    deleted_rows: deleted_rows(0, deleted),
                    file_row_group_idx: index,
                    ..(*dummy).clone()
                })
            })
            .collect();
        Arc::new(ParquetTable::new(row_groups))
    }

    #[test]
    fn deleted_rows_are_left_out_of_the_count() {
        let dispatch = Dispatch::spin_up(1, 32, None);
        let table = table_of_row_groups(&[5, 4], &[&[1, 3], &[]]);

        let batches = empty_projection_scan(dispatch.dispatcher(), &table, None, false)
            .collect()
            .unwrap();

        let rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(rows, 7);
    }

    #[test]
    fn deleted_rows_leave_gaps_in_the_row_indices() {
        let dispatch = Dispatch::spin_up(1, 32, None);
        let table = table_of_row_groups(&[5], &[&[1, 3]]);

        let batches = empty_projection_scan(dispatch.dispatcher(), &table, None, true)
            .collect()
            .unwrap();

        let positions: Vec<u32> = batches
            .iter()
            .flat_map(|batch| row_index(batch).values().to_vec())
            .collect();
        assert_eq!(positions, vec![0, 2, 4]);
    }

    /// Names rows 1 and 3 of every row group, with no lookup to wait for.
    #[derive(Clone)]
    struct SelectRowsOneAndThree;

    impl RowSelector for SelectRowsOneAndThree {
        type FileRows = ();
        type Pending = std::convert::Infallible;

        fn count_bytes_in_memory(&self, _file: &dispatch::io::OpenFile) -> usize {
            0
        }

        fn look_up_file(
            &mut self,
            _file: &dispatch::io::OpenFile,
            _io: &mut dispatch::OperatorIO,
        ) -> UnaryResult<crate::FileLookupProgress<(), Self::Pending>> {
            Ok(crate::FileLookupProgress::LookedUp(()))
        }

        fn resume_file_lookup(
            &mut self,
            pending: Self::Pending,
            _response: dispatch::io::ReadResponse,
            _io: &mut dispatch::OperatorIO,
        ) -> UnaryResult<crate::FileLookupProgress<(), Self::Pending>> {
            match pending {}
        }

        fn select_rows(
            &self,
            _file_rows: &(),
            _row_group: &RowGroupMetadata,
        ) -> UnaryResult<RowSelection> {
            Ok(RowSelection::Indices(vec![1, 3].into()))
        }
    }

    #[test]
    fn only_the_rows_a_selector_names_are_emitted() {
        let dispatch = Dispatch::spin_up(1, 32, None);
        let table = table_of_row_groups(&[5, 4], &[&[], &[]]);

        let batches = empty_projection_scan_of_selected_rows(
            dispatch.dispatcher(),
            &table,
            None,
            true,
            SelectRowsOneAndThree,
        )
        .collect()
        .unwrap();

        let positions: Vec<u32> = batches
            .iter()
            .flat_map(|batch| row_index(batch).values().to_vec())
            .collect();
        assert_eq!(positions, vec![1, 3, 1, 3]);
    }
}
