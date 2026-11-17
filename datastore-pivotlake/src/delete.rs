//! The dataflow of a DELETE, which reads and writes no row.

use catalog::{CommitAction, CommitQueue};
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use parquet_engine::build_affected_rows_spec;
use planner::catalog::Result as CatalogResult;
use planner::expression::Expression;

use crate::{CatalogTable, PendingDelete, PivotlakeDatastore};

/// Build the dataflow that stages the removal of the rows of `table` every
/// one of `conditions` is true of and emits their count, or refuse the
/// DELETE: it takes whole partitions out of the table, so its conditions
/// have to be ones a partition's values can be checked against, and must
/// cut no partition that holds a row. The count is of the version the
/// statement read; the transaction's commit selects what the conditions
/// remove from the live table. Driven by the binding's
/// `BoundTable::compile_delete` impl (see
/// [`parquet_engine::ParquetTableBinding`]).
pub(super) fn build_delete_spec(
    table: &CatalogTable,
    commit_actions: CommitQueue<PivotlakeDatastore>,
    conditions: &[Expression],
    dispatcher: &DataFlowDispatcher,
) -> CatalogResult<RecordBatchOperatorSpec> {
    let filter = table.bind_delete_filter(conditions)?;
    let deleted_rows = table.count_deleted_rows(&filter)?;
    let pending_delete: Box<dyn CommitAction<PivotlakeDatastore>> = Box::new(PendingDelete {
        table_id: table.id(),
        filter,
    });
    let stage_pending_delete = Box::new(move || commit_actions.push(pending_delete));
    build_affected_rows_spec(dispatcher, deleted_rows, stage_pending_delete)
}
