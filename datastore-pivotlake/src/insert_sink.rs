//! The dataflow that writes an INSERT's rows into a table's files.

use catalog::{CommitAction, CommitQueue};
use dispatch::RecordBatchOperatorSpec;

use crate::{CatalogTable, PendingInsert, PivotlakeDatastore};

/// Build the dataflow that writes `input`'s rows into `table` as Parquet and
/// emits the inserted-row count. Each finished file is pushed onto the
/// target's uploaded files for the statement's transaction to commit. Driven by
/// the binding's `BoundTable::compile_insert` impl (see [`super::binding::TableBinding`]).
pub(super) fn build_insert_spec(
    table: &CatalogTable,
    commit_actions: CommitQueue<PivotlakeDatastore>,
    input: RecordBatchOperatorSpec,
) -> crate::Result<RecordBatchOperatorSpec> {
    let target = table.insert_target();
    // The pending insert this statement commits with. Staged before any file
    // can land, so whatever lands is owned by a staged action the transaction
    // can commit or discard.
    let pending_insert: Box<dyn CommitAction<PivotlakeDatastore>> = Box::new(PendingInsert {
        table_id: table.id(),
        uploaded_files: target.uploaded_files.clone(),
    });
    let stage_pending_insert = Box::new(move || commit_actions.push(pending_insert));
    Ok(target.build_insert_spec(stage_pending_insert, input)?)
}
