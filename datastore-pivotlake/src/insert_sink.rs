//! The dataflow that writes an INSERT's rows into a table's files.

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{Field, Schema};
use crossbeam_deque::Injector;
use dispatch::RecordBatchOperatorSpec;
use parquet_engine::writing::upload::upload_files_spec;
use parquet_engine::writing::{Partitioning, encode_record_batches_spec, unshred_batches_spec};

use crate::{CatalogTable, CommitAction, PendingInsert};

/// Build the dataflow that writes `input`'s rows into `table` as Parquet and
/// emits the inserted-row count. Each finished file is pushed onto `uploaded_files`
/// for the statement's transaction to commit. This is
/// the write mirror of the scan's
/// [`table_input_with_filter_and_eq_predicates`](parquet_engine::table_input_with_filter_and_eq_predicates):
/// it wires the encode pipeline into the engine's upload sink. Driven by
/// the binding's `BoundTable::compile_insert` impl (see [`super::binding::TableBinding`]).
pub(super) fn build_insert_spec(
    table: &CatalogTable,
    commit_actions: Arc<Injector<Box<dyn CommitAction>>>,
    input: RecordBatchOperatorSpec,
) -> crate::Result<RecordBatchOperatorSpec> {
    let store = table.store();
    store.prepare_write()?;
    let uploaded_files = Arc::new(Injector::new());
    // The pending insert this statement commits with. Staged before any file
    // can land, so whatever lands is owned by a staged action the transaction
    // can commit or discard.
    let pending_insert: Box<dyn CommitAction> = Box::new(PendingInsert {
        table_id: table.id(),
        store: store.clone(),
        location: table.object_location().clone(),
        uploaded_files: uploaded_files.clone(),
    });
    let stage_pending_insert = Box::new(move || commit_actions.push(pending_insert));

    // In `INSERT INTO x VALUES (...)`, VALUES emits batches with unnamed columns
    // because only INSERT knows which table columns receive the rows. DuckDB has
    // already bound and cast each expression to its target column by position, so
    // stamp the table's declared schema onto the batches for Parquet files and
    // partitioning.
    let schema = Arc::new(Schema::new(
        table
            .columns()
            .into_iter()
            .map(|column| {
                // Every column is nullable: the catalog carries no NOT NULL
                // constraint, so a row is free to arrive with a NULL in any of
                // them. Marking the field OPTIONAL is what lets the written
                // leaves carry definition levels; readers do not lose the
                // no-NULLs fast path over it, since they refine nullability
                // from each chunk's null count rather than from this flag.
                let field = Field::new(
                    column.name,
                    planner::types::physical_arrow_type(&column.col_type),
                    true,
                );
                // The Arrow extension tag lives on the field, not on the data
                // type, so a variant built from `physical_arrow_type` alone is
                // indistinguishable from any other struct. Put the tag back, or
                // the shredding stage further down this pipeline does not
                // recognize the column and writes the documents unshredded.
                match column.col_type {
                    planner::types::Type::Variant => {
                        field.with_metadata(parquet_engine::variant_extension_metadata())
                    }
                    _ => field,
                }
            })
            .collect::<Vec<_>>(),
    ));
    let input = unshred_batches_spec(input).project({
        let schema = schema.clone();
        move || {
            let schema = schema.clone();
            move |batch| {
                // Selecting a variant column out of a table hands over the
                // physical shape that table shredded it into, while the column
                // declared here is the plain pair. Reassemble the documents, so
                // the rows are shredded for the file they land in rather than
                // carrying the source table's layout across.
                Ok(RecordBatch::try_new(
                    schema.clone(),
                    batch.columns().to_vec(),
                )?)
            }
        }
    });

    const TARGET_ROWS_PER_GROUP: usize = 128 * 1024;
    // This target measures retained Arrow data, not encoded Parquet bytes. The
    // collector applies it to each pending partition file and cuts only after
    // adding a complete batch. Compression and encoding normally make the
    // resulting files substantially smaller on disk.
    const TARGET_IN_MEMORY_BYTES_PER_FILE: usize = 900 * 1024 * 1024;
    let encoded = encode_record_batches_spec(
        input,
        schema,
        Partitioning::by_columns(table.partition_by()),
        table.sort_by().to_vec().into(),
        TARGET_ROWS_PER_GROUP,
        TARGET_IN_MEMORY_BYTES_PER_FILE,
    );
    Ok(upload_files_spec(
        store,
        table.object_location().clone(),
        table.columns().into(),
        uploaded_files,
        Some(stage_pending_insert),
        encoded,
    ))
}
