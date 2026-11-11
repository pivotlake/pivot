//! The dataflow that writes an INSERT's rows into a table's data files: the
//! Arrow schema the files are written under, the partition keys they are cut
//! by, and the engine's upload sink that lands them in the table's store.
//!
//! The sink knows nothing about what records the files afterwards: an
//! INSERT's transaction and a compaction both drain what it uploaded and
//! commit it through their own table format.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{Field, Schema};
use dispatch::{RecordBatchOperatorSpec, UnaryError};
use parquet_engine::writing::upload::{UploadedFiles, upload_files_spec};
use parquet_engine::writing::{encode_record_batches_spec, unshred_batches_spec};
use planner::catalog::Column;

use crate::Result;
use crate::partitioning::PartitionKeys;

/// The Arrow schema a table's files are written under: its declared columns
/// in order, each tagged the way the writer and the readers need. A column
/// `nullability` marks as not nullable is written REQUIRED, so a NULL in it
/// fails the write rather than landing in a file other readers trust. With
/// `field_ids`, each field also carries the Parquet field id (in `columns`
/// order) a format that resolves columns by id assigned it.
pub fn write_schema(
    columns: &[Column],
    nullability: &[bool],
    field_ids: Option<&[i32]>,
) -> Arc<Schema> {
    let fields = columns
        .iter()
        .enumerate()
        .map(|(position, column)| {
            let field = Field::new(
                column.name.clone(),
                planner::types::physical_arrow_type(&column.col_type),
                nullability[position],
            );
            // The Arrow extension tag lives on the field, not on the data
            // type, so a variant built from `physical_arrow_type` alone is
            // indistinguishable from any other struct. Put the tag back, or
            // the shredding stage further down this pipeline does not
            // recognize the column and writes the documents unshredded.
            let mut metadata = match column.col_type {
                planner::types::Type::Variant => parquet_engine::variant_extension_metadata(),
                _ => HashMap::new(),
            };
            if let Some(field_ids) = field_ids {
                metadata.insert(
                    parquet_engine::PARQUET_FIELD_ID_METADATA_KEY.to_string(),
                    field_ids[position].to_string(),
                );
            }
            field.with_metadata(metadata)
        })
        .collect::<Vec<_>>();
    Arc::new(Schema::new(fields))
}

/// Where an INSERT's files go and what shape they take: everything the sink
/// needs to know about the table it writes into.
pub struct InsertTarget {
    /// Where the files are written, the store (with the credentials its
    /// writes need) and the directory in it, and the queue each file is
    /// reported on. The statement's transaction holds the same value, so
    /// what it commits or discards is found where the sink wrote it.
    pub uploaded_files: UploadedFiles,
    pub columns: Vec<Column>,
    /// Per column, in `columns` order, whether a NULL may be written.
    pub nullability: Vec<bool>,
    /// The field id each column is written under, in `columns` order, for a
    /// table format that resolves columns by id.
    pub field_ids: Option<Vec<i32>>,
    /// How the files are cut by the table's partition spec, and what each
    /// file's partition values are.
    pub partition_keys: PartitionKeys,
    pub sort_by: Vec<String>,
}

impl InsertTarget {
    /// Build the dataflow that writes `input`'s rows into this target as
    /// Parquet and emits the inserted-row count. Each finished file is pushed
    /// onto the target's `uploaded_files` for the statement's transaction to
    /// commit; `on_start` runs once as the dataflow is built, before any file
    /// can land, which is where the caller stages what commits or discards
    /// them. This is the write mirror of the scan's
    /// [`table_input_with_filter_and_eq_predicates`](parquet_engine::table_input_with_filter_and_eq_predicates):
    /// it wires the encode pipeline into the engine's upload sink. Called on
    /// the coordinating thread, where the store's write credentials are primed
    /// for the workers.
    pub fn build_insert_spec(
        &self,
        on_start: Box<dyn FnOnce() + Send>,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec> {
        self.uploaded_files.store().prepare_write()?;
        // In `INSERT INTO x VALUES (...)`, VALUES emits batches with unnamed columns
        // because only INSERT knows which table columns receive the rows. The
        // planner has already bound and cast each expression to its target column
        // by position, so stamp the table's declared schema onto the batches, and
        // append the derived partition keys after it.
        let written_schema =
            write_schema(&self.columns, &self.nullability, self.field_ids.as_deref());
        let batch_schema = Arc::new(Schema::new(
            written_schema
                .fields()
                .iter()
                .map(|field| field.as_ref().clone())
                .chain(self.partition_keys.derived_fields())
                .collect::<Vec<_>>(),
        ));
        // Selecting a variant column out of a table hands over the physical shape
        // that table shredded it into, while the column declared here is the
        // plain pair. Reassemble the documents, so the rows are shredded for the
        // file they land in rather than carrying the source table's layout across.
        let partition_keys = Arc::new(self.partition_keys.clone());
        let input = unshred_batches_spec(input).project({
            let batch_schema = batch_schema.clone();
            move || {
                let partition_keys = Arc::clone(&partition_keys);
                let batch_schema = batch_schema.clone();
                move |batch| {
                    let mut columns = batch.columns().to_vec();
                    columns.extend(
                        partition_keys
                            .derive(batch.columns())
                            .map_err(|error| UnaryError::Operator(Box::new(error)))?,
                    );
                    // The checked constructor refuses a NULL under a REQUIRED
                    // column, which fails the INSERT rather than storing it.
                    Ok(RecordBatch::try_new(batch_schema.clone(), columns)?)
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
            batch_schema,
            self.partition_keys.partitioning(),
            self.sort_by.clone().into(),
            TARGET_ROWS_PER_GROUP,
            TARGET_IN_MEMORY_BYTES_PER_FILE,
        );
        Ok(upload_files_spec(
            self.uploaded_files.clone(),
            self.columns.clone().into(),
            Some(on_start),
            encoded,
        ))
    }
}
