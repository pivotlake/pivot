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
use crossbeam_deque::Injector;
use dispatch::RecordBatchOperatorSpec;
use object_storage::{ObjectPath, ObjectStore};
use parquet_engine::writing::upload::{UploadedFile, upload_files_spec};
use parquet_engine::writing::{encode_record_batches_spec, unshred_batches_spec};
use planner::catalog::Column;

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
    /// The store the table's files live in, with the credentials its writes
    /// need.
    pub store: Arc<dyn ObjectStore>,
    /// The table's directory under the store, which every uploaded file's
    /// path is relative to.
    pub directory: ObjectPath,
    /// Where under the table new files are written: nothing for a table that
    /// keeps its files flat, `data` for one laid out the way other writers
    /// expect. Paths reported for the files carry it.
    pub file_prefix: ObjectPath,
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

/// Build the dataflow that writes `input`'s rows into `target` as Parquet and
/// emits the inserted-row count. Each finished file is pushed onto
/// `uploaded_files` for the statement's transaction to commit. This is the
/// write mirror of the scan's
/// [`table_input_with_filter_and_eq_predicates`](parquet_engine::table_input_with_filter_and_eq_predicates):
/// it wires the encode pipeline into the engine's upload sink.
pub fn build_insert_spec(
    target: &InsertTarget,
    uploaded_files: Arc<Injector<UploadedFile>>,
    input: RecordBatchOperatorSpec,
) -> object_storage::Result<RecordBatchOperatorSpec> {
    target.store.prepare_write()?;

    // In `INSERT INTO x VALUES (...)`, VALUES emits batches with unnamed columns
    // because only INSERT knows which table columns receive the rows. The
    // planner has already bound and cast each expression to its target column
    // by position, so stamp the table's declared schema onto the batches, and
    // append the derived partition keys after it.
    let written_schema = write_schema(
        &target.columns,
        &target.nullability,
        target.field_ids.as_deref(),
    );
    let batch_schema = Arc::new(Schema::new(
        written_schema
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .chain(target.partition_keys.derived_fields())
            .collect::<Vec<_>>(),
    ));
    // The batch is first stamped with every column nullable, which checks
    // everything about the columns but their nulls, then restamped with the
    // table's own nullability. A NULL headed for a REQUIRED column is the
    // encoder's to refuse, as a write error, not this projection's to panic
    // over: the checked constructor would.
    let nullable_stamp = write_schema(
        &target.columns,
        &vec![true; target.columns.len()],
        target.field_ids.as_deref(),
    );
    let partition_keys = Arc::new(target.partition_keys.clone());
    let input = unshred_batches_spec(input).project({
        let batch_schema = batch_schema.clone();
        move || {
            let partition_keys = Arc::clone(&partition_keys);
            let nullable_stamp = nullable_stamp.clone();
            let batch_schema = batch_schema.clone();
            move |batch| {
                // Selecting a variant column out of a table hands over the
                // physical shape that table shredded it into, while the column
                // declared here is the plain pair. Reassemble the documents, so
                // the rows are shredded for the file they land in rather than
                // carrying the source table's layout across.
                let checked =
                    RecordBatch::try_new(nullable_stamp.clone(), batch.columns().to_vec())
                        .expect("INSERT binding matches the target table schema");
                let row_count = checked.num_rows();
                let mut columns = checked.columns().to_vec();
                columns.extend(partition_keys.derive(checked.columns()).expect(
                    "the transforms accepted these column types when the insert was bound",
                ));
                // SAFETY: `checked` just verified the written columns against
                // the same fields in count, data type and row count, and the
                // derived columns were computed from them, one value per row,
                // in the types their fields declare; `batch_schema` differs
                // from `nullable_stamp` only in which fields are nullable and
                // in the derived fields appended, and a null under a required
                // field is refused by the encoder's leaf builder rather than
                // stored.
                unsafe { RecordBatch::new_unchecked(batch_schema.clone(), columns, row_count) }
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
        target.partition_keys.partitioning(),
        target.sort_by.clone().into(),
        TARGET_ROWS_PER_GROUP,
        TARGET_IN_MEMORY_BYTES_PER_FILE,
    );
    Ok(upload_files_spec(
        target.store.clone(),
        target.directory.clone(),
        target.file_prefix.clone(),
        target.columns.clone().into(),
        uploaded_files,
        encoded,
    ))
}
