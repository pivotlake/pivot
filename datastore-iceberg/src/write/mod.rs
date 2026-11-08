//! Writing into a catalog table: what an INSERT reads off the loaded table
//! it writes into, the dataflow that writes its files, and the action that
//! commits them when the transaction does.
//!
//! The files go through the same encode and upload pipeline every datastore
//! writes with, through the table's store, the one the index reads it with. A table
//! partitioned by a transform of a column (a day of a timestamp, a bucket of
//! an id) has the transformed value computed per row ahead of the encoder,
//! which keys the files by it and records it as each file's partition value
//! without writing it: exactly what the manifest records and what another
//! engine reading the table expects.

mod commit;
mod data_files;

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, new_empty_array};
use arrow_schema::{DataType, Field, Schema};
use catalog::{CommitAction, CommitQueue};
use dispatch::{RecordBatchOperatorSpec, UnaryError};
use iceberg::arrow::type_to_arrow_type;
use iceberg::spec::{FormatVersion, PartitionSpec, PrimitiveType, Schema as IcebergSchema, Type};
use iceberg::transform::{TransformFunction, create_transform_function};
use object_storage::ObjectPath;
use parquet_engine::writing::upload::{
    UploadedFile, UploadedFiles, delete_uploaded_files, upload_files_spec,
};
use parquet_engine::writing::{Partitioning, encode_record_batches_spec, unshred_batches_spec};
use planner::catalog::{Column, Result as CatalogResult};

use crate::table::{LoadedTable, fetch_manifest_list_size};
use crate::write::data_files::AddedFile;
use crate::{Error, IcebergDatastore, Result};

/// The table property naming where new data files go, when a table keeps
/// them away from its location's `data/`. The crate honours its twin for
/// metadata objects itself.
const DATA_PATH_PROPERTY: &str = "write.data.path";

/// How a table's partition spec keys an INSERT's files: one key column per
/// partition field, named after the field. An identity field named after
/// its source column is keyed by the column itself; every other field's key
/// is derived per row, appended to the batch after the written columns and
/// dropped before the file is written.
pub(crate) struct PartitionKeys {
    /// The partition field names, in spec order.
    key_columns: Vec<String>,
    /// The keys computed per row, in the order they are appended.
    derived: Vec<DerivedKey>,
    /// Per partition field, in spec order, the Iceberg type of its value.
    pub value_types: Vec<PrimitiveType>,
}

/// A partition key computed from a column by the field's transform.
struct DerivedKey {
    name: String,
    source_column: usize,
    /// The Arrow type the transform takes its input in: the Iceberg type's
    /// own. A source column of another physical type (a string view, a
    /// narrow decimal) is cast to it first.
    input_type: DataType,
    result_type: DataType,
    transform: Arc<dyn TransformFunction>,
}

impl PartitionKeys {
    /// The keys for `spec` over a table of `columns`, or why the INSERT cannot
    /// partition by it: a source that is not one of the columns, a field
    /// named like a column (its key could not be told from the column), or a
    /// transform the spec crate cannot compute for the source's type.
    fn resolve(
        table: &str,
        schema: &IcebergSchema,
        spec: &PartitionSpec,
        columns: &[Column],
    ) -> Result<Self> {
        let mut key_columns = Vec::with_capacity(spec.fields().len());
        let mut derived = Vec::new();
        let mut value_types = Vec::with_capacity(spec.fields().len());
        for field in spec.fields() {
            let no_source = || Error::UnsupportedPartitionSource {
                table: table.to_string(),
                field: field.name.clone(),
            };
            let source_name = schema
                .name_by_field_id(field.source_id)
                .ok_or_else(no_source)?;
            let source_column = columns
                .iter()
                .position(|column| column.name == source_name)
                .ok_or_else(no_source)?;
            let source_type = schema
                .field_by_name(source_name)
                .and_then(|source| source.field_type.as_primitive_type().cloned())
                .ok_or_else(no_source)?;
            let unsupported = |message: String| Error::UnsupportedPartitionTransform {
                table: table.to_string(),
                field: field.name.clone(),
                transform: field.transform.to_string(),
                message,
            };
            let value_type = match field
                .transform
                .result_type(&Type::Primitive(source_type.clone()))
                .map_err(|error| unsupported(error.to_string()))?
            {
                Type::Primitive(primitive) => primitive,
                other => return Err(unsupported(format!("its values would be `{other}`"))),
            };
            value_types.push(value_type);
            if field.transform == iceberg::spec::Transform::Identity && field.name == source_name {
                key_columns.push(field.name.clone());
                continue;
            }
            if columns.iter().any(|column| column.name == field.name) {
                return Err(Error::PartitionFieldNamedLikeColumn {
                    table: table.to_string(),
                    field: field.name.clone(),
                });
            }
            let transform = create_transform_function(&field.transform)
                .map_err(|error| unsupported(error.to_string()))?;
            let input_type = type_to_arrow_type(&Type::Primitive(source_type))
                .map_err(|error| unsupported(error.to_string()))?;
            // Run the transform once on no rows: what it refuses, it refuses
            // now, as the INSERT's error, and what it produces names the key
            // column's type.
            let result_type = transform
                .transform(new_empty_array(&input_type))
                .map_err(|error| unsupported(error.to_string()))?
                .data_type()
                .clone();
            derived.push(DerivedKey {
                name: field.name.clone(),
                source_column,
                input_type,
                result_type,
                transform: Arc::from(transform),
            });
            key_columns.push(field.name.clone());
        }
        Ok(Self {
            key_columns,
            derived,
            value_types,
        })
    }

    /// How the encoder splits the files: by every key column, the derived
    /// ones (appended last) read for the key and not written.
    fn partitioning(&self) -> Partitioning {
        Partitioning {
            key_columns: self.key_columns.clone().into(),
            unwritten_trailing_columns: self.derived.len(),
        }
    }

    /// The fields of the derived key columns, in the order they are appended.
    fn derived_fields(&self) -> impl Iterator<Item = Field> + '_ {
        self.derived
            .iter()
            .map(|key| Field::new(key.name.clone(), key.result_type.clone(), true))
    }

    /// The derived key columns of a batch whose columns are `columns`, in the
    /// order they are appended.
    fn derive(&self, columns: &[ArrayRef]) -> iceberg::Result<Vec<ArrayRef>> {
        self.derived
            .iter()
            .map(|key| {
                let source = &columns[key.source_column];
                let input = if source.data_type() == &key.input_type {
                    source.clone()
                } else {
                    arrow_cast::cast(source, &key.input_type)?
                };
                key.transform.transform(input)
            })
            .collect()
    }
}

/// What an INSERT reads off the table it writes into, beyond what a scan
/// reads: all of it from the metadata the catalog returned, arranged as the
/// writer takes it.
impl LoadedTable {
    /// How the table's partition spec keys the INSERT's files, or why the
    /// table cannot be written: a format version whose manifests are not
    /// written, or a partition field whose key cannot be computed.
    pub(crate) fn partition_keys(&self) -> Result<PartitionKeys> {
        // A version 1 table's manifests have no sequence numbers, and this
        // datastore writes none.
        if self.metadata.format_version() < FormatVersion::V2 {
            return Err(Error::UnsupportedFormatVersion {
                table: self.name.to_string(),
                version: self.metadata.format_version() as u8,
            });
        }
        PartitionKeys::resolve(
            &self.name.to_string(),
            self.metadata.current_schema(),
            self.metadata.default_partition_spec(),
            self.columns(),
        )
    }

    /// The Iceberg field id of each column, in column order: the id a file's
    /// columns are written under, so a reader resolving columns by id finds
    /// them.
    pub(crate) fn field_ids(&self) -> Vec<i32> {
        let schema = self.metadata.current_schema();
        self.columns()
            .iter()
            .map(|column| {
                schema
                    .field_by_name(&column.name)
                    .expect("the table's columns are its schema's top-level fields")
                    .id
            })
            .collect()
    }

    /// Where under the store new data files go: where the table's
    /// `write.data.path` says, else `data/` under its location.
    pub(crate) fn data_dir(&self) -> Result<ObjectPath> {
        match self.metadata.properties().get(DATA_PATH_PROPERTY) {
            Some(path) => self.store().key_of(path.trim_end_matches('/')),
            None => Ok(self.store().key_of(self.metadata.location())?.join("data")),
        }
    }

    /// The Arrow schema the table's files are written under: its columns in
    /// order, each with its Iceberg field id, nullable as the table declares
    /// it. A field the table declares `required` is written REQUIRED, and a
    /// NULL headed for it fails the write.
    fn build_file_schema(&self) -> Arc<Schema> {
        let field_ids = self.field_ids();
        let nullability = self.nullability();
        let fields = self
            .columns()
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
                metadata.insert(
                    parquet_engine::PARQUET_FIELD_ID_METADATA_KEY.to_string(),
                    field_ids[position].to_string(),
                );
                field.with_metadata(metadata)
            })
            .collect::<Vec<_>>();
        Arc::new(Schema::new(fields))
    }
}

/// Build the dataflow that writes `input`'s rows into `table` as Parquet
/// files and emits the inserted-row count, or refuse the INSERT before any
/// file is written when the table cannot be written. Each finished file is
/// pushed onto `uploaded_files` for the statement's transaction to commit.
pub(crate) fn build_insert_spec(
    table: &Arc<LoadedTable>,
    commit_actions: CommitQueue<IcebergDatastore>,
    input: RecordBatchOperatorSpec,
) -> Result<RecordBatchOperatorSpec> {
    let partition_keys = Arc::new(table.partition_keys()?);
    let data_dir = table.data_dir()?;
    let store = table.store().store();
    let uploaded_files = UploadedFiles::new();
    // The pending insert this statement commits with. Staged before any file
    // can land, so whatever lands is owned by a staged action the transaction
    // can commit or discard.
    let pending_insert: Box<dyn CommitAction<IcebergDatastore>> = Box::new(PendingInsert {
        table: table.clone(),
        partition_keys: partition_keys.clone(),
        data_dir: data_dir.clone(),
        uploaded_files: uploaded_files.clone(),
    });
    let stage_pending_insert = Box::new(move || commit_actions.push(pending_insert));

    // In `INSERT INTO x VALUES (...)`, VALUES emits batches with unnamed
    // columns because only INSERT knows which table columns receive the rows.
    // The planner has already bound and cast each expression to its target
    // column by position, so stamp the table's declared schema onto the
    // batches, and append the derived partition keys after it.
    let written_schema = table.build_file_schema();
    let batch_schema = Arc::new(Schema::new(
        written_schema
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .chain(partition_keys.derived_fields())
            .collect::<Vec<_>>(),
    ));
    // Selecting a variant column out of a table hands over the physical shape
    // that table shredded it into, while the column declared here is the
    // plain pair. Reassemble the documents, so the rows are shredded for the
    // file they land in rather than carrying the source table's layout across.
    let input = unshred_batches_spec(input).project({
        let partition_keys = partition_keys.clone();
        let batch_schema = batch_schema.clone();
        move || {
            let partition_keys = partition_keys.clone();
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
    // This target measures retained Arrow data, not encoded Parquet bytes.
    // The collector applies it to each pending partition file and cuts only
    // after adding a complete batch. Compression and encoding normally make
    // the resulting files substantially smaller on disk.
    const TARGET_IN_MEMORY_BYTES_PER_FILE: usize = 900 * 1024 * 1024;
    let encoded = encode_record_batches_spec(
        input,
        batch_schema,
        partition_keys.partitioning(),
        table.sort_by.clone().into(),
        TARGET_ROWS_PER_GROUP,
        TARGET_IN_MEMORY_BYTES_PER_FILE,
    );
    Ok(upload_files_spec(
        store.clone(),
        data_dir,
        table.columns().to_vec().into(),
        uploaded_files.queue(),
        Some(stage_pending_insert),
        encoded,
    ))
}

/// The files one INSERT uploaded, waiting for its transaction to append
/// them to the table in one snapshot. Staged by the statement's upload stage
/// as the dataflow starts; the files arrive on `uploaded_files` as it runs, so
/// a sink that wrote no rows appends nothing. Carries the table the files
/// were written into, with the partition keys and the directory they were
/// written under, so the commit records them as they were written and a
/// discard can remove them.
pub(crate) struct PendingInsert {
    pub table: Arc<LoadedTable>,
    pub partition_keys: Arc<PartitionKeys>,
    pub data_dir: ObjectPath,
    pub uploaded_files: UploadedFiles,
}

impl CommitAction<IcebergDatastore> for PendingInsert {
    /// Commit the files as one append snapshot and publish the table at the
    /// landed version. A commit that is refused or fails leaves the files it
    /// did not land as a discard does: removed, since no snapshot names them.
    fn apply(self: Box<Self>, datastore: &IcebergDatastore) -> CatalogResult<()> {
        let uploaded = self.uploaded_files.take();
        if uploaded.is_empty() {
            return Ok(());
        }
        let committed = self.commit_and_publish(datastore, &uploaded);
        if committed.is_err() {
            delete_uploaded_files(
                self.table.store().store().as_ref(),
                &self.data_dir,
                &uploaded,
            );
        }
        Ok(committed?)
    }

    fn discard(self: Box<Self>) {
        self.uploaded_files
            .delete_from(self.table.store().store().as_ref(), &self.data_dir);
    }
}

impl PendingInsert {
    /// Commit `uploaded` as one append snapshot of the table as the index
    /// holds it now, through the client the last refresh built, and publish
    /// the table at the landed version into the index, so the next query
    /// sees the rows before the background refresh does.
    fn commit_and_publish(
        &self,
        datastore: &IcebergDatastore,
        uploaded: &[UploadedFile],
    ) -> Result<()> {
        let store = self.table.store();
        let files: Vec<AddedFile> = uploaded
            .iter()
            .map(|uploaded| AddedFile {
                uri: store.uri_of(&self.data_dir.join(uploaded.file.path.as_str())),
                size: uploaded.file.size,
                partition: uploaded.partition.clone(),
                stats: parquet_engine::aggregate_file_stats(&uploaded.row_groups),
            })
            .collect();
        // The commit is built on the table as the index has it: the bound
        // version, or a later one a refresh brought since.
        let current = datastore
            .current_table(&self.table.name)
            .unwrap_or_else(|| self.table.table.clone());
        let table = self.commit_append(&datastore.catalog(), &current, &files)?;
        let manifest_list_size = fetch_manifest_list_size(&self.table.name, &table, store)?
            .expect("an append leaves the table with a current snapshot");
        datastore.publish_table(&self.table.name, table, store.clone(), manifest_list_size);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, Int64Array, StringViewArray, TimestampMicrosecondArray};
    use iceberg::spec::{NestedField, Transform};
    use planner::types::Type as PivotType;

    use super::*;

    fn events_schema() -> IcebergSchema {
        IcebergSchema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(2, "at", Type::Primitive(PrimitiveType::Timestamp)).into(),
                NestedField::optional(3, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap()
    }

    fn events_columns() -> Vec<Column> {
        vec![
            Column {
                name: "id".to_string(),
                col_type: PivotType::Int64,
            },
            Column {
                name: "at".to_string(),
                col_type: PivotType::Timestamp,
            },
            Column {
                name: "name".to_string(),
                col_type: PivotType::Utf8,
            },
        ]
    }

    fn spec(fields: &[(&str, &str, Transform)]) -> PartitionSpec {
        let mut builder = PartitionSpec::builder(Arc::new(events_schema()));
        for (source, name, transform) in fields {
            builder = builder
                .add_partition_field(*source, *name, *transform)
                .unwrap();
        }
        builder.build().unwrap()
    }

    /// A spec as a catalog could hand it over, unchecked against the schema:
    /// what the builder refuses to build, the metadata may still say.
    fn unchecked_spec(source_id: i32, name: &str, transform: &str) -> PartitionSpec {
        serde_json::from_str(&format!(
            r#"{{"spec-id":0,"fields":[{{"source-id":{source_id},"field-id":1000,"name":"{name}","transform":"{transform}"}}]}}"#
        ))
        .unwrap()
    }

    #[test]
    fn a_day_of_a_timestamp_and_a_bucket_of_a_string_are_derived_per_row() {
        let keys = PartitionKeys::resolve(
            "db.events",
            &events_schema(),
            &spec(&[
                ("at", "at_day", Transform::Day),
                ("name", "name_bucket", Transform::Bucket(8)),
                ("id", "id", Transform::Identity),
            ]),
            &events_columns(),
        )
        .unwrap();
        let day_micros = 86_400_000_000i64;
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(TimestampMicrosecondArray::from(vec![0, 3 * day_micros + 5])),
            Arc::new(StringViewArray::from(vec!["ann", "bob"])),
        ];

        let derived = keys.derive(&columns).unwrap();

        assert_eq!(keys.key_columns, ["at_day", "name_bucket", "id"]);
        assert_eq!(
            keys.value_types,
            [PrimitiveType::Date, PrimitiveType::Int, PrimitiveType::Long]
        );
        assert_eq!(keys.partitioning().unwritten_trailing_columns, 2);
        assert_eq!(
            derived
                .iter()
                .map(|column| column.data_type().clone())
                .collect::<Vec<_>>(),
            [DataType::Date32, DataType::Int32]
        );
        let days = derived[0]
            .as_any()
            .downcast_ref::<arrow_array::Date32Array>()
            .unwrap();
        assert_eq!(days.values().as_ref(), [0, 3]);
        let buckets = derived[1]
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        assert!(
            buckets
                .values()
                .iter()
                .all(|bucket| (0..8).contains(bucket))
        );
        assert_eq!(
            keys.derived_fields()
                .map(|field| (field.name().clone(), field.data_type().clone()))
                .collect::<Vec<_>>(),
            [
                ("at_day".to_string(), DataType::Date32),
                ("name_bucket".to_string(), DataType::Int32),
            ]
        );
    }

    #[test]
    fn a_partition_field_named_like_a_column_is_refused() {
        let error = PartitionKeys::resolve(
            "db.events",
            &events_schema(),
            &unchecked_spec(2, "name", "day"),
            &events_columns(),
        )
        .err()
        .expect("the plan is refused");

        assert!(
            matches!(error, Error::PartitionFieldNamedLikeColumn { ref field, .. } if field == "name"),
            "{error}"
        );
    }

    #[test]
    fn a_transform_the_source_type_does_not_take_is_refused_when_planned() {
        let error = PartitionKeys::resolve(
            "db.events",
            &events_schema(),
            &unchecked_spec(3, "name_day", "day"),
            &events_columns(),
        )
        .err()
        .expect("the plan is refused");

        assert!(
            matches!(error, Error::UnsupportedPartitionTransform { ref field, .. } if field == "name_day"),
            "{error}"
        );
    }
}
