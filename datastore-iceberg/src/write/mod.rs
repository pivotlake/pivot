//! Writing into a catalog table: what an INSERT needs to know about its
//! target, the dataflow that writes its files, and the action that commits
//! them when the transaction does.
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
mod manifest_writer;

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, new_empty_array};
use arrow_schema::{DataType, Field, Schema};
use crossbeam_deque::Injector;
use dispatch::RecordBatchOperatorSpec;
use iceberg::arrow::type_to_arrow_type;
use iceberg::spec::{FormatVersion, PartitionSpec, PrimitiveType, Schema as IcebergSchema, Type};
use iceberg::transform::{TransformFunction, create_transform_function};
use iceberg::{NamespaceIdent, TableIdent};
use object_storage::ObjectPath;
use parquet_engine::writing::upload::{UploadedFile, upload_files_spec};
use parquet_engine::writing::{Partitioning, encode_record_batches_spec, unshred_batches_spec};
use planner::catalog::{Column, SchemaQualifiedTableName};

use crate::columns::table_columns;
use crate::store::TableStore;
use crate::table::{LoadedTable, identity_columns};
use crate::write::data_files::AddedFile;
use crate::{CommitAction, Error, IcebergDatastore, Result, drain_injector};

/// The table property naming where new data files go, and the one naming
/// where metadata objects go, when a table keeps them away from its
/// location's `data/` and `metadata/`.
const DATA_PATH_PROPERTY: &str = "write.data.path";
const METADATA_PATH_PROPERTY: &str = "write.metadata.path";

/// A catalog table's identifier in the catalog: its namespace is the
/// schema, its name the table.
pub(crate) fn table_ident(name: &SchemaQualifiedTableName) -> TableIdent {
    TableIdent::new(NamespaceIdent::new(name.schema.clone()), name.table.clone())
}

/// A table an INSERT writes into, as the transaction's index has it: the
/// store its files go through (the index's, whose vended keys the refresh
/// renews under a long write); the shape its files take; and where under the
/// store they and the commit's manifests are written.
pub(crate) struct WriteTarget {
    pub name: SchemaQualifiedTableName,
    pub store: TableStore,
    pub columns: Vec<Column>,
    /// Per column, whether a NULL may be written: an Iceberg field that is
    /// `required` is written REQUIRED, and a NULL headed for it fails the
    /// write.
    pub nullability: Vec<bool>,
    /// The Iceberg field id each column is written under, so a reader
    /// resolving columns by id finds them.
    pub field_ids: Vec<i32>,
    pub partition_spec_id: i32,
    pub partition_keys: PartitionKeys,
    /// The columns the table's sort order sorts by, as far as an INSERT can
    /// honour it: the identity sort fields, in order.
    pub sort_by: Vec<String>,
    /// Where under the store new data files go, as the table names it.
    pub data_dir: ObjectPath,
    /// Where under the store the commit's manifests go.
    pub metadata_dir: ObjectPath,
}

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
    fn plan(
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
            derived_key_columns: self.derived.len(),
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

/// `table`, as the transaction's index holds it, as an INSERT's target, or
/// why it cannot be written: a format version whose manifests are not
/// written, a column type Pivot cannot represent, a partition field whose
/// key cannot be computed.
pub(crate) fn write_target(table: &LoadedTable) -> Result<WriteTarget> {
    let name = &table.name;
    let table_name = name.to_string();
    let store = table.store().clone();
    let metadata = table.metadata.as_ref();
    // A version 1 table's manifests have no sequence numbers, and this
    // datastore writes none.
    if metadata.format_version() < FormatVersion::V2 {
        return Err(Error::UnsupportedFormatVersion {
            table: table_name,
            version: metadata.format_version() as u8,
        });
    }
    let schema = metadata.current_schema();
    let columns = table_columns(&table_name, schema)?.columns().to_vec();
    let mut field_ids = Vec::with_capacity(columns.len());
    let mut nullability = Vec::with_capacity(columns.len());
    for column in &columns {
        let field = schema
            .field_by_name(&column.name)
            .expect("the table's columns are its schema's top-level fields");
        field_ids.push(field.id);
        nullability.push(!field.required);
    }
    let partition_spec = metadata.default_partition_spec();
    let partition_keys = PartitionKeys::plan(&table_name, schema, partition_spec, &columns)?;
    let sort_by = identity_columns(
        schema,
        metadata
            .default_sort_order()
            .fields
            .iter()
            .map(|field| (field.source_id, &field.transform)),
    );
    let table_dir = store.key_of(metadata.location())?;
    let properties = metadata.properties();
    let data_dir = match properties.get(DATA_PATH_PROPERTY) {
        Some(path) => store.key_of(path.trim_end_matches('/'))?,
        None => table_dir.join("data"),
    };
    let metadata_dir = match properties.get(METADATA_PATH_PROPERTY) {
        Some(path) => store.key_of(path.trim_end_matches('/'))?,
        None => table_dir.join("metadata"),
    };
    Ok(WriteTarget {
        name: name.clone(),
        store,
        columns,
        nullability,
        field_ids,
        partition_spec_id: partition_spec.spec_id(),
        partition_keys,
        sort_by,
        data_dir,
        metadata_dir,
    })
}

/// The Arrow schema `target`'s files are written under: its columns in
/// order, each with its Iceberg field id, nullable as the table declares
/// it (`nullability`).
fn write_schema(target: &WriteTarget, nullability: &[bool]) -> Arc<Schema> {
    let fields = target
        .columns
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
                target.field_ids[position].to_string(),
            );
            field.with_metadata(metadata)
        })
        .collect::<Vec<_>>();
    Arc::new(Schema::new(fields))
}

/// Build the dataflow that writes `input`'s rows into `target` as Parquet
/// files and emits the inserted-row count. Each finished file is pushed onto
/// `uploaded_files` for the statement's transaction to commit.
pub(crate) fn build_insert_spec(
    target: &Arc<WriteTarget>,
    uploaded_files: Arc<Injector<UploadedFile>>,
    input: RecordBatchOperatorSpec,
) -> Result<RecordBatchOperatorSpec> {
    let store = target.store.store();
    store.prepare_write()?;

    // In `INSERT INTO x VALUES (...)`, VALUES emits batches with unnamed
    // columns because only INSERT knows which table columns receive the rows.
    // The planner has already bound and cast each expression to its target
    // column by position, so stamp the table's declared schema onto the
    // batches, and append the derived partition keys after it.
    let written_schema = write_schema(target, &target.nullability);
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
    let nullable_stamp = write_schema(target, &vec![true; target.columns.len()]);
    let input = unshred_batches_spec(input).project({
        let target = Arc::clone(target);
        let batch_schema = batch_schema.clone();
        move || {
            let target = Arc::clone(&target);
            let nullable_stamp = nullable_stamp.clone();
            let batch_schema = batch_schema.clone();
            move |batch| {
                // Selecting a variant column out of a table hands over the
                // physical shape that table shredded it into, while the
                // column declared here is the plain pair. Reassemble the
                // documents, so the rows are shredded for the file they land
                // in rather than carrying the source table's layout across.
                let checked =
                    RecordBatch::try_new(nullable_stamp.clone(), batch.columns().to_vec())
                        .expect("INSERT binding matches the target table schema");
                let row_count = checked.num_rows();
                let mut columns = checked.columns().to_vec();
                columns.extend(target.partition_keys.derive(checked.columns()).expect(
                    "the transforms accepted these column types when the insert was bound",
                ));
                // SAFETY: `checked` just verified the written columns against
                // the same fields in count, data type and row count, and the
                // derived columns were computed from them, one value per row,
                // in the types their fields declare; `batch_schema` differs
                // from `nullable_stamp` only in which fields are nullable,
                // and a null under a required field is refused by the
                // encoder's leaf builder rather than stored.
                unsafe { RecordBatch::new_unchecked(batch_schema.clone(), columns, row_count) }
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
        target.partition_keys.partitioning(),
        target.sort_by.clone().into(),
        TARGET_ROWS_PER_GROUP,
        TARGET_IN_MEMORY_BYTES_PER_FILE,
    );
    Ok(upload_files_spec(
        store.clone(),
        target.data_dir.clone(),
        target.columns.clone().into(),
        uploaded_files,
        encoded,
    ))
}

/// The files one INSERT uploaded, waiting for its transaction to append
/// them to the table in one snapshot. Staged when the statement's sink is
/// built; the files arrive on `uploaded_files` as the dataflow runs, so a
/// sink that wrote no rows appends nothing.
pub(crate) struct PendingInsert {
    pub target: Arc<WriteTarget>,
    pub uploaded_files: Arc<Injector<UploadedFile>>,
}

impl CommitAction for PendingInsert {
    /// Commit the files as one append snapshot, through a catalog client of
    /// the commit's own, and publish the table at the landed version into the
    /// datastore's index, so the next query sees the rows before the
    /// background refresh does.
    fn apply(self: Box<Self>, datastore: &IcebergDatastore) -> Result<()> {
        let files: Vec<AddedFile> = drain_injector(&self.uploaded_files)
            .into_iter()
            .map(|uploaded| AddedFile {
                uri: self
                    .target
                    .store
                    .uri_of(&self.target.data_dir.join(uploaded.file.path.as_str())),
                size: uploaded.file.size,
                partition: uploaded.partition,
                stats: parquet_engine::aggregate_file_stats(&uploaded.row_groups),
            })
            .collect();
        if files.is_empty() {
            return Ok(());
        }
        let catalog = datastore.catalog_client()?;
        let committed =
            commit::commit_append(&catalog, &datastore.dispatcher, &self.target, &files)?;
        datastore.publish_table(
            &self.target.name,
            committed.table,
            self.target.store.clone(),
            committed.manifest_list_size,
        );
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
        let keys = PartitionKeys::plan(
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
        assert_eq!(keys.partitioning().derived_key_columns, 2);
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
        let error = PartitionKeys::plan(
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
        let error = PartitionKeys::plan(
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
