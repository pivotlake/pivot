//! One data file as the datastore holds it, and as a manifest records it.
//!
//! The datastore keeps a file by its store identity, its partition tuple and
//! its column statistics as Pivot's typed Arrow scalars, the shape every
//! writer and the compaction planner work with. A manifest records the same
//! file by full URI with Iceberg values keyed by field id. The two conversions
//! here are inverses, so a file written, read back and re-emitted by a later
//! commit is recorded unchanged.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, Date32Array, Datum as ArrowDatum, Decimal64Array, Decimal128Array,
    Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, Scalar,
    StringViewArray, TimestampMicrosecondArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, FormatVersion, Literal,
    ManifestEntry, ManifestFile, PrimitiveLiteral, PrimitiveType, Schema, Struct, TableMetadata,
    Transform, Type as IcebergType,
};
use object_storage::{FileRef, ObjectPath};
use parquet_engine::{FileStats, PartitionValues};
use planner::catalog::Column;
use planner::types::Type;

use crate::columns::to_pivot_type;
use crate::datum::array_to_datum;
use crate::file_index::{IndexedFile, SortKeyEncoder};
use crate::layout::TableLocation;
use crate::{Error, Result};

/// String bounds are cut to this many code points: a bound only has to
/// bracket the file's values, and a long string would otherwise be recorded
/// in full in every manifest and manifest-list summary.
const STRING_BOUND_CODE_POINTS: usize = 16;

/// Where a committed file sits in its table's metadata: what a later commit
/// needs to re-emit the entry as EXISTING (carried into a rewritten manifest)
/// or DELETED (retired), both of which must keep the numbers the file was
/// committed with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManifestOrigin {
    pub snapshot_id: i64,
    pub sequence_number: i64,
    pub file_sequence_number: i64,
    pub first_row_id: Option<i64>,
    /// The manifest the entry was read from, as the manifest list names it:
    /// one copy shared by every file of the manifest.
    pub manifest_path: Arc<str>,
}

/// One data file: its store identity, the partition tuple a partitioned
/// INSERT stamps on it (`None` for an unpartitioned table; keyed by partition
/// field name, each value in the type the field's transform produces), its
/// statistics, the sort order its rows are in, and, once committed, where
/// its table's metadata records it.
#[derive(Clone)]
pub struct DataFileEntry {
    pub file: FileRef,
    pub partition: Option<PartitionValues>,
    /// The file's statistics: its row count, and per column the bounds and
    /// null count. Written into the manifest, so a file that has none cannot
    /// be committed.
    pub stats: Option<Arc<FileStats>>,
    /// The id of the table sort order the file's rows are in, as its manifest
    /// entry records it: [`UNSORTED_ORDER_ID`](crate::schema::UNSORTED_ORDER_ID)
    /// for a file written in no order, `None` for one whose entry says
    /// nothing, which every reader takes as unsorted: a file adopted as it
    /// lay, whose order nobody knows. A file a writer of this process
    /// ordered gets its id from
    /// [`Table::find_sort_order_id_of_rows_sorted_by`](crate::Table::find_sort_order_id_of_rows_sorted_by).
    pub sort_order_id: Option<i32>,
    pub origin: Option<ManifestOrigin>,
}

/// The Pivot type of the declared column `column`.
fn find_pivot_type<'a>(columns: &'a [Column], column: &str) -> Option<&'a Type> {
    columns
        .iter()
        .find(|declared| declared.name == column)
        .map(|declared| &declared.col_type)
}

/// The Iceberg primitive type of `schema`'s top-level field `name`, and its
/// id.
fn find_primitive_field(schema: &Schema, name: &str) -> Option<(i32, PrimitiveType)> {
    let field = schema.field_by_name(name)?;
    let primitive = field.field_type.as_primitive_type()?.clone();
    Some((field.id, primitive))
}

/// Per field of the table's default partition spec, in spec order, the
/// Iceberg type of its values: the field's transform's result over its
/// source field in the current schema.
fn read_partition_value_types(
    metadata: &TableMetadata,
    file: &ObjectPath,
) -> Result<Vec<PrimitiveType>> {
    let invalid = |message: String| Error::InvalidDataFile {
        file: file.to_string(),
        message,
    };
    let partition_type = metadata
        .default_partition_spec()
        .partition_type(metadata.current_schema())
        .map_err(|error| invalid(error.to_string()))?;
    partition_type
        .fields()
        .iter()
        .map(|field| {
            field
                .field_type
                .as_primitive_type()
                .cloned()
                .ok_or_else(|| {
                    invalid(format!("partition field `{}` is not primitive", field.name))
                })
        })
        .collect()
}

/// The Pivot type the values of the partition field `field` are kept as:
/// the source column's own for an identity field, else the Pivot reading
/// of the transform's result type. `None` for a field whose source is not
/// a declared column.
fn find_partition_pivot_type(
    schema: &Schema,
    columns: &[Column],
    field: &iceberg::spec::PartitionField,
    value_type: &PrimitiveType,
) -> Option<Type> {
    match field.transform {
        Transform::Identity => schema
            .name_by_field_id(field.source_id)
            .and_then(|name| find_pivot_type(columns, name))
            .cloned(),
        _ => to_pivot_type(&IcebergType::Primitive(value_type.clone())),
    }
}

/// The partition tuple `partition` holds, laid out by the table's default
/// partition spec, as the datastore keys it: partition field name to typed
/// scalar of the declared `columns`' Pivot types, a NULL value as a null
/// scalar. `None` for an unpartitioned table.
pub(crate) fn read_partition_values(
    metadata: &TableMetadata,
    columns: &[Column],
    partition: &Struct,
    file: &ObjectPath,
) -> Result<Option<PartitionValues>> {
    let fields = metadata.default_partition_spec().fields();
    if fields.is_empty() {
        return Ok(None);
    }
    let schema = metadata.current_schema();
    let value_types = read_partition_value_types(metadata, file)?;
    let mut values = PartitionValues::new();
    for (position, (field, value_type)) in fields.iter().zip(&value_types).enumerate() {
        let Some(pivot_type) = find_partition_pivot_type(schema, columns, field, value_type) else {
            continue;
        };
        let scalar = match partition.fields().get(position) {
            Some(Some(Literal::Primitive(literal))) => literal_to_scalar(literal, &pivot_type),
            _ => Some(null_scalar(&pivot_type)),
        };
        if let Some(scalar) = scalar {
            values.insert(field.name.clone(), scalar);
        }
    }
    Ok(Some(values))
}

impl DataFileEntry {
    /// This entry as a manifest of the table `metadata` describes records
    /// it: its field ids from the current schema, its tuple laid out by the
    /// default partition spec, its path as a URI under `location`. Requires
    /// the entry's statistics: a manifest entry carries the file's row
    /// count, and a file without one cannot be committed.
    pub fn create_data_file(
        &self,
        metadata: &TableMetadata,
        location: &TableLocation,
    ) -> Result<DataFile> {
        let schema = metadata.current_schema();
        let stats = self.stats.as_ref().ok_or_else(|| Error::MissingFileStats {
            file: self.file.path.to_string(),
        })?;
        let record_count = stats
            .num_records
            .and_then(|count| u64::try_from(count).ok())
            .ok_or_else(|| Error::MissingFileStats {
                file: self.file.path.to_string(),
            })?;
        let mut lower_bounds = HashMap::new();
        let mut upper_bounds = HashMap::new();
        let mut null_value_counts = HashMap::new();
        for (name, min) in &stats.min_values {
            let Some(max) = stats.max_values.get(name) else {
                continue;
            };
            let Some((field_id, primitive)) = find_primitive_field(schema, name) else {
                continue;
            };
            // A bound Iceberg cannot hold (a `UInt64` above `i64::MAX`, a NaN)
            // leaves the column unbounded for that file, which prunes nothing.
            let (Some(lower), Some(upper)) = (
                array_to_datum(min.as_ref(), &primitive).map(truncate_lower),
                array_to_datum(max.as_ref(), &primitive).and_then(truncate_upper),
            ) else {
                continue;
            };
            lower_bounds.insert(field_id, lower);
            upper_bounds.insert(field_id, upper);
        }
        for (name, nulls) in &stats.null_counts {
            if let (Some((field_id, _)), Ok(nulls)) =
                (find_primitive_field(schema, name), u64::try_from(*nulls))
            {
                null_value_counts.insert(field_id, nulls);
            }
        }
        let partition = self.build_partition_struct(metadata)?;
        let mut builder = DataFileBuilder::default();
        builder
            .content(DataContentType::Data)
            .file_path(location.uri_for(&self.file.path)?)
            .file_format(DataFileFormat::Parquet)
            .partition(partition)
            .partition_spec_id(metadata.default_partition_spec().spec_id())
            .record_count(record_count)
            .file_size_in_bytes(self.file.size)
            .lower_bounds(lower_bounds)
            .upper_bounds(upper_bounds)
            .null_value_counts(null_value_counts)
            .first_row_id(match metadata.format_version() {
                FormatVersion::V1 | FormatVersion::V2 => None,
                FormatVersion::V3 => self.origin.as_ref().and_then(|origin| origin.first_row_id),
            });
        if let Some(sort_order_id) = self.sort_order_id {
            builder.sort_order_id(sort_order_id);
        }
        builder.build().map_err(|error| Error::InvalidDataFile {
            file: self.file.path.to_string(),
            message: error.to_string(),
        })
    }

    /// The entry's partition tuple laid out as the spec orders its fields,
    /// each value in the Iceberg type its field's transform produces. An
    /// identity field the tuple lacks is a NULL partition value (a column the
    /// file does not have reads as NULL); a derived field's value must be
    /// there. A partitioned table's entry must carry a tuple: an Iceberg
    /// tuple has no "unknown" value, and an all-null tuple would make every
    /// other reader prune the file wrongly.
    fn build_partition_struct(&self, metadata: &TableMetadata) -> Result<Struct> {
        let fields = metadata.default_partition_spec().fields();
        if fields.is_empty() {
            return Ok(Struct::empty());
        }
        let tuple = self
            .partition
            .as_ref()
            .ok_or_else(|| Error::PartitionNotConstant {
                file: self.file.path.to_string(),
            })?;
        let invalid = |message: String| Error::InvalidDataFile {
            file: self.file.path.to_string(),
            message,
        };
        let value_types = read_partition_value_types(metadata, &self.file.path)?;
        let mut values = Vec::with_capacity(fields.len());
        for (field, value_type) in fields.iter().zip(&value_types) {
            let value = match tuple.get(&field.name) {
                Some(scalar) => {
                    let (array, _) = ArrowDatum::get(scalar);
                    if array.is_null(0) {
                        None
                    } else {
                        let datum = array_to_datum(array, value_type).ok_or_else(|| {
                            invalid(format!(
                                "partition value of `{}` has no Iceberg form in `{value_type}`",
                                field.name
                            ))
                        })?;
                        Some(Literal::Primitive(datum.literal().clone()))
                    }
                }
                None if field.transform == Transform::Identity => None,
                None => {
                    return Err(invalid(format!(
                        "the file has no value for partition field `{}`",
                        field.name
                    )));
                }
            };
            values.push(value);
        }
        Ok(Struct::from_iter(values))
    }

    /// `entry`, read from `manifest`, as the datastore holds it. Inherits what
    /// a manifest leaves implicit on its ADDED entries: the snapshot that
    /// added the file, its sequence numbers, and (`first_row_id` being the
    /// caller's running assignment) its first row id.
    pub fn from_manifest_entry(
        entry: &ManifestEntry,
        manifest: &ManifestFile,
        first_row_id: Option<i64>,
        metadata: &TableMetadata,
        columns: &[Column],
        location: &TableLocation,
    ) -> Result<DataFileEntry> {
        IndexedFile::from_manifest_entry(
            entry.clone(),
            manifest,
            &Arc::from(manifest.manifest_path.as_str()),
            metadata.current_schema().schema_id(),
            metadata.default_partition_spec().spec_id(),
            first_row_id,
            &SortKeyEncoder::new(metadata, columns, &[])?,
            metadata,
            location,
        )?
        .to_entry(metadata, columns)
    }
}

/// `literal` as a one-value Arrow scalar of `pivot_type`'s physical type, or
/// `None` for a pairing that does not read (a different writer's bound).
pub(crate) fn literal_to_scalar(
    literal: &PrimitiveLiteral,
    pivot_type: &Type,
) -> Option<Scalar<ArrayRef>> {
    let array: ArrayRef = match (literal, pivot_type) {
        (PrimitiveLiteral::Boolean(value), Type::Boolean) => {
            Arc::new(BooleanArray::from(vec![*value]))
        }
        (PrimitiveLiteral::Int(value), Type::Int8) => {
            Arc::new(Int8Array::from(vec![i8::try_from(*value).ok()?]))
        }
        (PrimitiveLiteral::Int(value), Type::Int16) => {
            Arc::new(Int16Array::from(vec![i16::try_from(*value).ok()?]))
        }
        (PrimitiveLiteral::Int(value), Type::Int32) => Arc::new(Int32Array::from(vec![*value])),
        (PrimitiveLiteral::Int(value), Type::UInt8) => {
            Arc::new(UInt8Array::from(vec![u8::try_from(*value).ok()?]))
        }
        (PrimitiveLiteral::Int(value), Type::UInt16) => {
            Arc::new(UInt16Array::from(vec![u16::try_from(*value).ok()?]))
        }
        (PrimitiveLiteral::Int(value), Type::Date) => Arc::new(Date32Array::from(vec![*value])),
        (PrimitiveLiteral::Long(value), Type::Int64) => Arc::new(Int64Array::from(vec![*value])),
        (PrimitiveLiteral::Long(value), Type::UInt32) => {
            Arc::new(UInt32Array::from(vec![u32::try_from(*value).ok()?]))
        }
        (PrimitiveLiteral::Long(value), Type::UInt64) => {
            Arc::new(UInt64Array::from(vec![u64::try_from(*value).ok()?]))
        }
        (PrimitiveLiteral::Long(value), Type::Timestamp) => {
            Arc::new(TimestampMicrosecondArray::from(vec![*value]))
        }
        (PrimitiveLiteral::Long(value), Type::TimestampTz) => Arc::new(
            TimestampMicrosecondArray::from(vec![*value])
                .with_timezone(planner::types::UTC_TIMEZONE),
        ),
        (PrimitiveLiteral::Float(value), Type::Float32) => {
            Arc::new(Float32Array::from(vec![value.0]))
        }
        (PrimitiveLiteral::Double(value), Type::Float64) => {
            Arc::new(Float64Array::from(vec![value.0]))
        }
        (PrimitiveLiteral::String(value), Type::Utf8) => {
            Arc::new(StringViewArray::from(vec![value.as_str()]))
        }
        (PrimitiveLiteral::Int128(unscaled), Type::Decimal { precision, scale }) => {
            if *precision <= planner::types::MAX_DECIMAL64_PRECISION {
                Arc::new(
                    Decimal64Array::from(vec![i64::try_from(*unscaled).ok()?])
                        .with_precision_and_scale(*precision, *scale)
                        .ok()?,
                )
            } else {
                Arc::new(
                    Decimal128Array::from(vec![*unscaled])
                        .with_precision_and_scale(*precision, *scale)
                        .ok()?,
                )
            }
        }
        _ => return None,
    };
    Some(Scalar::new(array))
}

/// A one-row all-null scalar of `pivot_type`'s physical type.
fn null_scalar(pivot_type: &Type) -> Scalar<ArrayRef> {
    Scalar::new(arrow_array::new_null_array(
        &planner::types::physical_arrow_type(pivot_type),
        1,
    ))
}

/// A lower bound, cut to [`STRING_BOUND_CODE_POINTS`] when it is a string:
/// a prefix is at or below every value it was cut from.
fn truncate_lower(datum: Datum) -> Datum {
    match datum.literal() {
        PrimitiveLiteral::String(value) if value.chars().count() > STRING_BOUND_CODE_POINTS => {
            Datum::string(
                value
                    .chars()
                    .take(STRING_BOUND_CODE_POINTS)
                    .collect::<String>(),
            )
        }
        _ => datum,
    }
}

/// An upper bound, cut to [`STRING_BOUND_CODE_POINTS`] when it is a string:
/// the prefix with its last code point incremented is above every value it
/// was cut from. A prefix whose every code point is already the largest has
/// no such successor, which leaves the column unbounded above.
fn truncate_upper(datum: Datum) -> Option<Datum> {
    match datum.literal() {
        PrimitiveLiteral::String(value) if value.chars().count() > STRING_BOUND_CODE_POINTS => {
            let mut prefix: Vec<char> = value.chars().take(STRING_BOUND_CODE_POINTS).collect();
            while let Some(last) = prefix.pop() {
                if let Some(next) = ((last as u32 + 1)..=char::MAX as u32).find_map(char::from_u32)
                {
                    prefix.push(next);
                    return Some(Datum::string(prefix.into_iter().collect::<String>()));
                }
            }
            None
        }
        _ => Some(datum),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::build_schema;
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Date32Type;
    use arrow_schema::DataType;
    use iceberg::spec::{
        FormatVersion, ManifestStatus, PartitionSpec, PartitionSpecRef, TableMetadataBuilder,
    };
    use object_storage::{LocalStore, ObjectPath, ObjectStore};

    fn column(name: &str, col_type: Type) -> Column {
        Column {
            name: name.to_string(),
            col_type,
        }
    }

    /// A table over `columns`, partitioned by `partition_by`, in a temporary
    /// local store.
    struct Fixture {
        _dir: tempfile::TempDir,
        location: TableLocation,
        metadata: TableMetadata,
        columns: Vec<Column>,
    }

    impl Fixture {
        fn new(columns: Vec<Column>, partition_by: &[&str]) -> Self {
            let schema = build_schema(&columns).unwrap();
            let partition_by: Vec<String> = partition_by.iter().map(|c| c.to_string()).collect();
            let spec = crate::schema::build_partition_spec(&schema, &partition_by).unwrap();
            Self::with_spec(columns, spec)
        }

        /// A table partitioned by `transform` of the column `source`, the
        /// partition field named `field`.
        fn partitioned_by_transform(
            columns: Vec<Column>,
            source: &str,
            field: &str,
            transform: Transform,
        ) -> Self {
            let schema = build_schema(&columns).unwrap();
            let spec = PartitionSpec::builder(Arc::new(schema))
                .add_partition_field(source, field, transform)
                .unwrap()
                .build()
                .unwrap()
                .into_unbound();
            Self::with_spec(columns, spec)
        }

        fn with_spec(columns: Vec<Column>, spec: iceberg::spec::UnboundPartitionSpec) -> Self {
            let dir = tempfile::tempdir().unwrap();
            let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(dir.path()).unwrap());
            let location = TableLocation::new(store, ObjectPath::new("t")).unwrap();
            let schema = build_schema(&columns).unwrap();
            let metadata = TableMetadataBuilder::new(
                schema,
                spec,
                iceberg::spec::SortOrder::unsorted_order(),
                location.table_uri(),
                FormatVersion::V3,
                HashMap::new(),
            )
            .unwrap()
            .build()
            .unwrap()
            .metadata;
            Self {
                _dir: dir,
                location,
                metadata,
                columns,
            }
        }

        fn unpartitioned_spec() -> PartitionSpecRef {
            Arc::new(PartitionSpec::unpartition_spec())
        }
    }

    fn scalar(array: impl Array + 'static) -> Scalar<ArrayRef> {
        Scalar::new(Arc::new(array))
    }

    fn stats(
        rows: i64,
        bounds: &[(&str, ArrayRef, ArrayRef)],
        nulls: &[(&str, i64)],
    ) -> Arc<FileStats> {
        Arc::new(FileStats {
            num_records: Some(rows),
            min_values: bounds
                .iter()
                .map(|(name, min, _)| (name.to_string(), min.clone()))
                .collect(),
            max_values: bounds
                .iter()
                .map(|(name, _, max)| (name.to_string(), max.clone()))
                .collect(),
            null_counts: nulls
                .iter()
                .map(|(name, count)| (name.to_string(), *count))
                .collect(),
        })
    }

    fn manifest_for(path: &str, snapshot_id: i64, sequence_number: i64) -> ManifestFile {
        ManifestFile {
            manifest_path: path.to_string(),
            manifest_length: 0,
            partition_spec_id: 0,
            content: iceberg::spec::ManifestContentType::Data,
            sequence_number,
            min_sequence_number: sequence_number,
            added_snapshot_id: snapshot_id,
            added_files_count: None,
            existing_files_count: None,
            deleted_files_count: None,
            added_rows_count: None,
            existing_rows_count: None,
            deleted_rows_count: None,
            partitions: None,
            key_metadata: None,
            first_row_id: None,
        }
    }

    #[test]
    fn an_entry_round_trips_through_a_manifest_record() {
        let fixture = Fixture::new(
            vec![
                column("id", Type::UInt64),
                column("name", Type::Utf8),
                column("day", Type::Date),
            ],
            &["day"],
        );
        let entry = DataFileEntry {
            file: FileRef {
                path: ObjectPath::new("pivot-1.parquet"),
                size: 123,
            },
            partition: Some(PartitionValues::from([(
                "day".to_string(),
                scalar(Date32Array::from(vec![20_000])),
            )])),
            stats: Some(stats(
                7,
                &[
                    (
                        "id",
                        Arc::new(UInt64Array::from(vec![1])),
                        Arc::new(UInt64Array::from(vec![9])),
                    ),
                    (
                        "name",
                        Arc::new(StringViewArray::from(vec!["a"])),
                        Arc::new(StringViewArray::from(vec!["z"])),
                    ),
                ],
                &[("name", 2)],
            )),
            sort_order_id: None,
            origin: None,
        };

        let data_file = entry
            .create_data_file(&fixture.metadata, &fixture.location)
            .unwrap();
        let manifest_entry = ManifestEntry {
            status: ManifestStatus::Added,
            snapshot_id: None,
            sequence_number: None,
            file_sequence_number: None,
            data_file: data_file.clone(),
        };
        let read = DataFileEntry::from_manifest_entry(
            &manifest_entry,
            &manifest_for("m0.avro", 41, 3),
            Some(100),
            &fixture.metadata,
            &fixture.columns,
            &fixture.location,
        )
        .unwrap();

        assert_eq!(data_file.record_count(), 7);
        assert_eq!(data_file.lower_bounds()[&1], Datum::long(1));
        assert_eq!(data_file.upper_bounds()[&2], Datum::string("z"));
        assert_eq!(data_file.null_value_counts()[&2], 2);
        assert_eq!(
            data_file.partition().fields(),
            &[Some(Literal::Primitive(PrimitiveLiteral::Int(20_000)))]
        );
        assert_eq!(read.file, entry.file);
        assert!(parquet_engine::scalar_values_equal(
            read.partition.as_ref().unwrap(),
            entry.partition.as_ref().unwrap()
        ));
        let read_stats = read.stats.unwrap();
        assert_eq!(read_stats.num_records, Some(7));
        assert_eq!(
            read_stats.min_values["id"].as_ref().data_type(),
            &DataType::UInt64
        );
        assert_eq!(read_stats.null_counts["name"], 2);
        assert_eq!(
            read.origin.unwrap(),
            ManifestOrigin {
                snapshot_id: 41,
                sequence_number: 3,
                file_sequence_number: 3,
                first_row_id: Some(100),
                manifest_path: Arc::from("m0.avro"),
            }
        );
    }

    #[test]
    fn an_existing_entry_keeps_its_own_numbers() {
        let fixture = Fixture::new(vec![column("id", Type::Int64)], &[]);
        let entry = DataFileEntry {
            file: FileRef {
                path: ObjectPath::new("pivot-1.parquet"),
                size: 1,
            },
            partition: None,
            stats: Some(stats(1, &[], &[])),
            sort_order_id: None,
            origin: None,
        };
        let data_file = entry
            .create_data_file(&fixture.metadata, &fixture.location)
            .unwrap();

        let read = DataFileEntry::from_manifest_entry(
            &ManifestEntry {
                status: ManifestStatus::Existing,
                snapshot_id: Some(5),
                sequence_number: Some(2),
                file_sequence_number: Some(2),
                data_file,
            },
            &manifest_for("m1.avro", 9, 8),
            None,
            &fixture.metadata,
            &fixture.columns,
            &fixture.location,
        )
        .unwrap();

        let origin = read.origin.unwrap();
        assert_eq!((origin.snapshot_id, origin.sequence_number), (5, 2));
        assert!(read.partition.is_none());
    }

    #[test]
    fn bounds_iceberg_cannot_hold_are_left_out() {
        let fixture = Fixture::new(
            vec![column("big", Type::UInt64), column("f", Type::Float64)],
            &[],
        );
        let entry = DataFileEntry {
            file: FileRef {
                path: ObjectPath::new("pivot-1.parquet"),
                size: 1,
            },
            partition: None,
            stats: Some(stats(
                1,
                &[
                    (
                        "big",
                        Arc::new(UInt64Array::from(vec![0])),
                        Arc::new(UInt64Array::from(vec![u64::MAX])),
                    ),
                    (
                        "f",
                        Arc::new(Float64Array::from(vec![f64::NAN])),
                        Arc::new(Float64Array::from(vec![1.0])),
                    ),
                ],
                &[],
            )),
            sort_order_id: None,
            origin: None,
        };

        let data_file = entry
            .create_data_file(&fixture.metadata, &fixture.location)
            .unwrap();

        assert!(data_file.lower_bounds().is_empty());
        assert!(data_file.upper_bounds().is_empty());
    }

    #[test]
    fn long_string_bounds_are_cut_to_a_bracketing_prefix() {
        let long = "abcdefghijklmnopqrstuvwxyz";

        let lower = truncate_lower(Datum::string(long));
        let upper = truncate_upper(Datum::string(long)).unwrap();

        assert_eq!(lower, Datum::string("abcdefghijklmnop"));
        assert_eq!(upper, Datum::string("abcdefghijklmnoq"));
        assert_eq!(
            truncate_lower(Datum::string("short")),
            Datum::string("short")
        );
    }

    #[test]
    fn a_partitioned_table_refuses_an_entry_without_a_tuple() {
        let fixture = Fixture::new(vec![column("day", Type::Date)], &["day"]);
        let entry = DataFileEntry {
            file: FileRef {
                path: ObjectPath::new("pivot-1.parquet"),
                size: 1,
            },
            partition: None,
            stats: Some(stats(1, &[], &[])),
            sort_order_id: None,
            origin: None,
        };

        let error = entry
            .create_data_file(&fixture.metadata, &fixture.location)
            .unwrap_err();

        assert!(
            matches!(error, Error::PartitionNotConstant { .. }),
            "{error}"
        );
    }

    #[test]
    fn a_null_partition_value_round_trips_as_a_null_scalar() {
        let fixture = Fixture::new(vec![column("name", Type::Utf8)], &["name"]);
        let entry = DataFileEntry {
            file: FileRef {
                path: ObjectPath::new("pivot-1.parquet"),
                size: 1,
            },
            partition: Some(PartitionValues::from([(
                "name".to_string(),
                null_scalar(&Type::Utf8),
            )])),
            stats: Some(stats(1, &[], &[])),
            sort_order_id: None,
            origin: None,
        };

        let data_file = entry
            .create_data_file(&fixture.metadata, &fixture.location)
            .unwrap();
        let read = read_partition_values(
            &fixture.metadata,
            &fixture.columns,
            data_file.partition(),
            &entry.file.path,
        )
        .unwrap()
        .unwrap();

        assert_eq!(data_file.partition().fields(), &[None]);
        assert!(read["name"].get().0.is_null(0));
        let _ = Fixture::unpartitioned_spec();
    }

    /// A derived partition value (a day of a timestamp) is recorded in the
    /// type its transform produces, keyed by the field's name, and reads back
    /// as a scalar of that type.
    #[test]
    fn a_derived_partition_value_is_recorded_in_its_transforms_type() {
        let fixture = Fixture::partitioned_by_transform(
            vec![column("at", Type::Timestamp)],
            "at",
            "at_day",
            Transform::Day,
        );
        let entry = DataFileEntry {
            file: FileRef {
                path: ObjectPath::new("pivot-1.parquet"),
                size: 1,
            },
            partition: Some(PartitionValues::from([(
                "at_day".to_string(),
                scalar(Date32Array::from(vec![19_723])),
            )])),
            stats: Some(stats(1, &[], &[])),
            sort_order_id: None,
            origin: None,
        };

        let data_file = entry
            .create_data_file(&fixture.metadata, &fixture.location)
            .unwrap();
        let read = read_partition_values(
            &fixture.metadata,
            &fixture.columns,
            data_file.partition(),
            &entry.file.path,
        )
        .unwrap()
        .unwrap();

        assert_eq!(
            data_file.partition().fields(),
            &[Some(Literal::date(19_723))]
        );
        let (day, _) = read["at_day"].get();
        assert_eq!(day.as_primitive::<Date32Type>().value(0), 19_723);
    }

    /// A derived partition field the tuple lacks is an error, not a NULL: a
    /// NULL day would hide the file from every reader pruning by day.
    #[test]
    fn a_missing_derived_partition_value_is_refused() {
        let fixture = Fixture::partitioned_by_transform(
            vec![column("at", Type::Timestamp)],
            "at",
            "at_day",
            Transform::Day,
        );
        let entry = DataFileEntry {
            file: FileRef {
                path: ObjectPath::new("pivot-1.parquet"),
                size: 1,
            },
            partition: Some(PartitionValues::new()),
            stats: Some(stats(1, &[], &[])),
            sort_order_id: None,
            origin: None,
        };

        let error = entry
            .create_data_file(&fixture.metadata, &fixture.location)
            .unwrap_err();

        assert!(error.to_string().contains("at_day"), "{error}");
    }

    #[test]
    fn an_entry_without_stats_cannot_be_recorded() {
        let fixture = Fixture::new(vec![column("id", Type::Int64)], &[]);
        let entry = DataFileEntry {
            file: FileRef {
                path: ObjectPath::new("pivot-1.parquet"),
                size: 1,
            },
            partition: None,
            stats: None,
            sort_order_id: None,
            origin: None,
        };

        let error = entry
            .create_data_file(&fixture.metadata, &fixture.location)
            .unwrap_err();

        assert!(matches!(error, Error::MissingFileStats { .. }), "{error}");
    }
}
