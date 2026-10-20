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

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type,
    Int32Type, Int64Type, TimestampMicrosecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Date32Array, Datum as ArrowDatum, Decimal64Array,
    Decimal128Array, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
    Scalar, StringViewArray, TimestampMicrosecondArray, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array,
};
use arrow_schema::{DataType, TimeUnit};
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, FormatVersion, Literal,
    ManifestEntry, ManifestFile, ManifestStatus, PartitionSpecRef, PrimitiveLiteral, PrimitiveType,
    SchemaRef, Struct,
};
use object_storage::FileRef;
use parquet_engine::{FileStats, PartitionValues};
use planner::catalog::Column;
use planner::types::Type;

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
    /// The manifest the entry was read from, as the manifest list names it.
    pub manifest_path: String,
}

/// One data file: its store identity, the partition tuple a partitioned
/// INSERT stamps on it (`None` for an unpartitioned table), its statistics,
/// and, once committed, where its table's metadata records it.
#[derive(Clone)]
pub struct DataFileEntry {
    pub file: FileRef,
    pub partition: Option<PartitionValues>,
    /// The file's statistics: its row count, and per column the bounds and
    /// null count. Written into the manifest, so a file that has none cannot
    /// be committed.
    pub stats: Option<Arc<FileStats>>,
    pub origin: Option<ManifestOrigin>,
}

/// What converting an entry needs to know about its table: the schema the
/// field ids come from, the partition spec the tuple is laid out by, the
/// declared columns' Pivot types, and where URIs are formed.
pub struct EntryContext<'a> {
    pub schema: &'a SchemaRef,
    pub partition_spec: &'a PartitionSpecRef,
    pub columns: &'a [Column],
    pub location: &'a TableLocation,
    /// The table's format version: a version 2 table has no row lineage,
    /// so its entries record no first row id.
    pub format_version: FormatVersion,
}

impl EntryContext<'_> {
    fn pivot_type(&self, column: &str) -> Option<&Type> {
        self.columns
            .iter()
            .find(|declared| declared.name == column)
            .map(|declared| &declared.col_type)
    }

    /// The Iceberg primitive type of the top-level field `name`, and its id.
    fn primitive_field(&self, name: &str) -> Option<(i32, PrimitiveType)> {
        let field = self.schema.field_by_name(name)?;
        let primitive = field.field_type.as_primitive_type()?.clone();
        Some((field.id, primitive))
    }
}

/// `entry` as the manifest records it. Requires the entry's statistics: a
/// manifest entry carries the file's row count, and a file without one
/// cannot be committed.
pub fn to_data_file(entry: &DataFileEntry, ctx: &EntryContext<'_>) -> Result<DataFile> {
    let stats = entry
        .stats
        .as_ref()
        .ok_or_else(|| Error::MissingFileStats {
            file: entry.file.path.to_string(),
        })?;
    let record_count = stats
        .num_records
        .and_then(|count| u64::try_from(count).ok())
        .ok_or_else(|| Error::MissingFileStats {
            file: entry.file.path.to_string(),
        })?;
    let mut lower_bounds = HashMap::new();
    let mut upper_bounds = HashMap::new();
    let mut null_value_counts = HashMap::new();
    for (name, min) in &stats.min_values {
        let Some(max) = stats.max_values.get(name) else {
            continue;
        };
        let Some((field_id, primitive)) = ctx.primitive_field(name) else {
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
        if let (Some((field_id, _)), Ok(nulls)) = (ctx.primitive_field(name), u64::try_from(*nulls))
        {
            null_value_counts.insert(field_id, nulls);
        }
    }
    let partition = build_partition_struct(entry, ctx)?;
    let mut builder = DataFileBuilder::default();
    builder
        .content(DataContentType::Data)
        .file_path(ctx.location.uri_for(&entry.file.path)?)
        .file_format(DataFileFormat::Parquet)
        .partition(partition)
        .partition_spec_id(ctx.partition_spec.spec_id())
        .record_count(record_count)
        .file_size_in_bytes(entry.file.size)
        .lower_bounds(lower_bounds)
        .upper_bounds(upper_bounds)
        .null_value_counts(null_value_counts)
        .first_row_id(match ctx.format_version {
            FormatVersion::V1 | FormatVersion::V2 => None,
            FormatVersion::V3 => entry.origin.as_ref().and_then(|origin| origin.first_row_id),
        });
    builder.build().map_err(|error| Error::InvalidDataFile {
        file: entry.file.path.to_string(),
        message: error.to_string(),
    })
}

/// The entry's partition tuple laid out as the spec orders its fields. A
/// column the tuple lacks is a NULL partition value. A partitioned table's
/// entry must carry a tuple: an Iceberg tuple has no "unknown" value, and an
/// all-null tuple would make every other reader prune the file wrongly.
fn build_partition_struct(entry: &DataFileEntry, ctx: &EntryContext<'_>) -> Result<Struct> {
    let fields = ctx.partition_spec.fields();
    if fields.is_empty() {
        return Ok(Struct::empty());
    }
    let tuple = entry
        .partition
        .as_ref()
        .ok_or_else(|| Error::PartitionNotConstant {
            file: entry.file.path.to_string(),
        })?;
    let mut values = Vec::with_capacity(fields.len());
    for field in fields {
        let name = ctx
            .schema
            .name_by_field_id(field.source_id)
            .ok_or_else(|| Error::InvalidDataFile {
                file: entry.file.path.to_string(),
                message: format!("partition field {} has no source column", field.name),
            })?;
        let value = match tuple.get(name) {
            Some(scalar) => {
                let (array, _) = ArrowDatum::get(scalar);
                if array.is_null(0) {
                    None
                } else {
                    let (_, primitive) =
                        ctx.primitive_field(name)
                            .ok_or_else(|| Error::InvalidDataFile {
                                file: entry.file.path.to_string(),
                                message: format!("partition column `{name}` is not primitive"),
                            })?;
                    let datum = array_to_datum(array, &primitive).ok_or_else(|| {
                        Error::InvalidDataFile {
                            file: entry.file.path.to_string(),
                            message: format!(
                                "partition value of `{name}` has no Iceberg form in `{primitive}`"
                            ),
                        }
                    })?;
                    Some(Literal::Primitive(datum.literal().clone()))
                }
            }
            None => None,
        };
        values.push(value);
    }
    Ok(Struct::from_iter(values))
}

/// `entry` as it reads back from the manifest it is about to be written to:
/// the same file, with its partition tuple and statistics rebuilt from their
/// Iceberg form. A writer's own entry carries the scalars its files were
/// encoded from, which live in the worker pool's memory and are bounded as
/// written, not as a manifest records them; the table's state keeps the
/// recorded form, owned by the heap, for as long as the file is live.
pub fn normalized_entry(entry: &DataFileEntry, ctx: &EntryContext<'_>) -> Result<DataFileEntry> {
    let data_file = to_data_file(entry, ctx)?;
    Ok(DataFileEntry {
        file: entry.file.clone(),
        partition: read_partition_values(data_file.partition(), ctx)?,
        stats: Some(Arc::new(read_stats(&data_file, ctx))),
        origin: entry.origin.clone(),
    })
}

/// `entry`, read from `manifest`, as the datastore holds it. Inherits what a
/// manifest leaves implicit on its ADDED entries: the snapshot that added the
/// file, its sequence numbers, and (`first_row_id` being the caller's
/// running assignment) its first row id.
pub fn from_manifest_entry(
    entry: &ManifestEntry,
    manifest: &ManifestFile,
    first_row_id: Option<i64>,
    ctx: &EntryContext<'_>,
) -> Result<DataFileEntry> {
    let data = entry.data_file();
    let path = ctx.location.file_path_for_uri(data.file_path())?;
    let added = entry.status() == ManifestStatus::Added;
    let snapshot_id = entry.snapshot_id().unwrap_or(manifest.added_snapshot_id);
    let sequence_number = match entry.sequence_number() {
        Some(number) => number,
        None if added => manifest.sequence_number,
        None => {
            return Err(Error::InvalidDataFile {
                file: path.to_string(),
                message: "an existing entry carries no sequence number".to_string(),
            });
        }
    };
    let file_sequence_number = match entry.file_sequence_number {
        Some(number) => number,
        None if added => manifest.sequence_number,
        None => sequence_number,
    };
    let partition = read_partition_values(data.partition(), ctx)?;
    let stats = Arc::new(read_stats(data, ctx));
    Ok(DataFileEntry {
        file: FileRef {
            path,
            size: data.file_size_in_bytes(),
        },
        partition,
        stats: Some(stats),
        origin: Some(ManifestOrigin {
            snapshot_id,
            sequence_number,
            file_sequence_number,
            first_row_id,
            manifest_path: manifest.manifest_path.clone(),
        }),
    })
}

/// The partition tuple `partition` holds, as the datastore keys it: column
/// name to typed scalar, a NULL value as a null scalar. `None` for an
/// unpartitioned table.
fn read_partition_values(
    partition: &Struct,
    ctx: &EntryContext<'_>,
) -> Result<Option<PartitionValues>> {
    let fields = ctx.partition_spec.fields();
    if fields.is_empty() {
        return Ok(None);
    }
    let mut values = PartitionValues::new();
    for (position, field) in fields.iter().enumerate() {
        let Some(name) = ctx.schema.name_by_field_id(field.source_id) else {
            continue;
        };
        let Some(pivot_type) = ctx.pivot_type(name) else {
            continue;
        };
        let scalar = match partition.fields().get(position) {
            Some(Some(Literal::Primitive(literal))) => literal_to_scalar(literal, pivot_type),
            _ => Some(null_scalar(pivot_type)),
        };
        if let Some(scalar) = scalar {
            values.insert(name.to_string(), scalar);
        }
    }
    Ok(Some(values))
}

/// The statistics `data` records, as the datastore keys them: by column name,
/// as typed scalars of the column's Pivot type.
fn read_stats(data: &DataFile, ctx: &EntryContext<'_>) -> FileStats {
    let mut stats = FileStats {
        num_records: i64::try_from(data.record_count()).ok(),
        min_values: HashMap::new(),
        max_values: HashMap::new(),
        null_counts: HashMap::new(),
    };
    for column in ctx.columns {
        let Some(field_id) = ctx.schema.field_id_by_name(&column.name) else {
            continue;
        };
        let bound = |bounds: &HashMap<i32, Datum>| {
            bounds
                .get(&field_id)
                .and_then(|datum| literal_to_scalar(datum.literal(), &column.col_type))
                .map(|scalar| scalar.into_inner())
        };
        if let (Some(min), Some(max)) = (bound(data.lower_bounds()), bound(data.upper_bounds())) {
            stats.min_values.insert(column.name.clone(), min);
            stats.max_values.insert(column.name.clone(), max);
        }
        if let Some(nulls) = data.null_value_counts().get(&field_id)
            && let Ok(nulls) = i64::try_from(*nulls)
        {
            stats.null_counts.insert(column.name.clone(), nulls);
        }
    }
    stats
}

/// The single value in `array` as an Iceberg value of `primitive`, or `None`
/// when the value has no such form: a null, an unsigned value above the
/// signed range, a NaN, or an Arrow type that is not a Pivot column's.
fn array_to_datum(array: &dyn Array, primitive: &PrimitiveType) -> Option<Datum> {
    if array.len() != 1 || array.is_null(0) {
        return None;
    }
    let datum = match (array.data_type(), primitive) {
        (DataType::Boolean, PrimitiveType::Boolean) => Datum::bool(array.as_boolean().value(0)),
        (DataType::Int8, PrimitiveType::Int) => {
            Datum::int(i32::from(array.as_primitive::<Int8Type>().value(0)))
        }
        (DataType::Int16, PrimitiveType::Int) => {
            Datum::int(i32::from(array.as_primitive::<Int16Type>().value(0)))
        }
        (DataType::Int32, PrimitiveType::Int) => {
            Datum::int(array.as_primitive::<Int32Type>().value(0))
        }
        (DataType::UInt8, PrimitiveType::Int) => {
            Datum::int(i32::from(array.as_primitive::<UInt8Type>().value(0)))
        }
        (DataType::UInt16, PrimitiveType::Int) => {
            Datum::int(i32::from(array.as_primitive::<UInt16Type>().value(0)))
        }
        (DataType::Int64, PrimitiveType::Long) => {
            Datum::long(array.as_primitive::<Int64Type>().value(0))
        }
        (DataType::UInt32, PrimitiveType::Long) => {
            Datum::long(i64::from(array.as_primitive::<UInt32Type>().value(0)))
        }
        (DataType::UInt64, PrimitiveType::Long) => {
            Datum::long(i64::try_from(array.as_primitive::<UInt64Type>().value(0)).ok()?)
        }
        (DataType::Float32, PrimitiveType::Float) => {
            let value = array.as_primitive::<Float32Type>().value(0);
            if value.is_nan() {
                return None;
            }
            Datum::float(value)
        }
        (DataType::Float64, PrimitiveType::Double) => {
            let value = array.as_primitive::<Float64Type>().value(0);
            if value.is_nan() {
                return None;
            }
            Datum::double(value)
        }
        (DataType::Utf8, PrimitiveType::String) => Datum::string(array.as_string::<i32>().value(0)),
        (DataType::LargeUtf8, PrimitiveType::String) => {
            Datum::string(array.as_string::<i64>().value(0))
        }
        (DataType::Utf8View, PrimitiveType::String) => {
            Datum::string(array.as_string_view().value(0))
        }
        (DataType::Date32, PrimitiveType::Date) => {
            Datum::date(array.as_primitive::<Date32Type>().value(0))
        }
        (DataType::Timestamp(TimeUnit::Microsecond, None), PrimitiveType::Timestamp) => {
            Datum::timestamp_micros(array.as_primitive::<TimestampMicrosecondType>().value(0))
        }
        (DataType::Timestamp(TimeUnit::Microsecond, Some(_)), PrimitiveType::Timestamptz) => {
            Datum::timestamptz_micros(array.as_primitive::<TimestampMicrosecondType>().value(0))
        }
        (DataType::Decimal64(_, _), PrimitiveType::Decimal { .. }) => {
            let unscaled = i128::from(array.as_primitive::<Decimal64Type>().value(0));
            Datum::try_from_bytes(&unscaled.to_be_bytes(), primitive.clone()).ok()?
        }
        (DataType::Decimal128(_, _), PrimitiveType::Decimal { .. }) => {
            let unscaled = array.as_primitive::<Decimal128Type>().value(0);
            Datum::try_from_bytes(&unscaled.to_be_bytes(), primitive.clone()).ok()?
        }
        _ => return None,
    };
    Some(datum)
}

/// `literal` as a one-value Arrow scalar of `pivot_type`'s physical type, or
/// `None` for a pairing that does not read (a different writer's bound).
fn literal_to_scalar(literal: &PrimitiveLiteral, pivot_type: &Type) -> Option<Scalar<ArrayRef>> {
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
    use iceberg::spec::{FormatVersion, PartitionSpec, TableMetadataBuilder};
    use object_storage::{LocalStore, ObjectPath, ObjectStore};

    fn column(name: &str, col_type: Type) -> Column {
        Column {
            name: name.to_string(),
            col_type,
        }
    }

    /// A table over `columns`, partitioned by `partition_by`, in a temporary
    /// local store; the metadata's own schema and spec feed the context.
    struct Fixture {
        _dir: tempfile::TempDir,
        location: TableLocation,
        schema: SchemaRef,
        partition_spec: PartitionSpecRef,
        columns: Vec<Column>,
    }

    impl Fixture {
        fn new(columns: Vec<Column>, partition_by: &[&str]) -> Self {
            let dir = tempfile::tempdir().unwrap();
            let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(dir.path()).unwrap());
            let location = TableLocation::new(store, ObjectPath::new("t")).unwrap();
            let schema = build_schema(&columns).unwrap();
            let partition_by: Vec<String> = partition_by.iter().map(|c| c.to_string()).collect();
            let spec = crate::schema::build_partition_spec(&schema, &partition_by).unwrap();
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
                schema: metadata.current_schema().clone(),
                partition_spec: metadata.default_partition_spec().clone(),
                columns,
            }
        }

        fn context(&self) -> EntryContext<'_> {
            EntryContext {
                schema: &self.schema,
                partition_spec: &self.partition_spec,
                columns: &self.columns,
                location: &self.location,
                format_version: FormatVersion::V3,
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
            origin: None,
        };

        let data_file = to_data_file(&entry, &fixture.context()).unwrap();
        let manifest_entry = ManifestEntry {
            status: ManifestStatus::Added,
            snapshot_id: None,
            sequence_number: None,
            file_sequence_number: None,
            data_file: data_file.clone(),
        };
        let read = from_manifest_entry(
            &manifest_entry,
            &manifest_for("m0.avro", 41, 3),
            Some(100),
            &fixture.context(),
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
                manifest_path: "m0.avro".to_string(),
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
            origin: None,
        };
        let data_file = to_data_file(&entry, &fixture.context()).unwrap();

        let read = from_manifest_entry(
            &ManifestEntry {
                status: ManifestStatus::Existing,
                snapshot_id: Some(5),
                sequence_number: Some(2),
                file_sequence_number: Some(2),
                data_file,
            },
            &manifest_for("m1.avro", 9, 8),
            None,
            &fixture.context(),
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
            origin: None,
        };

        let data_file = to_data_file(&entry, &fixture.context()).unwrap();

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
            origin: None,
        };

        let error = to_data_file(&entry, &fixture.context()).unwrap_err();

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
            origin: None,
        };

        let data_file = to_data_file(&entry, &fixture.context()).unwrap();
        let read = read_partition_values(data_file.partition(), &fixture.context())
            .unwrap()
            .unwrap();

        assert_eq!(data_file.partition().fields(), &[None]);
        assert!(read["name"].get().0.is_null(0));
        let _ = Fixture::unpartitioned_spec();
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
            origin: None,
        };

        let error = to_data_file(&entry, &fixture.context()).unwrap_err();

        assert!(matches!(error, Error::MissingFileStats { .. }), "{error}");
    }
}
