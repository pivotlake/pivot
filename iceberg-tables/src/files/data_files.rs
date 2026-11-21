//! One data file as the datastore holds it, and as a manifest records it.
//!
//! The datastore keeps a file by its store identity, its partition tuple and
//! its column statistics as Pivot's typed Arrow scalars, the shape every
//! writer and the compaction planner work with. A manifest records the same
//! file by full URI with Iceberg values keyed by field id. The two conversions
//! here are inverses, so a file written, read back and re-emitted by a later
//! commit is recorded unchanged. A file is recorded under the table's
//! default partition spec and read back under the spec it was written
//! under, so a file of an earlier spec is read and never re-emitted.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, Date32Array, Datum as ArrowDatum, Decimal64Array, Decimal128Array,
    Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, Scalar,
    StringViewArray, TimestampMicrosecondArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, Literal, PartitionSpec,
    PrimitiveLiteral, PrimitiveType, Schema, Struct, TableMetadata, Transform, Type as IcebergType,
};
use object_storage::{FileRef, ObjectPath};
use parquet_engine::{FileStats, PartitionValues};
use planner::catalog::Column;
use planner::types::Type;

use crate::files::partition::Partition;
use crate::metadata::layout::TableLocation;
use crate::types::columns::to_pivot_type;
use crate::types::datum::array_to_datum;
use crate::{Error, Result};

/// String bounds of a column outside the table's sort order are cut to this
/// many code points: such a bound only has to bracket the file's values, and
/// a long string would otherwise be recorded in full in every manifest entry
/// of the file.
const STRING_BOUND_CODE_POINTS: usize = 16;

/// One data file a commit adds: its store identity, the partition tuple a
/// partitioned INSERT stamps on it (`None` for an unpartitioned table; keyed
/// by partition field name, each value in the type the field's transform
/// produces), its statistics, and the sort order its rows are in. A commit
/// writes it under the table's default partition spec.
#[derive(Clone)]
pub struct DataFileEntry {
    pub file: FileRef,
    pub partition: Option<PartitionValues>,
    /// The file's statistics: its row count, and per column the bounds and
    /// null count. Written into the manifest, so a file that has none cannot
    /// be committed.
    pub stats: Option<Arc<FileStats>>,
    /// The id of the table sort order the file's rows are in, as its manifest
    /// entry records it: [`UNSORTED_ORDER_ID`](crate::types::schema::UNSORTED_ORDER_ID)
    /// for a file written in no order, `None` for one whose entry says
    /// nothing, which every reader takes as unsorted: a file adopted as it
    /// lay, whose order nobody knows. A file a writer of this process
    /// ordered gets its id from
    /// [`Table::find_sort_order_id_of_rows_sorted_by`](crate::Table::find_sort_order_id_of_rows_sorted_by).
    pub sort_order_id: Option<i32>,
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

/// Per field of `partition_spec`, a partition spec of the table `metadata`
/// describes, in spec order, the Iceberg type of its values: the field's
/// transform's result over its source field in the current schema.
fn read_partition_value_types(
    metadata: &TableMetadata,
    partition_spec: &PartitionSpec,
    file: &ObjectPath,
) -> Result<Vec<PrimitiveType>> {
    let invalid = |message: String| Error::InvalidDataFile {
        file: file.to_string(),
        message,
    };
    let partition_type = partition_spec
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

/// The tuple of `partition`, a partition of a spec `metadata` lists which
/// the file `file` sits in, as the datastore keys it: laid out by the
/// partition spec the file was written under, that spec's field name to
/// typed scalar of the declared `columns`' Pivot types, a NULL value as a
/// null scalar. `None` for a spec without fields.
pub(crate) fn read_partition_values(
    metadata: &TableMetadata,
    columns: &[Column],
    partition: &Partition,
    file: &ObjectPath,
) -> Result<Option<PartitionValues>> {
    let partition_spec = metadata
        .partition_spec_by_id(partition.spec_id())
        .expect("a partition is read under a spec its table lists");
    let fields = partition_spec.fields();
    if fields.is_empty() {
        return Ok(None);
    }
    let schema = metadata.current_schema();
    let value_types = read_partition_value_types(metadata, partition_spec, file)?;
    let tuple = partition.tuple().fields();
    let mut values = PartitionValues::new();
    for (position, (field, value_type)) in fields.iter().zip(&value_types).enumerate() {
        let Some(pivot_type) = find_partition_pivot_type(schema, columns, field, value_type) else {
            continue;
        };
        let scalar = match tuple.get(position) {
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
    ///
    /// A column of the default sort order is bounded by the minimum and
    /// maximum the statistics hold, whole, however long: where a file lies
    /// on the sort key is read from them, and bounds cut to a prefix give
    /// the files whose values share it one range, so files that lie apart
    /// read as overlapping, as do the files a rewrite of them writes. A
    /// string bound of any other column is cut to
    /// `STRING_BOUND_CODE_POINTS` code points.
    pub fn create_data_file(
        &self,
        metadata: &TableMetadata,
        location: &TableLocation,
    ) -> Result<DataFile> {
        let schema = metadata.current_schema();
        let sort_column_ids: Vec<i32> = metadata
            .default_sort_order()
            .fields
            .iter()
            .map(|field| field.source_id)
            .collect();
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
                array_to_datum(min.as_ref(), &primitive),
                array_to_datum(max.as_ref(), &primitive),
            ) else {
                continue;
            };
            let (lower, upper) = if sort_column_ids.contains(&field_id) {
                (lower, upper)
            } else {
                // A cut maximum without a successor leaves the column
                // unbounded too.
                let Some(upper) = truncate_upper(upper) else {
                    continue;
                };
                (truncate_lower(lower), upper)
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
            .null_value_counts(null_value_counts);
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
        let value_types = read_partition_value_types(
            metadata,
            metadata.default_partition_spec(),
            &self.file.path,
        )?;
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
}
