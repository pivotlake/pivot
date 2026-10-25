//! A data file an INSERT uploaded, as a manifest records it: by full
//! location, with the partition tuple its rows share and its column bounds as
//! Iceberg values keyed by field id.

use std::collections::HashMap;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type,
    Int32Type, Int64Type, TimestampMicrosecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, Datum as ArrowDatum};
use arrow_schema::{DataType, TimeUnit};
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, Literal, PartitionSpecRef,
    PrimitiveLiteral, PrimitiveType, SchemaRef, Struct,
};
use parquet_engine::{FileStats, PartitionValues};

use crate::{Error, Result};

/// String bounds are cut to this many code points: a bound only has to
/// bracket the file's values, and a long string would otherwise be recorded
/// in full in every manifest and manifest-list summary.
const STRING_BOUND_CODE_POINTS: usize = 16;

/// One file an INSERT wrote: where it is, how big, the partition values its
/// rows share keyed by partition field name (`None` for an unpartitioned
/// table), and its statistics.
pub(crate) struct AddedFile {
    pub uri: String,
    pub size: u64,
    pub partition: Option<PartitionValues>,
    pub stats: FileStats,
}

/// What recording a file needs to know about its table: the schema the field
/// ids come from, the partition spec the tuple is laid out by, and per spec
/// field the Iceberg type of its partition value.
pub(crate) struct EntryContext<'a> {
    pub schema: &'a SchemaRef,
    pub partition_spec: &'a PartitionSpecRef,
    pub partition_value_types: &'a [PrimitiveType],
}

impl EntryContext<'_> {
    /// The Iceberg primitive type of the top-level field `name`, and its id.
    fn primitive_field(&self, name: &str) -> Option<(i32, PrimitiveType)> {
        let field = self.schema.field_by_name(name)?;
        let primitive = field.field_type.as_primitive_type()?.clone();
        Some((field.id, primitive))
    }
}

/// `file` as the manifest records it: an ADDED Parquet data file with its
/// row count, size, partition tuple, and per column its bounds and null
/// count. A file without a row count cannot be committed.
pub(crate) fn to_data_file(file: &AddedFile, ctx: &EntryContext<'_>) -> Result<DataFile> {
    let record_count = file
        .stats
        .num_records
        .and_then(|count| u64::try_from(count).ok())
        .ok_or_else(|| Error::MissingFileStats {
            file: file.uri.clone(),
        })?;
    let mut lower_bounds = HashMap::new();
    let mut upper_bounds = HashMap::new();
    let mut null_value_counts = HashMap::new();
    for (name, min) in &file.stats.min_values {
        let Some(max) = file.stats.max_values.get(name) else {
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
    for (name, nulls) in &file.stats.null_counts {
        if let (Some((field_id, _)), Ok(nulls)) = (ctx.primitive_field(name), u64::try_from(*nulls))
        {
            null_value_counts.insert(field_id, nulls);
        }
    }
    let partition = build_partition_struct(file, ctx)?;
    let mut builder = DataFileBuilder::default();
    builder
        .content(DataContentType::Data)
        .file_path(file.uri.clone())
        .file_format(DataFileFormat::Parquet)
        .partition(partition)
        .partition_spec_id(ctx.partition_spec.spec_id())
        .record_count(record_count)
        .file_size_in_bytes(file.size)
        .lower_bounds(lower_bounds)
        .upper_bounds(upper_bounds)
        .null_value_counts(null_value_counts);
    builder.build().map_err(|error| Error::InvalidDataFile {
        file: file.uri.clone(),
        message: error.to_string(),
    })
}

/// The file's partition tuple laid out as the spec orders its fields, each
/// value in the Iceberg type its transform produces. A partitioned table's
/// file must carry a value for every field: an Iceberg tuple has no
/// "unknown" value, and a missing one would make every other reader prune
/// the file wrongly.
fn build_partition_struct(file: &AddedFile, ctx: &EntryContext<'_>) -> Result<Struct> {
    let fields = ctx.partition_spec.fields();
    if fields.is_empty() {
        return Ok(Struct::empty());
    }
    let invalid = |message: String| Error::InvalidDataFile {
        file: file.uri.clone(),
        message,
    };
    let tuple = file
        .partition
        .as_ref()
        .ok_or_else(|| invalid("the file of a partitioned table has no partition values".into()))?;
    let mut values = Vec::with_capacity(fields.len());
    for (field, value_type) in fields.iter().zip(ctx.partition_value_types) {
        let scalar = tuple.get(&field.name).ok_or_else(|| {
            invalid(format!(
                "the file has no value for partition field `{}`",
                field.name
            ))
        })?;
        let (array, _) = ArrowDatum::get(scalar);
        if array.is_null(0) {
            values.push(None);
            continue;
        }
        let datum = array_to_datum(array, value_type).ok_or_else(|| {
            invalid(format!(
                "the value of partition field `{}` has no Iceberg form in `{value_type}`",
                field.name
            ))
        })?;
        values.push(Some(Literal::Primitive(datum.literal().clone())));
    }
    Ok(Struct::from_iter(values))
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
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Date32Array, Int64Array, Scalar, StringViewArray};
    use iceberg::spec::{NestedField, PartitionSpec, Schema, Transform, Type};

    use super::*;

    fn orders_schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::optional(2, "customer", Type::Primitive(PrimitiveType::String))
                        .into(),
                    NestedField::optional(3, "at", Type::Primitive(PrimitiveType::Timestamp))
                        .into(),
                ])
                .build()
                .unwrap(),
        )
    }

    fn scalar(array: impl Array + 'static) -> Scalar<ArrayRef> {
        Scalar::new(Arc::new(array) as ArrayRef)
    }

    fn stats() -> FileStats {
        FileStats {
            num_records: Some(3),
            min_values: HashMap::from([(
                "id".to_string(),
                Arc::new(Int64Array::from(vec![1])) as ArrayRef,
            )]),
            max_values: HashMap::from([(
                "id".to_string(),
                Arc::new(Int64Array::from(vec![9])) as ArrayRef,
            )]),
            null_counts: HashMap::from([("customer".to_string(), 1)]),
        }
    }

    #[test]
    fn a_file_is_recorded_with_its_bounds_and_its_transformed_partition_values() {
        let schema = orders_schema();
        let spec: PartitionSpecRef = Arc::new(
            PartitionSpec::builder(schema.clone())
                .add_partition_field("customer", "customer", Transform::Identity)
                .unwrap()
                .add_partition_field("at", "at_day", Transform::Day)
                .unwrap()
                .build()
                .unwrap(),
        );
        let file = AddedFile {
            uri: "s3://lake/t/data/f.parquet".to_string(),
            size: 512,
            partition: Some(HashMap::from([
                (
                    "customer".to_string(),
                    scalar(StringViewArray::from(vec!["ann"])),
                ),
                (
                    "at_day".to_string(),
                    scalar(Date32Array::from(vec![20_000])),
                ),
            ])),
            stats: stats(),
        };
        let ctx = EntryContext {
            schema: &schema,
            partition_spec: &spec,
            partition_value_types: &[PrimitiveType::String, PrimitiveType::Date],
        };

        let data_file = to_data_file(&file, &ctx).unwrap();

        assert_eq!(data_file.record_count(), 3);
        assert_eq!(data_file.file_size_in_bytes(), 512);
        assert_eq!(data_file.lower_bounds()[&1], Datum::long(1));
        assert_eq!(data_file.upper_bounds()[&1], Datum::long(9));
        assert_eq!(data_file.null_value_counts()[&2], 1);
        assert_eq!(
            data_file.partition().fields(),
            &[Some(Literal::string("ann")), Some(Literal::date(20_000)),]
        );
    }

    #[test]
    fn a_partitioned_files_missing_partition_value_is_refused() {
        let schema = orders_schema();
        let spec: PartitionSpecRef = Arc::new(
            PartitionSpec::builder(schema.clone())
                .add_partition_field("at", "at_day", Transform::Day)
                .unwrap()
                .build()
                .unwrap(),
        );
        let file = AddedFile {
            uri: "s3://lake/t/data/f.parquet".to_string(),
            size: 512,
            partition: Some(HashMap::new()),
            stats: stats(),
        };
        let ctx = EntryContext {
            schema: &schema,
            partition_spec: &spec,
            partition_value_types: &[PrimitiveType::Date],
        };

        let error = to_data_file(&file, &ctx).unwrap_err();

        assert!(error.to_string().contains("at_day"), "{error}");
    }

    #[test]
    fn long_string_bounds_are_cut_to_a_bracketing_prefix() {
        let lower = truncate_lower(Datum::string("abcdefghijklmnopqrstuvwxyz"));
        let upper = truncate_upper(Datum::string("abcdefghijklmnopqrstuvwxyz")).unwrap();

        assert_eq!(lower, Datum::string("abcdefghijklmnop"));
        assert_eq!(upper, Datum::string("abcdefghijklmnoq"));
    }
}
