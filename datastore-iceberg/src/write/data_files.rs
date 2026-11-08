//! A data file an INSERT uploaded, as a manifest records it: by full
//! location, with the partition tuple its rows share and its column bounds as
//! Iceberg values keyed by field id.

use std::collections::HashMap;

use arrow_array::Datum as ArrowDatum;
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, Literal, PartitionSpec,
    PrimitiveLiteral, PrimitiveType, Schema, Struct,
};
use parquet_engine::{FileStats, PartitionValues};

use crate::datum::array_to_datum;
use crate::write::PendingInsert;
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

impl PendingInsert {
    /// `file` as the manifest records it, by the table the INSERT was bound
    /// to: the schema its columns were named and numbered by, the partition
    /// spec its tuple is laid out by, and the Iceberg type of each partition
    /// value. A column renamed since is still recorded under its field id,
    /// which a rename keeps; one dropped since is recorded under an id no
    /// reader asks for.
    pub(super) fn manifest_entry_of(&self, file: &AddedFile) -> Result<DataFile> {
        file.to_manifest_entry(
            self.table.metadata.current_schema(),
            self.table.metadata.default_partition_spec(),
            &self.partition_keys.value_types,
        )
    }
}

/// The Iceberg primitive type of `schema`'s top-level field `name`, and its
/// id.
fn find_primitive_field(schema: &Schema, name: &str) -> Option<(i32, PrimitiveType)> {
    let field = schema.field_by_name(name)?;
    let primitive = field.field_type.as_primitive_type()?.clone();
    Some((field.id, primitive))
}

impl AddedFile {
    /// This file as the manifest records it: an ADDED Parquet data file with its
    /// row count, size, partition tuple laid out by `spec` in `value_types`, and
    /// per column of `schema` its bounds and null count. A file without a row
    /// count cannot be committed.
    fn to_manifest_entry(
        &self,
        schema: &Schema,
        spec: &PartitionSpec,
        value_types: &[PrimitiveType],
    ) -> Result<DataFile> {
        let record_count = self
            .stats
            .num_records
            .and_then(|count| u64::try_from(count).ok())
            .ok_or_else(|| Error::MissingFileStats {
                file: self.uri.clone(),
            })?;
        let mut lower_bounds = HashMap::new();
        let mut upper_bounds = HashMap::new();
        let mut null_value_counts = HashMap::new();
        for (name, min) in &self.stats.min_values {
            let Some(max) = self.stats.max_values.get(name) else {
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
        for (name, nulls) in &self.stats.null_counts {
            if let (Some((field_id, _)), Ok(nulls)) =
                (find_primitive_field(schema, name), u64::try_from(*nulls))
            {
                null_value_counts.insert(field_id, nulls);
            }
        }
        let partition = self.build_partition_struct(spec, value_types)?;
        let mut builder = DataFileBuilder::default();
        builder
            .content(DataContentType::Data)
            .file_path(self.uri.clone())
            .file_format(DataFileFormat::Parquet)
            .partition(partition)
            .partition_spec_id(spec.spec_id())
            .record_count(record_count)
            .file_size_in_bytes(self.size)
            .lower_bounds(lower_bounds)
            .upper_bounds(upper_bounds)
            .null_value_counts(null_value_counts);
        builder.build().map_err(|error| Error::InvalidDataFile {
            file: self.uri.clone(),
            message: error.to_string(),
        })
    }

    /// The file's partition tuple laid out as `spec` orders its fields, each
    /// value in the Iceberg type its transform produces, `value_types` in the
    /// same order. A partitioned table's file must carry a value for every
    /// field: an Iceberg tuple has no "unknown" value, and a missing one would
    /// make every other reader prune the file wrongly.
    fn build_partition_struct(
        &self,
        spec: &PartitionSpec,
        value_types: &[PrimitiveType],
    ) -> Result<Struct> {
        let fields = spec.fields();
        if fields.is_empty() {
            return Ok(Struct::empty());
        }
        let invalid = |message: String| Error::InvalidDataFile {
            file: self.uri.clone(),
            message,
        };
        let tuple = self.partition.as_ref().ok_or_else(|| {
            invalid("the file of a partitioned table has no partition values".into())
        })?;
        let mut values = Vec::with_capacity(fields.len());
        for (field, value_type) in fields.iter().zip(value_types) {
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

    use arrow_array::{
        Array, ArrayRef, Date32Array, Float64Array, Int64Array, Scalar, StringViewArray,
    };
    use iceberg::spec::{NestedField, Transform, Type};

    use super::*;

    fn orders_schema() -> Arc<Schema> {
        Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::optional(2, "customer", Type::Primitive(PrimitiveType::String))
                        .into(),
                    NestedField::optional(3, "at", Type::Primitive(PrimitiveType::Timestamp))
                        .into(),
                    NestedField::optional(4, "amount", Type::Primitive(PrimitiveType::Double))
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
        let spec = PartitionSpec::builder(schema.clone())
            .add_partition_field("customer", "customer", Transform::Identity)
            .unwrap()
            .add_partition_field("at", "at_day", Transform::Day)
            .unwrap()
            .build()
            .unwrap();
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

        let data_file = file
            .to_manifest_entry(
                &schema,
                &spec,
                &[PrimitiveType::String, PrimitiveType::Date],
            )
            .unwrap();

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
        let spec = PartitionSpec::builder(schema.clone())
            .add_partition_field("at", "at_day", Transform::Day)
            .unwrap()
            .build()
            .unwrap();
        let file = AddedFile {
            uri: "s3://lake/t/data/f.parquet".to_string(),
            size: 512,
            partition: Some(HashMap::new()),
            stats: stats(),
        };

        let error = file
            .to_manifest_entry(&schema, &spec, &[PrimitiveType::Date])
            .unwrap_err();

        assert!(error.to_string().contains("at_day"), "{error}");
    }

    #[test]
    fn long_string_bounds_are_cut_to_a_bracketing_prefix() {
        let lower = truncate_lower(Datum::string("abcdefghijklmnopqrstuvwxyz"));
        let upper = truncate_upper(Datum::string("abcdefghijklmnopqrstuvwxyz")).unwrap();

        assert_eq!(lower, Datum::string("abcdefghijklmnop"));
        assert_eq!(upper, Datum::string("abcdefghijklmnoq"));
    }

    #[test]
    fn a_long_string_of_the_largest_code_points_has_no_upper_bound() {
        let largest = char::MAX.to_string().repeat(STRING_BOUND_CODE_POINTS + 1);

        let upper = truncate_upper(Datum::string(&largest));
        let lower = truncate_lower(Datum::string(&largest));

        assert_eq!(upper, None);
        assert_eq!(
            lower,
            Datum::string(&largest[..largest.len() - char::MAX.len_utf8()])
        );
    }

    #[test]
    fn a_bound_iceberg_cannot_hold_leaves_the_column_unbounded() {
        let schema = orders_schema();
        let spec = PartitionSpec::builder(schema.clone()).build().unwrap();
        let mut stats = stats();
        stats.min_values.insert(
            "amount".to_string(),
            Arc::new(Float64Array::from(vec![1.5])) as ArrayRef,
        );
        stats.max_values.insert(
            "amount".to_string(),
            Arc::new(Float64Array::from(vec![f64::NAN])) as ArrayRef,
        );
        let file = AddedFile {
            uri: "s3://lake/t/data/f.parquet".to_string(),
            size: 512,
            partition: None,
            stats,
        };

        let data_file = file.to_manifest_entry(&schema, &spec, &[]).unwrap();

        assert_eq!(data_file.lower_bounds()[&1], Datum::long(1));
        assert!(!data_file.lower_bounds().contains_key(&4));
        assert!(!data_file.upper_bounds().contains_key(&4));
    }
}
