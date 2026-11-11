//! A table as the cross-datastore catalog describes it: the columns it
//! declares with what they cost, and the files it holds, read from the
//! table's file index a partition at a time and from the files' footers a
//! batch at a time.

use std::collections::{BTreeMap, HashMap};
use std::ops::ControlFlow;

use catalog::datastore::{DatastoreColumnMetadata, DatastoreFileMetadata, DatastoreTableMetadata};
use iceberg::spec::{
    DataFile, Datum, Literal, PrimitiveType, Schema, TableMetadata, Transform, Type,
};
use iceberg::{Error as SpecError, ErrorKind};
use object_storage::FileRef;
use parquet_engine::{FileRowGroups, leaf_count};
use planner::catalog::SchemaQualifiedTableName;
use serde_json::{Value, value::RawValue};

use crate::files::file_index::{IndexedFile, Partition};
use crate::table::Table;
use crate::{Error, Result};

/// How many files' footers are held at a time.
const FILES_PER_READ: usize = 1024;

impl Table {
    /// This table as the catalog describes it under `name` and `id`, each
    /// file named by `name_file`. Every file's footer is taken, a batch of
    /// files at a time, and of each only what is reported is kept: a footer
    /// the cache holds is not read, and none read is cached. The files are
    /// listed in the order the index holds them, partition by partition.
    pub fn describe(
        &self,
        name: SchemaQualifiedTableName,
        id: String,
        name_file: &dyn Fn(&IndexedFile) -> String,
    ) -> Result<DatastoreTableMetadata> {
        let column_positions: HashMap<&str, usize> = self
            .columns()
            .iter()
            .enumerate()
            .map(|(position, column)| (column.name.as_str(), position))
            .collect();
        let mut column_bytes = vec![(0u64, 0u64); self.columns().len()];
        let mut files = Vec::new();
        let mut total_rows = 0u64;
        self.visit_partitions(&mut |_, index| {
            let partition_files: Vec<&IndexedFile> = index.files().map(|file| &**file).collect();
            for batch in partition_files.chunks(FILES_PER_READ) {
                let named: Vec<FileRef> = batch.iter().map(|file| file.file.clone()).collect();
                let footers = self.read_footers_unrecorded(&named)?;
                for (file, footer) in batch.iter().zip(&footers) {
                    total_rows += file.record_count();
                    files.push(DatastoreFileMetadata {
                        path: name_file(file),
                        bytes: file.file.size,
                        bytes_uncompressed: add_file_bytes(
                            &mut column_bytes,
                            &column_positions,
                            footer,
                        ),
                        partition: format_partition(
                            self.metadata(),
                            &file.partition,
                            file.schema_id,
                        )?,
                        min_max_stats: format_min_max_stats(
                            self.metadata().current_schema(),
                            &file.data,
                        )
                        .map_err(Error::spec)?,
                    });
                }
            }
            Ok(ControlFlow::Continue(()))
        })?;
        let columns = self
            .columns()
            .iter()
            .zip(column_bytes)
            .enumerate()
            .map(
                |(position, (column, (bytes, bytes_uncompressed)))| DatastoreColumnMetadata {
                    name: column.name.clone(),
                    column_type: column.col_type.clone(),
                    position,
                    bytes,
                    bytes_uncompressed,
                    is_partition_key: self.partition_by().contains(&column.name),
                    is_sort_key: self.sort_by().contains(&column.name),
                },
            )
            .collect();
        Ok(DatastoreTableMetadata {
            name,
            id,
            columns,
            sort_by: self.sort_by().to_vec(),
            partition_by: self.partition_by().to_vec(),
            total_rows,
            bytes: files.iter().map(|file| file.bytes).sum(),
            bytes_uncompressed: files.iter().map(|file| file.bytes_uncompressed).sum(),
            files,
        })
    }
}

/// Add to `totals`, per declared column (found by name in
/// `column_positions`), what the column costs in the file `footer`
/// describes: the bytes it occupies in storage, and what they hold decoded;
/// and return what the whole file holds decoded. A footer lays its chunks
/// out by the declared columns, a chunk per leaf, so a nested column is
/// charged for every leaf it spans; a file that does not carry a column
/// contributes nothing to it.
fn add_file_bytes(
    totals: &mut [(u64, u64)],
    column_positions: &HashMap<&str, usize>,
    footer: &FileRowGroups,
) -> u64 {
    let mut file_bytes_uncompressed = 0;
    for row_group in &footer.row_groups {
        let mut leaf = 0;
        for field in row_group.schema.fields() {
            let leaves = leaf_count(field);
            if let Some(&position) = column_positions.get(field.name().as_str()) {
                for chunk in &row_group.columns[leaf..leaf + leaves] {
                    totals[position].0 += chunk.total_compressed_size.max(0) as u64;
                    totals[position].1 += chunk.total_uncompressed_size.max(0) as u64;
                }
            }
            leaf += leaves;
        }
        file_bytes_uncompressed += row_group
            .columns
            .iter()
            .map(|chunk| chunk.total_uncompressed_size.max(0) as u64)
            .sum::<u64>();
    }
    file_bytes_uncompressed
}

/// `partition`, of a file whose manifest was written under the schema
/// `schema_id`, as the spec it was written under names it: `field=value`
/// pairs in the spec's order, comma-separated, each value in the plain
/// text its transform reads as (a day as its date, a bucket as its number)
/// and a NULL as nothing. Empty for a partition of an unpartitioned spec.
fn format_partition(
    metadata: &TableMetadata,
    partition: &Partition,
    schema_id: i32,
) -> Result<String> {
    let spec_id = partition.spec_id();
    let spec = metadata.partition_spec_by_id(spec_id).ok_or_else(|| {
        Error::spec(SpecError::new(
            ErrorKind::DataInvalid,
            format!("partition spec {spec_id} is not one the table's metadata lists"),
        ))
    })?;
    // A file whose schema the table no longer lists is taken to have the
    // current one's fields.
    let schema = metadata
        .schema_by_id(schema_id)
        .unwrap_or_else(|| metadata.current_schema());
    let value_types = spec.partition_type(schema).map_err(Error::spec)?;
    let values = partition.tuple().fields();
    let pairs: Vec<String> = spec
        .fields()
        .iter()
        .zip(value_types.fields())
        .enumerate()
        .map(|(position, (field, value_type))| {
            let value = match values.get(position) {
                Some(Some(value)) if field.transform != Transform::Void => field
                    .transform
                    .to_human_string(&value_type.field_type, Some(value)),
                _ => String::new(),
            };
            format!("{}={value}", field.name)
        })
        .collect();
    Ok(pairs.join(","))
}

/// Column names mapped to the `min` and `max` bounds a file's manifest entry
/// records: numbers (decimals included, at their full precision) and
/// booleans are JSON scalars, other types strings. A column without both
/// bounds, or with a bound that is not finite, is left out.
fn format_min_max_stats(schema: &Schema, data_file: &DataFile) -> iceberg::Result<String> {
    let mut stats = BTreeMap::new();
    for (field_id, min) in data_file.lower_bounds() {
        let (Some(name), Some(max)) = (
            schema.name_by_field_id(*field_id),
            data_file.upper_bounds().get(field_id),
        ) else {
            continue;
        };
        let (Some(min), Some(max)) = (format_bound(min)?, format_bound(max)?) else {
            continue;
        };
        stats.insert(name, BTreeMap::from([("min", min), ("max", max)]));
    }
    Ok(serde_json::to_string(&stats).expect("column bounds contain only valid JSON"))
}

/// A bound as one JSON value; `None` for one that is not finite.
fn format_bound(datum: &Datum) -> iceberg::Result<Option<Box<RawValue>>> {
    let value =
        Literal::from(datum.clone()).try_into_json(&Type::Primitive(datum.data_type().clone()))?;
    let json = match (datum.data_type(), value) {
        // Iceberg represents NaN and infinities as JSON null.
        (_, Value::Null) => return Ok(None),
        // A decimal stays a number, never rounded through a float.
        (PrimitiveType::Decimal { .. }, Value::String(decimal)) => decimal,
        (_, value) => value.to_string(),
    };
    Ok(Some(
        RawValue::from_string(json).expect("Iceberg bounds render as valid JSON"),
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use iceberg::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, FormatVersion, NestedField,
        PartitionSpec, SortOrder, Struct, TableMetadataBuilder,
    };
    use serde_json::json;

    use super::*;
    use crate::files::file_index::PartitionInterner;

    fn file(lower: HashMap<i32, Datum>, upper: HashMap<i32, Datum>) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("file.parquet".into())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(1)
            .record_count(1)
            .lower_bounds(lower)
            .upper_bounds(upper)
            .build()
            .unwrap()
    }

    #[test]
    fn a_day_partition_is_shown_by_its_fields_name_and_its_date() {
        let schema = Schema::builder()
            .with_fields([Arc::new(NestedField::optional(
                1,
                "at",
                Type::Primitive(PrimitiveType::Timestamp),
            ))])
            .build()
            .unwrap();
        let spec = PartitionSpec::builder(Arc::new(schema.clone()))
            .add_partition_field("at", "at_day", Transform::Day)
            .unwrap()
            .build()
            .unwrap()
            .into_unbound();
        let metadata = TableMetadataBuilder::new(
            schema,
            spec,
            SortOrder::unsorted_order(),
            "file:///t".to_string(),
            FormatVersion::V3,
            HashMap::new(),
        )
        .unwrap()
        .build()
        .unwrap()
        .metadata;
        let spec_id = metadata.default_partition_spec().spec_id();
        let schema_id = metadata.current_schema().schema_id();
        let partitions = PartitionInterner::default();
        let new_years_day =
            partitions.intern(spec_id, &Struct::from_iter([Some(Literal::date(19_723))]));
        let unknown_day = partitions.intern(spec_id, &Struct::from_iter([None]));

        let shown = format_partition(&metadata, &new_years_day, schema_id).unwrap();
        let shown_unknown = format_partition(&metadata, &unknown_day, schema_id).unwrap();

        assert_eq!(shown, "at_day=2024-01-01");
        assert_eq!(shown_unknown, "at_day=");
    }

    #[test]
    fn catalog_bounds_preserve_logical_types_and_decimal_precision() {
        let values = [
            ("boolean", Datum::bool(true)),
            ("integer", Datum::long(i64::MAX)),
            ("float", Datum::double(1.25)),
            ("date", Datum::date_from_str("2020-01-02").unwrap()),
            (
                "timestamp",
                Datum::timestamp_from_str("2020-01-02T03:04:05.123456").unwrap(),
            ),
            (
                "decimal",
                Datum::decimal_from_str("-123456789012345678901234567890123456.78").unwrap(),
            ),
            ("quoted\"name", Datum::string("a\"b\\c\n")),
        ];
        let schema = Schema::builder()
            .with_fields(values.iter().enumerate().map(|(id, (name, value))| {
                Arc::new(NestedField::optional(
                    id as i32 + 1,
                    *name,
                    Type::Primitive(value.data_type().clone()),
                ))
            }))
            .build()
            .unwrap();
        let bounds: HashMap<_, _> = values
            .into_iter()
            .enumerate()
            .map(|(id, (_, value))| (id as i32 + 1, value))
            .collect();

        let rendered = format_min_max_stats(&schema, &file(bounds.clone(), bounds)).unwrap();
        let stats: Value = serde_json::from_str(&rendered).unwrap();

        let expected: Value = serde_json::from_str(
            r#"{
                "boolean": true,
                "integer": 9223372036854775807,
                "float": 1.25,
                "date": "2020-01-02",
                "timestamp": "2020-01-02T03:04:05.123456",
                "decimal": -123456789012345678901234567890123456.78,
                "quoted\"name": "a\"b\\c\n"
            }"#,
        )
        .unwrap();
        for (name, value) in expected.as_object().unwrap() {
            assert_eq!(stats[name], json!({ "min": value, "max": value }));
        }
        assert!(rendered.contains("-123456789012345678901234567890123456.78"));
        assert!(stats["decimal"]["min"].is_number());
    }

    #[test]
    fn catalog_bounds_omit_missing_columns_endpoints_and_non_finite_numbers() {
        let schema = Schema::builder()
            .with_fields([Arc::new(NestedField::optional(
                1,
                "value",
                Type::Primitive(PrimitiveType::Double),
            ))])
            .build()
            .unwrap();
        let bounds = |id, value| HashMap::from([(id, Datum::double(value))]);
        let files = [
            file(HashMap::new(), HashMap::new()),
            file(bounds(1, 0.0), HashMap::new()),
            file(HashMap::new(), bounds(1, 1.0)),
            file(bounds(2, 0.0), bounds(2, 1.0)),
            file(bounds(1, f64::NAN), bounds(1, 1.0)),
            file(bounds(1, 0.0), bounds(1, f64::NAN)),
            file(bounds(1, f64::NEG_INFINITY), bounds(1, 1.0)),
            file(bounds(1, 0.0), bounds(1, f64::INFINITY)),
        ];

        let stats: Vec<_> = files
            .iter()
            .map(|file| format_min_max_stats(&schema, file).unwrap())
            .collect();

        assert!(stats.iter().all(|stats| stats == "{}"));
    }
}
