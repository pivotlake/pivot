//! A table as the cross-datastore catalog describes it: the columns it
//! declares with what they cost, and the files it holds, read through the
//! parts of the table's manifests a few partitions at a time and from the
//! files' footers a batch at a time.

use std::collections::{BTreeMap, HashMap};

use catalog::datastore::{DatastoreColumnMetadata, DatastoreFileMetadata, DatastoreTableMetadata};
use iceberg::spec::{Datum, Literal, PrimitiveType, Schema, TableMetadata, Transform, Type};
use iceberg::{Error as SpecError, ErrorKind};
use object_storage::{FileRef, ObjectPath};
use parquet_engine::{FileRowGroups, leaf_count};
use planner::catalog::SchemaQualifiedTableName;
use serde_json::{Value, value::RawValue};

use crate::files::manifest_parts::BoundSide;
use crate::files::partition::Partition;
use crate::table::ManifestReader;
use crate::table::Table;
use crate::{Error, Result};

/// How many files' footers are held at a time.
const FILES_PER_READ: usize = 1024;

/// How many partitions' files are read off the manifests' parts at a time.
const PARTITIONS_PER_LOAD: usize = 16;

impl Table {
    /// This table as the catalog describes it under `name` and `id`, each
    /// file named by `name_file`, which is handed the path the table names
    /// it by. Every file's footer is taken, a batch of files at a time, and
    /// of each only what is reported is kept: a footer the cache holds is
    /// not read, and none read is cached. The files are listed partition
    /// by partition, each partition's manifest by manifest, as the
    /// manifests list them.
    pub fn describe(
        &self,
        name: SchemaQualifiedTableName,
        id: String,
        name_file: &dyn Fn(&ObjectPath) -> Result<String>,
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
        // Every field the current schema names: the bounds of each are
        // reported.
        let schema = self.metadata().current_schema();
        let field_ids: Vec<i32> = (1..=schema.highest_field_id()).collect();
        let partitions = self.list_partitions()?;
        for partitions in partitions.chunks(PARTITIONS_PER_LOAD) {
            let mut reader = ManifestReader::new(self, field_ids.clone());
            let loaded = reader.load_slices(partitions)?;
            // Each file of these partitions: the slice that lists it, and
            // its place there.
            let listed_files = loaded.iter().flatten().flat_map(|slice| {
                (0..slice.files.count_files()).map(move |position| (&**slice, position))
            });
            let listed_files: Vec<_> = listed_files.collect();
            for batch in listed_files.chunks(FILES_PER_READ) {
                let named: Vec<FileRef> = batch
                    .iter()
                    .map(|(slice, position)| FileRef {
                        path: ObjectPath::new(slice.files.path(*position)),
                        size: slice.files.size(*position),
                    })
                    .collect();
                let footers = self.read_footers_unrecorded(&named)?;
                let footers = batch.iter().zip(&named).zip(&footers);
                for (((slice, position), file), footer) in footers {
                    let listed = &slice.files;
                    total_rows += listed.record_count(*position);
                    let bounds = listed.stats_field_ids().iter().filter_map(|field_id| {
                        let bounds = slice.find_bounds(*field_id)?;
                        let read = |side| bounds.read_datum(side, *position);
                        Some((*field_id, read(BoundSide::Lower), read(BoundSide::Upper)))
                    });
                    files.push(DatastoreFileMetadata {
                        path: name_file(&file.path)?,
                        bytes: file.size,
                        bytes_uncompressed: add_file_bytes(
                            &mut column_bytes,
                            &column_positions,
                            footer,
                        ),
                        partition: format_partition(
                            self.metadata(),
                            &slice.partition,
                            listed.schema_id(),
                        )?,
                        min_max_stats: format_min_max_stats(schema, bounds).map_err(Error::spec)?,
                    });
                }
            }
        }
        // A column is a partition key when a field of the default partition
        // spec is the column itself or a transform of it.
        let schema = self.metadata().current_schema();
        let partition_sources: Vec<&str> = self
            .metadata()
            .default_partition_spec()
            .fields()
            .iter()
            .filter_map(|field| schema.name_by_field_id(field.source_id))
            .collect();
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
                    is_partition_key: partition_sources.contains(&column.name.as_str()),
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

/// Column names mapped to the `min` and `max` of `bounds`, what a file's
/// manifest entry records of each field, by field id, below and above:
/// numbers (decimals included, at their full precision) and booleans are
/// JSON scalars, other types strings. A column without both bounds, or
/// with a bound that is not finite, is left out.
fn format_min_max_stats(
    schema: &Schema,
    bounds: impl Iterator<Item = (i32, Option<Datum>, Option<Datum>)>,
) -> iceberg::Result<String> {
    let mut stats = BTreeMap::new();
    for (field_id, min, max) in bounds {
        let (Some(name), Some(min), Some(max)) = (schema.name_by_field_id(field_id), min, max)
        else {
            continue;
        };
        let (Some(min), Some(max)) = (format_bound(&min)?, format_bound(&max)?) else {
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

    use iceberg::spec::NestedField;
    use serde_json::json;

    use super::*;

    /// What a file's entry records of each field `lower` or `upper`
    /// bounds, by field id.
    fn file(
        lower: HashMap<i32, Datum>,
        upper: HashMap<i32, Datum>,
    ) -> Vec<(i32, Option<Datum>, Option<Datum>)> {
        let mut field_ids: Vec<i32> = lower.keys().chain(upper.keys()).copied().collect();
        field_ids.sort_unstable();
        field_ids.dedup();
        field_ids
            .into_iter()
            .map(|field_id| {
                let read = |bounds: &HashMap<i32, Datum>| bounds.get(&field_id).cloned();
                (field_id, read(&lower), read(&upper))
            })
            .collect()
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

        let recorded = file(bounds.clone(), bounds).into_iter();
        let rendered = format_min_max_stats(&schema, recorded).unwrap();
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
            .into_iter()
            .map(|file| format_min_max_stats(&schema, file.into_iter()).unwrap())
            .collect();

        assert!(stats.iter().all(|stats| stats == "{}"));
    }
}
