//! A table as the cross-datastore catalog describes it: the columns it
//! declares with what they cost, and the files it holds, read through the
//! parts of the table's manifests about a manifest's worth at a time and
//! from the files' footers a batch at a time.

use std::collections::{BTreeMap, HashMap};

use catalog::datastore::{DatastoreColumnMetadata, DatastoreFileMetadata, DatastoreTableMetadata};
use iceberg::spec::{
    Literal, PrimitiveLiteral, PrimitiveType, Schema, TableMetadata, Transform, Type,
};
use iceberg::{Error as SpecError, ErrorKind};
use object_storage::{FileRef, ObjectPath};
use parquet_engine::{BoundSide, FileRowGroups, leaf_count};
use planner::catalog::SchemaQualifiedTableName;
use serde_json::{Value, value::RawValue};

use crate::files::manifest_list::ListedManifest;
use crate::files::partition::Partition;
use crate::table::{MANIFESTS_PER_READ, ManifestReader, Table};
use crate::types::partitioning::format_partition_fields;
use crate::{Error, Result};

/// How many files' footers are held at a time.
const FILES_PER_READ: usize = 1024;

/// How many bytes of manifests are read, and held as their parts, at a
/// time: one manifest of about the size manifests are merged up to, or the
/// smaller ones that make that up between them.
const MANIFEST_BYTES_PER_LOAD: u64 = 8 * 1024 * 1024;

/// `manifests` cut into the runs a description reads together, in order:
/// each as many manifests as [`MANIFEST_BYTES_PER_LOAD`] and
/// [`MANIFESTS_PER_READ`] allow, and one at the least.
fn batch_manifests<'m>(manifests: &'m [&'m ListedManifest]) -> Vec<&'m [&'m ListedManifest]> {
    let mut batches = Vec::new();
    let mut batch_start = 0;
    let mut batch_bytes = 0;
    for (position, manifest) in manifests.iter().enumerate() {
        let manifest_bytes = manifest.file().manifest_length.max(0) as u64;
        let is_batch_full = batch_bytes + manifest_bytes > MANIFEST_BYTES_PER_LOAD
            || position - batch_start == MANIFESTS_PER_READ;
        if position > batch_start && is_batch_full {
            batches.push(&manifests[batch_start..position]);
            batch_start = position;
            batch_bytes = 0;
        }
        batch_bytes += manifest_bytes;
    }
    if batch_start < manifests.len() {
        batches.push(&manifests[batch_start..]);
    }
    batches
}

impl Table {
    /// This table as the catalog describes it under `name` and `id`, each
    /// file named by `name_file`, which is handed the path the table names
    /// it by. Every file's footer is taken, a batch of files at a time, and
    /// of each only what is reported is kept: a footer the cache holds is
    /// not read, and none read is cached. The files are listed manifest by
    /// manifest, in manifest-list order, each manifest's partition by
    /// partition in their order, as the manifest lists them. A manifest is
    /// read and parsed once at most, whatever the cache holds: every file
    /// it lists is reported before the next manifests are taken.
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
        let manifests: Vec<&ListedManifest> =
            self.manifest_list().list_data_manifests().iter().collect();
        for manifests in batch_manifests(&manifests) {
            let mut reader = ManifestReader::new(self, field_ids.clone());
            let loaded = reader.load_manifests(manifests)?;
            // Each file these manifests list: the slice that lists it, and
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
                let footers = self.read_footers_without_caching(&named)?;
                let footers = batch.iter().zip(&named).zip(&footers);
                for (((slice, position), file), footer) in footers {
                    let listed = &slice.files;
                    total_rows += listed.record_count(*position);
                    let bounds = listed.stats_field_ids().iter().filter_map(|field_id| {
                        let bounds = slice.find_bounds(*field_id)?;
                        let bound_type = bounds.read_bound_type()?;
                        let read = |side| bounds.read_literal(side, *position);
                        let (min, max) = (read(BoundSide::Lower), read(BoundSide::Upper));
                        Some((*field_id, bound_type, min, max))
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
            partition_by: format_partition_fields(schema, self.metadata().default_partition_spec()),
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
                Some(Some(value)) if field.transform != Transform::Void => {
                    let rendered_type = find_rendered_type(&value_type.field_type, value);
                    field.transform.to_human_string(&rendered_type, Some(value))
                }
                _ => String::new(),
            };
            format!("{}={value}", field.name)
        })
        .collect();
    Ok(pairs.join(","))
}

/// The type `value`, a partition's value of a field of `field_type`, is
/// rendered under: the field's, or the integer type of the value's own
/// width when that is the other one. A partition is held in the width
/// whoever named it read, which is the narrow one of a manifest written
/// before the field's column was widened and the wide one of the table's
/// current schema, whichever schema the field's type is taken from.
fn find_rendered_type(field_type: &Type, value: &Literal) -> Type {
    match (field_type, value) {
        (Type::Primitive(PrimitiveType::Long), Literal::Primitive(PrimitiveLiteral::Int(_))) => {
            Type::Primitive(PrimitiveType::Int)
        }
        (Type::Primitive(PrimitiveType::Int), Literal::Primitive(PrimitiveLiteral::Long(_))) => {
            Type::Primitive(PrimitiveType::Long)
        }
        _ => field_type.clone(),
    }
}

/// What a file's manifest entry records of one field: the field's id, the
/// type its bounds are recorded in, and its lower and upper bound, `None`
/// for one the entry does not record.
type RecordedBounds<'t> = (
    i32,
    &'t PrimitiveType,
    Option<PrimitiveLiteral>,
    Option<PrimitiveLiteral>,
);

/// Column names mapped to the `min` and `max` of `bounds`, what a file's
/// manifest entry records of each field: numbers (decimals included, at
/// their full precision) and booleans are JSON scalars, other types
/// strings. A column without both bounds, or with a bound that is not
/// finite, is left out.
fn format_min_max_stats<'t>(
    schema: &Schema,
    bounds: impl Iterator<Item = RecordedBounds<'t>>,
) -> iceberg::Result<String> {
    let mut stats = BTreeMap::new();
    for (field_id, bound_type, min, max) in bounds {
        let (Some(name), Some(min), Some(max)) = (schema.name_by_field_id(field_id), min, max)
        else {
            continue;
        };
        let (Some(min), Some(max)) = (
            format_bound(bound_type, min)?,
            format_bound(bound_type, max)?,
        ) else {
            continue;
        };
        stats.insert(name, BTreeMap::from([("min", min), ("max", max)]));
    }
    Ok(serde_json::to_string(&stats).expect("column bounds contain only valid JSON"))
}

/// `bound`, a value of `bound_type`, as one JSON value; `None` for one that
/// is not finite.
fn format_bound(
    bound_type: &PrimitiveType,
    bound: PrimitiveLiteral,
) -> iceberg::Result<Option<Box<RawValue>>> {
    let value = Literal::Primitive(bound).try_into_json(&Type::Primitive(bound_type.clone()))?;
    let json = match (bound_type, value) {
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
        DataContentType, DataFileBuilder, DataFileFormat, Datum, FormatVersion, Manifest,
        ManifestContentType, ManifestEntry, ManifestMetadata, ManifestStatus, NestedField,
        PartitionSpec,
    };
    use serde_json::json;

    use super::*;
    use crate::files::manifest_parts::GroupedManifest;

    /// The stats of a file whose manifest entry records the `lower` and the
    /// `upper` bounds of the fields of `schema`, by field id, read off the
    /// parts the entry's manifest is cut into.
    fn format_file_stats(
        schema: &Schema,
        lower: &HashMap<i32, Datum>,
        upper: &HashMap<i32, Datum>,
    ) -> String {
        let data = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("/file".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(1)
            .record_count(1)
            .lower_bounds(lower.clone())
            .upper_bounds(upper.clone())
            .build()
            .unwrap();
        let entry = ManifestEntry::builder()
            .status(ManifestStatus::Added)
            .data_file(data)
            .build();
        let stored = Arc::new(schema.clone());
        let metadata = ManifestMetadata {
            schema: stored.clone(),
            schema_id: stored.schema_id(),
            partition_spec: PartitionSpec::builder(stored).build().unwrap(),
            format_version: FormatVersion::V2,
            content: ManifestContentType::Data,
        };
        let manifest = Manifest::new(metadata, vec![entry]);
        let grouped = GroupedManifest::group("table", manifest).unwrap();
        let mut field_ids: Vec<i32> = lower.keys().chain(upper.keys()).copied().collect();
        field_ids.sort_unstable();
        field_ids.dedup();
        let columns = grouped
            .partitions()
            .next()
            .unwrap()
            .build_columns(&field_ids);
        let recorded = field_ids
            .iter()
            .zip(&columns)
            .filter_map(|(field_id, bounds)| {
                let bound_type = bounds.read_bound_type()?;
                let read = |side| bounds.read_literal(side, 0);
                let (min, max) = (read(BoundSide::Lower), read(BoundSide::Upper));
                Some((*field_id, bound_type, min, max))
            });
        format_min_max_stats(schema, recorded).unwrap()
    }

    /// A schema of one optional field for each of `values`, named and typed
    /// as the value is, with the values as bounds of the fields, by field
    /// id.
    fn build_schema_holding(values: Vec<(&str, Datum)>) -> (Schema, HashMap<i32, Datum>) {
        let fields = values.iter().enumerate().map(|(position, (name, value))| {
            Arc::new(NestedField::optional(
                position as i32 + 1,
                *name,
                Type::Primitive(value.data_type().clone()),
            ))
        });
        let schema = Schema::builder().with_fields(fields).build().unwrap();
        let bounds = values
            .into_iter()
            .enumerate()
            .map(|(position, (_, value))| (position as i32 + 1, value))
            .collect();
        (schema, bounds)
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
        let (schema, bounds) = build_schema_holding(values.to_vec());

        let rendered = format_file_stats(&schema, &bounds, &bounds);
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
    fn catalog_bounds_of_every_other_type_read_as_their_manifest_recorded_them() {
        let (schema, bounds) = build_schema_holding(vec![
            ("int", Datum::int(-7)),
            ("float", Datum::float(1.5_f32)),
            ("time", Datum::time_from_str("03:04:05.123456").unwrap()),
            (
                "timestamptz",
                Datum::timestamptz_from_str("2020-01-02T03:04:05.123456+00:00").unwrap(),
            ),
            (
                "timestamp_ns",
                Datum::timestamp_nanos(1_577_934_245_123_456_789),
            ),
            (
                "timestamptz_ns",
                Datum::timestamptz_nanos(1_577_934_245_123_456_789),
            ),
            (
                "uuid",
                Datum::uuid_from_str("a1a2a3a4-b1b2-c1c2-d1d2-d3d4d5d6d7d8").unwrap(),
            ),
            ("fixed", Datum::fixed([0, 255])),
            ("binary", Datum::binary([1, 2, 3])),
        ]);

        let rendered = format_file_stats(&schema, &bounds, &bounds);

        let stats: Value = serde_json::from_str(&rendered).unwrap();
        let expected = json!({
            "int": -7,
            "float": 1.5,
            "time": "03:04:05.123456",
            "timestamptz": "2020-01-02T03:04:05.123456+00:00",
            "timestamp_ns": "2020-01-02T03:04:05.123456789",
            "timestamptz_ns": "2020-01-02T03:04:05.123456789+00:00",
            "uuid": "a1a2a3a4-b1b2-c1c2-d1d2-d3d4d5d6d7d8",
            "fixed": "00ff",
            "binary": "010203",
        });
        for (name, value) in expected.as_object().unwrap() {
            assert_eq!(stats[name], json!({ "min": value, "max": value }));
        }
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
            (HashMap::new(), HashMap::new()),
            (bounds(1, 0.0), HashMap::new()),
            (HashMap::new(), bounds(1, 1.0)),
            (bounds(2, 0.0), bounds(2, 1.0)),
            (bounds(1, f64::NAN), bounds(1, 1.0)),
            (bounds(1, 0.0), bounds(1, f64::NAN)),
            (bounds(1, f64::NEG_INFINITY), bounds(1, 1.0)),
            (bounds(1, 0.0), bounds(1, f64::INFINITY)),
        ];

        let stats: Vec<_> = files
            .iter()
            .map(|(lower, upper)| format_file_stats(&schema, lower, upper))
            .collect();

        assert!(stats.iter().all(|stats| stats == "{}"));
    }
}
