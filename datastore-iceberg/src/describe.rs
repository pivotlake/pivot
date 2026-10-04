//! System catalog metadata. Footer reads are batched to bound retained metadata.

use crate::Result;
use crate::manifests::{FileDescriptor, live_files};
use crate::table::LoadedTable;
use catalog::datastore::{DatastoreColumnMetadata, DatastoreFileMetadata, DatastoreTableMetadata};
use iceberg::spec::{DataFile, Datum, Literal, PrimitiveType, Schema, Type};
use serde_json::{Value, value::RawValue};
use std::collections::BTreeMap;

pub(crate) fn describe_table(table: &LoadedTable) -> Result<DatastoreTableMetadata> {
    let manifests = table.load_all_manifests()?;
    let files: Vec<_> = manifests
        .iter()
        .flat_map(live_files)
        .map(|file| (file.manifest, file.data))
        .collect();
    let mut column_bytes = vec![(0u64, 0u64); table.columns().len()];
    let mut described = Vec::with_capacity(files.len());
    // Bound peak memory by processing and releasing one batch of parsed footers
    // at a time instead of loading every footer for the table into this query.
    for chunk in files.chunks(1024) {
        let entries: Vec<_> = chunk
            .iter()
            .map(|(_, file)| FileDescriptor::of(file))
            .collect();
        for ((manifest, file), loaded) in chunk.iter().zip(table.read_footers(&entries)?) {
            let mut uncompressed = 0u64;
            for group in &loaded.row_groups {
                for (position, column) in group.columns.iter().enumerate() {
                    let bytes = column.total_uncompressed_size.max(0) as u64;
                    column_bytes[position].0 += column.total_compressed_size.max(0) as u64;
                    column_bytes[position].1 += bytes;
                    uncompressed += bytes;
                }
            }
            described.push(DatastoreFileMetadata {
                path: file.file_path().to_string(),
                bytes: file.file_size_in_bytes(),
                bytes_uncompressed: uncompressed,
                partition: manifest
                    .metadata()
                    .partition_spec()
                    .partition_to_path(file.partition(), manifest.metadata().schema.clone())
                    .replace('/', ","),
                min_max_stats: format_min_max_stats(table.metadata.current_schema(), file)
                    .map_err(Box::new)?,
            });
        }
    }
    described.sort_unstable_by(|left, right| left.path.cmp(&right.path));
    let columns = table
        .columns()
        .iter()
        .enumerate()
        .map(|(position, column)| {
            let (bytes, bytes_uncompressed) = column_bytes[position];
            DatastoreColumnMetadata {
                name: column.name.clone(),
                column_type: column.col_type.clone(),
                position,
                bytes,
                bytes_uncompressed,
                is_partition_key: table.partition_by.contains(&column.name),
                is_sort_key: table.sort_by.contains(&column.name),
            }
        })
        .collect();
    Ok(DatastoreTableMetadata {
        name: table.name.clone(),
        id: table.uuid.clone(),
        columns,
        sort_by: table.sort_by.clone(),
        partition_by: table.partition_by.clone(),
        total_rows: files.iter().map(|(_, file)| file.record_count()).sum(),
        bytes: described.iter().map(|file| file.bytes).sum(),
        bytes_uncompressed: described.iter().map(|file| file.bytes_uncompressed).sum(),
        files: described,
    })
}

/// Column names mapped to `min`/`max` bounds, matching the Pivotlake catalog:
/// numbers (including decimals) and booleans are JSON scalars; other types
/// are strings. Incomplete bounds and non-finite numbers are omitted.
fn format_min_max_stats(schema: &Schema, data_file: &DataFile) -> iceberg::Result<String> {
    let mut stats = BTreeMap::new();
    for (field_id, min) in data_file.lower_bounds() {
        let (Some(name), Some(max)) = (
            schema.name_by_field_id(*field_id),
            data_file.upper_bounds().get(field_id),
        ) else {
            continue;
        };
        let (Some(min), Some(max)) = (bound_to_json(min)?, bound_to_json(max)?) else {
            continue;
        };
        stats.insert(name, BTreeMap::from([("min", min), ("max", max)]));
    }
    Ok(serde_json::to_string(&stats).expect("column bounds contain only valid JSON"))
}

fn bound_to_json(datum: &Datum) -> iceberg::Result<Option<Box<RawValue>>> {
    let value =
        Literal::from(datum.clone()).try_into_json(&Type::Primitive(datum.data_type().clone()))?;
    let json = match (datum.data_type(), value) {
        // Iceberg represents NaN and infinities as JSON null.
        (_, Value::Null) => return Ok(None),
        // Preserve decimals as numbers without rounding them through f64.
        (PrimitiveType::Decimal { .. }, Value::String(decimal)) => decimal,
        (_, value) => value.to_string(),
    };
    Ok(Some(
        RawValue::from_string(json).expect("Iceberg bounds render as valid JSON"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use iceberg::spec::{DataContentType, DataFileBuilder, DataFileFormat, NestedField};
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::Arc;

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
