//! The Iceberg table-metadata document (the JSON a `loadTable` returns) and the
//! mapping from its schema to planner columns. Only what resolving and scanning
//! need is modeled: the current schema and the current snapshot's manifest
//! list; everything else in the document is ignored.

use crate::{Error, Result};
use planner::catalog::Column;
use planner::types::Type;

/// The subset of an Iceberg table-metadata document a scan needs. Handles both
/// format v1 (a single top-level `schema`) and v2 (a `schemas` list selected by
/// `current-schema-id`).
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) struct TableMetadata {
    #[serde(default)]
    current_snapshot_id: Option<i64>,
    #[serde(default)]
    snapshots: Vec<Snapshot>,
    #[serde(default)]
    current_schema_id: Option<i32>,
    #[serde(default)]
    schemas: Vec<IcebergSchema>,
    /// Format v1's single inline schema.
    #[serde(default)]
    schema: Option<IcebergSchema>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) struct Snapshot {
    snapshot_id: i64,
    /// The Avro manifest-list file enumerating the snapshot's manifests. Very
    /// old v1 writers recorded a `manifests` array instead; those tables are
    /// rejected on read.
    #[serde(default)]
    manifest_list: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) struct IcebergSchema {
    #[serde(default)]
    schema_id: Option<i32>,
    fields: Vec<IcebergField>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct IcebergField {
    name: String,
    /// A primitive type is a JSON string (`"long"`, `"string"`, ...); a nested
    /// type (struct/list/map) is a JSON object. Kept raw so the mapping can
    /// report exactly what it rejected.
    #[serde(rename = "type")]
    field_type: serde_json::Value,
}

impl TableMetadata {
    /// The manifest-list URI of the table's current snapshot, or `None` for a
    /// table with no snapshot yet (created, never written to). Errors when the
    /// metadata is inconsistent (a current snapshot id that no snapshot
    /// carries) or predates manifest lists.
    pub(crate) fn find_current_manifest_list(&self) -> Result<Option<String>> {
        // Both "absent" and the sentinel -1 mean "no current snapshot".
        let Some(snapshot_id) = self.current_snapshot_id.filter(|id| *id != -1) else {
            return Ok(None);
        };
        let snapshot = self
            .snapshots
            .iter()
            .find(|s| s.snapshot_id == snapshot_id)
            .ok_or_else(|| {
                Error::Metadata(format!(
                    "current-snapshot-id {snapshot_id} not present in the snapshots list"
                ))
            })?;
        let manifest_list = snapshot.manifest_list.clone().ok_or_else(|| {
            Error::Metadata(format!(
                "snapshot {snapshot_id} carries no manifest-list (a pre-manifest-list v1 table)"
            ))
        })?;
        Ok(Some(manifest_list))
    }

    /// The current schema's columns as planner [`Column`]s, in field order.
    /// `table` names the table in errors.
    pub(crate) fn map_columns(&self, table: &str) -> Result<Vec<Column>> {
        self.find_current_schema()?
            .fields
            .iter()
            .map(|field| {
                let col_type = map_field_type(&field.field_type).ok_or_else(|| {
                    Error::UnsupportedColumnType {
                        table: table.to_string(),
                        column: field.name.clone(),
                        iceberg_type: field.field_type.to_string(),
                    }
                })?;
                Ok(Column {
                    name: field.name.clone(),
                    col_type,
                })
            })
            .collect()
    }

    /// The schema `current-schema-id` names (format v2), or the single inline
    /// `schema` (format v1).
    fn find_current_schema(&self) -> Result<&IcebergSchema> {
        if let Some(current_id) = self.current_schema_id {
            return self
                .schemas
                .iter()
                .find(|schema| schema.schema_id == Some(current_id))
                .ok_or_else(|| {
                    Error::Metadata(format!(
                        "current-schema-id {current_id} not present in the schemas list"
                    ))
                });
        }
        self.schema
            .as_ref()
            .ok_or_else(|| Error::Metadata("metadata carries no schema".to_string()))
    }
}

/// Map one Iceberg field type to the planner [`Type`] a column of it scans as,
/// or `None` for a type pivot has no counterpart for (which fails the resolve,
/// naming the column - never a column that decodes wrong).
fn map_field_type(field_type: &serde_json::Value) -> Option<Type> {
    // A nested type (struct/list/map) is a JSON object, not a string.
    let name = field_type.as_str()?;
    match name {
        "boolean" => Some(Type::Boolean),
        "int" => Some(Type::Int32),
        "long" => Some(Type::Int64),
        "double" => Some(Type::Float64),
        "string" => Some(Type::Utf8),
        "date" => Some(Type::Date),
        // Not supported: timestamp/timestamptz (iceberg stores microseconds,
        // but the engine decodes parquet timestamps as epoch seconds - mapping
        // them would silently misread every value by a factor of a million),
        // float (pivot has no 32-bit float type), decimal, uuid, time, binary,
        // fixed - and anything this list does not know.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> TableMetadata {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn maps_a_v2_schema_to_planner_columns() {
        let metadata = parse(
            r#"{
                "format-version": 2,
                "current-schema-id": 1,
                "schemas": [
                    {"schema-id": 0, "fields": []},
                    {"schema-id": 1, "fields": [
                        {"id": 1, "name": "id", "required": true, "type": "long"},
                        {"id": 2, "name": "name", "required": false, "type": "string"},
                        {"id": 3, "name": "score", "required": false, "type": "double"}
                    ]}
                ]
            }"#,
        );

        let columns = metadata.map_columns("events").unwrap();

        let mapped: Vec<(&str, &Type)> = columns
            .iter()
            .map(|c| (c.name.as_str(), &c.col_type))
            .collect();
        assert_eq!(
            mapped,
            vec![
                ("id", &Type::Int64),
                ("name", &Type::Utf8),
                ("score", &Type::Float64),
            ]
        );
    }

    #[test]
    fn maps_a_v1_inline_schema() {
        let metadata = parse(
            r#"{
                "format-version": 1,
                "schema": {"fields": [
                    {"id": 1, "name": "flag", "required": true, "type": "boolean"}
                ]}
            }"#,
        );

        let columns = metadata.map_columns("t").unwrap();

        assert_eq!(columns[0].col_type, Type::Boolean);
    }

    #[test]
    fn rejects_an_unsupported_column_type_naming_the_column() {
        let metadata = parse(
            r#"{
                "current-schema-id": 0,
                "schemas": [{"schema-id": 0, "fields": [
                    {"id": 1, "name": "amount", "required": true, "type": "decimal(10, 2)"}
                ]}]
            }"#,
        );

        let error = metadata.map_columns("orders").unwrap_err().to_string();

        assert!(error.contains("amount"), "unexpected error: {error}");
        assert!(
            error.contains("decimal(10, 2)"),
            "unexpected error: {error}"
        );
    }

    // The engine decodes parquet timestamps as epoch seconds while iceberg
    // stores microseconds, so timestamp columns must fail the resolve loudly
    // instead of silently misreading every value.
    #[test]
    fn rejects_timestamp_columns() {
        let metadata = parse(
            r#"{
                "current-schema-id": 0,
                "schemas": [{"schema-id": 0, "fields": [
                    {"id": 1, "name": "event_ts", "required": true, "type": "timestamptz"}
                ]}]
            }"#,
        );

        let error = metadata.map_columns("events").unwrap_err().to_string();

        assert!(error.contains("event_ts"), "unexpected error: {error}");
    }

    #[test]
    fn rejects_a_nested_column_type() {
        let metadata = parse(
            r#"{
                "current-schema-id": 0,
                "schemas": [{"schema-id": 0, "fields": [
                    {"id": 1, "name": "tags", "required": false,
                     "type": {"type": "list", "element-id": 2, "element": "string", "element-required": false}}
                ]}]
            }"#,
        );

        assert!(metadata.map_columns("t").is_err());
    }

    #[test]
    fn resolves_the_current_snapshots_manifest_list() {
        let metadata = parse(
            r#"{
                "current-snapshot-id": 42,
                "snapshots": [
                    {"snapshot-id": 41, "manifest-list": "s3://b/w/snap-41.avro"},
                    {"snapshot-id": 42, "manifest-list": "s3://b/w/snap-42.avro"}
                ]
            }"#,
        );

        let manifest_list = metadata.find_current_manifest_list().unwrap();

        assert_eq!(manifest_list.as_deref(), Some("s3://b/w/snap-42.avro"));
    }

    #[test]
    fn a_table_without_snapshots_has_no_manifest_list() {
        assert_eq!(
            parse("{}").find_current_manifest_list().unwrap(),
            None,
            "absent current-snapshot-id"
        );
        assert_eq!(
            parse(r#"{"current-snapshot-id": -1}"#)
                .find_current_manifest_list()
                .unwrap(),
            None,
            "-1 sentinel"
        );
    }

    #[test]
    fn errors_when_the_current_snapshot_is_missing_from_the_list() {
        let metadata = parse(r#"{"current-snapshot-id": 7, "snapshots": []}"#);

        assert!(metadata.find_current_manifest_list().is_err());
    }
}
