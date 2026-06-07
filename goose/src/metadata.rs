//! The goose catalog **snapshot** — the metadata committed as a versioned JSON
//! object at `_goose_log/<version>.json`.
//!
//! This is the read-side contract with the writer (ingest, which owns the
//! compare-and-swap commit loop): the writer publishes a new snapshot per
//! commit; goose reads the highest-versioned one to resolve a table name to the
//! set of Parquet **data files** that back it. A snapshot is self-contained —
//! schemas → tables → columns + data-file locations — and relocatable, because
//! goose-managed files are recorded by a path *relative* to the catalog root.
//!
//! The shape mirrors the reference `goose` extension's metadata, with two
//! read-oriented additions on [`DataFile`]: an optional byte `size` (lets the
//! reader skip a HEAD / suffix-range probe to locate a remote footer) and the
//! existing best-effort `row_count`. Serialization is plain `serde_json` rather
//! than a hand-rolled encoder.

use serde::{Deserialize, Serialize};

/// Bumped if the on-disk snapshot layout changes incompatibly. Readers reject a
/// snapshot whose `format_version` they don't understand.
pub const FORMAT_VERSION: i64 = 1;

/// One column of a table: a name and its SQL type spelled exactly as the writer
/// recorded it (e.g. `"INTEGER"`, `"VARCHAR"`, `"DECIMAL(18,2)"`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Column {
    pub name: String,
    #[serde(rename = "type")]
    pub type_sql: String,
}

/// A single Parquet file backing a table.
///
/// `location` is self-describing — how to read the file follows from its shape,
/// so there is no separate internal/external flag:
/// * **relative** (e.g. `_goose_data/main/events/<uuid>.parquet`) — a
///   goose-managed file under the catalog root; resolved against the root store.
/// * **absolute** (`s3://…`, `gs://…`, `/abs/path`, `file://…`) — a file
///   registered in place, read from wherever it lives.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataFile {
    pub location: String,
    /// Total file size in bytes, if the writer knew it at commit time. When
    /// present the reader can compute the footer offset directly; when `None`
    /// it probes (suffix-range GET for remote, `stat` for local).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    /// Best-effort row count; `0` when unknown.
    #[serde(default)]
    pub row_count: i64,
}

/// A table: its name, ordered columns, and the data files that compose it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Table {
    pub name: String,
    pub columns: Vec<Column>,
    #[serde(default)]
    pub files: Vec<DataFile>,
}

/// A schema (namespace) containing tables. Tables are stored as an array in
/// writer-defined (deterministic) order; look them up by name via
/// [`Schema::table`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Schema {
    pub name: String,
    #[serde(default)]
    pub tables: Vec<Table>,
}

impl Schema {
    pub fn table(&self, name: &str) -> Option<&Table> {
        self.tables.iter().find(|t| t.name == name)
    }
}

/// A complete catalog snapshot at a given version. `version` is monotonic: `0`
/// is the implicit empty catalog (before any commit); the first commit is `1`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogSnapshot {
    pub format_version: i64,
    pub version: i64,
    #[serde(default)]
    pub schemas: Vec<Schema>,
}

/// Error decoding a snapshot's bytes.
#[derive(Debug, thiserror::Error)]
pub enum MetadataError {
    #[error("snapshot is not valid json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported snapshot format_version {found} (this build understands {expected})")]
    UnsupportedFormat { found: i64, expected: i64 },
}

impl CatalogSnapshot {
    /// The implicit pre-commit state: version 0, a single empty `main` schema.
    pub fn empty() -> Self {
        Self {
            format_version: FORMAT_VERSION,
            version: 0,
            schemas: vec![Schema {
                name: "main".to_string(),
                tables: Vec::new(),
            }],
        }
    }

    /// Parse a snapshot from its JSON bytes, rejecting an unknown
    /// `format_version`.
    pub fn from_slice(bytes: &[u8]) -> Result<Self, MetadataError> {
        let snap: CatalogSnapshot = serde_json::from_slice(bytes)?;
        if snap.format_version != FORMAT_VERSION {
            return Err(MetadataError::UnsupportedFormat {
                found: snap.format_version,
                expected: FORMAT_VERSION,
            });
        }
        Ok(snap)
    }

    /// Serialize to JSON bytes. (Used by tests and any in-process writer; the
    /// production writer is ingest.)
    pub fn to_vec(&self) -> Vec<u8> {
        // Serialization of our own owned types cannot fail.
        serde_json::to_vec(self).expect("snapshot serialization is infallible")
    }

    pub fn schema(&self, name: &str) -> Option<&Schema> {
        self.schemas.iter().find(|s| s.name == name)
    }

    pub fn table(&self, schema: &str, table: &str) -> Option<&Table> {
        self.schema(schema)?.table(table)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> CatalogSnapshot {
        CatalogSnapshot {
            format_version: FORMAT_VERSION,
            version: 2,
            schemas: vec![Schema {
                name: "main".to_string(),
                tables: vec![Table {
                    name: "events".to_string(),
                    columns: vec![
                        Column { name: "id".into(), type_sql: "INTEGER".into() },
                        Column { name: "name".into(), type_sql: "VARCHAR".into() },
                    ],
                    files: vec![
                        DataFile {
                            location: "_goose_data/main/events/a.parquet".into(),
                            size: Some(4096),
                            row_count: 3,
                        },
                        DataFile {
                            location: "s3://other/logs/b.parquet".into(),
                            size: None,
                            row_count: 0,
                        },
                    ],
                }],
            }],
        }
    }

    #[test]
    fn round_trips_through_json() {
        let snap = sample();
        let bytes = snap.to_vec();
        let back = CatalogSnapshot::from_slice(&bytes).unwrap();
        assert_eq!(snap, back);
    }

    #[test]
    fn lookups_resolve_by_name() {
        let snap = sample();
        let t = snap.table("main", "events").expect("table exists");
        assert_eq!(t.columns.len(), 2);
        assert_eq!(t.files.len(), 2);
        assert!(snap.table("main", "missing").is_none());
        assert!(snap.table("nope", "events").is_none());
    }

    #[test]
    fn size_is_omitted_when_absent_and_parsed_when_present() {
        let json = br#"{
            "format_version": 1, "version": 1,
            "schemas": [{"name":"main","tables":[{"name":"t","columns":[],
              "files":[{"location":"a.parquet"},
                       {"location":"b.parquet","size":99,"row_count":7}]}]}]
        }"#;
        let snap = CatalogSnapshot::from_slice(json).unwrap();
        let files = &snap.table("main", "t").unwrap().files;
        assert_eq!(files[0].size, None);
        assert_eq!(files[0].row_count, 0); // serde default
        assert_eq!(files[1].size, Some(99));
        assert_eq!(files[1].row_count, 7);
    }

    #[test]
    fn rejects_unknown_format_version() {
        let json = br#"{"format_version": 999, "version": 1, "schemas": []}"#;
        assert!(matches!(
            CatalogSnapshot::from_slice(json),
            Err(MetadataError::UnsupportedFormat { found: 999, expected: 1 })
        ));
    }

    #[test]
    fn empty_is_version_zero_with_main_schema() {
        let snap = CatalogSnapshot::empty();
        assert_eq!(snap.version, 0);
        assert!(snap.schema("main").unwrap().tables.is_empty());
    }
}
