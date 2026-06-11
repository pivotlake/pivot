//! The **table manifest**: a database's durable record of which tables exist and
//! where each table's Parquet data lives.
//!
//! It's a single JSON document at the `_pivot_manifest.json` key of the
//! database's [`ObjectStore`] — so the same [`load`]/[`insert`] serve any
//! database, whether its store is a local directory or S3 / GCS. Opening the
//! catalog [`load`]s it; every `CREATE TABLE` [`insert`]s into it.
//!
//! The manifest tracks only table *definitions* (name, declared schema, and the
//! location of the data). The Parquet row-group metadata itself is read from
//! those locations when the catalog loads — the manifest is the small, durable
//! index that points at them. The declared columns are kept here (not re-derived
//! from the Parquet on load) because they are the authoritative logical schema —
//! possibly a deliberate reinterpretation of the physical types — and because an
//! empty table has no Parquet to derive them from.

use crate::sql_type::{pivot_type_to_sql, sql_type_to_pivot};
use crate::store::ObjectStore;
use planner::catalog::Column;
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error(transparent)]
    Store(#[from] crate::store::StoreError),
    #[error("malformed table manifest: {0}")]
    Parse(String),
}

pub type Result<T> = std::result::Result<T, ManifestError>;

/// Key of the manifest document within the database's object store.
const MANIFEST_KEY: &str = "_pivot_manifest.json";
/// On-disk format version of the manifest document.
const MANIFEST_VERSION: u32 = 1;

/// One table's entry in the manifest: its identity, declared schema, and where
/// its Parquet data lives. Serializes directly as the on-disk record; the
/// columns round-trip through their SQL type spelling (see [`sql_columns`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestEntry {
    pub name: String,
    #[serde(with = "sql_columns")]
    pub columns: Vec<Column>,
    /// Where this table's Parquet data lives, relative to the database root (or
    /// an absolute local path) — a directory for a local database, a key prefix
    /// for a remote one. Never carries an object-store scheme.
    pub location: String,
}

/// Every table the database has, recovered from its manifest. Empty for a fresh
/// or in-memory database.
pub fn load(store: &dyn ObjectStore) -> Result<Vec<ManifestEntry>> {
    Ok(read_doc(store)?.tables)
}

/// Durably record a newly-created table (read-modify-write the document).
pub fn insert(store: &dyn ObjectStore, entry: &ManifestEntry) -> Result<()> {
    let mut doc = read_doc(store)?;
    doc.tables.push(entry.clone());
    let bytes = serde_json::to_vec_pretty(&doc)
        .map_err(|e| ManifestError::Parse(format!("serializing manifest: {e}")))?;
    store.put(MANIFEST_KEY, &bytes)?;
    Ok(())
}

fn read_doc(store: &dyn ObjectStore) -> Result<ManifestDoc> {
    let Some(bytes) = store.get(MANIFEST_KEY)? else {
        return Ok(ManifestDoc::default());
    };
    let doc: ManifestDoc = serde_json::from_slice(&bytes)
        .map_err(|e| ManifestError::Parse(format!("{MANIFEST_KEY}: {e}")))?;
    if doc.version > MANIFEST_VERSION {
        return Err(ManifestError::Parse(format!(
            "{MANIFEST_KEY}: manifest version {} is newer than this build understands ({MANIFEST_VERSION})",
            doc.version
        )));
    }
    Ok(doc)
}

/// Serialized form of the whole manifest document.
#[derive(Serialize, Deserialize)]
struct ManifestDoc {
    version: u32,
    tables: Vec<ManifestEntry>,
}

impl Default for ManifestDoc {
    fn default() -> Self {
        Self {
            version: MANIFEST_VERSION,
            tables: Vec::new(),
        }
    }
}

/// Columns serialize as `{name, type}` pairs, with `type` spelled the way the
/// planner declared it — the authoritative logical schema, which round-trips
/// even when it deliberately reinterprets the physical Parquet type.
mod sql_columns {
    use super::*;
    use serde::de::Error as _;
    use serde::{Deserializer, Serializer};

    /// The on-disk shape of one column.
    #[derive(Serialize, Deserialize)]
    struct ColumnSql {
        name: String,
        #[serde(rename = "type")]
        type_sql: String,
    }

    pub fn serialize<S: Serializer>(
        columns: &[Column],
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serde::Serialize::serialize(
            &columns
                .iter()
                .map(|c| ColumnSql {
                    name: c.name.clone(),
                    type_sql: pivot_type_to_sql(&c.col_type).to_string(),
                })
                .collect::<Vec<_>>(),
            serializer,
        )
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Vec<Column>, D::Error> {
        let columns: Vec<ColumnSql> = serde::Deserialize::deserialize(deserializer)?;
        columns
            .into_iter()
            .map(|c| {
                let col_type = sql_type_to_pivot(&c.type_sql).ok_or_else(|| {
                    D::Error::custom(format!(
                        "unknown type `{}` for column `{}`",
                        c.type_sql, c.name
                    ))
                })?;
                Ok(Column {
                    name: c.name,
                    col_type,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::LocalStore;
    use planner::types::Type;

    fn entry(name: &str) -> ManifestEntry {
        ManifestEntry {
            name: name.into(),
            columns: vec![Column {
                name: "a".into(),
                col_type: Type::Int32,
            }],
            location: name.to_string(),
        }
    }

    #[test]
    fn round_trips_through_the_store() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::new(dir.path());
        assert!(load(&store).unwrap().is_empty());

        insert(&store, &entry("first")).unwrap();
        insert(&store, &entry("second")).unwrap();

        // A fresh read (as a restart would do) sees both.
        assert_eq!(load(&store).unwrap(), vec![entry("first"), entry("second")]);
    }
}
