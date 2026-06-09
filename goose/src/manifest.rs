//! The **table manifest**: a database's durable record of which tables exist and
//! where each table's Parquet data lives.
//!
//! It's a single JSON document at the `_pivot_manifest.json` key of the
//! database's [`ObjectStore`] — so the same [`load`]/[`insert`] serve any
//! database, in-memory ([`MemoryStore`](crate::store::MemoryStore)) or persisted
//! (local directory / S3 / GCS). Opening the catalog [`load`]s it; every
//! `CREATE TABLE` [`insert`]s into it.
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
/// its Parquet data lives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManifestEntry {
    pub name: String,
    pub columns: Vec<Column>,
    /// Where this table's Parquet data lives, relative to the database root (or
    /// an absolute local path) — a directory for a local database, a key prefix
    /// for a remote one. Never carries an object-store scheme.
    pub location: String,
}

/// Every table the database has, recovered from its manifest. Empty for a fresh
/// or in-memory database.
pub fn load(store: &dyn ObjectStore) -> Result<Vec<ManifestEntry>> {
    read_doc(store)?
        .tables
        .into_iter()
        .map(TableRecord::into_entry)
        .collect()
}

/// Durably record a newly-created table (read-modify-write the document).
pub fn insert(store: &dyn ObjectStore, entry: &ManifestEntry) -> Result<()> {
    let mut doc = read_doc(store)?;
    doc.tables.push(TableRecord::from_entry(entry));
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
    tables: Vec<TableRecord>,
}

impl Default for ManifestDoc {
    fn default() -> Self {
        Self {
            version: MANIFEST_VERSION,
            tables: Vec::new(),
        }
    }
}

/// Serialized form of one table entry.
#[derive(Serialize, Deserialize)]
struct TableRecord {
    name: String,
    columns: Vec<ColumnRecord>,
    location: String,
}

/// Serialized form of one column: name plus its SQL type spelling (the same
/// spelling the planner declared), so the type round-trips through the manifest.
#[derive(Serialize, Deserialize)]
struct ColumnRecord {
    name: String,
    #[serde(rename = "type")]
    type_sql: String,
}

impl TableRecord {
    fn from_entry(entry: &ManifestEntry) -> Self {
        Self {
            name: entry.name.clone(),
            columns: entry
                .columns
                .iter()
                .map(|c| ColumnRecord {
                    name: c.name.clone(),
                    type_sql: pivot_type_to_sql(&c.col_type).to_string(),
                })
                .collect(),
            location: entry.location.clone(),
        }
    }

    fn into_entry(self) -> Result<ManifestEntry> {
        let columns = self
            .columns
            .into_iter()
            .map(|c| {
                let col_type = sql_type_to_pivot(&c.type_sql).ok_or_else(|| {
                    ManifestError::Parse(format!(
                        "unknown type `{}` for column `{}` of table `{}`",
                        c.type_sql, c.name, self.name
                    ))
                })?;
                Ok(Column {
                    name: c.name,
                    col_type,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(ManifestEntry {
            name: self.name,
            columns,
            location: self.location,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::MemoryStore;
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
        let store = MemoryStore::new();
        assert!(load(&store).unwrap().is_empty());

        insert(&store, &entry("first")).unwrap();
        insert(&store, &entry("second")).unwrap();

        // A fresh read (as a restart would do) sees both.
        assert_eq!(load(&store).unwrap(), vec![entry("first"), entry("second")]);
    }
}
