//! Pivot's database-level table index, stored as `_pivot_manifest.json`.
//!
//! This manifest answers only which tables exist, where their roots are, and
//! which table format the database keeps them in. Each table's schema,
//! version, and active files come from its own metadata in that format.

use planner::catalog::SchemaQualifiedTableName;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid as TableId;

use crate::format::TableFormat;
use object_storage::{ObjectPath, ObjectStore};

/// Key of the [`CatalogManifest`] document within the database's object store.
const MANIFEST_KEY: &str = "_pivot_manifest.json";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] object_storage::StoreError),
    #[error("manifest json: {0}")]
    Json(#[from] serde_json::Error),
    /// A table was written to a schema the manifest does not hold. Schemas own
    /// their tables, so there is nowhere to put it.
    #[error("schema `{0}` does not exist")]
    MissingSchema(String),
    /// A schema names a table whose identity has no recorded location: half of
    /// the table's registration is missing, so the manifest is inconsistent.
    #[error("table `{table}`: the manifest records identity {id} but no location for it")]
    MissingTableLocation { table: String, id: TableId },
    /// A table was removed by a name the manifest does not hold. The caller
    /// resolved the table against live state, so the manifest is out of sync.
    #[error("table `{0}` does not exist")]
    MissingTable(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The schema an entry belongs to when the document omits the field: the
/// default schema, which every datastore defines.
fn default_schema_name() -> String {
    planner::DEFAULT_SCHEMA_NAME.to_string()
}

/// The schema list of a document that omits the field: the default schema
/// alone, which every datastore defines.
fn default_schemas() -> Vec<CatalogManifestSchemaEntry> {
    vec![CatalogManifestSchemaEntry::new(default_schema_name())]
}

/// One schema's entry in the [`CatalogManifest`]: its name, and the identity
/// each of its table names resolves to.
///
/// Only the names live here. A name is meaningful just inside one schema (two
/// schemas can each hold an `events`), so it nests; an identity is the table
/// itself and is unique across the database, so where its data lives is
/// recorded once, at the top level. Going through the identity means a rename
/// touches only this map, and the storage it points at never has to move.
#[derive(Clone, Serialize, Deserialize)]
pub struct CatalogManifestSchemaEntry {
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) table_ids: HashMap<String, TableId>,
}

impl CatalogManifestSchemaEntry {
    fn new(name: String) -> Self {
        Self {
            name,
            table_ids: HashMap::new(),
        }
    }
}

/// A dropped table's tombstone: the identity and storage location the table
/// held, kept so vacuum can delete the storage once the retention window has
/// passed (a query that bound the table before the drop may still be reading
/// its files). `retention_ms` is the table's own deleted-file retention,
/// captured at drop time because the metadata it lived in is itself part of
/// the storage awaiting deletion.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct DroppedTableEntry {
    pub(crate) id: TableId,
    pub(crate) location: ObjectPath,
    pub(crate) dropped_at_ms: u64,
    pub(crate) retention_ms: u64,
}

/// The database manifest: the index of which schemas and tables exist, where
/// each table's data lives, and the format every table is kept in. A
/// monotonically increasing `version` records how many times it has changed.
#[derive(Clone, Serialize, Deserialize)]
pub struct CatalogManifest {
    pub(crate) version: u64,
    /// The format every table of this database is kept in. Absent from the
    /// document of a database created before the field existed, which is
    /// Delta Lake; always written for a database created since.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) table_format: Option<TableFormat>,
    /// Every schema the database defines, always including the default one,
    /// each naming the tables inside it.
    #[serde(default = "default_schemas")]
    pub(crate) schemas: Vec<CatalogManifestSchemaEntry>,
    /// Where each table's data lives, keyed by the table's identity. Kept
    /// outside the schemas because an identity is unique across the database
    /// and a location belongs to the table, not to whichever schema currently
    /// names it.
    #[serde(default)]
    pub(crate) table_locations: HashMap<TableId, ObjectPath>,
    /// Tables that were dropped but whose storage has not been reclaimed yet.
    /// Vacuum deletes each one's storage after its retention window and then
    /// removes the tombstone.
    #[serde(default)]
    pub(crate) dropped_tables: Vec<DroppedTableEntry>,
}

impl Default for CatalogManifest {
    /// A brand-new database as a document that records no format reads: no
    /// tables, and the default schema alone.
    fn default() -> Self {
        Self {
            version: 0,
            table_format: None,
            schemas: default_schemas(),
            table_locations: HashMap::new(),
            dropped_tables: Vec::new(),
        }
    }
}

impl CatalogManifest {
    /// Load the database manifest, or an empty one if the database is brand new.
    pub fn load(store: &dyn ObjectStore) -> Result<Self> {
        Ok(Self::load_existing(store)?.unwrap_or_default())
    }

    /// Load the database manifest, or `None` if the database has none yet.
    pub(crate) fn load_existing(store: &dyn ObjectStore) -> Result<Option<Self>> {
        match store.get(&ObjectPath::new(MANIFEST_KEY))? {
            Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            None => Ok(None),
        }
    }

    /// The format this database keeps its tables in: the recorded one, or
    /// Delta Lake for a document written before the field existed.
    pub(crate) fn table_format(&self) -> TableFormat {
        self.table_format
            .unwrap_or(TableFormat::RECORDED_BY_OMISSION)
    }

    /// Read-modify-write the database manifest as one atomic update: `mutate`
    /// runs against the current document (an empty one if the database is brand
    /// new) and the result replaces it without losing any concurrent writer's
    /// update — the store serializes the whole cycle behind a file lock (local)
    /// or retries it under a version-conditional swap (remote), so `mutate` may
    /// run more than once, each time on a fresh document.
    pub(crate) fn update<T, E: From<Error>>(
        store: &dyn ObjectStore,
        mut mutate: impl FnMut(&mut CatalogManifest) -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E> {
        let mut outcome = None;
        store
            .update(&ObjectPath::new(MANIFEST_KEY), &mut |current| {
                let loaded = match current {
                    Some(bytes) => {
                        serde_json::from_slice(&bytes).map_err(|error| E::from(Error::Json(error)))
                    }
                    None => Ok(Self::default()),
                };
                let result = loaded.and_then(|mut manifest| {
                    let value = mutate(&mut manifest)?;
                    let bytes = serde_json::to_vec(&manifest)
                        .map_err(|error| E::from(Error::Json(error)))?;
                    Ok((value, bytes))
                });
                match result {
                    Ok((value, bytes)) => {
                        outcome = Some(Ok(value));
                        Some(bytes)
                    }
                    // A failed mutation writes nothing; the error surfaces below.
                    Err(error) => {
                        outcome = Some(Err(error));
                        None
                    }
                }
            })
            .map_err(|error| E::from(Error::Store(error)))?;
        outcome.expect("the store ran the update closure at least once")
    }

    /// Whether this document has never been written: what a database that is
    /// being created looks like, as opposed to one written before the format
    /// field existed.
    pub(crate) fn is_fresh(&self) -> bool {
        self.version == 0
    }

    /// Register a table inside its schema under `id`, replacing whatever that
    /// name resolved to before. Errors if the schema does not exist: a table
    /// only ever lives inside one.
    pub(crate) fn upsert_table(
        &mut self,
        name: &SchemaQualifiedTableName,
        id: TableId,
        location: ObjectPath,
    ) -> Result<()> {
        let schema = self
            .schemas
            .iter_mut()
            .find(|s| s.name == name.schema)
            .ok_or_else(|| Error::MissingSchema(name.schema.clone()))?;
        schema.table_ids.insert(name.table.clone(), id);
        self.table_locations.insert(id, location);
        self.version += 1;
        Ok(())
    }

    /// Unregister the table `name` resolves to, leaving a tombstone so vacuum
    /// can delete the table's storage once `retention_ms` has passed (counted
    /// from `dropped_at_ms`). Errors if the schema or the name is missing: the
    /// caller resolved the table against live state, so a manifest that
    /// disagrees is out of sync, not a no-op.
    pub(crate) fn remove_table(
        &mut self,
        name: &SchemaQualifiedTableName,
        dropped_at_ms: u64,
        retention_ms: u64,
    ) -> Result<TableId> {
        let schema = self
            .schemas
            .iter_mut()
            .find(|s| s.name == name.schema)
            .ok_or_else(|| Error::MissingSchema(name.schema.clone()))?;
        let id = schema
            .table_ids
            .remove(&name.table)
            .ok_or_else(|| Error::MissingTable(name.to_string()))?;
        let location =
            self.table_locations
                .remove(&id)
                .ok_or_else(|| Error::MissingTableLocation {
                    table: name.to_string(),
                    id,
                })?;
        self.dropped_tables.push(DroppedTableEntry {
            id,
            location,
            dropped_at_ms,
            retention_ms,
        });
        self.version += 1;
        Ok(id)
    }

    /// The dropped-table tombstones awaiting storage reclamation.
    pub(crate) fn dropped_tables(&self) -> &[DroppedTableEntry] {
        &self.dropped_tables
    }

    /// Forget the tombstone for dropped table `id`: its storage has been
    /// deleted, so there is nothing left to reclaim. Unknown ids are a no-op,
    /// so reclaiming a tombstone another writer already forgot converges.
    pub(crate) fn remove_dropped_table(&mut self, id: &TableId) {
        let before = self.dropped_tables.len();
        self.dropped_tables.retain(|entry| entry.id != *id);
        if self.dropped_tables.len() != before {
            self.version += 1;
        }
    }

    /// The identity `name` currently resolves to, or `None`: how a refresh
    /// decides whether the table it holds is still the one the manifest names.
    pub(crate) fn table_id(&self, name: &SchemaQualifiedTableName) -> Option<TableId> {
        self.schemas
            .iter()
            .find(|s| s.name == name.schema)?
            .table_ids
            .get(&name.table)
            .copied()
    }

    /// Every table the database holds: its schema-qualified name, its identity,
    /// and where its data lives. A named identity with no recorded location is
    /// an inconsistent manifest, reported as an error rather than skipped:
    /// silently dropping the entry would make the table vanish from the
    /// catalog.
    pub(crate) fn tables(
        &self,
    ) -> impl Iterator<Item = Result<(SchemaQualifiedTableName, TableId, &ObjectPath)>> {
        self.schemas.iter().flat_map(move |schema| {
            schema.table_ids.iter().map(move |(table, id)| {
                let name = SchemaQualifiedTableName::new(schema.name.clone(), table.clone());
                let location =
                    self.table_locations
                        .get(id)
                        .ok_or_else(|| Error::MissingTableLocation {
                            table: name.to_string(),
                            id: *id,
                        })?;
                Ok((name, *id, location))
            })
        })
    }

    /// Whether this manifest already lists `schema`.
    pub(crate) fn contains_schema(&self, schema: &str) -> bool {
        self.schemas.iter().any(|s| s.name == schema)
    }

    /// Add a schema and bump the manifest version. Callers check
    /// [`contains_schema`](Self::contains_schema) first to decide whether a
    /// duplicate is an error; this stays idempotent so a caller that tolerates
    /// one cannot double-list it.
    pub(crate) fn add_schema(&mut self, schema: String) {
        if self.contains_schema(&schema) {
            return;
        }
        self.schemas.push(CatalogManifestSchemaEntry::new(schema));
        self.version += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_creates_the_manifest_and_mutates_it_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let store = object_storage::LocalStore::new(dir.path()).unwrap();

        CatalogManifest::update(&store, |manifest| {
            manifest.add_schema("logs".to_string());
            Ok::<(), Error>(())
        })
        .unwrap();

        let manifest = CatalogManifest::load(&store).unwrap();
        assert!(manifest.contains_schema("logs"));
        assert_eq!(manifest.version, 1);
    }

    #[test]
    fn a_failed_mutation_stores_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let store = object_storage::LocalStore::new(dir.path()).unwrap();

        let error = CatalogManifest::update(&store, |_| {
            Err::<(), _>(Error::MissingSchema("nope".to_string()))
        })
        .unwrap_err();

        assert!(matches!(error, Error::MissingSchema(_)));
        assert!(store.get(&ObjectPath::new(MANIFEST_KEY)).unwrap().is_none());
    }

    #[test]
    fn a_document_without_a_format_is_a_delta_database_and_stays_so_when_rewritten() {
        let manifest: CatalogManifest =
            serde_json::from_str(r#"{"version":3,"schemas":[{"name":"main","table_ids":{}}]}"#)
                .unwrap();

        let rewritten = serde_json::to_string(&manifest).unwrap();

        assert_eq!(manifest.table_format(), TableFormat::Delta);
        assert!(!rewritten.contains("table_format"), "{rewritten}");
    }

    #[test]
    fn a_recorded_format_round_trips_and_accepts_the_deltalake_alias() {
        let iceberg: CatalogManifest =
            serde_json::from_str(r#"{"version":0,"table_format":"iceberg"}"#).unwrap();
        let aliased: CatalogManifest =
            serde_json::from_str(r#"{"version":0,"table_format":"deltalake"}"#).unwrap();

        let rewritten = serde_json::to_string(&iceberg).unwrap();

        assert_eq!(iceberg.table_format(), TableFormat::Iceberg);
        assert_eq!(aliased.table_format(), TableFormat::Delta);
        assert!(
            rewritten.contains(r#""table_format":"iceberg""#),
            "{rewritten}"
        );
    }
}
