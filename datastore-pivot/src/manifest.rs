//! Pivot's database-level table index, stored as `_pivot_manifest.json`.
//!
//! This manifest answers only which tables exist and where their roots are.
//! Each table's schema, version, and active files come from its Delta log.

use planner::catalog::SchemaQualifiedTableName;
use planner::expression::CompareType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid as TableId;

use arrow_array::{ArrayRef, Scalar};
pub use parquet_engine::{
    FileStats, PartitionValues, pivot_scalar, scalar_equal, scalar_values_equal,
    scalar_values_from_row,
};

use crate::FileRef;
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

/// One file at the Delta log level: its store identity ([`FileRef`]), the
/// optional partition tuple a partitioned INSERT stamps on it, and its Parquet
/// statistics. Each optional field is `None` when the file carries no such
/// metadata (an unpartitioned table, or a file whose footer stats were not
/// read).
#[derive(Clone)]
pub struct DeltaFileEntry {
    pub file: FileRef,
    pub partition: Option<PartitionValues>,
    /// The file's Parquet statistics, persisted into the Delta `Add` action's
    /// `stats`. `None` for a file we did not write (adopted at CREATE) or reloaded
    /// from the log, where the stats are not re-committed.
    pub stats: Option<Arc<FileStats>>,
}

impl DeltaFileEntry {
    /// An entry with no partition/sort metadata (unpartitioned + unsorted table,
    /// or a writer that doesn't record it).
    pub fn new(file: FileRef) -> Self {
        Self {
            file,
            partition: None,
            stats: None,
        }
    }

    /// Whether this file *can* hold a row matching every partition filter — a
    /// soft test, so it never wrongly drops a file. A filter on a non-partition
    /// column, an entry with no recorded tuple, or a tuple missing the column all
    /// keep the entry (absence of a value is not proof of a mismatch). Only a
    /// recorded partition value that differs from the filter's excludes it.
    /// Both sides are Pivot-native typed scalars.
    pub fn maybe_matches_partition(
        &self,
        partition_by: &[String],
        filters: &[PartitionEqFilter],
    ) -> bool {
        let Some(tuple) = self.partition.as_ref() else {
            return true;
        };
        filters.iter().all(|filter| {
            !partition_by.iter().any(|c| c == &filter.column)
                || match tuple.get(&filter.column) {
                    Some(value) => scalar_equal(value, &filter.value).unwrap_or(true),
                    None => true,
                }
        })
    }

    /// Whether this file *can* hold a row matching every stat filter — a soft
    /// test (like [`maybe_matches_partition`](Self::maybe_matches_partition)), so
    /// it never wrongly drops a file. A file with no recorded stats, or a filter
    /// on a column the stats don't bound, keeps the file. Only a min/max range
    /// that proves no row can match excludes it. Skips a whole file (all its row
    /// groups) before the finer row-group stats pruning looks inside it.
    pub fn maybe_matches_stats(&self, filters: &[ColumnStatFilter]) -> bool {
        let Some(stats) = self.stats.as_ref() else {
            return true;
        };
        filters.iter().all(|filter| {
            let (Some(min), Some(max)) = (
                stats.min_values.get(&filter.column),
                stats.max_values.get(&filter.column),
            ) else {
                return true;
            };
            !parquet_engine::bounds_eliminate(
                &Scalar::new(min.clone()),
                &Scalar::new(max.clone()),
                filter.compare_type,
                &filter.value,
            )
            .unwrap_or(false)
        })
    }
}

/// A `partition column = constant` predicate the query pushed down, with the
/// constant retained as Pivot's typed Arrow scalar.
/// [`DeltaFileEntry::maybe_matches_partition`] uses it to skip a file whose
/// recorded partition value can't match *before* its footer is fetched — the
/// HTTP a stats prune can't save, since stats live in the footer.
#[derive(Clone, Debug)]
pub struct PartitionEqFilter {
    pub column: String,
    pub value: Scalar<ArrayRef>,
}

/// A `column <cmp> constant` range predicate the query pushed down, retained as
/// Pivot's typed Arrow scalar. [`DeltaFileEntry::maybe_matches_stats`] uses it to
/// skip a whole file whose Parquet stats prove no row can match, before the finer
/// row-group pruning descends into the file. Plain top-level columns only: a
/// variant path's stats live in a shredded leaf, pruned per row group.
#[derive(Clone, Debug)]
pub struct ColumnStatFilter {
    pub column: String,
    pub compare_type: CompareType,
    pub value: Scalar<ArrayRef>,
}

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
/// its files). `retention_ms` is the table's own `deletedFileRetentionDuration`,
/// captured at drop time because the Delta log it lived in is itself part of
/// the storage awaiting deletion.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct DroppedTableEntry {
    pub(crate) id: TableId,
    pub(crate) location: ObjectPath,
    pub(crate) dropped_at_ms: u64,
    pub(crate) retention_ms: u64,
}

/// The database manifest: the index of which schemas and tables exist and where
/// each table's data lives. A monotonically increasing `version` records how
/// many times it has changed.
#[derive(Clone, Serialize, Deserialize)]
pub struct CatalogManifest {
    pub(crate) version: u64,
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
    /// A brand-new database: no tables, and the default schema alone.
    fn default() -> Self {
        Self {
            version: 0,
            schemas: default_schemas(),
            table_locations: HashMap::new(),
            dropped_tables: Vec::new(),
        }
    }
}

impl CatalogManifest {
    /// Load the database manifest, or an empty one if the database is brand new.
    pub fn load(store: &dyn ObjectStore) -> Result<Self> {
        match store.get(&ObjectPath::new(MANIFEST_KEY))? {
            Some(bytes) => Ok(serde_json::from_slice(&bytes)?),
            None => Ok(Self::default()),
        }
    }

    /// Read-modify-write the database manifest as one atomic update: `mutate`
    /// runs against the current document (an empty one if the database is brand
    /// new) and the result replaces it without losing any concurrent writer's
    /// update — the store serializes the whole cycle behind a file lock (local)
    /// or retries it under a version-conditional swap (remote), so `mutate` may
    /// run more than once, each time on a fresh document.
    pub(crate) fn update<T>(
        store: &dyn ObjectStore,
        mut mutate: impl FnMut(&mut CatalogManifest) -> Result<T>,
    ) -> Result<T> {
        let mut outcome = None;
        store.update(&ObjectPath::new(MANIFEST_KEY), &mut |current| {
            let loaded = match current {
                Some(bytes) => serde_json::from_slice(&bytes).map_err(Error::Json),
                None => Ok(Self::default()),
            };
            let result = loaded.and_then(|mut manifest| {
                let value = mutate(&mut manifest)?;
                let bytes = serde_json::to_vec(&manifest)?;
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
        })?;
        outcome.expect("the store ran the update closure at least once")
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
    use arrow_array::{Array, Datum, Int64Array, RecordBatch, StringArray, StringViewArray};
    use arrow_schema::{DataType, Field, Schema};

    fn scalar<T: Array + 'static>(array: T) -> Scalar<ArrayRef> {
        Scalar::new(Arc::new(array))
    }

    fn manifest_entry(partition: HashMap<String, Scalar<ArrayRef>>) -> DeltaFileEntry {
        DeltaFileEntry {
            file: FileRef {
                path: ObjectPath::new("part.parquet"),
                size: 1,
            },
            partition: Some(partition),
            stats: None,
        }
    }

    #[test]
    fn record_batch_row_becomes_typed_scalar_values() {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("service", DataType::Utf8, false),
                Field::new("shard", DataType::Int64, false),
            ])),
            vec![
                Arc::new(StringArray::from(vec!["a", "b"])),
                Arc::new(Int64Array::from(vec![10, 20])),
            ],
        )
        .unwrap();

        let value =
            scalar_values_from_row(&batch, &["service".to_string(), "shard".to_string()], 1)
                .unwrap();

        let service = value.get("service").unwrap().get().0;
        assert_eq!(service.data_type(), &DataType::Utf8View);
        assert_eq!(
            service
                .as_any()
                .downcast_ref::<StringViewArray>()
                .unwrap()
                .value(0),
            "b"
        );
        let shard = value.get("shard").unwrap().get().0;
        assert_eq!(
            shard
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0),
            20
        );
    }

    #[test]
    fn partition_pruning_compares_typed_scalars_softly() {
        let entry = manifest_entry(HashMap::from([(
            "service".to_string(),
            scalar(StringViewArray::from(vec!["api"])),
        )]));
        let partition_by = ["service".to_string()];
        let filter = |value| PartitionEqFilter {
            column: "service".to_string(),
            value: scalar(StringViewArray::from(vec![value])),
        };

        assert!(entry.maybe_matches_partition(&partition_by, &[filter("api")]));
        assert!(!entry.maybe_matches_partition(&partition_by, &[filter("worker")]));

        // A mismatched physical type or absent field is unknown, so pruning
        // keeps the file instead of risking a false negative.
        let mismatched = PartitionEqFilter {
            column: "service".to_string(),
            value: scalar(Int64Array::from(vec![1])),
        };
        assert!(entry.maybe_matches_partition(&partition_by, &[mismatched]));
        assert!(
            manifest_entry(HashMap::new())
                .maybe_matches_partition(&partition_by, &[filter("worker")])
        );
    }

    #[test]
    fn update_creates_the_manifest_and_mutates_it_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let store = object_storage::LocalStore::new(dir.path()).unwrap();

        CatalogManifest::update(&store, |manifest| {
            manifest.add_schema("logs".to_string());
            Ok(())
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
    fn scalar_value_map_equality_is_independent_of_insertion_order() {
        let left = HashMap::from([
            (
                "service".to_string(),
                scalar(StringViewArray::from(vec!["api"])),
            ),
            ("shard".to_string(), scalar(Int64Array::from(vec![3]))),
        ]);
        let right = HashMap::from([
            ("shard".to_string(), scalar(Int64Array::from(vec![3]))),
            (
                "service".to_string(),
                scalar(StringViewArray::from(vec!["api"])),
            ),
        ]);

        assert!(scalar_values_equal(&left, &right));
    }
}
