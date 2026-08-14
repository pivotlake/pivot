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

use arrow_array::{
    Array, ArrayRef, Datum, RecordBatch, Scalar, StringArray, StringViewArray, UInt32Array,
};
use arrow_schema::{ArrowError, DataType};
use arrow_select::take::take;

use crate::FileRef;
use crate::store::{ObjectPath, ObjectStore};

/// Key of the [`CatalogManifest`] document within the database's object store.
const MANIFEST_KEY: &str = "_pivot_manifest.json";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] crate::store::StoreError),
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
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Select `columns` from one row of `batch` as named, typed Pivot scalars.
/// Plain Arrow UTF-8 is normalized to Pivot's physical `Utf8View`, matching
/// SQL string constants and Parquet footer values.
pub fn scalar_values_from_row(
    batch: &RecordBatch,
    columns: &[String],
    row: usize,
) -> Result<HashMap<String, Scalar<ArrayRef>>, ArrowError> {
    if row >= batch.num_rows() {
        return Err(ArrowError::InvalidArgumentError(format!(
            "row {row} is outside a batch of {} rows",
            batch.num_rows()
        )));
    }
    let schema = batch.schema();
    columns
        .iter()
        .map(|name| {
            let index = schema.index_of(name)?;
            Ok((name.clone(), pivot_scalar(batch.column(index), row)))
        })
        .collect()
}

/// Equality for two maps of typed Arrow scalars. Arrow scalars intentionally
/// do not implement Rust's `Eq`, so compare their one-value arrays with Arrow's
/// typed comparison kernels instead.
pub fn scalar_values_equal(
    left: &HashMap<String, Scalar<ArrayRef>>,
    right: &HashMap<String, Scalar<ArrayRef>>,
) -> bool {
    left.len() == right.len()
        && left.iter().all(|(name, left_value)| {
            right
                .get(name)
                .is_some_and(|right_value| scalar_equal(left_value, right_value) == Some(true))
        })
}

/// SQL-style equality between two typed scalars. Two null partition values are
/// the same partition; one null and one value differ. A type/kernel mismatch is
/// unknown (`None`) so callers performing soft pruning can retain the file.
pub(crate) fn scalar_equal(left: &Scalar<ArrayRef>, right: &Scalar<ArrayRef>) -> Option<bool> {
    let left_array = left.get().0;
    let right_array = right.get().0;
    if left_array.data_type() != right_array.data_type() {
        return None;
    }
    match (left_array.is_null(0), right_array.is_null(0)) {
        (true, true) => Some(true),
        (true, false) | (false, true) => Some(false),
        (false, false) => arrow_ord::cmp::eq(left as &dyn Datum, right as &dyn Datum)
            .ok()
            .filter(|result| result.is_valid(0))
            .map(|result| result.value(0)),
    }
}

pub fn pivot_scalar(array: &ArrayRef, row: usize) -> Scalar<ArrayRef> {
    // The scalar must own its memory outright: `array` may be a zero-copy view
    // into a dispatch worker's read buffer, while the scalar lands in manifest
    // entries that outlive the scan and drop on non-worker threads. A plain
    // slice would keep (and later mis-drop) the worker buffer.
    let value: ArrayRef = match array.data_type() {
        DataType::Utf8 => {
            let strings = array
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("Utf8 array has StringArray representation");
            Arc::new(StringViewArray::from(vec![
                (!strings.is_null(row)).then(|| strings.value(row)),
            ]))
        }
        // `slice` would keep referencing the scan-backed string buffers. Here
        // `gc` copies the selected value into new buffers, detaching the scalar
        // from the dispatch worker's memory before it enters the manifest.
        DataType::Utf8View => {
            let strings = array
                .as_any()
                .downcast_ref::<StringViewArray>()
                .expect("Utf8View array has StringViewArray representation");
            Arc::new(strings.slice(row, 1).gc())
        }
        _ => take(array, &UInt32Array::from(vec![row as u32]), None)
            .expect("one-row take supports every physical column type"),
    };
    Scalar::new(value)
}

/// A file's partition tuple: each partition column's name mapped to the typed
/// value every row of the file carries for it.
pub type PartitionValues = HashMap<String, Scalar<ArrayRef>>;

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

/// A file's Parquet statistics, aggregated over its row groups, as they are
/// persisted into the Delta `Add` action's `stats`: the row count, and per-column
/// min/max and null count for every column whose type carries them (a column
/// missing from a map simply records no stat, which is sound: it is never pruned).
#[derive(Debug, Clone)]
pub struct FileStats {
    /// The file's row count. `None` when it could not be read (a reload whose log
    /// entry recorded no `numRecords`), so an unknown count is never mistaken for
    /// an empty file and a later pass can fill it in.
    pub num_records: Option<i64>,
    /// Per-column min/max as single-element Arrow arrays (the value's own type).
    pub min_values: HashMap<String, ArrayRef>,
    pub max_values: HashMap<String, ArrayRef>,
    pub null_counts: HashMap<String, i64>,
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
            !crate::parquet::bounds_eliminate(
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

/// A table removed from the live name index whose managed storage is waiting
/// for its retention window to pass. The table identity is the key in
/// [`CatalogManifest::dropped_tables`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct DroppedTableEntry {
    pub(crate) location: ObjectPath,
    pub(crate) delete_after_unix_ms: u64,
}

impl CatalogManifestSchemaEntry {
    fn new(name: String) -> Self {
        Self {
            name,
            table_ids: HashMap::new(),
        }
    }
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
    /// Managed table roots awaiting deferred physical deletion. Keeping these
    /// in the durable manifest lets vacuum finish cleanup after a restart.
    #[serde(default)]
    pub(crate) dropped_tables: HashMap<TableId, DroppedTableEntry>,
}

impl Default for CatalogManifest {
    /// A brand-new database: no tables, and the default schema alone.
    fn default() -> Self {
        Self {
            version: 0,
            schemas: default_schemas(),
            table_locations: HashMap::new(),
            dropped_tables: HashMap::new(),
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

    /// Write the database manifest, overwriting the previous version.
    pub fn store(&self, store: &dyn ObjectStore) -> Result<()> {
        store.put(&ObjectPath::new(MANIFEST_KEY), &serde_json::to_vec(self)?)?;
        Ok(())
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
        self.dropped_tables.remove(&id);
        self.version += 1;
        Ok(())
    }

    /// The identity currently registered under `name`, if any.
    pub(crate) fn table_id(&self, name: &SchemaQualifiedTableName) -> Option<TableId> {
        self.schemas
            .iter()
            .find(|schema| schema.name == name.schema)?
            .table_ids
            .get(&name.table)
            .copied()
    }

    /// Remove `name` only when it still resolves to `expected_id`, moving its
    /// location into the durable dropped-table set for deferred vacuum. Returns
    /// `false` without changing anything when the name is absent or has been
    /// rebound to another table incarnation.
    pub(crate) fn tombstone_table(
        &mut self,
        name: &SchemaQualifiedTableName,
        expected_id: TableId,
        delete_after_unix_ms: u64,
    ) -> Result<bool> {
        if self.table_id(name) != Some(expected_id) {
            return Ok(false);
        }
        let location = self
            .table_locations
            .get(&expected_id)
            .cloned()
            .ok_or_else(|| Error::MissingTableLocation {
                table: name.to_string(),
                id: expected_id,
            })?;
        let schema = self
            .schemas
            .iter_mut()
            .find(|schema| schema.name == name.schema)
            .expect("the table lookup found its schema");
        schema.table_ids.remove(&name.table);
        self.table_locations.remove(&expected_id);
        self.dropped_tables.insert(
            expected_id,
            DroppedTableEntry {
                location,
                delete_after_unix_ms,
            },
        );
        self.version += 1;
        Ok(true)
    }

    /// Every managed table root waiting for deferred physical deletion.
    pub(crate) fn dropped_tables(&self) -> impl Iterator<Item = (TableId, &DroppedTableEntry)> {
        self.dropped_tables.iter().map(|(id, entry)| (*id, entry))
    }

    pub(crate) fn dropped_table(&self, id: TableId) -> Option<&DroppedTableEntry> {
        self.dropped_tables.get(&id)
    }

    /// Forget a tombstone after vacuum has deleted its managed storage. The
    /// expected entry guard prevents an old vacuum candidate from removing a
    /// newer record that happens to use the same identity.
    pub(crate) fn remove_dropped_table(
        &mut self,
        id: TableId,
        expected: &DroppedTableEntry,
    ) -> bool {
        if self.dropped_tables.get(&id) != Some(expected) {
            return false;
        }
        self.dropped_tables.remove(&id);
        self.version += 1;
        true
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
    use arrow_array::{Int64Array, StringArray, StringViewArray};
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
