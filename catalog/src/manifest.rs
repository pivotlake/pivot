//! Pivot's database-level table index, stored as `_pivot_manifest.json`.
//!
//! This manifest answers only which tables exist and where their roots are.
//! Each table's schema, version, and active files come from its Delta log.

use planner::catalog::Column;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Datum, RecordBatch, Scalar, StringArray, StringViewArray};
use arrow_schema::{ArrowError, DataType};

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
        _ => array.slice(row, 1),
    };
    Scalar::new(value)
}

/// A file's partition tuple: each partition column's name mapped to the typed
/// value every row of the file carries for it.
pub type PartitionValues = HashMap<String, Scalar<ArrayRef>>;

/// The sort key's range within one file: the `sort_by` columns at the file's
/// first and last row (the file is sorted, so these bound every row). Each side
/// is a map of column names to typed Pivot scalars. Lets a reader prune a file
/// on a sort-key range predicate without fetching its footer.
#[derive(Clone, Debug)]
pub struct SortBounds {
    pub min: HashMap<String, Scalar<ArrayRef>>,
    pub max: HashMap<String, Scalar<ArrayRef>>,
}

/// One file in a table manifest: its store identity ([`FileRef`]) plus the
/// optional partition tuple and sort-key bounds the partitioning/sorting ingest
/// sink stamps on it. Both are `None` for files written without that metadata
/// (an unpartitioned/unsorted table, or compaction output today).
#[derive(Clone)]
pub struct ManifestEntry {
    pub file: FileRef,
    pub partition: Option<PartitionValues>,
    pub sort_bounds: Option<SortBounds>,
}

impl ManifestEntry {
    /// An entry with no partition/sort metadata (unpartitioned + unsorted table,
    /// or a writer that doesn't record it).
    pub fn new(file: FileRef) -> Self {
        Self {
            file,
            partition: None,
            sort_bounds: None,
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
}

/// A `partition column = constant` predicate the query pushed down, with the
/// constant retained as Pivot's typed Arrow scalar.
/// [`ManifestEntry::maybe_matches_partition`] uses it to skip a file whose
/// recorded partition value can't match *before* its footer is fetched — the
/// HTTP a stats prune can't save, since stats live in the footer.
#[derive(Clone, Debug)]
pub struct PartitionEqFilter {
    pub column: String,
    pub value: Scalar<ArrayRef>,
}

/// One table's in-memory Delta snapshot projected into the metadata Pivot's
/// existing Parquet scan path consumes.
///
/// [`CatalogTable`]: crate::catalog::CatalogTable
#[derive(Clone)]
pub struct TableManifest {
    pub version: u64,
    pub columns: Vec<Column>,
    /// Partition columns, in order (identity partitioning); empty = unpartitioned.
    pub partition_by: Vec<String>,
    /// Sort columns, in order; empty = unsorted.
    pub sort_by: Vec<String>,
    pub entries: Vec<ManifestEntry>,
}

/// One table's entry in the [`CatalogManifest`]: its name and the
/// [location](ObjectPath) its Parquet data lives at.
#[derive(Clone, Serialize, Deserialize)]
pub struct CatalogManifestTableEntry {
    pub(crate) name: String,
    pub(crate) location: ObjectPath,
}

impl CatalogManifestTableEntry {
    pub fn new(name: String, location: ObjectPath) -> Self {
        Self { name, location }
    }
}

/// The database manifest: the index of which tables exist and where their data
/// lives. A monotonically increasing `version` records how many times it has
/// changed.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct CatalogManifest {
    pub(crate) version: u64,
    pub(crate) tables: Vec<CatalogManifestTableEntry>,
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

    /// Add (or replace) a table entry and bump the manifest version.
    pub(crate) fn upsert(&mut self, entry: CatalogManifestTableEntry) {
        self.tables.retain(|t| t.name != entry.name);
        self.tables.push(entry);
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

    fn manifest_entry(partition: HashMap<String, Scalar<ArrayRef>>) -> ManifestEntry {
        ManifestEntry {
            file: FileRef {
                path: ObjectPath::new("part.parquet"),
                size: 1,
            },
            partition: Some(partition),
            sort_bounds: None,
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
