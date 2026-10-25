//! One data file as the catalog tracks it, whichever format recorded it: its
//! store identity, the partition tuple a partitioned INSERT stamped on it, and
//! its statistics; plus the file-level pruning a query's pushed predicates do
//! before any footer is consulted.

use std::sync::Arc;

use arrow_array::{ArrayRef, Scalar};
use object_storage::FileRef;
use parquet_engine::PushedPredicate;
pub use parquet_engine::{
    FileStats, PartitionValues, pivot_scalar, scalar_equal, scalar_values_equal,
    scalar_values_from_row,
};
use planner::catalog::Column;
use planner::expression::CompareType;

/// The equality predicates among `predicates` as partition-filter
/// candidates: the column name and the typed scalar. The table intersects
/// these with its partition columns, so yielding every equality (not just
/// ones on partition columns, which a binding cannot tell apart) is fine; a
/// non-partition column prunes no files.
pub fn partition_filter_candidates(
    predicates: &[PushedPredicate],
    columns: &[Column],
) -> Vec<PartitionEqFilter> {
    predicates
        .iter()
        .filter(|predicate| matches!(predicate.compare_type, CompareType::Equal))
        .filter_map(|predicate| {
            Some(PartitionEqFilter {
                column: columns.get(predicate.column_idx)?.name.clone(),
                value: predicate.value.clone(),
            })
        })
        .collect()
}

/// The predicates among `predicates` that bound a file's stats: a plain
/// top-level column comparison (no variant path) paired with its typed
/// constant. Variant paths are excluded: their stats live in a shredded
/// leaf, pruned per row group, not in the file's column stats. A column the
/// file's stats don't bound simply prunes no files.
pub fn stat_filter_candidates(
    predicates: &[PushedPredicate],
    columns: &[Column],
) -> Vec<ColumnStatFilter> {
    predicates
        .iter()
        .filter(|predicate| predicate.path.is_empty())
        .filter_map(|predicate| {
            Some(ColumnStatFilter {
                column: columns.get(predicate.column_idx)?.name.clone(),
                compare_type: predicate.compare_type,
                value: predicate.value.clone(),
            })
        })
        .collect()
}

/// One file at the table-metadata level: its store identity ([`FileRef`]), the
/// optional partition tuple a partitioned INSERT stamps on it, and its
/// statistics as the table's metadata recorded them. Each optional field is
/// `None` when the file carries no such metadata (an unpartitioned table, or a
/// file adopted without its stats being read).
#[derive(Clone)]
pub struct FileEntry {
    pub file: FileRef,
    pub partition: Option<PartitionValues>,
    /// The file's statistics, persisted into the table's metadata by the
    /// commit that added it. `None` for a file whose metadata recorded none.
    pub stats: Option<Arc<FileStats>>,
}

impl FileEntry {
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
/// [`FileEntry::maybe_matches_partition`] uses it to skip a file whose
/// recorded partition value can't match *before* its footer is fetched — the
/// HTTP a stats prune can't save, since stats live in the footer.
#[derive(Clone, Debug)]
pub struct PartitionEqFilter {
    pub column: String,
    pub value: Scalar<ArrayRef>,
}

/// A `column <cmp> constant` range predicate the query pushed down, retained as
/// Pivot's typed Arrow scalar. [`FileEntry::maybe_matches_stats`] uses it to
/// skip a whole file whose stats prove no row can match, before the finer
/// row-group pruning descends into the file. Plain top-level columns only: a
/// variant path's stats live in a shredded leaf, pruned per row group.
#[derive(Clone, Debug)]
pub struct ColumnStatFilter {
    pub column: String,
    pub compare_type: CompareType,
    pub value: Scalar<ArrayRef>,
}

/// The partition a file with these `stats` belongs to: each partition column
/// mapped to the value it holds when that value is constant across the file (its
/// min equals its max). Returns `None` if any partition column is absent from the
/// stats or varies within the file, so a file that does not sit in a single
/// partition is left for the caller to refuse or leave unstamped.
pub(crate) fn constant_partition(
    stats: &FileStats,
    partition_by: &[String],
) -> Option<PartitionValues> {
    let mut values = PartitionValues::new();
    for column in partition_by {
        let min = Scalar::new(ArrayRef::clone(stats.min_values.get(column)?));
        let max = Scalar::new(ArrayRef::clone(stats.max_values.get(column)?));
        if scalar_equal(&min, &max) != Some(true) {
            return None;
        }
        values.insert(column.clone(), min);
    }
    Some(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, Datum, Int64Array, RecordBatch, StringArray, StringViewArray};
    use arrow_schema::{DataType, Field, Schema};
    use object_storage::ObjectPath;
    use std::collections::HashMap;

    fn scalar<T: Array + 'static>(array: T) -> Scalar<ArrayRef> {
        Scalar::new(Arc::new(array))
    }

    fn entry_with_partition(partition: HashMap<String, Scalar<ArrayRef>>) -> FileEntry {
        FileEntry {
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
        let entry = entry_with_partition(HashMap::from([(
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
            entry_with_partition(HashMap::new())
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
