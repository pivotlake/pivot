//! Builds the observer a scan-adjacent filter feeds to populate the
//! [`QueryConditionCache`].
//!
//! The observer receives every filter batch (tagged with row-group metadata
//! columns) and its boolean mask, accumulates the passing row positions per
//! row group, and publishes a row group to the cache only once every one of
//! its rows has been seen. That completeness gate is what makes observation
//! safe under everything that can cut a scan short: dynamic Top-N pruning
//! skipping row groups at steal time, a LIMIT cancelling the query early,
//! dictionary pushdown pruning a row group mid-scan, and mixed scans where
//! already-cached row groups emit only their surviving rows. An incomplete
//! row group is simply never published.

#[cfg(test)]
use crate::catalog::condition_cache::CachedPositions;
use crate::catalog::condition_cache::QueryConditionCache;
use crate::parquet::{ParquetTable, row_index, visit_row_group_runs};
use arrow_array::{Array, BooleanArray, RecordBatch};
use planner::catalog::ConditionObserverFn;
use planner::condition_key::ConditionKey;
use std::cmp::Ordering;
use std::mem;
use std::sync::{Arc, Mutex};

/// One row group's observation state. Batches of one row group can arrive on
/// any worker (downstream stages steal), so the state is locked per row group
/// and shared through the one observer closure.
struct RowGroupObservation {
    file_path: Arc<str>,
    file_row_group_idx: usize,
    num_rows: u64,
    state: Mutex<ObservationState>,
}

enum ObservationState {
    Pending { rows_seen: u64, positions: Vec<u32> },
    Done,
}

/// Build the observer for one scan: `observations` aligns with the scanned
/// `parquet.row_groups()` (the space the batches' row-group ids index), with
/// already-`covered` row groups pre-marked done so their (position-filtered)
/// batches are ignored.
pub(super) fn build_condition_observer(
    cache: Arc<QueryConditionCache>,
    condition: ConditionKey,
    parquet: &ParquetTable,
    covered: &[bool],
) -> ConditionObserverFn {
    let observations: Vec<RowGroupObservation> = parquet
        .row_groups()
        .iter()
        .zip(covered)
        .map(|(row_group, covered)| RowGroupObservation {
            file_path: row_group.file_path.clone(),
            file_row_group_idx: row_group.file_row_group_idx,
            num_rows: row_group.num_rows as u64,
            state: Mutex::new(if *covered {
                ObservationState::Done
            } else {
                ObservationState::Pending {
                    rows_seen: 0,
                    positions: Vec::new(),
                }
            }),
        })
        .collect();
    Arc::new(move |batch: &RecordBatch, mask: &BooleanArray| {
        observe_batch(&observations, &cache, &condition, batch, mask);
    })
}

fn observe_batch(
    observations: &[RowGroupObservation],
    cache: &QueryConditionCache,
    condition: &ConditionKey,
    batch: &RecordBatch,
    mask: &BooleanArray,
) {
    let row_indices = row_index(batch);
    visit_row_group_runs(batch, |group, logical_start, logical_end| {
        let observation = &observations[group as usize];
        let mut state = observation.state.lock().unwrap();
        let ObservationState::Pending {
            rows_seen,
            positions,
        } = &mut *state
        else {
            return;
        };
        for logical in logical_start..logical_end {
            // A null mask slot excludes the row, matching how the filter
            // itself treats it.
            if mask.is_valid(logical) && mask.value(logical) {
                positions.push(row_indices.value(logical));
            }
        }
        *rows_seen += (logical_end - logical_start) as u64;
        match (*rows_seen).cmp(&observation.num_rows) {
            Ordering::Less => {}
            Ordering::Equal => {
                let mut positions = mem::take(positions);
                // Batches of one row group may arrive out of order (stolen
                // across workers); downstream consumers require sorted
                // positions.
                positions.sort_unstable();
                cache.insert(
                    observation.file_path.clone(),
                    observation.file_row_group_idx,
                    condition,
                    Arc::new(positions),
                    observation.num_rows,
                );
                *state = ObservationState::Done;
            }
            Ordering::Greater => {
                // More rows than the row group holds means the accounting is
                // broken; publishing would poison the cache, so drop it.
                debug_assert!(
                    false,
                    "observed more rows than the row group holds ({} > {})",
                    rows_seen, observation.num_rows
                );
                *state = ObservationState::Done;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::test_utils::dummy_row_group;
    use crate::parquet::with_row_group_metadata;
    use arrow_array::RecordBatchOptions;
    use arrow_array::{ArrayRef, Int32Array, Scalar};
    use arrow_schema::Schema;
    use planner::condition_key::{ConditionKeyOutcome, build_condition_key};
    use planner::expression::{Compare, CompareType, Expression, Ref};
    use planner::types::Type;

    fn test_condition() -> ConditionKey {
        let expression = Expression::Compare(Compare {
            left: Box::new(Expression::Ref(Ref {
                column_idx: 0,
                return_type: Type::Int32,
                name: None,
            })),
            right: Box::new(Expression::Constant(Scalar::new(
                Arc::new(Int32Array::from(vec![7])) as ArrayRef,
            ))),
            compare_type: CompareType::Equal,
            return_type: Type::Int8,
        });
        match build_condition_key(&[&expression], &["a".to_string()]) {
            ConditionKeyOutcome::Cacheable(key) => key,
            ConditionKeyOutcome::NotCacheable => unreachable!("a plain compare is cacheable"),
        }
    }

    fn table_of_rows(num_rows: i64) -> ParquetTable {
        let mut row_group = (*dummy_row_group()).clone();
        row_group.num_rows = num_rows;
        ParquetTable::new(vec![Arc::new(row_group)])
    }

    /// A metadata-only batch tagging `rows` rows of row group `group`, with row
    /// indices starting at `offset`.
    fn meta_batch(group: usize, offset: usize, rows: usize) -> RecordBatch {
        let empty = RecordBatch::try_new_with_options(
            Arc::new(Schema::empty()),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )
        .unwrap();
        with_row_group_metadata(empty, group, offset)
    }

    fn cache() -> Arc<QueryConditionCache> {
        Arc::new(QueryConditionCache::new(true, 1 << 20))
    }

    /// The sparse positions of a cached entry, or `None` for absent/dense.
    fn sparse(cached: Option<CachedPositions>) -> Option<Arc<Vec<u32>>> {
        match cached {
            Some(CachedPositions::Sparse(positions)) => Some(positions),
            Some(CachedPositions::Dense) | None => None,
        }
    }

    #[test]
    fn publishes_only_at_full_row_group_coverage() {
        let cache = cache();
        let condition = test_condition();
        let table = table_of_rows(6);
        let observer = build_condition_observer(cache.clone(), condition.clone(), &table, &[false]);

        observer(
            &meta_batch(0, 0, 4),
            &BooleanArray::from(vec![true, false, false, false]),
        );
        let after_partial = cache.stats().inserts;
        observer(&meta_batch(0, 4, 2), &BooleanArray::from(vec![false, true]));

        assert_eq!(after_partial, 0);
        let key: Arc<str> = table.row_groups()[0].file_path.clone();
        assert_eq!(
            sparse(cache.get(&key, 0, &condition)).as_deref(),
            Some(&vec![0, 5])
        );
    }

    #[test]
    fn out_of_order_batches_publish_sorted_positions() {
        let cache = cache();
        let condition = test_condition();
        let table = table_of_rows(6);
        let observer = build_condition_observer(cache.clone(), condition.clone(), &table, &[false]);

        observer(
            &meta_batch(0, 3, 3),
            &BooleanArray::from(vec![true, false, false]),
        );
        observer(
            &meta_batch(0, 0, 3),
            &BooleanArray::from(vec![true, false, false]),
        );

        let key: Arc<str> = table.row_groups()[0].file_path.clone();
        assert_eq!(
            sparse(cache.get(&key, 0, &condition)).as_deref(),
            Some(&vec![0, 3])
        );
    }

    #[test]
    fn null_mask_slots_exclude_their_rows() {
        let cache = cache();
        let condition = test_condition();
        let table = table_of_rows(3);
        let observer = build_condition_observer(cache.clone(), condition.clone(), &table, &[false]);

        observer(
            &meta_batch(0, 0, 3),
            &BooleanArray::from(vec![Some(true), None, Some(false)]),
        );

        let key: Arc<str> = table.row_groups()[0].file_path.clone();
        assert_eq!(
            sparse(cache.get(&key, 0, &condition)).as_deref(),
            Some(&vec![0])
        );
    }

    #[test]
    fn a_mostly_true_result_publishes_a_dense_marker() {
        let cache = cache();
        let condition = test_condition();
        let table = table_of_rows(4);
        let observer = build_condition_observer(cache.clone(), condition.clone(), &table, &[false]);

        observer(
            &meta_batch(0, 0, 4),
            &BooleanArray::from(vec![true, true, true, false]),
        );

        let key: Arc<str> = table.row_groups()[0].file_path.clone();
        assert!(matches!(
            cache.get(&key, 0, &condition),
            Some(CachedPositions::Dense)
        ));
    }

    #[test]
    fn a_precached_row_group_is_never_republished() {
        let cache = cache();
        let condition = test_condition();
        let table = table_of_rows(2);
        let observer = build_condition_observer(cache.clone(), condition.clone(), &table, &[true]);

        observer(&meta_batch(0, 0, 2), &BooleanArray::from(vec![true, true]));

        assert_eq!(cache.stats().inserts, 0);
    }
}
