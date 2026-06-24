//! Records which rows survive the filter and publishes them to the
//! [`QueryConditionCache`], so a repeat of the same query can replay the
//! surviving rows instead of decoding and re-filtering the data.
//!
//! ```text
//!   scan(metadata) -> filter -> [ConditionPopulator] -> rest of the query
//!                                       |
//!                                       +-- on finish: survivors -> cache
//! ```
//!
//! It sits directly above the filter on a metadata-carrying scan, so the rows it
//! sees are exactly the ones that passed the filter. It is the terminal consumer
//! of the `(global_row_group, row_idx)` metadata columns: it strips them before
//! forwarding, so downstream operators see the query's original schema.
//!
//! # Completeness across workers
//!
//! The filter runs as one instance per worker (siblings), and work-stealing means
//! a single row group's surviving rows can be split across several of them. A
//! cache entry must list *every* survivor of a group (a partial set would later
//! drop matching rows), so each worker accumulates locally and merges into a
//! shared map at `finish`; the last sibling to finish publishes the now-complete
//! map in one shot. Until then nothing is exposed.

use crate::QueryConditionCache;
use crate::parquet::reading::record_batch_metadata::{METADATA_COLUMN_COUNT, accumulate_row_indices};
use arrow_array::RecordBatch;
use dispatch::{Sender, Unary, UnaryFactory, UnaryResult};
use std::collections::HashMap;
use std::mem;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Upper bound on the number of surviving rows worth caching for one filter.
///
/// Replay only pays off when survivors are sparse enough that whole data pages
/// have none and can be skipped; past this many rows the filter is effectively
/// non-selective — replay would re-decode nearly every page (no faster than a
/// plain scan) and the in-flight row groups can exhaust the page-cache ring and
/// stall. Above the bound we record the filter as "skipped" and take the plain
/// path. Roughly the order of a large table's data-page count.
const MAX_CACHED_SURVIVORS: usize = 131_072;

/// State shared by every sibling [`ConditionPopulator`]: where survivors are
/// merged, and the countdown that elects the last finisher to publish.
struct Shared {
    cache: Arc<QueryConditionCache>,
    /// Stable hash of the *full* filter these survivors belong to (the cache key).
    filter: u64,
    /// Survivors merged across siblings, keyed by global row group. Built up at
    /// each sibling's `finish`, drained and published by the last one.
    merged: Mutex<HashMap<u32, Vec<u32>>>,
    /// Siblings that have not yet finished. Starts at the worker count; the
    /// sibling that decrements it to zero owns publishing.
    siblings_left: AtomicUsize,
    /// Whether to drop the metadata columns before forwarding. `true` when *we*
    /// turned metadata on for a plain filter (downstream wants the original
    /// schema); `false` when metadata was already on for late materialization (a
    /// downstream `Materialize` still needs those columns).
    strip_metadata: bool,
}

/// Builds one [`ConditionPopulator`] per worker, all sharing one [`Shared`].
///
/// `Clone` is how the per-worker siblings come to share state: build it once with
/// the worker count, then hand each worker a clone (the `Arc` is shared, the
/// countdown and merge map are the same).
#[derive(Clone)]
pub struct ConditionPopulatorFactory {
    shared: Arc<Shared>,
}

impl ConditionPopulatorFactory {
    /// One factory whose clones share a single accumulator. `worker_count` must be
    /// the number of sibling workers (and hence clones), so the countdown elects
    /// exactly one last finisher. `strip_metadata` is `true` for a plain filter
    /// (we added the metadata columns and remove them again) and `false` for late
    /// materialization (they were already there and are consumed downstream).
    pub fn new(
        cache: Arc<QueryConditionCache>,
        filter: u64,
        worker_count: usize,
        strip_metadata: bool,
    ) -> Self {
        Self {
            shared: Arc::new(Shared {
                cache,
                filter,
                merged: Mutex::new(HashMap::new()),
                siblings_left: AtomicUsize::new(worker_count),
                strip_metadata,
            }),
        }
    }
}

impl UnaryFactory<RecordBatch, RecordBatch> for ConditionPopulatorFactory {
    type Unary = ConditionPopulator;

    fn build_unary(self) -> ConditionPopulator {
        ConditionPopulator {
            shared: self.shared,
            local: HashMap::new(),
        }
    }
}

/// One worker's populator: accumulates its own surviving rows by global row
/// group, then contributes them to the shared map at `finish`.
pub struct ConditionPopulator {
    shared: Arc<Shared>,
    /// This worker's survivors so far (no lock on the hot path); merged into
    /// `shared.merged` at `finish`.
    local: HashMap<u32, Vec<u32>>,
}

impl Unary<RecordBatch, RecordBatch> for ConditionPopulator {
    /// Record this batch's (already filter-surviving) rows, then forward it. When
    /// we added the metadata columns ourselves, strip them first so downstream
    /// sees the query's original schema; otherwise forward unchanged.
    fn consume<S: Sender<RecordBatch>>(
        &mut self,
        batch: RecordBatch,
        sender: &mut S,
    ) -> UnaryResult<()> {
        accumulate_row_indices(&batch, &mut self.local);

        let forwarded = if self.shared.strip_metadata {
            let data_columns = batch.num_columns() - METADATA_COLUMN_COUNT;
            batch.project(&(0..data_columns).collect::<Vec<_>>())?
        } else {
            batch
        };
        sender.send(forwarded)?;
        Ok(())
    }

    /// Merge this worker's survivors into the shared map; the last sibling to
    /// finish publishes the complete per-row-group sets to the cache.
    fn finish<S: Sender<RecordBatch>>(&mut self, _sender: &mut S) -> UnaryResult<bool> {
        {
            let mut merged = self.shared.merged.lock().unwrap();
            for (group, indices) in self.local.drain() {
                merged.entry(group).or_default().extend(indices);
            }
        }

        // `fetch_sub` returns the previous value, so `== 1` means we just took the
        // countdown to zero: every sibling has merged, the map is complete, and we
        // are the one to publish it. `AcqRel` pairs with the other siblings'
        // decrements so their merges are visible to this publish.
        if self.shared.siblings_left.fetch_sub(1, Ordering::AcqRel) == 1 {
            let complete = mem::take(&mut *self.shared.merged.lock().unwrap());
            let survivors: usize = complete.values().map(|rows| rows.len()).sum();
            if survivors > MAX_CACHED_SURVIVORS {
                // Non-selective: not worth replaying, and large enough to risk
                // stalling the ring. Record the decision so we don't re-populate.
                self.shared.cache.mark_skipped(self.shared.filter);
            } else {
                self.shared.cache.publish(self.shared.filter, complete);
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::reading::record_batch_metadata::with_row_group_metadata;
    use arrow_array::{ArrayRef, Int32Array};
    use arrow_schema::{DataType, Field, Schema};
    use dispatch::test_utils::CollectSender;

    /// A one-data-column batch of `rows` values, tagged with global row group
    /// `group` and row indices `offset..offset+rows` (the shape a metadata scan
    /// emits after a filter has kept these rows).
    fn meta_batch(group: usize, offset: usize, rows: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let col: ArrayRef = Arc::new(Int32Array::from_iter_values(0..rows as i32));
        let batch = RecordBatch::try_new(schema, vec![col]).unwrap();
        with_row_group_metadata(batch, group, offset)
    }

    fn stripping_populator(cache: &Arc<QueryConditionCache>, workers: usize) -> ConditionPopulator {
        ConditionPopulatorFactory::new(cache.clone(), 42, workers, true).build_unary()
    }

    #[test]
    fn strips_metadata_columns_for_a_plain_filter() {
        let cache = Arc::new(QueryConditionCache::default());
        let mut populator = stripping_populator(&cache, 1);
        let mut out = CollectSender::<RecordBatch>::default();

        populator.consume(meta_batch(5, 0, 3), &mut out).unwrap();

        assert_eq!(out.items[0].num_columns(), 1);
        assert_eq!(out.items[0].num_rows(), 3);
    }

    #[test]
    fn forwards_metadata_untouched_when_not_stripping() {
        let cache = Arc::new(QueryConditionCache::default());
        let mut populator = ConditionPopulatorFactory::new(cache.clone(), 42, 1, false).build_unary();
        let mut out = CollectSender::<RecordBatch>::default();

        populator.consume(meta_batch(5, 0, 3), &mut out).unwrap();

        assert_eq!(out.items[0].num_columns(), 3); // 1 data + 2 metadata columns kept
    }

    #[test]
    fn publishes_survivors_when_the_only_worker_finishes() {
        let cache = Arc::new(QueryConditionCache::default());
        let mut populator = stripping_populator(&cache, 1);
        let mut out = CollectSender::<RecordBatch>::default();

        populator.consume(meta_batch(5, 0, 3), &mut out).unwrap();
        populator.consume(meta_batch(8, 0, 2), &mut out).unwrap();
        populator.finish(&mut out).unwrap();

        assert_eq!(cache.lookup(42, 5).unwrap().as_ref(), &[0, 1, 2]);
        assert_eq!(cache.lookup(42, 8).unwrap().as_ref(), &[0, 1]);
    }

    #[test]
    fn one_row_groups_survivors_split_across_workers_publish_as_one_complete_set() {
        let cache = Arc::new(QueryConditionCache::default());
        let factory = ConditionPopulatorFactory::new(cache.clone(), 42, 2, true);
        let mut worker_a = factory.clone().build_unary();
        let mut worker_b = factory.build_unary();
        let mut out = CollectSender::<RecordBatch>::default();

        worker_a.consume(meta_batch(5, 0, 2), &mut out).unwrap(); // rows 0,1
        worker_b.consume(meta_batch(5, 2, 2), &mut out).unwrap(); // rows 2,3
        worker_a.finish(&mut out).unwrap();
        assert!(cache.lookup(42, 5).is_none()); // nothing until the last sibling finishes
        worker_b.finish(&mut out).unwrap();

        assert_eq!(cache.lookup(42, 5).unwrap().as_ref(), &[0, 1, 2, 3]);
    }
}
