//! Drops rows that cannot pass the window's row limit while a worker is
//! still consuming.
//!
//! The limit keeps the rows whose function value is at most some number. The
//! final value of a row is only known once its whole partition is sorted, but
//! a value computed over a subset of the partition is never larger than the
//! final one: every supported function counts rows or tie groups that precede
//! the row, and a subset has no more of those. So a row that already exceeds
//! the limit within one worker's run of a bucket can be dropped for good.
//!
//! [`RunLimiter`] applies that in two places. Every new run is cut as soon as
//! it is sorted. That alone leaves up to `limit` rows per partition in every
//! run, so the limiter also merges a bucket's runs and cuts the result:
//!
//! - all of them, once the bucket's rows have doubled, which brings the
//!   bucket back to at most `limit` rows per partition;
//! - the runs that were never merged, once there are many of them, which
//!   keeps a bucket that receives a few surviving rows at a time from
//!   collecting one tiny run per sort. A row takes part in this kind of merge
//!   once, so its cost stays proportional to the rows received.
//!
//! # When limiting stops
//!
//! A worker can only drop rows of a partition it has received more than
//! `limit` rows of. When it receives few rows per partition, the cuts drop
//! almost nothing and the checks and merges are pure cost. So the limiter
//! compares the rows the worker has received with the partitions they belong
//! to, and when there are too few rows per partition the consumer drops its
//! limiter and leaves the limit to the output phase.
//!
//! The row count is exact. The partition count is exact for one bucket right
//! after all its rows were merged into one run, where each partition appears
//! once. Buckets are split by a hash of the partition key, so every bucket
//! holds about the same number of partitions, and the worker's count is that
//! bucket's count times the number of buckets.
//!
//! The comparison waits until the worker has received
//! `MIN_RECEIVED_ROWS_TO_JUDGE` rows. Before that, large partitions also
//! have few rows each. An input whose partitions pass the limit only later
//! than that loses its limiter too, and is then held in full until the
//! output phase.
//!
//! Separately, the table of cutoffs that rejects rows on arrival is given up
//! once it holds more than `MAX_CUTOFF_PARTITIONS` partitions, so its memory
//! stays small. The cuts of runs and merged buckets continue without it.
//!
//! # Where the surviving rows live
//!
//! A run is sorted into slabs sized for all its rows. When most of those rows
//! are dropped, keeping the rest in place would hold the whole slab for a few
//! rows. The survivors are instead copied into slabs sized for them, taken
//! from an allocator used for nothing else, so they sit densely together and
//! the buffers the full run was sorted into are free to be reused.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, RecordBatch, RecordBatchOptions};
use arrow_buffer::BooleanBuffer;
use arrow_schema::ArrowError;

use super::bucket_runs::{BucketRuns, SortedRun};
use super::partition_cutoffs::PartitionCutoffs;
use crate::RECORD_BATCH_SIZE;
use crate::arrays::{take, take_chunked};
use crate::memory::SlabAllocator;
use crate::operations::unary::order_by::{KWayMergePlan, MergeRun};
use crate::operations::unary::window::evaluate::WindowEvaluator;
use crate::operations::unary::window::{WindowFunction, WindowLayout};

/// The fewest rows a bucket holds before all its runs are merged to apply the
/// limit again. Below one batch the merge would free too little to matter.
const MIN_ROWS_FOR_FULL_MERGE: usize = RECORD_BATCH_SIZE;

/// The most never-merged runs a bucket holds before they are merged into
/// one. Every run costs a fixed amount of bookkeeping however few rows it
/// has, and a merge of this many runs is still cheap.
const MAX_UNMERGED_RUNS: usize = 16;

/// How many rows a worker receives before it judges whether limiting pays
/// (see [`RunLimiter::is_worth_keeping`]).
const MIN_RECEIVED_ROWS_TO_JUDGE: usize = 1 << 20;

/// Limiting pays when the worker has received at least this many times
/// `limit` rows per partition. At exactly `limit` rows per partition nothing
/// is past the limit, and at twice that about half the rows are.
const MIN_LIMITS_RECEIVED_PER_PARTITION: usize = 2;

/// The most partitions a worker keeps cutoffs for. Every cutoff holds a copy
/// of a row and a table entry, so this bounds the memory the cutoffs take.
/// Past this size the limiter gives the cutoffs up and keeps cutting runs
/// and merged buckets.
const MAX_CUTOFF_PARTITIONS: usize = 16_384;

/// Applies the window's row limit to one worker's runs (see the module docs).
pub struct RunLimiter {
    layout: Arc<WindowLayout>,
    node_id: usize,
    node_count: usize,
    /// Ring memory that holds only rows that survived a cut. Taken on first
    /// use.
    survivor_allocator: Option<SlabAllocator>,
    /// Rejects rows past their partition's cutoff as they arrive. Created
    /// from the first run, whose schema gives the key types, and only for a
    /// limit on `row_number()` over keys the cutoff table covers. Given up
    /// for good once it holds more than [`MAX_CUTOFF_PARTITIONS`].
    cutoffs: Option<PartitionCutoffs>,
    /// Whether creating `cutoffs` has been attempted.
    cutoffs_decided: bool,
    /// Rows the worker has received, before any were dropped.
    received_row_count: usize,
    /// Whether the worker has received enough rows per partition for the
    /// limit to drop a useful share of them, as of the last time that was
    /// judged.
    receives_enough_rows_per_partition: bool,
    /// Scratch space reused from batch to batch.
    surviving_positions: Vec<(u32, u32)>,
    cutoff_positions: Vec<(u32, u32)>,
    rows_before_cutoffs: Vec<u32>,
}

impl RunLimiter {
    /// `None` when the window has no row limit.
    pub fn try_new(layout: &Arc<WindowLayout>, node_id: usize, node_count: usize) -> Option<Self> {
        layout.spec.row_limit?;
        Some(Self {
            layout: layout.clone(),
            node_id,
            node_count,
            survivor_allocator: None,
            cutoffs: None,
            cutoffs_decided: false,
            received_row_count: 0,
            receives_enough_rows_per_partition: true,
            surviving_positions: Vec::new(),
            cutoff_positions: Vec::new(),
            rows_before_cutoffs: Vec::new(),
        })
    }

    /// Whether limiting while consuming still pays. `false` once the worker
    /// has received too few rows per partition (see the module docs). The
    /// output phase applies the limit either way, so a consumer that drops
    /// its limiter produces the same result.
    pub fn is_worth_keeping(&self) -> bool {
        self.receives_enough_rows_per_partition
    }

    /// Returns `batch`, which the worker has just received, without the rows
    /// that are past their partition's cutoff (see [`PartitionCutoffs`]).
    /// The kept rows are copied into `allocator`'s memory when any row is
    /// dropped.
    pub fn drop_rows_past_cutoffs(
        &mut self,
        batch: RecordBatch,
        allocator: &mut SlabAllocator,
    ) -> Result<RecordBatch, ArrowError> {
        self.received_row_count += batch.num_rows();
        let Some(cutoffs) = &mut self.cutoffs else {
            return Ok(batch);
        };
        cutoffs.collect_rows_before_cutoffs(&batch, &mut self.rows_before_cutoffs)?;
        if self.rows_before_cutoffs.len() == batch.num_rows() {
            return Ok(batch);
        }
        let columns = batch
            .columns()
            .iter()
            .map(|column| take(allocator, column, &self.rows_before_cutoffs))
            .collect::<Result<Vec<ArrayRef>, _>>()?;
        let options =
            RecordBatchOptions::new().with_row_count(Some(self.rows_before_cutoffs.len()));
        RecordBatch::try_new_with_options(batch.schema(), columns, &options)
    }

    /// Returns `run` without the rows that exceed the limit within it.
    /// `allocator` is the worker's own, used for what does not outlive the
    /// call.
    ///
    /// `whole_bucket` names the bucket when `run` is everything the worker
    /// holds for it. The run then also sets the bucket's cutoffs, and its
    /// partitions are counted to judge whether limiting still pays.
    pub fn limit_run(
        &mut self,
        run: SortedRun,
        allocator: &mut SlabAllocator,
        whole_bucket: Option<usize>,
    ) -> Result<SortedRun, ArrowError> {
        let row_limit = self
            .layout
            .spec
            .row_limit
            .expect("a run limiter exists only for a window with a row limit");
        let limit = i64::try_from(row_limit.limit).unwrap_or(i64::MAX);
        let limits_row_number =
            self.layout.spec.functions[row_limit.function_index] == WindowFunction::RowNumber;
        let cutoff_bucket = whole_bucket.filter(|_| limits_row_number);

        let mut limited_run = Vec::with_capacity(run.len());
        self.cutoff_positions.clear();
        let mut evaluator = WindowEvaluator::for_row_limit(&self.layout, row_limit, &run)?;
        if whole_bucket.is_some() && self.received_row_count >= MIN_RECEIVED_ROWS_TO_JUDGE {
            let worker_partition_count = evaluator.count_partitions() * self.layout.bucket_count;
            let rows_for_limiting_to_pay = worker_partition_count
                .saturating_mul(usize::try_from(row_limit.limit).unwrap_or(usize::MAX))
                .saturating_mul(MIN_LIMITS_RECEIVED_PER_PARTITION);
            self.receives_enough_rows_per_partition =
                self.received_row_count >= rows_for_limiting_to_pay;
        }
        for (batch_index, batch) in run.iter().enumerate() {
            let (function_values, rows_within_limit) = evaluator.evaluate_next_batch(allocator)?;
            let rows_within_limit =
                rows_within_limit.expect("a row limit evaluator marks the rows within the limit");
            if cutoff_bucket.is_some() {
                // The row numbered `limit` is its partition's cutoff.
                let row_numbers = function_values[0].as_primitive::<Int64Type>().values();
                for (row, &row_number) in row_numbers.iter().enumerate() {
                    if row_number == limit {
                        self.cutoff_positions.push((batch_index as u32, row as u32));
                    }
                }
            }
            let surviving_count = rows_within_limit.count_set_bits();
            if surviving_count == batch.num_rows() {
                limited_run.push(batch.clone());
            } else if surviving_count > 0 {
                limited_run.push(self.copy_surviving_rows(batch, &rows_within_limit)?);
            }
        }

        if let (Some(bucket), Some(first_batch)) = (cutoff_bucket, run.first()) {
            if !self.cutoffs_decided {
                self.cutoffs = PartitionCutoffs::try_new(&self.layout, &first_batch.schema());
                self.cutoffs_decided = true;
            }
            if let Some(cutoffs) = &mut self.cutoffs {
                cutoffs.replace_bucket_cutoffs(bucket, &run, &self.cutoff_positions)?;
                if cutoffs.partition_count() > MAX_CUTOFF_PARTITIONS {
                    self.cutoffs = None;
                }
            }
        }
        Ok(limited_run)
    }

    /// Copies the rows of `batch` whose bit of `survives` is set into slabs
    /// sized for them. The copy shares no memory with `batch`.
    fn copy_surviving_rows(
        &mut self,
        batch: &RecordBatch,
        survives: &BooleanBuffer,
    ) -> Result<RecordBatch, ArrowError> {
        self.surviving_positions.clear();
        self.surviving_positions
            .extend(survives.set_indices().map(|row| (0, row as u32)));
        let survivor_allocator = self
            .survivor_allocator
            .get_or_insert_with(|| SlabAllocator::new(false));
        let columns = batch
            .columns()
            .iter()
            .map(|column| {
                take_chunked(
                    survivor_allocator,
                    std::slice::from_ref(column),
                    &self.surviving_positions,
                )
            })
            .collect::<Result<Vec<ArrayRef>, _>>()?;
        RecordBatch::try_new(batch.schema(), columns)
    }

    /// Merges runs of the bucket and applies the limit to the result, when
    /// the bucket has grown enough to be worth it (see the module docs).
    pub fn merge_bucket_if_due(
        &mut self,
        bucket_index: usize,
        bucket: &mut BucketRuns,
        allocator: &mut SlabAllocator,
    ) -> Result<(), ArrowError> {
        if bucket.has_doubled_since_full_merge(MIN_ROWS_FOR_FULL_MERGE) {
            let runs = bucket.take_runs();
            let limited_run = self.merge_and_limit(runs, allocator, Some(bucket_index))?;
            bucket.replace_with_fully_merged(limited_run);
        } else if bucket.unmerged_run_count() > MAX_UNMERGED_RUNS {
            let unmerged_runs = bucket.take_unmerged_runs();
            let limited_run = self.merge_and_limit(unmerged_runs, allocator, None)?;
            bucket.push_merged_run(limited_run);
        }
        Ok(())
    }

    /// Merges `runs` into one run on the calling worker and applies the limit
    /// to it. `whole_bucket` is as for [`limit_run`](Self::limit_run).
    ///
    /// The output phase spreads a bucket's merge over the worker pool. This
    /// is the same merge, run start to finish by the one worker that owns the
    /// runs.
    fn merge_and_limit(
        &mut self,
        runs: Vec<SortedRun>,
        allocator: &mut SlabAllocator,
        whole_bucket: Option<usize>,
    ) -> Result<SortedRun, ArrowError> {
        let runs = runs
            .into_iter()
            .map(|run| MergeRun::new(run, self.node_id))
            .collect();
        let sort_key = self.layout.sort_key.clone();
        let merged = match KWayMergePlan::try_new(sort_key, runs, self.node_count)? {
            KWayMergePlan::Empty => return Ok(Vec::new()),
            KWayMergePlan::Identity(merged) => merged,
            KWayMergePlan::Parallel(slices) => {
                let mut merged = None;
                for slice in slices {
                    // Only the slice that finishes last returns the whole
                    // output.
                    merged = slice.execute(allocator)?.or(merged);
                }
                merged.expect("the last slice of a merge returns its output")
            }
        };
        let merged_run = merged
            .into_batches()
            .into_iter()
            .map(|batch| batch.into_batch())
            .collect();
        self.limit_run(merged_run, allocator, whole_bucket)
    }
}
