//! One worker's consume phase: turns input batches into sorted runs, one
//! list of runs per hash bucket.

pub mod bucket_runs;
mod bucketed_rows;
mod partition_cutoffs;
mod row_limit;

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::ArrowError;

use super::WindowLayout;
use super::output::{SharedBuckets, WindowOutputter};
use crate::RECORD_BATCH_SIZE;
use crate::arrays::take_chunked;
use crate::memory::SlabAllocator;
use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::order_by::ChunkedRowSorter;
use crate::operations::unary::pipeline_breaker::Consumer;
use bucket_runs::{BucketRuns, SortedRun};
use bucketed_rows::BucketedRows;
use row_limit::RunLimiter;

/// Receives one worker's share of the input (see the [module docs](super)).
///
/// ```text
///                │  RecordBatch: an input batch
///                ▼
///    ┌───────────────────────┐   Hash the partition keys of every row to a
///    │ BucketedRows::push    │   bucket and note which rows belong to each
///    │                       │   bucket. The batch is kept as it arrived.
///    └───────────────────────┘
///                │  RecordBatch + BucketGrouping: the batches as they
///                │  arrived, once enough rows are pending
///                ▼
///    ┌───────────────────────┐   Sort each bucket's pending rows into one
///    │ sort_pending_into_    │   sorted run. This is the first copy of the
///    │   runs                │   rows.
///    └───────────────────────┘
///                │  SortedRun: one run per bucket
///                ▼
///    ┌───────────────────────┐   Only with a row limit: drop the rows that
///    │ RunLimiter            │   can no longer pass it.
///    └───────────────────────┘
///                │  SortedRun: the run without the dropped rows
///                ▼
///    ┌───────────────────────┐   Per bucket, the sorted runs this worker has
///    │ BucketRuns            │   produced so far.
///    └───────────────────────┘
///                │  Vec<Vec<SortedRun>>: every bucket's runs, handed to
///                ▼  the output phase when the input ends
/// ```
///
/// These are steps inside this one operator, connected by method calls, not
/// operators connected by channels.
pub struct WindowConsumer {
    layout: Arc<WindowLayout>,
    shared: Arc<SharedBuckets>,
    worker_id: usize,
    node_id: usize,
    pending: BucketedRows,
    runs_by_bucket: Vec<BucketRuns>,
    /// `None` for a window without a row limit.
    run_limiter: Option<RunLimiter>,
    /// Ring memory the worker's sorted runs land in. Taken on first use, so
    /// a worker that receives no rows holds none.
    allocator: Option<SlabAllocator>,
}

impl WindowConsumer {
    pub fn new(
        layout: Arc<WindowLayout>,
        shared: Arc<SharedBuckets>,
        worker_id: usize,
        node_id: usize,
        node_count: usize,
    ) -> Self {
        Self {
            pending: BucketedRows::new(layout.spec.partition_by.clone(), layout.bucket_count),
            runs_by_bucket: (0..layout.bucket_count)
                .map(|_| BucketRuns::default())
                .collect(),
            run_limiter: RunLimiter::try_new(&layout, node_id, node_count),
            allocator: None,
            layout,
            shared,
            worker_id,
            node_id,
        }
    }

    /// Sorts every bucket's pending rows into one run and adds it to the
    /// bucket's runs. Nothing is left pending.
    ///
    /// Only row positions are sorted. Each bucket's rows are then gathered
    /// once, in sorted order, into batch-sized chunks on slab memory. The
    /// comparison is bound to the pending batches once and reused for every
    /// bucket.
    fn sort_pending_into_runs(&mut self) -> Result<(), ArrowError> {
        if self.pending.row_count() == 0 {
            return Ok(());
        }
        let (batches, groupings) = self.pending.take_batches_by_bucket();
        if self.layout.sort_key.is_empty() {
            // Without keys there is one bucket and nothing to sort: the
            // batches form a run in the order they arrived, uncopied.
            return self.add_run(0, batches);
        }

        let schema = batches[0].schema();
        // The batches' columns regrouped per column, which is how a gather
        // across batches reads them.
        let batches_of_column: Vec<Vec<ArrayRef>> = (0..schema.fields().len())
            .map(|column| {
                batches
                    .iter()
                    .map(|batch| batch.column(column).clone())
                    .collect()
            })
            .collect();
        let mut row_sorter = ChunkedRowSorter::try_new(&self.layout.sort_key, &batches)?;
        // One list of positions, refilled for each bucket.
        let mut positions = Vec::new();
        for bucket in 0..self.layout.bucket_count {
            positions.clear();
            for (batch_index, grouping) in groupings.iter().enumerate() {
                grouping.append_bucket_positions(bucket, batch_index as u32, &mut positions);
            }
            if positions.is_empty() {
                continue;
            }
            row_sorter.sort(&mut positions);
            let allocator = self
                .allocator
                .get_or_insert_with(|| SlabAllocator::new(false));
            let run = positions
                .chunks(RECORD_BATCH_SIZE)
                .map(|chunk_positions| {
                    let columns = batches_of_column
                        .iter()
                        .map(|column_batches| {
                            take_chunked(allocator, column_batches, chunk_positions)
                        })
                        .collect::<Result<Vec<ArrayRef>, _>>()?;
                    RecordBatch::try_new(schema.clone(), columns)
                })
                .collect::<Result<SortedRun, ArrowError>>()?;
            self.add_run(bucket, run)?;
        }
        Ok(())
    }

    /// Adds a freshly sorted run to `bucket`, applying the row limit to it
    /// when the window has one.
    fn add_run(&mut self, bucket: usize, run: SortedRun) -> Result<(), ArrowError> {
        let bucket_runs = &mut self.runs_by_bucket[bucket];
        let Some(run_limiter) = &mut self.run_limiter else {
            bucket_runs.push_run(run);
            return Ok(());
        };
        let allocator = self
            .allocator
            .get_or_insert_with(|| SlabAllocator::new(false));
        // The first run of a bucket is everything the worker holds for it.
        let whole_bucket = bucket_runs.is_empty().then_some(bucket);
        let limited_run = run_limiter.limit_run(run, allocator, whole_bucket)?;
        bucket_runs.push_run(limited_run);
        run_limiter.merge_bucket_if_due(bucket, bucket_runs, allocator)?;
        if !run_limiter.is_worth_keeping() {
            self.run_limiter = None;
        }
        Ok(())
    }
}

impl Consumer<RecordBatch, RecordBatch> for WindowConsumer {
    type Outputter = WindowOutputter;

    fn consume(
        &mut self,
        batch: RecordBatch,
        _sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let batch = match &mut self.run_limiter {
            None => batch,
            Some(run_limiter) => {
                let allocator = self
                    .allocator
                    .get_or_insert_with(|| SlabAllocator::new(false));
                run_limiter.drop_rows_past_cutoffs(batch, allocator)?
            }
        };
        if batch.num_rows() == 0 {
            return Ok(());
        }
        self.pending.push(batch)?;
        if self.pending.row_count() >= self.layout.pending_row_capacity {
            self.sort_pending_into_runs()?;
        }
        Ok(())
    }

    fn into_outputter(mut self) -> unary::Result<Option<WindowOutputter>> {
        self.sort_pending_into_runs()?;
        let runs_by_bucket = self
            .runs_by_bucket
            .iter_mut()
            .map(BucketRuns::take_runs)
            .collect();
        self.shared
            .arrive(self.worker_id, self.node_id, runs_by_bucket);
        Ok(Some(WindowOutputter::new(
            self.layout,
            self.shared,
            self.node_id,
            self.allocator,
        )))
    }
}
