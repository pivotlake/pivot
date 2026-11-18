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
use crate::memory::{BUFFER_SIZE, SlabAllocator, memory_ctx};
use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::order_by::{ChunkedRowSorter, MergeRun};
use crate::operations::unary::pipeline_breaker::Consumer;
use bucket_runs::{BucketRuns, SortedRun};
use bucketed_rows::BucketedRows;
use row_limit::RunLimiter;

/// How many of its node's buffers a worker lends to the workers of other
/// nodes on top of what the rows it received call for. The bytes those
/// workers write are only about equal to the estimate, and a run does not
/// fill its last buffer.
const SPARE_LENT_BUFFERS: usize = 4;

/// Receives one worker's share of the input (see the [module docs](super)).
///
/// ```text
///                │  RecordBatch: an input batch
///                ▼
///    ┌───────────────────────┐   Hash the partition keys of every row to a
///    │ BucketedRows::push    │   bucket and copy the batch so that each
///    │                       │   bucket's rows are next to each other.
///    └───────────────────────┘
///                │  RecordBatch + BucketGrouping: the batches in bucket
///                │  order, once enough rows are pending
///                ▼
///    ┌───────────────────────┐   Sort each bucket's pending rows into one
///    │ sort_pending_into_    │   sorted run, written into the memory of
///    │   runs                │   the bucket's NUMA node.
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
    /// Ring memory of the worker's own node: its runs of that node's buckets
    /// land in it. Taken on first use, so a worker that receives no rows
    /// holds none.
    allocator: Option<SlabAllocator>,
    /// Per node, ring memory of that node, lent by its workers: this
    /// worker's runs of that node's buckets land in it. Taken on first use,
    /// and never for the worker's own node.
    allocators_of_other_nodes: Vec<Option<SlabAllocator>>,
    /// How many bytes of rows this worker has received, and how many of its
    /// node's buffers it has lent to the workers of other nodes.
    received_bytes: usize,
    lent_buffer_count: usize,
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
            allocators_of_other_nodes: (0..node_count).map(|_| None).collect(),
            received_bytes: 0,
            lent_buffer_count: 0,
            layout,
            shared,
            worker_id,
            node_id,
        }
    }

    /// Lends the workers of the other nodes enough of this node's buffers
    /// for the rows this worker has received so far.
    ///
    /// The runs of this node's buckets are written into this node's memory
    /// by every worker (see [`SharedBuckets::lend_buffer`]). Every node
    /// receives about the same number of rows and a node's buckets hold an
    /// equal share of them, so the workers of the other nodes together write
    /// about as many bytes into this node as this node's workers write into
    /// theirs. A worker can measure the second, so that is what it lends,
    /// with a few buffers to spare. A worker that still finds none lent
    /// writes to its own node instead.
    fn lend_buffers_for_received_rows(&mut self) {
        let node_count = self.allocators_of_other_nodes.len();
        let bytes_for_other_nodes = self.received_bytes / node_count * (node_count - 1);
        let buffers_due = SPARE_LENT_BUFFERS + bytes_for_other_nodes.div_ceil(BUFFER_SIZE);
        while self.lent_buffer_count < buffers_due {
            self.shared
                .lend_buffer(self.node_id, memory_ctx().get_write_buffer(false));
            self.lent_buffer_count += 1;
        }
    }

    /// The allocator a run of `bucket` is written with: one of the bucket's
    /// node. It takes the fields it works on instead of `&mut self` because
    /// its callers hold other fields of the consumer at the same time.
    fn select_run_allocator<'allocators>(
        allocator: &'allocators mut Option<SlabAllocator>,
        allocators_of_other_nodes: &'allocators mut [Option<SlabAllocator>],
        shared: &SharedBuckets,
        node_id: usize,
        bucket: usize,
    ) -> &'allocators mut SlabAllocator {
        let bucket_node = shared.find_node_of_bucket(bucket);
        if bucket_node == node_id {
            allocator.get_or_insert_with(|| SlabAllocator::new(false))
        } else {
            allocators_of_other_nodes[bucket_node].get_or_insert_with(|| {
                SlabAllocator::new_from_lent_buffers(shared.lent_buffers_of_node(bucket_node))
            })
        }
    }

    /// Sorts every bucket's pending rows into one run and adds it to the
    /// bucket's runs. Nothing is left pending.
    ///
    /// Only row positions are sorted. Each bucket's rows are then gathered
    /// once, in sorted order, into batch-sized chunks on slab memory of the
    /// bucket's node. The comparison is bound to the pending batches once
    /// and reused for every bucket.
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
            let allocator = Self::select_run_allocator(
                &mut self.allocator,
                &mut self.allocators_of_other_nodes,
                &self.shared,
                self.node_id,
                bucket,
            );
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
        let allocator = Self::select_run_allocator(
            &mut self.allocator,
            &mut self.allocators_of_other_nodes,
            &self.shared,
            self.node_id,
            bucket,
        );
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
        if self.allocators_of_other_nodes.len() > 1 {
            self.received_bytes += batch.get_array_memory_size();
            self.lend_buffers_for_received_rows();
        }
        self.pending.push(batch)?;
        if self.pending.row_count() >= self.layout.pending_row_capacity {
            self.sort_pending_into_runs()?;
        }
        Ok(())
    }

    fn into_outputter(mut self) -> unary::Result<Option<WindowOutputter>> {
        self.sort_pending_into_runs()?;
        // A sorted run was written into its bucket's node. Without keys
        // nothing is sorted and the run is the batches as they arrived, on
        // this worker's node.
        let runs_are_on_bucket_node = !self.layout.sort_key.is_empty();
        let runs_by_bucket = self
            .runs_by_bucket
            .iter_mut()
            .enumerate()
            .map(|(bucket, bucket_runs)| {
                let run_node = if runs_are_on_bucket_node {
                    self.shared.find_node_of_bucket(bucket)
                } else {
                    self.node_id
                };
                bucket_runs
                    .take_runs()
                    .into_iter()
                    .map(|run| MergeRun::new(run, run_node))
                    .collect()
            })
            .collect();
        self.shared.arrive(self.worker_id, runs_by_bucket);
        Ok(Some(WindowOutputter::new(
            self.layout,
            self.shared,
            self.node_id,
            self.allocator,
        )))
    }
}
