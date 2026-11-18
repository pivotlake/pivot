//! The output phase: merges each bucket's runs, evaluates the window
//! functions over the sorted bucket, and emits it.
//!
//! A worker that finishes consuming adds its runs to each bucket's shared
//! list. When the last worker has done so, it queues every bucket for
//! planning, and from then on every worker takes tasks until no bucket is
//! left:
//!
//! ```text
//!  PlanMerge(bucket's runs)
//!      │  split the merge into independent slices (KWayMergePlan)
//!      ▼
//!  MergeSlice ... MergeSlice      one task each, run by any worker
//!      │  the worker that finishes the bucket's last slice continues:
//!      ▼
//!  evaluate the sorted bucket batch by batch and send it downstream
//! ```
//!
//! A bucket whose runs need no merging (a single run, or runs that already
//! line up end to end) skips the slices and is evaluated by the worker that
//! planned it.
//!
//! # NUMA nodes
//!
//! Reading another node's memory is several times slower than reading the
//! worker's own, and the output phase reads every row more than once. So
//! each bucket is given to one node, the buckets in turn so the nodes share
//! the rows evenly, and everything about the bucket is kept on that node:
//!
//! - The consume phase writes every worker's run of the bucket into that
//!   node's memory, also the workers of other nodes (see
//!   [`SharedBuckets::lend_buffer`]). Each row crosses to its bucket's node
//!   once, as part of a copy that is made anyway.
//! - All of the bucket's tasks wait on that node. Its merged rows are then
//!   written to that node's memory, and merging, evaluating the functions
//!   and sending the rows downstream read only local memory.
//!
//! A worker takes tasks of its own node first, and of the other nodes when
//! its own has none, so a node that finishes early helps with what is left.
//! With one node none of this changes anything.
//!
//! # The order slices are merged in
//!
//! A planned bucket's slices wait in [`PlannedBuckets`], which hands out the
//! slices of the bucket planned first until it has none left, then those of
//! the next. Two things follow from that, and both matter on a machine with
//! many workers:
//!
//! - A planner adds its bucket in one step, however many slices it has.
//!   When every worker finishes planning at about the same moment, as they
//!   do at the start of the output phase, adding slices one at a time to a
//!   queue shared by a node's workers makes them wait for each other for
//!   longer than the planning itself takes.
//! - A node's workers merge one bucket together and finish it before they
//!   start the next. A slice is merged faster that way than when the node's
//!   workers are spread over many buckets at once, because the workers on a
//!   bucket read the same runs.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::SchemaRef;
use crossbeam_deque::{Injector, Steal};
use crossbeam_queue::SegQueue;

use super::WindowLayout;
use super::evaluate::WindowEvaluator;
use crate::Topology;
use crate::arrays::accumulator::BatchAccumulator;
use crate::gather_barrier::GatherBarrier;
use crate::memory::{SlabAllocator, WriteBuffer};
use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::order_by::{KWayMergePlan, KWayMergeTask, MergeRun, MergedOutput};
use crate::operations::unary::pipeline_breaker::{OutputStatus, Outputter};
use crate::waker::waker_set;

/// One unit of output-phase work.
enum BucketTask {
    /// Plan the merge of the runs of the bucket with this index.
    PlanMerge(usize),
    /// Produce one slice of a bucket's merged output.
    MergeSlice(KWayMergeTask),
}

/// The planned buckets of one node whose slices wait to be merged, in the
/// order their planning finished (see the module docs).
struct PlannedBuckets {
    /// One place per bucket of the operator, filled in the order the buckets
    /// are added. Each holds a bucket's slices that no worker has started.
    slices_by_arrival: Box<[OnceLock<SegQueue<KWayMergeTask>>]>,
    /// How many places have been given to planners.
    arrived: AtomicUsize,
    /// The first place that may still have slices.
    oldest: AtomicUsize,
}

impl PlannedBuckets {
    fn new(bucket_count: usize) -> Self {
        Self {
            slices_by_arrival: (0..bucket_count).map(|_| OnceLock::new()).collect(),
            arrived: AtomicUsize::new(0),
            oldest: AtomicUsize::new(0),
        }
    }

    /// Adds a planned bucket's slices behind the buckets already added.
    fn add_bucket(&self, slices: Vec<KWayMergeTask>) {
        let bucket_slices = SegQueue::new();
        for slice in slices {
            bucket_slices.push(slice);
        }
        let place = self.arrived.fetch_add(1, Ordering::AcqRel);
        if self.slices_by_arrival[place].set(bucket_slices).is_err() {
            unreachable!("a place is given to one planner");
        }
    }

    /// Takes a slice of the oldest bucket that has one. Returns `None` when
    /// no slice waits.
    fn take_slice(&self) -> Option<KWayMergeTask> {
        loop {
            let oldest = self.oldest.load(Ordering::Acquire);
            // A place that was given out and is not filled yet is filled
            // within moments. Until then nothing is handed out, also not from
            // the places behind it, and the caller asks again.
            let bucket_slices = self.slices_by_arrival.get(oldest)?.get()?;
            if let Some(slice) = bucket_slices.pop() {
                return Some(slice);
            }
            // The bucket has no slices left. Whichever worker notices first
            // moves on to the next; the others find it already moved.
            let _ = self.oldest.compare_exchange(
                oldest,
                oldest + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
    }
}

/// Coordination shared by every worker of one window operator.
pub struct SharedBuckets {
    topology: Topology,
    /// Per bucket, the runs of the workers that have finished consuming.
    /// Each worker adds its own runs, so the workers share that work, which
    /// grows with the number of workers times the number of buckets, and
    /// each one reads only runs it wrote itself. Workers add to a bucket at
    /// once without waiting for each other.
    runs_by_bucket: Box<[SegQueue<MergeRun>]>,
    /// Counts the workers that have added their runs. The last one queues
    /// the buckets' merges.
    workers_arrived: GatherBarrier<()>,
    /// Per node, the buckets whose merge no worker has started to plan.
    unplanned_by_node: Box<[Injector<usize>]>,
    /// Per node, the planned buckets whose slices wait to be merged.
    planned_by_node: Box<[PlannedBuckets]>,
    /// Per node, ring buffers of that node which its workers lent to the
    /// workers of the other nodes (see [`lend_buffer`](Self::lend_buffer)).
    lent_buffers_by_node: Box<[Arc<SegQueue<WriteBuffer>>]>,
    /// Buckets not yet emitted. Starts at the bucket count, so it only
    /// reaches zero after the last worker has arrived and every bucket has
    /// either been found empty or been emitted.
    buckets_remaining: AtomicUsize,
}

impl SharedBuckets {
    pub fn new(layout: &WindowLayout, topology: Topology) -> Self {
        Self {
            runs_by_bucket: (0..layout.bucket_count).map(|_| SegQueue::new()).collect(),
            workers_arrived: GatherBarrier::new(topology.total_workers()),
            unplanned_by_node: (0..topology.node_count).map(|_| Injector::new()).collect(),
            planned_by_node: (0..topology.node_count)
                .map(|_| PlannedBuckets::new(layout.bucket_count))
                .collect(),
            lent_buffers_by_node: (0..topology.node_count)
                .map(|_| Arc::new(SegQueue::new()))
                .collect(),
            buckets_remaining: AtomicUsize::new(layout.bucket_count),
            topology,
        }
    }

    /// The node that `bucket` is given to (see the module docs).
    pub fn find_node_of_bucket(&self, bucket: usize) -> usize {
        bucket % self.topology.node_count
    }

    /// Lends `buffer`, a ring buffer of node `node_id` taken by one of that
    /// node's workers, to the workers of the other nodes.
    ///
    /// A worker can only take buffers of its own node, and it has to write
    /// its runs of the other nodes' buckets into those nodes' memory. So
    /// while rows arrive, every worker lends buffers of its node here, and
    /// the workers of the other nodes write those runs with an allocator
    /// that draws on them (see
    /// [`lent_buffers_of_node`](Self::lent_buffers_of_node)).
    pub fn lend_buffer(&self, node_id: usize, buffer: WriteBuffer) {
        self.lent_buffers_by_node[node_id].push(buffer);
    }

    /// The buffers of node `node_id` lent to the workers of other nodes and
    /// not yet taken by one.
    pub fn lent_buffers_of_node(&self, node_id: usize) -> Arc<SegQueue<WriteBuffer>> {
        self.lent_buffers_by_node[node_id].clone()
    }

    /// Hands over the runs per bucket of worker `worker_id`, which has
    /// finished consuming. The last worker to arrive starts the output
    /// phase.
    pub fn arrive(&self, worker_id: usize, runs_by_bucket: Vec<Vec<MergeRun>>) {
        for (bucket, runs) in runs_by_bucket.into_iter().enumerate() {
            for run in runs {
                self.runs_by_bucket[bucket].push(run);
            }
        }
        self.workers_arrived
            .arrive_at(worker_id, (), |_| self.queue_bucket_merges());
    }

    /// Queues every bucket that has rows for planning. Every worker has
    /// added its runs by now.
    fn queue_bucket_merges(&self) {
        // No worker writes runs any more, so the lent buffers that were not
        // needed go back to their node's pool.
        for lent_buffers in self.lent_buffers_by_node.iter() {
            while lent_buffers.pop().is_some() {}
        }
        for (bucket, runs) in self.runs_by_bucket.iter().enumerate() {
            if runs.is_empty() {
                self.finish_bucket();
            } else {
                let bucket_node = self.find_node_of_bucket(bucket);
                self.unplanned_by_node[bucket_node].push(bucket);
                waker_set().notify_one_near(bucket_node);
            }
        }
    }

    /// Makes the slices of a planned bucket of node `node_id` available to
    /// the workers.
    fn queue_slices(&self, node_id: usize, slices: Vec<KWayMergeTask>) {
        let slice_count = slices.len();
        self.planned_by_node[node_id].add_bucket(slices);
        for _ in 0..slice_count {
            waker_set().notify_one_near(node_id);
        }
    }

    /// Takes a task for a worker on node `node_id`: a bucket to plan while
    /// one waits, and a slice otherwise. Both are taken from the worker's
    /// own node when it has one, otherwise from the first other node that
    /// does.
    fn claim_task(&self, node_id: usize) -> Option<BucketTask> {
        let node_count = self.topology.node_count;
        let nodes_from_own = (0..node_count).map(|offset| (node_id + offset) % node_count);
        for node in nodes_from_own.clone() {
            let unplanned = &self.unplanned_by_node[node];
            // Checking for emptiness first costs a plain load, where a steal
            // from an empty queue costs more and is repeated by every idle
            // worker until the last bucket is emitted.
            while !unplanned.is_empty() {
                match unplanned.steal() {
                    Steal::Success(bucket) => return Some(BucketTask::PlanMerge(bucket)),
                    Steal::Retry => continue,
                    Steal::Empty => break,
                }
            }
        }
        for node in nodes_from_own {
            if let Some(slice) = self.planned_by_node[node].take_slice() {
                return Some(BucketTask::MergeSlice(slice));
            }
        }
        None
    }

    /// Records that one more bucket needs no further work.
    fn finish_bucket(&self) {
        if self.buckets_remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            // Workers waiting for the last bucket can now finish.
            waker_set().notify_all();
        }
    }

    fn every_bucket_finished(&self) -> bool {
        self.buckets_remaining.load(Ordering::Acquire) == 0
    }
}

/// One worker's output phase (see the [module docs](super)).
///
/// ```text
///                │  Vec<MergeRun>: one bucket's runs from every worker
///                ▼
///    ┌───────────────────────┐   Per bucket, merge every worker's runs into
///    │ KWayMergePlan         │   one sorted sequence. The merge is split
///    │                       │   into slices that workers run in parallel.
///    └───────────────────────┘
///                │  MergedOutput: one bucket in sorted order
///                ▼
///    ┌───────────────────────┐   Find where partitions and tie groups start,
///    │ WindowEvaluator       │   compute each function, apply the row limit.
///    └───────────────────────┘
///                │  RecordBatch: input columns + one column per function
///                ▼
/// ```
///
/// These are steps inside this one operator. The file's module docs describe
/// how the steps are divided into tasks among the workers.
pub struct WindowOutputter {
    layout: Arc<WindowLayout>,
    shared: Arc<SharedBuckets>,
    node_id: usize,
    allocator: Option<SlabAllocator>,
    /// Gathers the rows that pass the row limit into full batches, which is
    /// what the operator downstream is best served by. Created from the first
    /// batch that has rows to drop, so a window without a limit has none.
    rows_within_limit: Option<BatchAccumulator>,
    /// The input schema followed by one field per function, built from the
    /// first batch this worker emits.
    output_schema: Option<SchemaRef>,
}

impl WindowOutputter {
    pub fn new(
        layout: Arc<WindowLayout>,
        shared: Arc<SharedBuckets>,
        node_id: usize,
        allocator: Option<SlabAllocator>,
    ) -> Self {
        Self {
            rows_within_limit: None,
            output_schema: None,
            layout,
            shared,
            node_id,
            allocator,
        }
    }

    fn run_task(
        &mut self,
        task: BucketTask,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        match task {
            BucketTask::PlanMerge(bucket) => {
                // Only this task takes the bucket's runs, and every worker
                // added its runs before the bucket was queued.
                let bucket_runs = &self.shared.runs_by_bucket[bucket];
                let runs: Vec<MergeRun> = std::iter::from_fn(|| bucket_runs.pop()).collect();
                let plan = KWayMergePlan::try_new(
                    self.layout.sort_key.clone(),
                    runs,
                    self.shared.topology.node_count,
                )?;
                let already_sorted = match plan {
                    KWayMergePlan::Empty => {
                        self.shared.finish_bucket();
                        None
                    }
                    KWayMergePlan::Identity(sorted_bucket) => Some(sorted_bucket),
                    KWayMergePlan::Parallel(slices) => {
                        // The slices go to the bucket's own node even when a
                        // worker of another node planned it, because that is
                        // where the bucket's runs are.
                        let bucket_node = self.shared.find_node_of_bucket(bucket);
                        self.shared.queue_slices(bucket_node, slices);
                        None
                    }
                };
                if let Some(sorted_bucket) = already_sorted {
                    self.emit_bucket(sorted_bucket, sender)?;
                }
            }
            BucketTask::MergeSlice(slice) => {
                let allocator = self
                    .allocator
                    .get_or_insert_with(|| SlabAllocator::new(false));
                // Only the slice that finishes last returns the sorted bucket.
                if let Some(sorted_bucket) = slice.execute(allocator)? {
                    self.emit_bucket(sorted_bucket, sender)?;
                }
            }
        }
        Ok(())
    }

    /// Evaluates the window functions over one sorted bucket and sends its
    /// rows downstream.
    fn emit_bucket(
        &mut self,
        sorted_bucket: MergedOutput,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        let allocator = self
            .allocator
            .get_or_insert_with(|| SlabAllocator::new(false));
        let batches: Vec<RecordBatch> = sorted_bucket
            .into_batches()
            .into_iter()
            .map(|batch| batch.into_batch())
            .collect();
        let mut evaluator = WindowEvaluator::for_output(&self.layout, &batches)?;
        for batch in batches {
            let (function_columns, within_limit) = evaluator.evaluate_next_batch(allocator)?;
            let output_schema = match &self.output_schema {
                Some(output_schema) => output_schema,
                None => self
                    .output_schema
                    .insert(self.layout.build_output_schema(&batch.schema())?),
            };
            let output_columns: Vec<ArrayRef> = batch
                .columns()
                .iter()
                .cloned()
                .chain(function_columns)
                .collect();
            let output_batch = RecordBatch::try_new(output_schema.clone(), output_columns)?;
            let Some(within_limit) = within_limit else {
                sender.send(output_batch)?;
                continue;
            };
            let rows_are_waiting = self
                .rows_within_limit
                .as_ref()
                .is_some_and(|waiting| !waiting.is_empty());
            let kept_count = within_limit.count_set_bits();
            if kept_count == 0 {
                continue;
            }
            if kept_count == output_batch.num_rows() && !rows_are_waiting {
                // Nothing to drop and nothing ahead of it: send it as it is.
                sender.send(output_batch)?;
                continue;
            }
            self.rows_within_limit
                .get_or_insert_with(|| BatchAccumulator::new(output_batch.schema(), allocator))
                .append_batch_by_mask_flushing(
                    &output_batch,
                    &within_limit,
                    allocator,
                    &mut |full_batch| sender.send(full_batch).map_err(unary::Error::from),
                )?;
        }
        self.shared.finish_bucket();
        Ok(())
    }
}

impl Outputter<RecordBatch> for WindowOutputter {
    fn output(
        &mut self,
        sender: &mut dyn Sender<RecordBatch>,
        _io: &mut crate::io::OperatorIO,
    ) -> unary::Result<OutputStatus> {
        if let Some(task) = self.shared.claim_task(self.node_id) {
            self.run_task(task, sender)?;
            // One task per call keeps the worker responsive to the rest of
            // its dataflow.
            return Ok(OutputStatus::Working);
        }
        if !self.shared.every_bucket_finished() {
            // Other workers are still consuming or merging. Their progress
            // feeds the queues.
            return Ok(OutputStatus::Working);
        }
        // Send the last rows that passed the row limit, which did not fill a
        // batch. Rows only wait here after an allocator was taken to copy
        // them.
        if let (Some(waiting), Some(allocator)) = (&mut self.rows_within_limit, &mut self.allocator)
            && !waiting.is_empty()
        {
            sender.send(waiting.take_batch(allocator)?)?;
        }
        Ok(OutputStatus::Done)
    }
}
