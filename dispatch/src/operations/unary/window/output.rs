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
//! Every worker buckets the rows it receives, so on a machine with several
//! NUMA nodes each bucket's runs are spread over all of them, and merging a
//! bucket has to read the other nodes' runs once wherever it runs. What can
//! stay local is everything after that read. So each bucket is given to one
//! node, the buckets in turn so the nodes share the rows evenly, and all of
//! its tasks are queued on that node: its merged rows are then written to
//! that node's memory, and evaluating the functions and sending the rows
//! downstream read only local memory.
//!
//! A worker takes tasks from its own node's queues first, and from the other
//! nodes' queues when its own are empty, so a node that finishes early helps
//! with what is left. With one node none of this changes anything.
//!
//! # Planning a few buckets at a time
//!
//! At most a quarter of the workers plan a merge at once. The others run
//! slices, or wait for the first slices to be queued.
//!
//! Without the limit every worker would start the output phase by planning
//! a bucket, and from then on all of them would go through the same steps
//! at the same moments: queue their slices, merge, evaluate. Each of these
//! steps is slower when every worker does it at once. The pushes to a
//! node's slice queue hold each other up. Merging and evaluating are bound
//! by memory, and a slice takes longer when every worker is merging than
//! when half of them are.
//!
//! With the limit, buckets are planned at a steady rate. Their slices, and
//! after the slices their evaluation, are then spread evenly over the
//! output phase, and at any moment the workers are divided among planning,
//! merging and evaluating.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::SchemaRef;
use crossbeam_deque::{Injector, Steal};
use crossbeam_queue::SegQueue;

use super::WindowLayout;
use super::consume::bucket_runs::SortedRun;
use super::evaluate::WindowEvaluator;
use crate::Topology;
use crate::arrays::accumulator::BatchAccumulator;
use crate::gather_barrier::GatherBarrier;
use crate::memory::SlabAllocator;
use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::order_by::{KWayMergePlan, KWayMergeTask, MergeRun, MergedOutput};
use crate::operations::unary::pipeline_breaker::{OutputStatus, Outputter};
use crate::waker::waker_set;

/// How many workers there are for each one that may plan a merge at the same
/// time (see the module docs).
const WORKERS_PER_PLANNING_WORKER: usize = 4;

/// One unit of output-phase work.
enum BucketTask {
    /// Plan the merge of the runs of the bucket with this index.
    PlanMerge(usize),
    /// Produce one slice of a bucket's merged output.
    MergeSlice(KWayMergeTask),
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
    /// Per node, the slices of planned merges that no worker has started.
    slices_by_node: Box<[Injector<KWayMergeTask>]>,
    /// How many workers are planning a merge right now.
    planning_workers: AtomicUsize,
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
            slices_by_node: (0..topology.node_count).map(|_| Injector::new()).collect(),
            planning_workers: AtomicUsize::new(0),
            buckets_remaining: AtomicUsize::new(layout.bucket_count),
            topology,
        }
    }

    /// Hands over the runs per bucket of worker `worker_id`, which runs on
    /// NUMA node `node_id` and has finished consuming. The last worker to
    /// arrive starts the output phase.
    pub fn arrive(&self, worker_id: usize, node_id: usize, runs_by_bucket: Vec<Vec<SortedRun>>) {
        for (bucket, runs) in runs_by_bucket.into_iter().enumerate() {
            for run in runs {
                self.runs_by_bucket[bucket].push(MergeRun::new(run, node_id));
            }
        }
        self.workers_arrived
            .arrive_at(worker_id, (), |_| self.queue_bucket_merges());
    }

    /// Queues every bucket that has rows for planning. Every worker has
    /// added its runs by now.
    fn queue_bucket_merges(&self) {
        for (bucket, runs) in self.runs_by_bucket.iter().enumerate() {
            if runs.is_empty() {
                self.finish_bucket();
            } else {
                let bucket_node = bucket % self.topology.node_count;
                self.unplanned_by_node[bucket_node].push(bucket);
                waker_set().notify_one_near(bucket_node);
            }
        }
    }

    fn queue_slice(&self, node_id: usize, slice: KWayMergeTask) {
        self.slices_by_node[node_id].push(slice);
        waker_set().notify_one_near(node_id);
    }

    /// Takes a task for a worker on node `node_id`: a bucket to plan while
    /// fewer workers are planning than may at once, and a slice otherwise.
    /// The worker that gets a bucket to plan calls
    /// [`finish_planning`](Self::finish_planning) once its slices are queued.
    fn claim_task(&self, node_id: usize) -> Option<BucketTask> {
        if self.try_start_planning() {
            if let Some(bucket) = Self::take_near(&self.unplanned_by_node, node_id) {
                return Some(BucketTask::PlanMerge(bucket));
            }
            self.finish_planning();
        }
        Self::take_near(&self.slices_by_node, node_id).map(BucketTask::MergeSlice)
    }

    /// Counts the calling worker among the planning ones when a bucket waits
    /// to be planned and fewer workers are planning than may at once.
    fn try_start_planning(&self) -> bool {
        // Idle workers ask until the last bucket is emitted. Checking with
        // plain loads first keeps them from writing to the shared count.
        if self.unplanned_by_node.iter().all(Injector::is_empty) {
            return false;
        }
        let max_planning_workers =
            (self.topology.total_workers() / WORKERS_PER_PLANNING_WORKER).max(1);
        if self.planning_workers.load(Ordering::Acquire) >= max_planning_workers {
            return false;
        }
        if self.planning_workers.fetch_add(1, Ordering::AcqRel) >= max_planning_workers {
            self.planning_workers.fetch_sub(1, Ordering::AcqRel);
            return false;
        }
        true
    }

    fn finish_planning(&self) {
        self.planning_workers.fetch_sub(1, Ordering::AcqRel);
    }

    /// Takes the oldest entry of the queue of node `node_id` when it has
    /// one, otherwise of the first other node's queue that does.
    fn take_near<T>(queues_by_node: &[Injector<T>], node_id: usize) -> Option<T> {
        let node_count = queues_by_node.len();
        for offset in 0..node_count {
            let queue = &queues_by_node[(node_id + offset) % node_count];
            // Checking for emptiness first costs a plain load, where a steal
            // from an empty queue costs more and is repeated by every idle
            // worker until the last bucket is emitted.
            if queue.is_empty() {
                continue;
            }
            loop {
                match queue.steal() {
                    Steal::Success(entry) => return Some(entry),
                    Steal::Retry => continue,
                    Steal::Empty => break,
                }
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
                        // worker of another node planned it, so every node
                        // gets the same share of the slices.
                        let bucket_node = bucket % self.shared.topology.node_count;
                        for slice in slices {
                            self.shared.queue_slice(bucket_node, slice);
                        }
                        None
                    }
                };
                self.shared.finish_planning();
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
