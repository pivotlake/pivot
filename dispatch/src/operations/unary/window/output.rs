//! The output phase: merges each bucket's runs, evaluates the window
//! functions over the sorted bucket, and emits it.
//!
//! When the last worker finishes consuming, it groups every worker's runs by
//! bucket and queues one task per bucket. From then on every worker takes
//! tasks from its NUMA node's queue until no bucket is left:
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

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::SchemaRef;
use crossbeam_deque::{Injector, Steal};

use super::WindowLayout;
use super::consume::bucket_runs::SortedRun;
use super::evaluate::WindowEvaluator;
use crate::Topology;
use crate::arrays::accumulator::BatchAccumulator;
use crate::gather_barrier::GatherBarrier;
use crate::memory::SlabAllocator;
use crate::operations::channels::{NodeIdOutput, Sender};
use crate::operations::unary;
use crate::operations::unary::order_by::{KWayMergePlan, KWayMergeTask, MergeRun, MergedOutput};
use crate::operations::unary::pipeline_breaker::{OutputStatus, Outputter};
use crate::waker::{WakerSet, waker_set};

/// One unit of output-phase work.
enum BucketTask {
    /// Plan the merge of one bucket's runs.
    PlanMerge(Vec<MergeRun>),
    /// Produce one slice of a bucket's merged output.
    MergeSlice(KWayMergeTask),
}

/// Coordination shared by every worker of one window operator.
pub struct SharedBuckets {
    layout: Arc<WindowLayout>,
    topology: Topology,
    /// Collects every worker's NUMA node and its runs per bucket. The last
    /// worker to arrive queues the buckets' merges.
    worker_runs: GatherBarrier<(usize, Vec<Vec<SortedRun>>)>,
    tasks_by_node: Box<[Injector<BucketTask>]>,
    /// The wakers of the dataflow's workers, to wake one on the node a task
    /// was queued for.
    wakers: WakerSet,
    /// Buckets not yet emitted. Starts at the bucket count, so it only
    /// reaches zero after the last worker has arrived and every bucket has
    /// either been found empty or been emitted.
    buckets_remaining: AtomicUsize,
}

impl SharedBuckets {
    /// `topology` and `wakers` are those of the dataflow's workers.
    pub fn new(layout: Arc<WindowLayout>, topology: Topology, wakers: WakerSet) -> Self {
        Self {
            wakers,
            worker_runs: GatherBarrier::new(topology.total_workers()),
            tasks_by_node: (0..topology.node_count).map(|_| Injector::new()).collect(),
            buckets_remaining: AtomicUsize::new(layout.bucket_count),
            layout,
            topology,
        }
    }

    /// Hands over the runs per bucket of worker `worker_id`, which runs on
    /// NUMA node `node_id` and has finished consuming. The last worker to
    /// arrive starts the output phase.
    pub fn arrive(&self, worker_id: usize, node_id: usize, runs_by_bucket: Vec<Vec<SortedRun>>) {
        self.worker_runs
            .arrive_at(worker_id, (node_id, runs_by_bucket), |all_workers| {
                self.queue_bucket_merges(all_workers)
            });
    }

    /// Groups every worker's runs by bucket and queues one planning task per
    /// bucket that has rows.
    fn queue_bucket_merges(&self, mut all_workers: Vec<(usize, Vec<Vec<SortedRun>>)>) {
        for bucket in 0..self.layout.bucket_count {
            let runs: Vec<MergeRun> = all_workers
                .iter_mut()
                .flat_map(|(node_id, runs_by_bucket)| {
                    let node_id = *node_id;
                    std::mem::take(&mut runs_by_bucket[bucket])
                        .into_iter()
                        .map(move |run| MergeRun::new(run, node_id))
                })
                .collect();
            match runs.first() {
                None => self.finish_bucket(),
                Some(first_run) => {
                    let node_id = first_run.node_id();
                    self.queue_task(node_id, BucketTask::PlanMerge(runs));
                }
            }
        }
    }

    fn queue_task(&self, node_id: usize, task: BucketTask) {
        self.tasks_by_node[node_id].push(task);
        self.wakers.notify_one_near(node_id);
    }

    fn claim_task(&self, node_id: usize) -> Option<BucketTask> {
        loop {
            match self.tasks_by_node[node_id].steal() {
                Steal::Success(task) => return Some(task),
                Steal::Retry => continue,
                Steal::Empty => return None,
            }
        }
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
            BucketTask::PlanMerge(runs) => {
                let plan = KWayMergePlan::try_new(
                    self.layout.sort_key.clone(),
                    runs,
                    self.shared.topology.node_count,
                )?;
                match plan {
                    KWayMergePlan::Empty => self.shared.finish_bucket(),
                    KWayMergePlan::Identity(sorted_bucket) => {
                        self.emit_bucket(sorted_bucket, sender)?
                    }
                    KWayMergePlan::Parallel(slices) => {
                        for slice in slices {
                            self.shared
                                .queue_task(slice.node_id(), BucketTask::MergeSlice(slice));
                        }
                    }
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
