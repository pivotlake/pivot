//! Parallel ORDER BY over a complete input stream.
//!
//! This operator is a pipeline breaker. Each worker sorts incoming record
//! batches and joins adjacent batches that already form one ordered run. Once
//! input ends, workers on each NUMA node merge their runs locally. A second
//! k-way merge combines the node results. On a single-node machine that second
//! level is an identity operation and copies no rows.
//!
//! Merge tasks compare only key rows to construct a gather order, then apply
//! that order to every column through [`crate::arrays::take_chunked`].
//! Tasks run only on their selected NUMA node.

mod batch_sort;
mod k_way_merge;
mod keys;

pub use k_way_merge::{
    KWayMergePlan, KWayMergeTask, LocatedBatch, MergeRun, MergedMapping, MergedOutput, key_order_by,
};

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use arrow_array::{ArrayRef, RecordBatch};
use crossbeam_deque::{Injector, Steal};

use crate::Topology;
use crate::arrays::take;
use crate::gather_barrier::GatherBarrier;
use crate::memory::SlabAllocator;
use crate::operations::channels::{NodeIdOutput, Sender};
use crate::operations::unary;
use crate::operations::unary::factory::UnaryFactory;
use crate::operations::unary::order_by_limit::OrderBy;
use crate::operations::unary::pipeline_breaker::{Consumer, Outputter, PipelineBreaker};
use crate::operations::unary::{
    BatchesOutputter, CollectorFactory, InitializableOutputter, NormalizationBatches, Normalizer,
};
use crate::waker::waker_set;

use batch_sort::{batch_arrives_sorted, sorted_row_indices};
use keys::{KeyOrdering, RunRow, SelectedKeyOrdering, select_key_ordering, with_key_ordering};

/// Returns a valid row order for one batch. The identity order is returned
/// when the batch already satisfies `order_by`.
pub fn batch_sort_indices(
    order_by: &[OrderBy],
    batch: &RecordBatch,
) -> Result<Vec<u32>, arrow_schema::ArrowError> {
    let batches = std::slice::from_ref(batch);
    let selected = select_key_ordering(order_by, batches, batches)?;
    Ok(with_key_ordering!(selected, |ordering| {
        if batch_arrives_sorted(&mut ordering, batch.num_rows()) {
            (0..batch.num_rows() as u32).collect()
        } else {
            sorted_row_indices(&mut ordering, batch.num_rows())
        }
    }))
}

/// A maximal stretch of rows already in key order, held as the batch-sized
/// chunks in which it arrived.
pub(crate) struct SortedRun {
    chunks: Vec<RecordBatch>,
}

impl SortedRun {
    fn from_chunks(chunks: Vec<RecordBatch>) -> Self {
        Self { chunks }
    }

    fn starting_with(chunk: RecordBatch) -> Self {
        Self::from_chunks(vec![chunk])
    }

    fn extend_with(&mut self, chunk: RecordBatch) {
        self.chunks.push(chunk);
    }

    fn into_chunks(self) -> Vec<RecordBatch> {
        self.chunks
    }

    fn last_chunk(&self) -> &RecordBatch {
        self.chunks.last().expect("a run is never empty")
    }
}

/// Runs and ownership context produced by one worker during the consume phase.
pub(crate) struct WorkerRuns {
    runs: Vec<SortedRun>,
    node_id: usize,
    local_worker_id: usize,
    allocator: Option<SlabAllocator>,
}

impl NormalizationBatches for WorkerRuns {
    fn visit_batches_mut(&mut self, visit: &mut dyn FnMut(&mut RecordBatch)) {
        for run in &mut self.runs {
            for batch in &mut run.chunks {
                visit(batch);
            }
        }
    }
}

/// Coordination shared by every worker participating in one ORDER BY.
struct SharedMergeState {
    order_by: Arc<[OrderBy]>,
    topology: Topology,
    node_gathers: Box<[GatherBarrier<WorkerRuns>]>,
    node_outputs: Box<[OnceLock<MergedOutput>]>,
    nodes_remaining: AtomicUsize,
    tasks_by_node: Box<[Injector<MergeTask>]>,
    sorted_output: OnceLock<MergedOutput>,
    merges_complete: AtomicBool,
    output_claimed: AtomicBool,
}

#[derive(Clone, Copy)]
enum MergeLevel {
    Local(usize),
    Global,
}

struct MergeTask {
    level: MergeLevel,
    merge: KWayMergeTask,
}

impl NodeIdOutput for MergeTask {
    fn node_id(&self) -> usize {
        self.merge.node_id()
    }
}

impl SharedMergeState {
    fn start_local_merge(
        &self,
        node_id: usize,
        gathered_by_worker: Vec<WorkerRuns>,
    ) -> unary::Result<()> {
        let runs = gathered_by_worker
            .into_iter()
            .flat_map(|worker| worker.runs)
            .map(|run| MergeRun::new(run.into_chunks(), node_id))
            .collect();
        let plan = KWayMergePlan::try_new(self.order_by.clone(), runs, self.topology.node_count)?;
        self.schedule(plan, MergeLevel::Local(node_id))
    }

    fn schedule(&self, plan: KWayMergePlan, level: MergeLevel) -> unary::Result<()> {
        match plan {
            KWayMergePlan::Empty => self.complete_level(level, MergedOutput::empty()),
            KWayMergePlan::Identity(output) => self.complete_level(level, output),
            KWayMergePlan::Parallel(tasks) => {
                self.enqueue(level, tasks);
                Ok(())
            }
        }
    }

    /// Push one level's tasks onto their nodes' deques and wake workers there.
    fn enqueue(&self, level: MergeLevel, tasks: Vec<KWayMergeTask>) {
        for merge in tasks {
            let node_id = merge.node_id();
            self.tasks_by_node[node_id].push(MergeTask { level, merge });
            waker_set().notify_one_near(node_id);
        }
    }

    fn complete_level(&self, level: MergeLevel, output: MergedOutput) -> unary::Result<()> {
        match level {
            MergeLevel::Local(node_id) => self.complete_local_merge(node_id, output),
            MergeLevel::Global => {
                self.finish_output(output);
                Ok(())
            }
        }
    }

    fn complete_local_merge(&self, node_id: usize, output: MergedOutput) -> unary::Result<()> {
        self.node_outputs[node_id]
            .set(output)
            .unwrap_or_else(|_| panic!("NUMA node {node_id} completed its local merge twice"));
        if self.nodes_remaining.fetch_sub(1, Ordering::AcqRel) != 1 {
            return Ok(());
        }

        let node_runs = self
            .node_outputs
            .iter()
            .enumerate()
            .filter_map(|(node_id, output)| {
                let output = output
                    .get()
                    .expect("the final node observes every local merge")
                    .clone();
                (output.row_count() > 0).then(|| output.into_run(node_id))
            })
            .collect();
        let global =
            KWayMergePlan::try_new(self.order_by.clone(), node_runs, self.topology.node_count)?;
        self.schedule(global, MergeLevel::Global)
    }

    fn execute(&self, task: MergeTask, allocator: &mut SlabAllocator) -> unary::Result<()> {
        let MergeTask { level, merge } = task;
        if let Some(output) = merge.execute(allocator)? {
            self.complete_level(level, output)?;
        }
        Ok(())
    }

    fn claim_task(&self, node_id: usize) -> Option<MergeTask> {
        loop {
            match self.tasks_by_node[node_id].steal() {
                Steal::Success(task) => return Some(task),
                Steal::Retry => continue,
                Steal::Empty => return None,
            }
        }
    }

    fn finish_output(&self, output: MergedOutput) {
        self.sorted_output
            .set(output)
            .unwrap_or_else(|_| unreachable!("the output finishes once"));
        self.merges_complete.store(true, Ordering::Release);
        waker_set().notify_all();
    }
}

/// Builds one worker's [`OrderBySorter`].
pub struct OrderByFactory<O = RunMerger> {
    order_by: Arc<[OrderBy]>,
    node_id: usize,
    local_worker_id: usize,
    outputter: O,
}

type NormalizingOrderByFactory = OrderByFactory<Normalizer<WorkerRuns>>;
type OrderByCollectorFactory = CollectorFactory<WorkerRuns, RunMerger>;
type NormalizingOrderByFactories = (Vec<NormalizingOrderByFactory>, Vec<OrderByCollectorFactory>);

impl OrderByFactory<RunMerger> {
    /// Creates one factory per worker with shared merge coordination.
    pub fn create_for_workers(order_by: Vec<OrderBy>, topology: Topology) -> Vec<OrderByFactory> {
        let order_by: Arc<[OrderBy]> = order_by.into();
        let shared = Arc::new(SharedMergeState {
            order_by: order_by.clone(),
            topology,
            node_gathers: (0..topology.node_count)
                .map(|_| GatherBarrier::new(topology.workers_per_node))
                .collect(),
            node_outputs: (0..topology.node_count).map(|_| OnceLock::new()).collect(),
            nodes_remaining: AtomicUsize::new(topology.node_count),
            tasks_by_node: (0..topology.node_count).map(|_| Injector::new()).collect(),
            sorted_output: OnceLock::new(),
            merges_complete: AtomicBool::new(false),
            output_claimed: AtomicBool::new(false),
        });
        (0..topology.total_workers())
            .map(|worker_id| {
                let node_id = topology.node_of_worker(worker_id);
                let local_worker_id = topology.local_index_of_worker(worker_id);
                OrderByFactory {
                    order_by: order_by.clone(),
                    node_id,
                    local_worker_id,
                    outputter: RunMerger {
                        shared: shared.clone(),
                        node_id,
                        allocator: None,
                    },
                }
            })
            .collect()
    }
}

/// Build the two-breaker ORDER BY composition used for variant-bearing input.
/// The sorter first outputs through [`Normalizer`]; a single collector then
/// initializes the ordinary shared [`RunMerger`] outputters.
pub(crate) fn create_normalizing_for_workers(
    order_by: Vec<OrderBy>,
    topology: Topology,
    collector_worker: usize,
) -> NormalizingOrderByFactories {
    let direct = OrderByFactory::create_for_workers(order_by, topology);
    let normalizers = Normalizer::create_for_workers(topology.total_workers());
    let mut sorters = Vec::with_capacity(topology.total_workers());
    let mut collectors = Vec::with_capacity(topology.total_workers());
    for (worker, (factory, normalizer)) in direct.into_iter().zip(normalizers).enumerate() {
        let OrderByFactory {
            order_by,
            node_id,
            local_worker_id,
            outputter,
        } = factory;
        sorters.push(OrderByFactory {
            order_by,
            node_id,
            local_worker_id,
            outputter: normalizer,
        });
        collectors.push(CollectorFactory::new(outputter, worker == collector_worker));
    }
    (sorters, collectors)
}

impl<O, Out> UnaryFactory<RecordBatch, Out> for OrderByFactory<O>
where
    O: BatchesOutputter<WorkerRuns, Out> + Send + 'static,
    Out: 'static,
{
    type Unary = PipelineBreaker<RecordBatch, Out, OrderBySorter<O>>;

    fn build_unary(self) -> Self::Unary {
        PipelineBreaker::Consuming(OrderBySorter {
            order_by: self.order_by,
            node_id: self.node_id,
            local_worker_id: self.local_worker_id,
            open_run: None,
            completed_runs: Vec::new(),
            allocator: None,
            outputter: self.outputter,
        })
    }
}

/// Sorts batches received by one worker and combines adjacent ordered batches
/// into runs.
pub struct OrderBySorter<O = RunMerger> {
    order_by: Arc<[OrderBy]>,
    node_id: usize,
    local_worker_id: usize,
    /// Most recent run, retained separately so the next batch can extend it.
    open_run: Option<SortedRun>,
    completed_runs: Vec<SortedRun>,
    /// Ring memory the sorted copies of unsorted batches land in. Taken on
    /// first use, so a worker whose batches all arrive sorted holds none.
    allocator: Option<SlabAllocator>,
    outputter: O,
}

impl<O> OrderBySorter<O> {
    /// Returns the input unchanged when it is already ordered; otherwise
    /// returns a reordered batch backed by ring memory.
    fn sort_batch(&mut self, batch: RecordBatch) -> unary::Result<RecordBatch> {
        let chunks = std::slice::from_ref(&batch);
        let selected = select_key_ordering(&self.order_by, chunks, chunks)?;
        let sorted_indices = with_key_ordering!(selected, |ordering| {
            if batch_arrives_sorted(&mut ordering, batch.num_rows()) {
                return Ok(batch);
            }
            sorted_row_indices(&mut ordering, batch.num_rows())
        });

        let allocator = self
            .allocator
            .get_or_insert_with(|| SlabAllocator::new(false));
        let columns = batch
            .columns()
            .iter()
            .map(|column| take(allocator, column, &sorted_indices))
            .collect::<Result<Vec<ArrayRef>, _>>()?;
        RecordBatch::try_new(batch.schema(), columns).map_err(unary::Error::from)
    }

    /// Whether appending `batch` preserves the ordering of the open run.
    fn extends_open_run(&self, batch: &RecordBatch) -> unary::Result<bool> {
        let Some(run) = &self.open_run else {
            return Ok(false);
        };
        let run_tail = std::slice::from_ref(run.last_chunk());
        let selected = select_key_ordering(&self.order_by, run_tail, std::slice::from_ref(batch))?;
        let last_of_run = RunRow {
            chunk: 0,
            row: run.last_chunk().num_rows() - 1,
        };
        let first_of_batch = RunRow { chunk: 0, row: 0 };
        Ok(with_key_ordering!(selected, |ordering| {
            ordering.compare(last_of_run, first_of_batch) != std::cmp::Ordering::Greater
        }))
    }
}

impl<O, Out> Consumer<RecordBatch, Out> for OrderBySorter<O>
where
    O: BatchesOutputter<WorkerRuns, Out>,
{
    type Outputter = O;

    fn consume(&mut self, batch: RecordBatch, _sender: &mut dyn Sender<Out>) -> unary::Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let sorted = self.sort_batch(batch)?;
        if self.extends_open_run(&sorted)? {
            self.open_run
                .as_mut()
                .expect("extends_open_run saw a run")
                .extend_with(sorted);
        } else {
            if let Some(finished) = self.open_run.take() {
                self.completed_runs.push(finished);
            }
            self.open_run = Some(SortedRun::starting_with(sorted));
        }
        Ok(())
    }

    fn into_outputter(mut self) -> unary::Result<Option<Self::Outputter>> {
        self.completed_runs.extend(self.open_run.take());
        let group = WorkerRuns {
            runs: std::mem::take(&mut self.completed_runs),
            node_id: self.node_id,
            local_worker_id: self.local_worker_id,
            allocator: self.allocator.take(),
        };
        self.outputter.accept(group)?;
        Ok(Some(self.outputter))
    }
}

/// One worker's output phase: run node-local tasks until one worker emits the
/// completed global output.
pub struct RunMerger {
    shared: Arc<SharedMergeState>,
    node_id: usize,
    allocator: Option<SlabAllocator>,
}

impl BatchesOutputter<WorkerRuns, RecordBatch> for RunMerger {
    fn accept(&mut self, mut group: WorkerRuns) -> unary::Result<()> {
        debug_assert_eq!(group.node_id, self.node_id);
        self.allocator = group.allocator.take();
        let shared = self.shared.clone();
        if let Some(result) = self.shared.node_gathers[group.node_id].arrive_at(
            group.local_worker_id,
            group,
            |groups| shared.start_local_merge(self.node_id, groups),
        ) {
            result?;
        }
        Ok(())
    }
}

impl InitializableOutputter<WorkerRuns, RecordBatch> for RunMerger {
    fn initialize(&mut self, groups: Vec<WorkerRuns>) -> unary::Result<()> {
        let mut by_node: Vec<Vec<WorkerRuns>> = (0..self.shared.topology.node_count)
            .map(|_| Vec::new())
            .collect();
        let mut kept_allocator = false;
        for mut group in groups {
            if !kept_allocator
                && group.node_id == self.node_id
                && let Some(allocator) = group.allocator.take()
            {
                self.allocator = Some(allocator);
                kept_allocator = true;
            }
            by_node[group.node_id].push(group);
        }
        for (node_id, groups) in by_node.into_iter().enumerate() {
            self.shared.start_local_merge(node_id, groups)?;
        }
        Ok(())
    }
}

impl Outputter<RecordBatch> for RunMerger {
    fn output(&mut self, sender: &mut dyn Sender<RecordBatch>) -> unary::Result<bool> {
        if let Some(task) = self.shared.claim_task(self.node_id) {
            let allocator = self
                .allocator
                .get_or_insert_with(|| SlabAllocator::new(false));
            self.shared.execute(task, allocator)?;
            // One slice per call keeps the worker responsive to the rest of
            // its dataflow.
            return Ok(false);
        }
        if !self.shared.merges_complete.load(Ordering::Acquire) {
            // Other workers are still sorting or merging; their completions
            // will feed the queues.
            return Ok(false);
        }
        if !self.shared.output_claimed.swap(true, Ordering::AcqRel) {
            let output = self
                .shared
                .sorted_output
                .get()
                .expect("completed merges set the output");
            for batch in output.batches() {
                sender.send(batch.batch().clone())?;
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use crate::operations::unary::test_utils::{CollectSender, run_consumers};
    use crate::waker::{WakerSet, WorkerWaker, init_waker_set, init_worker_waker};
    use crate::worker::WORKER_IDX;
    use arrow_array::{Int64Array, StringViewArray};
    use arrow_schema::{DataType, Field, Schema};

    fn consumers(order_by: Vec<OrderBy>, worker_count: usize) -> Vec<OrderBySorter> {
        OrderByFactory::create_for_workers(order_by, Topology::single_node(worker_count))
            .into_iter()
            .map(|factory| match factory.build_unary() {
                PipelineBreaker::Consuming(sorter) => sorter,
                _ => unreachable!("factories build consuming breakers"),
            })
            .collect()
    }

    fn int_batch(values: &[i64]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values.to_vec()))]).unwrap()
    }

    fn string_int_batch(names: &[&str], values: &[i64]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8View, false),
            Field::new("key", DataType::Int64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringViewArray::from(names.to_vec())),
                Arc::new(Int64Array::from(values.to_vec())),
            ],
        )
        .unwrap()
    }

    fn ascending_by_key() -> Vec<OrderBy> {
        vec![OrderBy::new(0, false, true)]
    }

    #[test]
    fn shuffled_batches_across_workers_come_out_sorted() {
        init_test_free_pool(16);
        let worker_batches = vec![
            vec![int_batch(&[9, 3, 41]), int_batch(&[7, 7, 0])],
            vec![int_batch(&[25, -4, 8])],
            vec![int_batch(&[12, 1])],
        ];

        let output = run_consumers(consumers(ascending_by_key(), 3), worker_batches);

        assert_eq!(
            output.i64_column(0),
            vec![-4, 0, 1, 3, 7, 7, 8, 9, 12, 25, 41]
        );
    }

    #[test]
    fn two_nodes_merge_locally_then_globally() {
        init_test_free_pool(16);
        let topology = Topology {
            workers_per_node: 1,
            node_count: 2,
        };
        let node_wakers: Vec<_> = (0..2).map(|_| Arc::new(WorkerWaker::new(1))).collect();
        init_worker_waker(&node_wakers[0]);
        init_waker_set(WakerSet::new(node_wakers, 1));
        let mut consumers: Vec<_> =
            OrderByFactory::create_for_workers(ascending_by_key(), topology)
                .into_iter()
                .map(|factory| match factory.build_unary() {
                    PipelineBreaker::Consuming(sorter) => sorter,
                    _ => unreachable!("factories build consuming breakers"),
                })
                .collect();
        let mut ignored = CollectSender::new();
        consumers[0]
            .consume(int_batch(&[9, 1, 5]), &mut ignored)
            .unwrap();
        consumers[1]
            .consume(int_batch(&[8, 2, 4]), &mut ignored)
            .unwrap();

        let mut outputters = Vec::new();
        for (worker_id, consumer) in consumers.into_iter().enumerate() {
            WORKER_IDX.set(worker_id);
            outputters.push(consumer.into_outputter().unwrap().unwrap());
        }
        WORKER_IDX.set(0);
        let mut output = CollectSender::new();
        loop {
            let mut done = true;
            for outputter in &mut outputters {
                done &= outputter.output(&mut output).unwrap();
            }
            if done {
                break;
            }
        }

        assert_eq!(output.i64_column(0), vec![1, 2, 4, 5, 8, 9]);
    }

    #[test]
    fn presorted_batches_pass_through_as_the_same_arrays() {
        init_test_free_pool(16);
        let first = int_batch(&[1, 2, 3]);
        let second = int_batch(&[3, 5, 9]);
        let worker_batches = vec![vec![first.clone(), second.clone()]];

        let output = run_consumers(consumers(ascending_by_key(), 1), worker_batches);

        assert_eq!(output.items.len(), 2);
        assert!(Arc::ptr_eq(output.items[0].column(0), first.column(0)));
        assert!(Arc::ptr_eq(output.items[1].column(0), second.column(0)));
    }

    #[test]
    fn a_descending_key_reverses_the_order() {
        init_test_free_pool(16);
        let worker_batches = vec![vec![int_batch(&[3, 41, 9])], vec![int_batch(&[25, 7])]];

        let output = run_consumers(
            consumers(vec![OrderBy::new(0, true, true)], 2),
            worker_batches,
        );

        assert_eq!(output.i64_column(0), vec![41, 25, 9, 7, 3]);
    }

    #[test]
    fn string_keys_sort_by_their_bytes() {
        init_test_free_pool(16);
        let worker_batches = vec![
            vec![string_int_batch(
                &["pear", "apple", "a value long enough to leave the view"],
                &[1, 2, 3],
            )],
            vec![string_int_batch(&["banana", "quince"], &[4, 5])],
        ];

        let output = run_consumers(consumers(ascending_by_key(), 2), worker_batches);

        assert_eq!(
            output.string_column(0),
            vec![
                "a value long enough to leave the view",
                "apple",
                "banana",
                "pear",
                "quince"
            ]
        );
    }

    #[test]
    fn null_keys_take_the_general_path_and_order_first() {
        init_test_free_pool(16);
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, true)]));
        let with_null = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![Some(5), None, Some(2)]))],
        )
        .unwrap();
        let worker_batches = vec![vec![with_null], vec![int_batch(&[3])]];

        let output = run_consumers(consumers(ascending_by_key(), 2), worker_batches);

        let keys: Vec<Option<i64>> = output
            .items
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .iter()
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(keys, vec![None, Some(2), Some(3), Some(5)]);
    }

    #[test]
    fn a_second_key_breaks_ties_of_the_first() {
        init_test_free_pool(16);
        let worker_batches = vec![vec![string_int_batch(&["b", "a", "b", "a"], &[1, 9, 0, 4])]];
        let order_by = vec![OrderBy::new(0, false, true), OrderBy::new(1, false, true)];

        let output = run_consumers(consumers(order_by, 1), worker_batches);

        assert_eq!(output.string_column(0), vec!["a", "a", "b", "b"]);
        assert_eq!(output.i64_column(1), vec![4, 9, 0, 1]);
    }

    #[test]
    fn equal_keys_may_use_any_legal_sql_order() {
        init_test_free_pool(16);
        let worker_batches = vec![vec![string_int_batch(
            &["tie", "tie", "tie", "tie"],
            &[10, 20, 30, 40],
        )]];

        let output = run_consumers(consumers(ascending_by_key(), 1), worker_batches);

        let mut payloads = output.i64_column(1);
        payloads.sort_unstable();
        assert_eq!(payloads, vec![10, 20, 30, 40]);
    }

    #[test]
    fn empty_input_emits_nothing() {
        init_test_free_pool(16);

        let output = run_consumers(consumers(ascending_by_key(), 2), vec![vec![], vec![]]);

        assert!(output.items.is_empty());
    }

    /// Enough rows for the merge plan to split its output across several
    /// independently executable slices.
    #[test]
    fn large_shuffled_input_sorts_across_merge_slices() {
        init_test_free_pool(64);
        let rows_per_batch = 4_096;
        let batches: Vec<RecordBatch> = (0..8)
            .map(|batch_index| {
                let values: Vec<i64> = (0..rows_per_batch)
                    .map(|row| {
                        let scattered = (row * 7919 + batch_index * 13) % 100_000;
                        scattered as i64 - 50_000
                    })
                    .collect();
                int_batch(&values)
            })
            .collect();
        let worker_batches = vec![
            batches[..3].to_vec(),
            batches[3..6].to_vec(),
            batches[6..].to_vec(),
        ];

        let output = run_consumers(consumers(ascending_by_key(), 3), worker_batches);

        let keys = output.i64_column(0);
        assert_eq!(keys.len(), 8 * rows_per_batch);
        assert!(keys.windows(2).all(|pair| pair[0] <= pair[1]));
    }
}
