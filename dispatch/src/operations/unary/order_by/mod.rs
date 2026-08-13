//! ORDER BY: a pipeline breaker that emits its whole input sorted by a key,
//! parallelized the way rayon's stable merge sort is.
//!
//! Rayon's `par_mergesort` has three phases, and each maps onto the dispatch
//! workers this operator runs across:
//!
//! 1. **Chunk sort.** Rayon sorts fixed-size chunks of the slice in parallel,
//!    leaving already-ascending chunks untouched and concatenating adjacent
//!    ones. Here every worker sorts each record batch the moment it arrives in
//!    `consume` — while the batch is hot in its cache — and batches that
//!    arrive already in order extend the worker's current sorted *run* without
//!    being touched, so ordered input streams cost a comparison scan and
//!    nothing else.
//!
//! 2. **A merge tree over the runs.** Rayon recursively merges pairs of runs
//!    with fork-join. Dispatch has no fork-join, it has work stealing, so
//!    every worker delivers its runs to a [`GatherBarrier`] (the same
//!    gathering GROUP BY and ORDER BY LIMIT use), and the final worker to
//!    arrive builds the tree once as [`MergeForest`]: each node merges its
//!    two children's runs, and a node becomes runnable when both children
//!    complete.
//!
//! 3. **Parallel merges.** A merge of two large runs is not one serial walk:
//!    following rayon's `par_merge`, it splits at the larger run's midpoint
//!    (binary-searching that key in the smaller run) until the pieces are a
//!    batch's worth, and every piece is an independent [`MergeSliceTask`]
//!    writing its own output chunk. Tasks live where their inputs are hot,
//!    rayon's deque discipline: a worker that completes a node plans the
//!    parent merge it unblocked and pushes those slices onto its own LIFO
//!    deque — it just wrote half their input — and the leaf merges are routed
//!    at build time to the worker that sorted their runs. Idle workers steal
//!    from the far end of other deques, so the final merge of two million-row
//!    runs is still spread over every worker, not one worker's stall.
//!
//! A merge task produces its output chunk by walking the two runs' *keys*
//! ([`keys`]) into a row mapping and gathering every column through the
//! slab-backed chunked take ([`crate::arrays::take::take_chunked`]), so batch
//! memory stays on the ring throughout. The finished sort is a run of
//! batch-sized chunks, emitted as they are.

mod keys;
mod merge;
mod multiway;
mod partitioner;

pub use multiway::{MultiwayMergeSlice, merge_multiway_slice, plan_multiway_merge_slices};
pub use partitioner::{PartitionerFactory, SortedPiece};

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use arrow_array::{ArrayRef, RecordBatch};
use crossbeam_deque::{Injector, Steal, Stealer, Worker as SliceDeque};

use crate::arrays::take::{take, take_chunked};
use crate::gather_barrier::GatherBarrier;
use crate::memory::SlabAllocator;
use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::factory::UnaryFactory;
use crate::operations::unary::order_by_limit::OrderBy;
use crate::operations::unary::pipeline_breaker::{Consumer, Outputter, PipelineBreaker};
use crate::waker::waker_set;
use crate::worker::WORKER_IDX;

use keys::{KeyOrdering, RunRow, SelectedKeyOrdering, select_key_ordering};
use merge::{
    MergeSliceRows, batch_arrives_sorted, merge_slice_mapping, plan_merge_slices,
    sorted_row_indices,
};

/// Runs [`select_key_ordering`]'s result through one of the generic routines,
/// so every key shape's comparisons stay monomorphized in the loops.
macro_rules! with_key_ordering {
    ($selected:expr, |$ordering:ident| $body:expr) => {
        match $selected {
            SelectedKeyOrdering::FixedWidth(mut $ordering) => $body,
            SelectedKeyOrdering::ViewBytes(mut $ordering) => $body,
            SelectedKeyOrdering::General(mut $ordering) => $body,
        }
    };
}
pub(super) use with_key_ordering;

/// A maximal stretch of rows already in key order, held as the batch-sized
/// chunks it arrived (or was merged) in.
#[derive(Clone)]
struct SortedRun {
    chunks: Vec<RecordBatch>,
    /// The logical row each chunk starts at, for translating a row to its
    /// chunk.
    first_row_of_chunk: Vec<usize>,
    rows: usize,
}

impl SortedRun {
    fn from_chunks(chunks: Vec<RecordBatch>) -> Self {
        let mut first_row_of_chunk = Vec::with_capacity(chunks.len());
        let mut rows = 0;
        for chunk in &chunks {
            first_row_of_chunk.push(rows);
            rows += chunk.num_rows();
        }
        Self {
            chunks,
            first_row_of_chunk,
            rows,
        }
    }

    fn starting_with(chunk: RecordBatch) -> Self {
        Self::from_chunks(vec![chunk])
    }

    fn extend_with(&mut self, chunk: RecordBatch) {
        self.first_row_of_chunk.push(self.rows);
        self.rows += chunk.num_rows();
        self.chunks.push(chunk);
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn chunks(&self) -> &[RecordBatch] {
        &self.chunks
    }

    /// The `(chunk, row)` position of logical row `row`.
    fn locate(&self, row: usize) -> RunRow {
        let chunk = self
            .first_row_of_chunk
            .partition_point(|&first| first <= row)
            - 1;
        RunRow {
            chunk,
            row: row - self.first_row_of_chunk[chunk],
        }
    }

    fn last_chunk(&self) -> &RecordBatch {
        self.chunks.last().expect("a run is never empty")
    }
}

/// Where a merge node's finished run goes: into a parent node as one side of
/// its merge, or — at the tree's root — out as the sort's finished chunks.
#[derive(Clone, Copy)]
enum MergeResultDestination {
    Node { node: usize, side: ChildSide },
    Output,
}

/// One node of a merge tree: merges the runs its two children deliver, and
/// delivers the result onward.
struct MergeNode {
    delivers_to: MergeResultDestination,
    left: OnceLock<SortedRun>,
    right: OnceLock<SortedRun>,
    /// How many of the two inputs have arrived; the worker delivering the
    /// second one plans the node's merge.
    inputs_delivered: AtomicUsize,
    /// Slices still to merge; the worker finishing the last one completes the
    /// node.
    slices_remaining: AtomicUsize,
    /// Each slice's output chunk, in output order: the planning worker sets
    /// the row of slots, each slice fills exactly its own, and the counter
    /// above is the barrier that makes them all visible to the completer —
    /// disjoint set-once writes, so no lock anywhere.
    slice_results: OnceLock<Vec<OnceLock<RecordBatch>>>,
}

#[derive(Clone, Copy)]
enum ChildSide {
    Left,
    Right,
}

impl MergeNode {
    fn awaiting_children(delivers_to: MergeResultDestination) -> Self {
        Self {
            delivers_to,
            left: OnceLock::new(),
            right: OnceLock::new(),
            inputs_delivered: AtomicUsize::new(0),
            slices_remaining: AtomicUsize::new(0),
            slice_results: OnceLock::new(),
        }
    }

    fn input(&self, side: ChildSide) -> &OnceLock<SortedRun> {
        match side {
            ChildSide::Left => &self.left,
            ChildSide::Right => &self.right,
        }
    }
}

/// One independent piece of one node's merge, stealable by any worker.
struct MergeSliceTask {
    node: usize,
    slice: usize,
    rows: MergeSliceRows,
}

/// Everything one worker sorted, delivered to the gather barrier when its
/// consume phase ends. The barrier gathers in worker order, so a delivery's
/// position names the worker whose cache holds its runs.
type WorkerRuns = Vec<SortedRun>;

/// The state every worker's operator shares: the merge tree over the
/// gathered runs and each worker's merge-slice queues. Everything here
/// synchronizes through set-once slots and atomic counters — each writer
/// owns exactly one slot, and the counters are the barriers that publish
/// the slots to whoever acts last — so no worker ever blocks on a lock.
struct SharedMergeState {
    order_by: Arc<[OrderBy]>,
    forest: OnceLock<MergeForest>,
    /// Per-worker queues the forest build routes leaf merges through: each
    /// worker is assigned the slices over runs it sorted, and drains its own
    /// queue into its deque before helping anyone else.
    assigned_slices: Vec<Injector<MergeSliceTask>>,
    /// The stealing side of every worker's deque, for workers with nothing
    /// of their own left.
    slice_stealers: Vec<Stealer<MergeSliceTask>>,
    /// The finished output, rows in key order. The worker winning
    /// `output_claimed` emits it.
    sorted_output: OnceLock<Vec<RecordBatch>>,
    merges_complete: AtomicBool,
    output_claimed: AtomicBool,
}

/// The merge tree over every worker's runs, one node arena shared so any
/// worker helps any merge.
struct MergeForest {
    nodes: Vec<MergeNode>,
}

impl SharedMergeState {
    /// Lay the merge tree over every worker's runs and queue the merges that
    /// are ready. Called by the worker holding the run channel's receiver,
    /// once every worker has delivered. A sort with nothing to merge (a
    /// single run, or no sort keys at all) finishes here on the spot.
    fn build_merge_forest(&self, gathered_by_worker: Vec<WorkerRuns>) {
        let mut runs: Vec<(usize, SortedRun)> = Vec::new();
        for (owner, worker_runs) in gathered_by_worker.into_iter().enumerate() {
            for run in worker_runs {
                runs.push((owner, run));
            }
        }
        // No sort keys means the rows have no order to establish: the runs
        // simply concatenate. A single run needs no merge either way.
        if self.order_by.is_empty() || runs.len() <= 1 {
            let chunks = runs
                .into_iter()
                .flat_map(|(_, run)| run.chunks)
                .collect::<Vec<RecordBatch>>();
            self.finish_output(chunks);
            return;
        }

        let mut nodes = Vec::new();
        let mut leaf_deliveries = Vec::new();
        let mut run_pool: Vec<Option<(usize, SortedRun)>> = runs.into_iter().map(Some).collect();
        build_subtree(
            &mut nodes,
            &mut leaf_deliveries,
            0,
            run_pool.len(),
            MergeResultDestination::Output,
        );

        self.forest
            .set(MergeForest { nodes })
            .unwrap_or_else(|_| unreachable!("only the last sorting worker builds"));

        // Deliver the leaf runs. A node whose two inputs are both leaves gets
        // its merge planned right here; its slices are assigned to the worker
        // that sorted the larger input, where more of the data is warm.
        // (Adjacent runs mostly come from one worker, since each worker's
        // runs arrive as one contiguous block.)
        let mut first_leaf_of_node: HashMap<usize, (usize, usize)> = HashMap::new();
        for (run_index, node, side) in leaf_deliveries {
            let (owner, run) = run_pool[run_index].take().expect("each leaf delivers once");
            let rows = run.rows();
            match self.deliver_run(node, side, run) {
                Some(planned_slices) => {
                    let assignee = match first_leaf_of_node.remove(&node) {
                        Some((sibling_owner, sibling_rows)) if sibling_rows > rows => sibling_owner,
                        _ => owner,
                    };
                    for task in planned_slices {
                        self.assigned_slices[assignee].push(task);
                    }
                }
                None => {
                    first_leaf_of_node.insert(node, (owner, rows));
                }
            }
        }
        // The gather barrier wakes every worker once this build returns, so
        // the new assignments need no notify of their own.
    }

    /// Record the sort's finished chunks and wake everyone to emit them.
    fn finish_output(&self, chunks: Vec<RecordBatch>) {
        self.sorted_output
            .set(chunks)
            .unwrap_or_else(|_| unreachable!("the output finishes once"));
        self.merges_complete.store(true, Ordering::Release);
        // Parked workers must observe the completion to emit and finish; wake
        // them all.
        waker_set().notify_all();
    }

    /// Hand a completed run to `node` as its `side` input. The second input
    /// to arrive plans the node's merge; the planned slice tasks are returned
    /// so the caller queues them on the worker whose cache their inputs are
    /// hot in.
    fn deliver_run(
        &self,
        node_index: usize,
        side: ChildSide,
        run: SortedRun,
    ) -> Option<Vec<MergeSliceTask>> {
        let tree = self.forest.get().expect("delivery follows the build");
        let node = &tree.nodes[node_index];
        node.input(side)
            .set(run)
            .unwrap_or_else(|_| unreachable!("each side is delivered exactly once"));
        if node.inputs_delivered.fetch_add(1, Ordering::AcqRel) + 1 < 2 {
            return None;
        }

        let left = node.left.get().expect("both inputs delivered");
        let right = node.right.get().expect("both inputs delivered");
        let selected = select_key_ordering(&self.order_by, left.chunks(), right.chunks())
            .expect("both runs share the input schema");
        let slices = with_key_ordering!(selected, |ordering| plan_merge_slices(
            &mut ordering,
            left,
            right
        ));
        node.slice_results
            .set((0..slices.len()).map(|_| OnceLock::new()).collect())
            .unwrap_or_else(|_| unreachable!("only the second delivery plans"));
        node.slices_remaining.store(slices.len(), Ordering::Release);
        Some(
            slices
                .into_iter()
                .enumerate()
                .map(|(slice, rows)| MergeSliceTask {
                    node: node_index,
                    slice,
                    rows,
                })
                .collect(),
        )
    }

    /// Merge one slice of one node, and complete the node if it was the last.
    fn run_merge_slice(
        &self,
        task: MergeSliceTask,
        allocator: &mut SlabAllocator,
        local_slices: &SliceDeque<MergeSliceTask>,
    ) -> unary::Result<()> {
        let tree = self.forest.get().expect("tasks follow the build");
        let node = &tree.nodes[task.node];
        let left = node.left.get().expect("planned nodes have both inputs");
        let right = node.right.get().expect("planned nodes have both inputs");

        let selected = select_key_ordering(&self.order_by, left.chunks(), right.chunks())?;
        let mapping = with_key_ordering!(selected, |ordering| merge_slice_mapping(
            &mut ordering,
            left,
            right,
            &task.rows
        ));

        // Gather every column through the combined chunk list, the left run's
        // chunks first — the order the mapping's chunk indices were built for.
        let schema = left.chunks()[0].schema();
        let mut columns = Vec::with_capacity(schema.fields().len());
        for column in 0..schema.fields().len() {
            let chunks: Vec<ArrayRef> = left
                .chunks()
                .iter()
                .chain(right.chunks())
                .map(|chunk| chunk.column(column).clone())
                .collect();
            columns.push(take_chunked(allocator, &chunks, &mapping)?);
        }
        let merged = RecordBatch::try_new(schema, columns).map_err(unary::Error::from)?;

        node.slice_results.get().expect("tasks follow the plan")[task.slice]
            .set(merged)
            .unwrap_or_else(|_| unreachable!("each slice completes once"));
        if node.slices_remaining.fetch_sub(1, Ordering::AcqRel) == 1
            && let Some(planned_slices) = self.complete_node(task.node)
        {
            // The parent merge this unblocked reads chunks this worker just
            // wrote: keep its slices on this worker's own deque, stealable
            // from the far end, and wake anyone parked to come help.
            for slice_task in planned_slices {
                local_slices.push(slice_task);
            }
            waker_set().notify_all();
        }
        Ok(())
    }

    /// All of a node's slices are merged: assemble its run and deliver it
    /// upward, or finish the sort at the root. Returns the parent's slice
    /// tasks when this delivery unblocked its merge.
    fn complete_node(&self, node_index: usize) -> Option<Vec<MergeSliceTask>> {
        let tree = self.forest.get().expect("completion follows the build");
        let node = &tree.nodes[node_index];
        let chunks: Vec<RecordBatch> = node
            .slice_results
            .get()
            .expect("completion follows the plan")
            .iter()
            .map(|slot| slot.get().expect("every slice finished").clone())
            .collect();
        match node.delivers_to {
            MergeResultDestination::Node { node: parent, side } => {
                self.deliver_run(parent, side, SortedRun::from_chunks(chunks))
            }
            MergeResultDestination::Output => {
                self.finish_output(chunks);
                None
            }
        }
    }
}

/// Lay out the merge tree over `runs[first..past_last]` of the
/// run pool, rayon's recursive halving of the run list: one node per merge,
/// children built first so a node knows where its result goes. Records which
/// node each leaf run feeds.
fn build_subtree(
    nodes: &mut Vec<MergeNode>,
    leaf_deliveries: &mut Vec<(usize, usize, ChildSide)>,
    first: usize,
    past_last: usize,
    delivers_to: MergeResultDestination,
) {
    debug_assert!(past_last - first >= 2, "a merge needs two subtrees");
    let node_index = nodes.len();
    nodes.push(MergeNode::awaiting_children(delivers_to));

    let middle = first + (past_last - first) / 2;
    for (child_first, child_past_last, side) in [
        (first, middle, ChildSide::Left),
        (middle, past_last, ChildSide::Right),
    ] {
        if child_past_last - child_first == 1 {
            leaf_deliveries.push((child_first, node_index, side));
        } else {
            build_subtree(
                nodes,
                leaf_deliveries,
                child_first,
                child_past_last,
                MergeResultDestination::Node {
                    node: node_index,
                    side,
                },
            );
        }
    }
}

/// Builds one worker's [`OrderBySorter`]; all factories of one operator share
/// the merge state and one run channel. The first factory holds the channel's
/// receiver; the rest get `None`.
pub struct OrderByFactory {
    order_by: Arc<[OrderBy]>,
    shared: Arc<SharedMergeState>,
    runs_gather: Arc<GatherBarrier<WorkerRuns>>,
    /// This worker's own merge-slice deque; its stealer is in the shared
    /// state.
    local_slices: SliceDeque<MergeSliceTask>,
}

impl OrderByFactory {
    /// One factory per worker, sharing the run pool and merge queue.
    pub fn create_for_workers(order_by: Vec<OrderBy>, worker_count: usize) -> Vec<OrderByFactory> {
        let order_by: Arc<[OrderBy]> = order_by.into();
        let local_deques: Vec<SliceDeque<MergeSliceTask>> =
            (0..worker_count).map(|_| SliceDeque::new_lifo()).collect();
        let shared = Arc::new(SharedMergeState {
            order_by: order_by.clone(),
            forest: OnceLock::new(),
            assigned_slices: (0..worker_count).map(|_| Injector::new()).collect(),
            slice_stealers: local_deques.iter().map(|deque| deque.stealer()).collect(),
            sorted_output: OnceLock::new(),
            merges_complete: AtomicBool::new(false),
            output_claimed: AtomicBool::new(false),
        });
        let runs_gather = Arc::new(GatherBarrier::new(worker_count));
        local_deques
            .into_iter()
            .map(|local_slices| OrderByFactory {
                order_by: order_by.clone(),
                shared: shared.clone(),
                runs_gather: runs_gather.clone(),
                local_slices,
            })
            .collect()
    }
}

impl UnaryFactory<RecordBatch, RecordBatch> for OrderByFactory {
    type Unary = PipelineBreaker<RecordBatch, RecordBatch, OrderBySorter>;

    fn build_unary(self) -> Self::Unary {
        PipelineBreaker::Consuming(OrderBySorter {
            order_by: self.order_by,
            shared: self.shared,
            runs_gather: self.runs_gather,
            local_slices: self.local_slices,
            open_run: None,
            completed_runs: Vec::new(),
            allocator: None,
        })
    }
}

/// One worker's consume phase: sort each arriving batch and grow sorted
/// runs; ship the runs into the shared pool when the input drains.
pub struct OrderBySorter {
    order_by: Arc<[OrderBy]>,
    shared: Arc<SharedMergeState>,
    runs_gather: Arc<GatherBarrier<WorkerRuns>>,
    /// This worker's own merge-slice deque, carried through to its
    /// [`RunMerger`].
    local_slices: SliceDeque<MergeSliceTask>,
    /// The run the next batch may extend.
    open_run: Option<SortedRun>,
    completed_runs: Vec<SortedRun>,
    /// Ring memory the sorted copies of unsorted batches land in. Taken on
    /// first use, so a worker whose batches all arrive sorted holds none.
    allocator: Option<SlabAllocator>,
}

impl OrderBySorter {
    /// The batch with its rows in key order: the batch itself when they
    /// already are, a slab-backed sorted copy otherwise. The copy shares what
    /// it can with the source (a view column's value bytes stay where they
    /// were); a pipeline breaker retains its whole input regardless, so
    /// there is nothing to gain from a self-contained one.
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

    /// Whether `batch`'s first row sorts at or after the open run's last row,
    /// so appending keeps the run sorted.
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

impl Consumer<RecordBatch, RecordBatch> for OrderBySorter {
    type Outputter = RunMerger;

    fn consume(
        &mut self,
        batch: RecordBatch,
        _sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
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
        let allocator = self.allocator.take();
        // The final worker to arrive sees every worker's runs in worker order
        // and builds the forest; the barrier then wakes everyone to merge.
        self.runs_gather
            .arrive(std::mem::take(&mut self.completed_runs), |gathered| {
                self.shared.build_merge_forest(gathered)
            });
        // Every worker merges, whatever it consumed: the tasks are shared.
        Ok(Some(RunMerger {
            shared: self.shared,
            local_slices: self.local_slices,
            allocator: allocator.unwrap_or_else(|| SlabAllocator::new(false)),
        }))
    }
}

/// One worker's output phase: claim and run merge slices until the root run
/// is complete, then one worker emits the sorted chunks.
pub struct RunMerger {
    shared: Arc<SharedMergeState>,
    /// This worker's own merge slices, popped newest-first: the top of the
    /// deque is the merge whose inputs this worker wrote last.
    local_slices: SliceDeque<MergeSliceTask>,
    allocator: SlabAllocator,
}

impl RunMerger {
    /// The next merge slice to run, hottest first: this worker's own deque,
    /// then the slices the forest build assigned to it, and — with nothing of
    /// its own left — another worker's deque or assignments.
    fn claim_slice(&self) -> Option<MergeSliceTask> {
        if let Some(task) = self.local_slices.pop() {
            return Some(task);
        }
        let worker_index = WORKER_IDX.get();
        loop {
            match self.shared.assigned_slices[worker_index].steal_batch_and_pop(&self.local_slices)
            {
                Steal::Success(task) => return Some(task),
                Steal::Retry => continue,
                Steal::Empty => break,
            }
        }
        let worker_count = self.shared.slice_stealers.len();
        for offset in 1..worker_count {
            let other = (worker_index + offset) % worker_count;
            // The plain `is_empty` pre-checks keep the drained-queue polls
            // from pinning a crossbeam epoch each pass.
            let other_deque = &self.shared.slice_stealers[other];
            while !other_deque.is_empty() {
                match other_deque.steal() {
                    Steal::Success(task) => return Some(task),
                    Steal::Retry => continue,
                    Steal::Empty => break,
                }
            }
            // A worker busy elsewhere in its dataflow may not have drained
            // its assignments yet; its work must not strand.
            let other_assigned = &self.shared.assigned_slices[other];
            while !other_assigned.is_empty() {
                match other_assigned.steal() {
                    Steal::Success(task) => return Some(task),
                    Steal::Retry => continue,
                    Steal::Empty => break,
                }
            }
        }
        None
    }
}

impl Outputter<RecordBatch> for RunMerger {
    fn output(&mut self, sender: &mut dyn Sender<RecordBatch>) -> unary::Result<bool> {
        if let Some(task) = self.claim_slice() {
            self.shared
                .run_merge_slice(task, &mut self.allocator, &self.local_slices)?;
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
            let chunks = self
                .shared
                .sorted_output
                .get()
                .expect("completed merges set the output");
            for chunk in chunks {
                sender.send(chunk.clone())?;
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use crate::operations::unary::test_utils::run_consumers;
    use arrow_array::{Int64Array, StringViewArray};
    use arrow_schema::{DataType, Field, Schema};

    fn consumers(order_by: Vec<OrderBy>, worker_count: usize) -> Vec<OrderBySorter> {
        OrderByFactory::create_for_workers(order_by, worker_count)
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
    fn equal_keys_keep_their_arrival_order() {
        init_test_free_pool(16);
        let worker_batches = vec![vec![string_int_batch(
            &["tie", "tie", "tie", "tie"],
            &[10, 20, 30, 40],
        )]];

        let output = run_consumers(consumers(ascending_by_key(), 1), worker_batches);

        assert_eq!(output.i64_column(1), vec![10, 20, 30, 40]);
    }

    #[test]
    fn empty_input_emits_nothing() {
        init_test_free_pool(16);

        let output = run_consumers(consumers(ascending_by_key(), 2), vec![vec![], vec![]]);

        assert!(output.items.is_empty());
    }

    /// Enough rows that the merge of the two biggest runs splits into several
    /// slices, which is the parallel-merge path rayon's split drives.
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
