//! Groups sorted partition batches into file candidates.
//!
//! Every worker's [`FileCollector`] feeds the same pending runs, so the worker
//! that sorted a run queues it itself and no run waits on a single collecting
//! worker. Each partition key has a lock-free queue of runs and a count of the
//! bytes queued there that no file has claimed. A worker that finds
//! `target_in_memory_bytes_per_file` unclaimed takes that many bytes with a
//! compare-and-swap, so exactly one worker cuts each file, then pops the runs
//! and emits the file. Whole runs are popped, so the final run may take a file
//! past the target. The last worker to finish emits what each partition has
//! left. Files already sent downstream are outside this limit.
//!
//! Each input message is one sorted run and remembers its source NUMA node.
//! Ordered files are emitted as one local-merge request per participating node.
//! Unordered files proceed directly to row-group planning without copying.

use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use arrow_array::RecordBatch;
use arrow_row::OwnedRow;
use crossbeam_deque::{Injector, Steal};
use dispatch::{
    LocatedBatch, MergeRun, OrderBy, Sender, Topology, Unary, UnaryFactory, UnaryResult,
};

use super::error::WriteError;
use super::partition_sorter::SortedPartitionRun;
use super::types::{FileMergeContext, FileOrderInput, FilePlan, NodeMergeRequest, ReadyFile};
use crate::scalar_values_from_row;

/// One partition's sorted runs waiting to be cut into files.
struct PendingRuns {
    runs: Injector<SortedPartitionRun>,
    /// Bytes of `runs` that no file has claimed. A run is counted only after
    /// it is queued, so bytes a worker claims are there for it to pop.
    unclaimed_bytes: AtomicI64,
}

impl PendingRuns {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            runs: Injector::new(),
            unclaimed_bytes: AtomicI64::new(0),
        })
    }
}

/// What every worker's collector shares for one write.
struct SharedCollector {
    /// Every partition seen so far. A worker looks here only the first time it
    /// meets a partition key, then keeps the handle it found. Most of those
    /// lookups find a partition another worker registered, so they share the
    /// read lock; only registering a new partition takes the write lock. An
    /// unpartitioned table's one partition is registered before the workers
    /// start, so its write only ever takes the read lock.
    partitions: RwLock<HashMap<Option<OwnedRow>, Arc<PendingRuns>>>,
    next_file_id: AtomicU64,
    next_row_group_id: AtomicU64,
    /// Workers still to finish. The last one emits each partition's leftovers.
    remaining_workers: AtomicUsize,
}

pub(super) struct FileCollectorFactory {
    partition_column_names: Arc<[String]>,
    order_by: Arc<[OrderBy]>,
    target_rows_per_group: usize,
    target_in_memory_bytes_per_file: usize,
    max_file_size: Option<usize>,
    topology: Topology,
    shared: Arc<SharedCollector>,
}

pub(super) fn factories(
    partition_column_names: Arc<[String]>,
    order_by: Arc<[OrderBy]>,
    target_rows_per_group: usize,
    target_in_memory_bytes_per_file: usize,
    max_file_size: Option<usize>,
    topology: Topology,
) -> Vec<FileCollectorFactory> {
    let shared = Arc::new(SharedCollector {
        partitions: RwLock::new(HashMap::new()),
        next_file_id: AtomicU64::new(0),
        next_row_group_id: AtomicU64::new(0),
        remaining_workers: AtomicUsize::new(topology.total_workers()),
    });
    // An unpartitioned table has the one partition, `None`: register it now,
    // so no worker ever takes the write lock.
    if partition_column_names.is_empty() {
        shared
            .partitions
            .write()
            .expect("partition registry poisoned")
            .insert(None, PendingRuns::new());
    }
    (0..topology.total_workers())
        .map(|_| FileCollectorFactory {
            partition_column_names: partition_column_names.clone(),
            order_by: order_by.clone(),
            target_rows_per_group,
            target_in_memory_bytes_per_file,
            max_file_size,
            topology,
            shared: shared.clone(),
        })
        .collect()
}

impl UnaryFactory<SortedPartitionRun, FileOrderInput> for FileCollectorFactory {
    type Unary = FileCollector;

    fn build_unary(self) -> FileCollector {
        FileCollector {
            partition_column_names: self.partition_column_names,
            order_by: self.order_by,
            target_rows_per_group: self.target_rows_per_group,
            target_in_memory_bytes_per_file: self.target_in_memory_bytes_per_file,
            max_file_size: self.max_file_size,
            topology: self.topology,
            shared: self.shared,
            known_partitions: HashMap::new(),
        }
    }
}

/// The sorted runs of one file.
struct PendingFile {
    runs_by_node: Vec<Vec<Vec<RecordBatch>>>,
    row_count: usize,
    in_memory_bytes: usize,
}

impl PendingFile {
    fn new(node_count: usize) -> Self {
        Self {
            runs_by_node: (0..node_count).map(|_| Vec::new()).collect(),
            row_count: 0,
            in_memory_bytes: 0,
        }
    }

    fn push(&mut self, run: SortedPartitionRun) {
        self.row_count += run.batches.iter().map(RecordBatch::num_rows).sum::<usize>();
        self.in_memory_bytes += run.in_memory_bytes;
        self.runs_by_node[run.source_node].push(run.batches);
    }

    fn first_batch(&self) -> &RecordBatch {
        self.runs_by_node
            .iter()
            .flatten()
            .flatten()
            .next()
            .expect("a pending file is nonempty")
    }
}

pub(super) struct FileCollector {
    partition_column_names: Arc<[String]>,
    order_by: Arc<[OrderBy]>,
    target_rows_per_group: usize,
    target_in_memory_bytes_per_file: usize,
    max_file_size: Option<usize>,
    topology: Topology,
    shared: Arc<SharedCollector>,
    /// The partitions this worker has already looked up in `shared`.
    known_partitions: HashMap<Option<OwnedRow>, Arc<PendingRuns>>,
}

impl FileCollector {
    /// The shared pending runs of `partition_key`, registering the partition
    /// the first time any worker meets it.
    fn pending_runs(&mut self, partition_key: &Option<OwnedRow>) -> Arc<PendingRuns> {
        if let Some(pending) = self.known_partitions.get(partition_key) {
            return pending.clone();
        }
        let registered = self
            .shared
            .partitions
            .read()
            .expect("partition registry poisoned")
            .get(partition_key)
            .cloned();
        let pending = match registered {
            Some(pending) => pending,
            // Another worker may register the partition between the two
            // locks, so the write side looks again before inserting.
            None => self
                .shared
                .partitions
                .write()
                .expect("partition registry poisoned")
                .entry(partition_key.clone())
                .or_insert_with(PendingRuns::new)
                .clone(),
        };
        self.known_partitions
            .insert(partition_key.clone(), pending.clone());
        pending
    }

    /// Claims one file's worth of unclaimed bytes and pops its runs, or returns
    /// `None` when less than a file is unclaimed. Concurrent claimers each
    /// take whole runs past their share, so one of them may find the queue a
    /// few runs short and cut a slightly smaller file.
    fn try_claim_file(&self, pending: &PendingRuns) -> Option<PendingFile> {
        // A target no file can reach (compaction) leaves everything to finish.
        let target = i64::try_from(self.target_in_memory_bytes_per_file).unwrap_or(i64::MAX);
        let mut unclaimed = pending.unclaimed_bytes.load(Ordering::Acquire);
        loop {
            if unclaimed < target {
                return None;
            }
            match pending.unclaimed_bytes.compare_exchange_weak(
                unclaimed,
                unclaimed - target,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(current) => unclaimed = current,
            }
        }
        let file = self.pop_runs(pending, target as usize);
        // The claim took exactly `target`; settle it to what was popped.
        pending
            .unclaimed_bytes
            .fetch_sub(file.in_memory_bytes as i64 - target, Ordering::AcqRel);
        (file.row_count > 0).then_some(file)
    }

    /// Pops whole runs until the file holds `target_bytes` or none are queued.
    fn pop_runs(&self, pending: &PendingRuns, target_bytes: usize) -> PendingFile {
        let mut file = PendingFile::new(self.topology.node_count);
        while file.in_memory_bytes < target_bytes {
            match pending.runs.steal() {
                Steal::Success(run) => file.push(run),
                Steal::Empty => break,
                Steal::Retry => {}
            }
        }
        file
    }

    /// Assigns file and row-group identities, then emits either a ready file or
    /// one local-merge request for each participating NUMA node.
    fn emit_pending_file(
        &self,
        partition_key: &Option<OwnedRow>,
        pending_file: PendingFile,
        sender: &mut dyn Sender<FileOrderInput>,
    ) -> UnaryResult<()> {
        debug_assert!(pending_file.row_count > 0);
        let file_id = self.shared.next_file_id.fetch_add(1, Ordering::Relaxed);
        let row_group_count = pending_file.row_count.div_ceil(self.target_rows_per_group);
        let base_row_group_id = self
            .shared
            .next_row_group_id
            .fetch_add(row_group_count as u64, Ordering::Relaxed);
        let partition_values = match partition_key {
            Some(_) => Some(
                scalar_values_from_row(pending_file.first_batch(), &self.partition_column_names, 0)
                    .map_err(WriteError::from)?,
            ),
            None => None,
        };
        let plan = FilePlan {
            file_id,
            base_row_group_id,
            assembly_worker: (file_id as usize) % self.topology.total_workers(),
            partition: partition_values,
            target_rows_per_group: self.target_rows_per_group,
            max_file_size: self.max_file_size,
        };

        if self.order_by.is_empty() {
            let mut rows_by_node = vec![0usize; self.topology.node_count];
            let mut batches = Vec::new();
            for (node_id, runs) in pending_file.runs_by_node.into_iter().enumerate() {
                for batch in runs.into_iter().flatten() {
                    rows_by_node[node_id] += batch.num_rows();
                    batches.push(LocatedBatch::new(batch, node_id));
                }
            }
            let target_node = dispatch::dominant_node(&rows_by_node);
            sender.send(FileOrderInput::Ready(ReadyFile {
                plan,
                batches,
                row_count: pending_file.row_count,
                target_node,
            }))?;
            return Ok(());
        }

        let participating_nodes = pending_file
            .runs_by_node
            .iter()
            .filter(|runs| !runs.is_empty())
            .count();
        let context = Arc::new(FileMergeContext {
            plan,
            order_by: self.order_by.clone(),
            row_count: pending_file.row_count,
            local_outputs: (0..self.topology.node_count)
                .map(|_| OnceLock::new())
                .collect(),
            nodes_remaining: participating_nodes.into(),
        });
        for (node_id, runs) in pending_file.runs_by_node.into_iter().enumerate() {
            if runs.is_empty() {
                continue;
            }
            sender.send(FileOrderInput::Merge(NodeMergeRequest {
                context: context.clone(),
                node_id,
                runs: runs
                    .into_iter()
                    .map(|batches| MergeRun::new(batches, node_id))
                    .collect(),
            }))?;
        }
        Ok(())
    }
}

impl Unary<SortedPartitionRun, FileOrderInput> for FileCollector {
    fn consume(
        &mut self,
        partition_run: SortedPartitionRun,
        sender: &mut dyn Sender<FileOrderInput>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        if partition_run
            .batches
            .iter()
            .all(|batch| batch.num_rows() == 0)
        {
            return Ok(());
        }
        let partition_key = partition_run.partition_key.clone();
        let pending = self.pending_runs(&partition_key);
        let run_bytes = partition_run.in_memory_bytes as i64;
        pending.runs.push(partition_run);
        pending
            .unclaimed_bytes
            .fetch_add(run_bytes, Ordering::AcqRel);
        while let Some(file) = self.try_claim_file(&pending) {
            self.emit_pending_file(&partition_key, file, sender)?;
        }
        Ok(())
    }

    fn finish(&mut self, sender: &mut dyn Sender<FileOrderInput>) -> UnaryResult<bool> {
        // A worker reaches finish only after its last consume, so once every
        // other worker has been here no run is still on its way to a queue.
        if self.shared.remaining_workers.fetch_sub(1, Ordering::AcqRel) != 1 {
            return Ok(true);
        }
        let partitions = self
            .shared
            .partitions
            .read()
            .expect("partition registry poisoned");
        for (partition_key, pending) in partitions.iter() {
            let file = self.pop_runs(pending, usize::MAX);
            if file.row_count > 0 {
                self.emit_pending_file(partition_key, file, sender)?;
            }
        }
        Ok(true)
    }
}
