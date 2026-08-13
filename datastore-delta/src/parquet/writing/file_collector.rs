//! Groups sorted partition batches into file candidates.
//!
//! All batches for one write arrive at a single [`FileCollector`]. It keeps a
//! pending file for each partition key and emits that file after its retained
//! Arrow data reaches `target_in_memory_bytes_per_file`. The threshold applies
//! independently to each partition and may be exceeded by the final batch.
//! Queued messages and files already sent downstream are outside this limit.
//!
//! Each input message is one sorted run and remembers its source NUMA node.
//! Ordered files are emitted as one local-merge request per participating node.
//! Unordered files proceed directly to row-group planning without copying.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use arrow_array::RecordBatch;
use arrow_row::OwnedRow;
use dispatch::{
    LocatedBatch, MergeRun, OrderBy, Sender, Topology, Unary, UnaryFactory, UnaryResult,
};

use super::error::WriteError;
use super::partition_sorter::SortedPartitionRun;
use super::types::{FileMergeContext, FileOrderInput, FilePlan, NodeMergeRequest, ReadyFile};
use crate::scalar_values_from_row;

pub(super) struct FileCollectorFactory {
    partition_column_names: Arc<[String]>,
    order_by: Arc<[OrderBy]>,
    target_rows_per_group: usize,
    target_in_memory_bytes_per_file: usize,
    topology: Topology,
}

pub(super) fn factories(
    partition_column_names: Arc<[String]>,
    order_by: Arc<[OrderBy]>,
    target_rows_per_group: usize,
    target_in_memory_bytes_per_file: usize,
    topology: Topology,
) -> Vec<FileCollectorFactory> {
    (0..topology.total_workers())
        .map(|_| FileCollectorFactory {
            partition_column_names: partition_column_names.clone(),
            order_by: order_by.clone(),
            target_rows_per_group,
            target_in_memory_bytes_per_file,
            topology,
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
            topology: self.topology,
            pending_files_by_partition: HashMap::new(),
            next_file_id: 0,
            next_row_group_id: 0,
        }
    }
}

/// Sorted runs retained for the next file of one partition.
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
    topology: Topology,
    pending_files_by_partition: HashMap<Option<OwnedRow>, PendingFile>,
    next_file_id: u64,
    next_row_group_id: u64,
}

impl FileCollector {
    /// Assigns file and row-group identities, then emits either a ready file or
    /// one local-merge request for each participating NUMA node.
    fn emit_pending_file(
        &mut self,
        partition_key: Option<OwnedRow>,
        pending_file: PendingFile,
        sender: &mut dyn Sender<FileOrderInput>,
    ) -> UnaryResult<()> {
        debug_assert!(pending_file.row_count > 0);
        let file_id = self.next_file_id;
        self.next_file_id += 1;
        let row_group_count = pending_file.row_count.div_ceil(self.target_rows_per_group);
        let base_row_group_id = self.next_row_group_id;
        self.next_row_group_id += row_group_count as u64;
        let partition_values = match &partition_key {
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
    ) -> UnaryResult<()> {
        let run_row_count: usize = partition_run
            .batches
            .iter()
            .map(RecordBatch::num_rows)
            .sum();
        if run_row_count == 0 {
            return Ok(());
        }
        let pending_file = self
            .pending_files_by_partition
            .entry(partition_run.partition_key.clone())
            .or_insert_with(|| PendingFile::new(self.topology.node_count));
        pending_file.row_count += run_row_count;
        pending_file.in_memory_bytes += partition_run.in_memory_bytes;
        pending_file.runs_by_node[partition_run.source_node].push(partition_run.batches);
        if pending_file.in_memory_bytes >= self.target_in_memory_bytes_per_file {
            let completed_file = self
                .pending_files_by_partition
                .remove(&partition_run.partition_key)
                .expect("the pending file was inserted above");
            self.emit_pending_file(partition_run.partition_key, completed_file, sender)?;
        }
        Ok(())
    }

    fn finish(&mut self, sender: &mut dyn Sender<FileOrderInput>) -> UnaryResult<bool> {
        for (partition_key, pending_file) in std::mem::take(&mut self.pending_files_by_partition) {
            self.emit_pending_file(partition_key, pending_file, sender)?;
        }
        Ok(true)
    }
}
