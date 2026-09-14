//! Turns rows fetched in row-group order back into a file in sorted order.
//!
//! The sort ran over the keys alone and left its order with the ordered
//! materializer as `(row group, row)` pairs. The fetched batches arrive
//! carrying the metadata columns that say which row group and rows each
//! holds, so once every batch is in, the order resolves to `(batch, row)`
//! over them and the file is ready for row-group planning.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, RecordBatch, UInt32Array};
use dispatch::{LocatedBatch, Sender, Topology, Unary, UnaryFactory, UnaryResult};

use super::error::WriteError;
use super::types::{FilePlan, FileRows, ReadyFile};
use crate::RowOrder;
use crate::reading::record_batch_metadata::{global_row_group, row_index};

pub(super) struct MaterializedFileCollectorFactory {
    order: RowOrder,
    partition: Option<crate::PartitionValues>,
    target_rows_per_group: usize,
    max_file_size: usize,
    node_id: usize,
}

pub(super) fn factories(
    order: RowOrder,
    partition: Option<crate::PartitionValues>,
    target_rows_per_group: usize,
    max_file_size: usize,
    topology: Topology,
) -> Vec<MaterializedFileCollectorFactory> {
    (0..topology.total_workers())
        .map(|worker| MaterializedFileCollectorFactory {
            order: order.clone(),
            partition: partition.clone(),
            target_rows_per_group,
            max_file_size,
            node_id: topology.node_of_worker(worker),
        })
        .collect()
}

impl UnaryFactory<RecordBatch, ReadyFile> for MaterializedFileCollectorFactory {
    type Unary = MaterializedFileCollector;

    fn build_unary(self) -> MaterializedFileCollector {
        MaterializedFileCollector {
            order: self.order,
            partition: self.partition,
            target_rows_per_group: self.target_rows_per_group,
            max_file_size: self.max_file_size,
            node_id: self.node_id,
            batches: Vec::new(),
        }
    }
}

pub(super) struct MaterializedFileCollector {
    order: RowOrder,
    partition: Option<crate::PartitionValues>,
    target_rows_per_group: usize,
    max_file_size: usize,
    node_id: usize,
    batches: Vec<RecordBatch>,
}

impl Unary<RecordBatch, ReadyFile> for MaterializedFileCollector {
    fn consume(
        &mut self,
        batch: RecordBatch,
        _sender: &mut dyn Sender<ReadyFile>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        if batch.num_rows() > 0 {
            self.batches.push(batch);
        }
        Ok(())
    }

    fn finish(&mut self, sender: &mut dyn Sender<ReadyFile>) -> UnaryResult<bool> {
        if self.batches.is_empty() {
            return Ok(true);
        }
        let order = self
            .order
            .get()
            .expect("the order is published before any row is fetched");
        // Where each row group\'s rows sit: the first row index of every batch
        // holding some of them, in row order.
        let mut batches_by_group: HashMap<u32, Vec<(u32, usize)>> = HashMap::new();
        for (batch_index, batch) in self.batches.iter().enumerate() {
            let groups = global_row_group(batch);
            let group = groups
                .values()
                .as_any()
                .downcast_ref::<UInt32Array>()
                .expect("row group ids are unsigned")
                .value(groups.run_ends().get_start_physical_index());
            let first_row = row_index(batch).value(0);
            batches_by_group
                .entry(group)
                .or_default()
                .push((first_row, batch_index));
        }
        for batches in batches_by_group.values_mut() {
            batches.sort_unstable();
        }
        let mapping: Vec<(u32, u32)> = order
            .iter()
            .map(|&(group, row)| {
                let batches = &batches_by_group[&group];
                let position = batches.partition_point(|&(first_row, _)| first_row <= row) - 1;
                let (first_row, batch_index) = batches[position];
                (batch_index as u32, row - first_row)
            })
            .collect();
        let data_columns = self.batches[0].num_columns() - 2;
        let batches = std::mem::take(&mut self.batches)
            .into_iter()
            .map(|batch| {
                let batch = batch
                    .project(&(0..data_columns).collect::<Vec<_>>())
                    .map_err(WriteError::from)?;
                Ok(LocatedBatch::new(batch, self.node_id))
            })
            .collect::<Result<Vec<_>, WriteError>>()?;
        sender.send(ReadyFile {
            plan: FilePlan {
                file_id: 0,
                base_row_group_id: 0,
                assembly_worker: 0,
                partition: self.partition.clone(),
                target_rows_per_group: self.target_rows_per_group,
                max_file_size: Some(self.max_file_size),
            },
            batches,
            row_count: mapping.len(),
            rows: FileRows::Mapped(Arc::new(mapping)),
            target_node: self.node_id,
        })?;
        Ok(true)
    }
}
