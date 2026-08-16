//! Divides ordered files into row groups and column-encoding jobs.
//!
//! Row-group boundaries can cross merge output batches. Batch slicing remains
//! zero-copy, and each column job is routed to the NUMA node that contributes
//! the most rows to its row group.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use dispatch::{Sender, Topology, Unary, UnaryFactory, UnaryResult};

use super::shredding;
use super::types::{ColumnChunkJob, FileAssemblyInfo, ReadyFile, RowGroupContext};

pub(super) struct RowGroupPlannerFactory {
    node_count: usize,
}

pub(super) fn factories(topology: Topology) -> Vec<RowGroupPlannerFactory> {
    (0..topology.total_workers())
        .map(|_| RowGroupPlannerFactory {
            node_count: topology.node_count,
        })
        .collect()
}

impl UnaryFactory<ReadyFile, ColumnChunkJob> for RowGroupPlannerFactory {
    type Unary = RowGroupPlanner;

    fn build_unary(self) -> Self::Unary {
        RowGroupPlanner {
            node_count: self.node_count,
        }
    }
}

pub(super) struct RowGroupPlanner {
    node_count: usize,
}

impl Unary<ReadyFile, ColumnChunkJob> for RowGroupPlanner {
    fn consume(
        &mut self,
        file: ReadyFile,
        sender: &mut dyn Sender<ColumnChunkJob>,
    ) -> UnaryResult<()> {
        emit_column_chunk_jobs(file, self.node_count, sender)
    }
}

fn emit_column_chunk_jobs(
    file: ReadyFile,
    node_count: usize,
    sender: &mut dyn Sender<ColumnChunkJob>,
) -> UnaryResult<()> {
    let ReadyFile {
        plan,
        batches,
        row_count,
        ..
    } = file;
    debug_assert!(row_count > 0);
    let record_batches: Vec<RecordBatch> =
        batches.iter().map(|batch| batch.batch().clone()).collect();
    let shredding_plan = shredding::plan_file_shredding(&record_batches)?;

    // A fixed-width column is materialized into one ring slab by the encoder.
    debug_assert!(
        plan.target_rows_per_group <= dispatch::BUFFER_SIZE / 16,
        "a row group's widest column must fit one slab"
    );
    let row_group_count = row_count.div_ceil(plan.target_rows_per_group);
    let file_info = Arc::new(FileAssemblyInfo {
        file_id: plan.file_id,
        row_group_count,
        partition: plan.partition.clone(),
    });

    let mut rows_remaining = row_count;
    let mut batch_index = 0;
    let mut rows_consumed_from_batch = 0;
    for row_group_index in 0..row_group_count {
        let mut rows_remaining_in_group = plan.target_rows_per_group.min(rows_remaining);
        rows_remaining -= rows_remaining_in_group;
        let mut row_group_batches = Vec::new();
        let mut rows_by_node = vec![0usize; node_count];
        while rows_remaining_in_group > 0 {
            let located_batch = &batches[batch_index];
            let batch = located_batch.batch();
            let rows_available = batch.num_rows() - rows_consumed_from_batch;
            let rows_taken = rows_available.min(rows_remaining_in_group);
            row_group_batches.push(
                if rows_consumed_from_batch == 0 && rows_taken == batch.num_rows() {
                    batch.clone()
                } else {
                    batch.slice(rows_consumed_from_batch, rows_taken)
                },
            );
            rows_by_node[located_batch.node_id()] += rows_taken;
            rows_consumed_from_batch += rows_taken;
            rows_remaining_in_group -= rows_taken;
            if rows_consumed_from_batch == batch.num_rows() {
                batch_index += 1;
                rows_consumed_from_batch = 0;
            }
        }
        let target_node = rows_by_node
            .iter()
            .enumerate()
            .max_by_key(|(_, rows)| *rows)
            .map_or(0, |(node_id, _)| node_id);

        let context = Arc::new(RowGroupContext {
            row_group_id: plan.base_row_group_id + row_group_index as u64,
            assembly_worker: plan.assembly_worker,
            schema: shredding_plan.schema.clone(),
            file_info: file_info.clone(),
        });
        for column_index in 0..shredding_plan.schema.fields().len() {
            let column_batches: Arc<[ArrayRef]> = row_group_batches
                .iter()
                .map(|batch| batch.column(column_index).clone())
                .collect();
            sender.send(ColumnChunkJob {
                context: context.clone(),
                column_index,
                batches: column_batches,
                shredding: shredding_plan.column_shredding[column_index].clone(),
                target_node,
            })?;
        }
    }
    Ok(())
}
