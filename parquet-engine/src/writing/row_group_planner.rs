//! Divides ordered files into row groups and column jobs.
//!
//! Row-group boundaries can cross merge output batches. Batch slicing remains
//! zero-copy, and each column job is routed to the NUMA node that contributes
//! the most rows to its row group.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use dispatch::{Sender, Topology, Unary, UnaryFactory, UnaryResult};

use super::leaves;
use super::shredding;
use super::types::{
    ColumnChunkJob, FileAssemblyInfo, FileRows, ReadyFile, RowGroupContext, RowGroupRows,
};

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
        _io: &mut dispatch::OperatorIO,
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
        rows,
        row_count,
        target_node,
    } = file;
    debug_assert!(row_count > 0);
    let record_batches: Vec<RecordBatch> =
        batches.iter().map(|batch| batch.batch().clone()).collect();
    let shredding_plan = shredding::plan_file_shredding(&record_batches)?;
    // The file's leaves are numbered depth-first across its columns, so each
    // column's leaves start where the previous column's ended.
    let leaf_counts: Vec<usize> = shredding_plan
        .schema
        .fields()
        .iter()
        .map(|field| leaves::count_leaves(field))
        .collect();
    let leaf_count = leaf_counts.iter().sum();

    // A fixed-width leaf is materialized into one ring slab by the encoder.
    debug_assert!(
        plan.target_rows_per_group <= dispatch::BUFFER_SIZE / 16,
        "a row group's widest column must fit one slab"
    );
    let row_group_count = row_count.div_ceil(plan.target_rows_per_group);
    let file_info = Arc::new(FileAssemblyInfo {
        file_id: plan.file_id,
        row_group_count,
        partition: plan.partition.clone(),
        max_file_size: plan.max_file_size,
    });

    // Every row group's column job takes from the same per-column chunk list
    // and the same row order, so both are shared rather than sliced.
    let column_chunks: Vec<Arc<[ArrayRef]>> = (0..leaf_counts.len())
        .map(|column_index| {
            batches
                .iter()
                .map(|batch| batch.batch().column(column_index).clone())
                .collect()
        })
        .collect();
    let mapping = match rows {
        FileRows::InOrder => None,
        FileRows::Mapped(mapping) => {
            debug_assert_eq!(mapping.len(), row_count);
            Some(mapping)
        }
    };
    // Rows in order are cut at batch boundaries as the cursor walks them.
    let mut batch_index = 0;
    let mut rows_consumed_from_batch = 0;
    for row_group_index in 0..row_group_count {
        let first_row = row_group_index * plan.target_rows_per_group;
        let rows = first_row..(first_row + plan.target_rows_per_group).min(row_count);
        let (row_group_rows, target_node) = match &mapping {
            Some(mapping) => (
                RowGroupRows::Gather {
                    mapping: mapping.clone(),
                    rows,
                },
                target_node,
            ),
            None => {
                let mut runs = Vec::new();
                let mut rows_by_node = vec![0usize; node_count];
                let mut rows_remaining_in_group = rows.len();
                while rows_remaining_in_group > 0 {
                    let located_batch = &batches[batch_index];
                    let batch_rows = located_batch.batch().num_rows();
                    let rows_taken =
                        (batch_rows - rows_consumed_from_batch).min(rows_remaining_in_group);
                    runs.push((
                        batch_index,
                        rows_consumed_from_batch..rows_consumed_from_batch + rows_taken,
                    ));
                    rows_by_node[located_batch.node_id()] += rows_taken;
                    rows_consumed_from_batch += rows_taken;
                    rows_remaining_in_group -= rows_taken;
                    if rows_consumed_from_batch == batch_rows {
                        batch_index += 1;
                        rows_consumed_from_batch = 0;
                    }
                }
                (
                    RowGroupRows::Slices(runs),
                    dispatch::dominant_node(&rows_by_node),
                )
            }
        };
        let row_group_rows = Arc::new(row_group_rows);

        let context = Arc::new(RowGroupContext {
            row_group_id: plan.base_row_group_id + row_group_index as u64,
            assembly_worker: plan.assembly_worker,
            schema: shredding_plan.schema.clone(),
            leaf_count,
            file_info: file_info.clone(),
        });
        let mut first_leaf_index = 0;
        let mut jobs: Vec<ColumnChunkJob> = Vec::with_capacity(leaf_counts.len());
        for (column_index, &column_leaf_count) in leaf_counts.iter().enumerate() {
            jobs.push(ColumnChunkJob {
                context: context.clone(),
                column_index,
                first_leaf_index,
                chunks: column_chunks[column_index].clone(),
                rows: row_group_rows.clone(),
                shredding: shredding_plan.column_shredding[column_index].clone(),
                target_node,
            });
            first_leaf_index += column_leaf_count;
        }
        // Heaviest columns first, so the light ones fill in behind them
        // rather than a heavy one running alone at the row group's end.
        jobs.sort_by_key(|job| {
            std::cmp::Reverse(column_weight(shredding_plan.schema.field(job.column_index)))
        });
        for job in jobs {
            sender.send(job)?;
        }
    }
    Ok(())
}

/// A rough cost of taking and shredding a column, for ordering a row group's
/// jobs: a variant or byte-view column moves its values, a fixed-width one
/// only its cells.
fn column_weight(field: &arrow_schema::Field) -> usize {
    match field.data_type() {
        arrow_schema::DataType::Struct(_) => 3,
        arrow_schema::DataType::Utf8View | arrow_schema::DataType::BinaryView => 2,
        _ => 1,
    }
}
