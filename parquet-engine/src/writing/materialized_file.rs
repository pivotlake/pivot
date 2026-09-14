//! Lays fetched rows out in sorted order, a batch at a time.
//!
//! The sort ran over the keys alone and left its order with the materializer
//! as `(row group, row)` pairs. The fetched batches arrive carrying the
//! metadata columns that say which row group and rows each holds, roughly in
//! the order the rows are needed. The collector keeps them, and whenever the
//! next stretch of the order is wholly resident it gathers that stretch into
//! one sorted batch and lets go of every row group the stretch finished
//! with. The sorted batches make the file, in order, so the rows are copied
//! once and the decoded input shrinks as the copy grows.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, RecordBatch, UInt32Array};
use arrow_schema::Schema;
use dispatch::arrays::take_chunked;
use dispatch::memory::SlabAllocator;
use dispatch::{
    LocatedBatch, RECORD_BATCH_SIZE, Sender, Topology, Unary, UnaryFactory, UnaryResult,
};

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
            resident: HashMap::new(),
            rows_left: HashMap::new(),
            cursor: 0,
            sorted: Vec::new(),
            allocator: None,
            schema: None,
        }
    }
}

/// One fetched batch of a row group: its first row index, and the batch.
struct ResidentBatch {
    first_row: u32,
    batch: RecordBatch,
}

pub(super) struct MaterializedFileCollector {
    order: RowOrder,
    partition: Option<crate::PartitionValues>,
    target_rows_per_group: usize,
    max_file_size: usize,
    node_id: usize,
    /// The fetched batches of each row group still needed, by first row.
    resident: HashMap<u32, Vec<ResidentBatch>>,
    /// Per row group, how many of its rows the cursor has yet to pass.
    rows_left: HashMap<u32, usize>,
    /// The next row of the order to lay out.
    cursor: usize,
    /// The file so far, in order.
    sorted: Vec<RecordBatch>,
    allocator: Option<SlabAllocator>,
    /// The data columns\' schema, without the metadata columns.
    schema: Option<Arc<Schema>>,
}

impl MaterializedFileCollector {
    /// Lay out every stretch of the order that is wholly resident.
    fn advance(&mut self) -> Result<(), WriteError> {
        let order = self
            .order
            .get()
            .expect("the order is published before any row is fetched");
        if self.rows_left.is_empty() {
            for &(group, _) in order {
                *self.rows_left.entry(group).or_default() += 1;
            }
        }
        while self.cursor < order.len() {
            let end = (self.cursor + RECORD_BATCH_SIZE).min(order.len());
            let stretch = &order[self.cursor..end];
            // Each row of the stretch as (resident batch, row in it), with the
            // batches numbered as the gather will see them.
            let mut chunks: Vec<(u32, usize)> = Vec::new();
            let mut chunk_of: HashMap<(u32, usize), u32> = HashMap::new();
            let mut mapping = Vec::with_capacity(stretch.len());
            for &(group, row) in stretch {
                let Some(batches) = self.resident.get(&group) else {
                    return Ok(());
                };
                let position = batches.partition_point(|batch| batch.first_row <= row);
                if position == 0 {
                    return Ok(());
                }
                let position = position - 1;
                let batch = &batches[position];
                if row >= batch.first_row + batch.batch.num_rows() as u32 {
                    return Ok(());
                }
                let chunk = *chunk_of.entry((group, position)).or_insert_with(|| {
                    chunks.push((group, position));
                    chunks.len() as u32 - 1
                });
                mapping.push((chunk, row - batch.first_row));
            }
            let allocator = self
                .allocator
                .get_or_insert_with(|| SlabAllocator::new(false));
            let schema = self
                .schema
                .clone()
                .expect("a resident batch set the schema");
            let mut columns = Vec::with_capacity(schema.fields().len());
            for column in 0..schema.fields().len() {
                let sources: Vec<_> = chunks
                    .iter()
                    .map(|&(group, position)| {
                        self.resident[&group][position].batch.column(column).clone()
                    })
                    .collect();
                columns.push(take_chunked(allocator, &sources, &mapping)?);
            }
            self.sorted.push(RecordBatch::try_new(schema, columns)?);
            for &(group, _) in stretch {
                let left = self
                    .rows_left
                    .get_mut(&group)
                    .expect("every row group of the order is counted");
                *left -= 1;
                if *left == 0 {
                    self.resident.remove(&group);
                }
            }
            self.cursor = end;
        }
        Ok(())
    }
}

impl Unary<RecordBatch, ReadyFile> for MaterializedFileCollector {
    fn consume(
        &mut self,
        batch: RecordBatch,
        _sender: &mut dyn Sender<ReadyFile>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let groups = global_row_group(&batch);
        let group = groups
            .values()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("row group ids are unsigned")
            .value(groups.run_ends().get_start_physical_index());
        let first_row = row_index(&batch).value(0);
        if self.schema.is_none() {
            let data_columns = batch.num_columns() - 2;
            self.schema = Some(Arc::new(Schema::new(
                batch.schema().fields()[..data_columns].to_vec(),
            )));
        }
        let batches = self.resident.entry(group).or_default();
        let at = batches.partition_point(|resident| resident.first_row < first_row);
        batches.insert(at, ResidentBatch { first_row, batch });
        self.advance()?;
        Ok(())
    }

    fn finish(&mut self, sender: &mut dyn Sender<ReadyFile>) -> UnaryResult<bool> {
        if self.schema.is_none() {
            return Ok(true);
        }
        self.advance()?;
        let order_len = self.order.get().map_or(0, Vec::len);
        assert_eq!(
            self.cursor, order_len,
            "every row of the order was fetched before the file finished"
        );
        let batches: Vec<LocatedBatch> = std::mem::take(&mut self.sorted)
            .into_iter()
            .map(|batch| LocatedBatch::new(batch, self.node_id))
            .collect();
        sender.send(ReadyFile {
            plan: FilePlan {
                file_id: 0,
                base_row_group_id: 0,
                assembly_worker: 0,
                partition: self.partition.clone(),
                target_rows_per_group: self.target_rows_per_group,
                max_file_size: Some(self.max_file_size),
            },
            row_count: order_len,
            batches,
            rows: FileRows::InOrder,
            target_node: self.node_id,
        })?;
        Ok(true)
    }
}
