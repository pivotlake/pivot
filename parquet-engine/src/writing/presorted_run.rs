//! Adapts compaction scan batches to the write pipeline's sorted-run input.

use arrow_array::{ArrayRef, RecordBatch};
use arrow_row::{OwnedRow, RowConverter, SortField};
use dispatch::{Sender, Topology, Unary, UnaryFactory, UnaryResult};

use super::partition_sorter::SortedPartitionRun;

/// Wraps compaction scan batches without partitioning, sorting, or copying
/// them. Each decoded batch already belongs to the selected partition and
/// retains the order of its source file; the downstream merge combines
/// overlapping runs when the table has a sort key.
pub(super) struct PresortedRun {
    partition_key: Option<OwnedRow>,
    source_node: usize,
}

impl PresortedRun {
    pub(super) fn create_for_workers(
        partition: Option<&crate::PartitionValues>,
        partition_column_names: &[String],
        topology: Topology,
    ) -> Vec<Self> {
        let partition_key = partition.map(|partition| {
            let columns: Vec<ArrayRef> = partition_column_names
                .iter()
                .map(|name| {
                    partition
                        .get(name)
                        .unwrap_or_else(|| panic!("partition has no value for column `{name}`"))
                        .clone()
                        .into_inner()
                })
                .collect();
            let converter = RowConverter::new(
                columns
                    .iter()
                    .map(|column| SortField::new(column.data_type().clone()))
                    .collect(),
            )
            .expect("partition columns have row-compatible types");
            converter
                .convert_columns(&columns)
                .expect("partition values match their column types")
                .row(0)
                .owned()
        });
        (0..topology.total_workers())
            .map(|worker_id| Self {
                partition_key: partition_key.clone(),
                source_node: topology.node_of_worker(worker_id),
            })
            .collect()
    }
}

impl UnaryFactory<RecordBatch, SortedPartitionRun> for PresortedRun {
    type Unary = Self;

    fn build_unary(self) -> Self {
        self
    }
}

impl Unary<RecordBatch, SortedPartitionRun> for PresortedRun {
    fn consume(
        &mut self,
        batch: RecordBatch,
        sender: &mut dyn Sender<SortedPartitionRun>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        let _tagged = dispatch::memory::tagged(dispatch::memory::MemoryTag::Collect);
        if batch.num_rows() > 0 {
            let in_memory_bytes = batch.get_array_memory_size();
            sender.send(SortedPartitionRun {
                partition_key: self.partition_key.clone(),
                batches: vec![batch],
                in_memory_bytes,
                source_node: self.source_node,
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow_array::Int64Array;
    use arrow_schema::{DataType, Field, Schema};
    use dispatch::test_utils::CollectSender;

    #[test]
    fn compaction_wraps_a_batch_without_copying_or_reordering_it() {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![9, 5, 3, 8]))])
                .unwrap();
        let in_memory_bytes = batch.get_array_memory_size();
        let input_values = batch.column(0).clone();
        let mut wrapper = PresortedRun::create_for_workers(None, &[], Topology::single_node(1))
            .pop()
            .unwrap()
            .build_unary();
        let mut sender = CollectSender::new();

        wrapper
            .consume(
                batch,
                &mut sender,
                &mut dispatch::TestOperatorIO::default().io(),
            )
            .unwrap();

        assert_eq!(sender.items.len(), 1);
        assert!(sender.items[0].partition_key.is_none());
        assert_eq!(sender.items[0].in_memory_bytes, in_memory_bytes);
        assert!(Arc::ptr_eq(
            sender.items[0].batches[0].column(0),
            &input_values
        ));
    }
}
