//! Splits each input batch by table partition and sorts the rows within every
//! resulting batch.
//!
//! The output batches own their Arrow buffers. This copy is important because
//! input batches are often small views over decoded row-group buffers. The file
//! collector retains output while it forms files; retaining an input view would
//! keep the larger decoded allocation alive as well.
//!
//! A worker retains no input across `consume` calls. During a call it may hold
//! the input, a partition-clustered batch, and the independently owned output
//! batches derived from them.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_row::{OwnedRow, RowConverter, SortField};

use dispatch::arrays::take::{take, take_chunked};
use dispatch::memory::SlabAllocator;
use dispatch::{OrderBy, Sender, Topology, Unary, UnaryFactory, UnaryResult, batch_sort_indices};

/// One independently owned sorted run for a single table partition.
pub(super) struct SortedPartitionRun {
    /// Arrow's comparable encoding of the partition columns. `None` represents
    /// an unpartitioned table.
    pub(super) partition_key: Option<OwnedRow>,
    /// Batch-sized chunks of one sorted run.
    pub(super) batches: Vec<RecordBatch>,
    /// Approximate retained size of `batches`, used by the file-cut policy.
    pub(super) in_memory_bytes: usize,
    pub(super) source_node: usize,
}

pub(super) struct PartitionSorterFactory {
    partition_column_indices: Arc<[usize]>,
    order_by: Arc<[OrderBy]>,
    source_node: usize,
}

impl PartitionSorterFactory {
    pub(super) fn create_for_workers(
        partition_column_indices: Vec<usize>,
        order_by: Vec<OrderBy>,
        topology: Topology,
    ) -> Vec<PartitionSorterFactory> {
        let partition_column_indices: Arc<[usize]> = partition_column_indices.into();
        let order_by: Arc<[OrderBy]> = order_by.into();
        (0..topology.total_workers())
            .map(|worker_id| PartitionSorterFactory {
                partition_column_indices: partition_column_indices.clone(),
                order_by: order_by.clone(),
                source_node: topology.node_of_worker(worker_id),
            })
            .collect()
    }
}

impl UnaryFactory<RecordBatch, SortedPartitionRun> for PartitionSorterFactory {
    type Unary = PartitionSorter;

    fn build_unary(self) -> PartitionSorter {
        PartitionSorter {
            partition_column_indices: self.partition_column_indices,
            order_by: self.order_by,
            source_node: self.source_node,
            partition_key_converter: None,
            allocator: None,
        }
    }
}

pub(super) struct PartitionSorter {
    partition_column_indices: Arc<[usize]>,
    order_by: Arc<[OrderBy]>,
    source_node: usize,
    /// Initialized from the first nonempty batch because the key types come
    /// from its schema.
    partition_key_converter: Option<RowConverter>,
    allocator: Option<SlabAllocator>,
}

impl Unary<RecordBatch, SortedPartitionRun> for PartitionSorter {
    fn consume(
        &mut self,
        batch: RecordBatch,
        sender: &mut dyn Sender<SortedPartitionRun>,
    ) -> UnaryResult<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        for (partition_key, partition_batch) in self.split_by_partition(batch)? {
            let batches = self.copy_in_sort_order(partition_batch)?;
            let in_memory_bytes = batches.iter().map(RecordBatch::get_array_memory_size).sum();
            sender.send(SortedPartitionRun {
                partition_key,
                batches,
                in_memory_bytes,
                source_node: self.source_node,
            })?;
        }
        Ok(())
    }
}

impl PartitionSorter {
    /// Returns one batch per partition key. If the input contains several
    /// partitions, rows are clustered before the batch is sliced.
    fn split_by_partition(
        &mut self,
        batch: RecordBatch,
    ) -> UnaryResult<Vec<(Option<OwnedRow>, RecordBatch)>> {
        if self.partition_column_indices.is_empty() {
            return Ok(vec![(None, batch)]);
        }
        let partition_columns: Vec<ArrayRef> = self
            .partition_column_indices
            .iter()
            .map(|&column_index| batch.column(column_index).clone())
            .collect();
        let partition_key_converter = match &self.partition_key_converter {
            Some(converter) => converter,
            None => self.partition_key_converter.insert(
                RowConverter::new(
                    partition_columns
                        .iter()
                        .map(|column| SortField::new(column.data_type().clone()))
                        .collect(),
                )
                .map_err(dispatch::UnaryError::from)?,
            ),
        };
        let partition_keys = partition_key_converter
            .convert_columns(&partition_columns)
            .map_err(dispatch::UnaryError::from)?;

        let has_one_partition =
            (1..batch.num_rows()).all(|row| partition_keys.row(row) == partition_keys.row(0));
        if has_one_partition {
            return Ok(vec![(Some(partition_keys.row(0).owned()), batch)]);
        }

        let mut partition_row_order: Vec<u32> = (0..batch.num_rows() as u32).collect();
        partition_row_order.sort_unstable_by(|&left, &right| {
            partition_keys
                .row(left as usize)
                .cmp(&partition_keys.row(right as usize))
        });
        let allocator = self
            .allocator
            .get_or_insert_with(|| SlabAllocator::new(false));
        let output_columns = batch
            .columns()
            .iter()
            .map(|column| take(allocator, column, &partition_row_order))
            .collect::<Result<Vec<ArrayRef>, _>>()?;
        let clustered_batch = RecordBatch::try_new(batch.schema(), output_columns)
            .map_err(dispatch::UnaryError::from)?;

        let mut partition_batches = Vec::new();
        let mut partition_start = 0;
        for row in 1..=partition_row_order.len() {
            let partition_ends = row == partition_row_order.len()
                || partition_keys.row(partition_row_order[row] as usize)
                    != partition_keys.row(partition_row_order[partition_start] as usize);
            if partition_ends {
                partition_batches.push((
                    Some(
                        partition_keys
                            .row(partition_row_order[partition_start] as usize)
                            .owned(),
                    ),
                    clustered_batch.slice(partition_start, row - partition_start),
                ));
                partition_start = row;
            }
        }
        Ok(partition_batches)
    }

    /// Copies a single-partition batch into independently owned, batch-sized
    /// outputs in sort order.
    fn copy_in_sort_order(&mut self, batch: RecordBatch) -> UnaryResult<Vec<RecordBatch>> {
        let sorted_rows = batch_sort_indices(&self.order_by, &batch)?;
        let gather_order: Vec<(u32, u32)> = sorted_rows.into_iter().map(|row| (0, row)).collect();

        let allocator = self
            .allocator
            .get_or_insert_with(|| SlabAllocator::new(false));
        gather_order
            .chunks(dispatch::RECORD_BATCH_SIZE)
            .map(|batch_rows| {
                let output_columns = batch
                    .columns()
                    .iter()
                    .map(|column| take_chunked(allocator, std::slice::from_ref(column), batch_rows))
                    .collect::<Result<Vec<ArrayRef>, _>>()?;
                RecordBatch::try_new(batch.schema(), output_columns)
                    .map_err(dispatch::UnaryError::from)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, Int64Array, StringViewArray};
    use arrow_schema::{DataType, Field, Schema};
    use dispatch::memory::init_test_free_pool;
    use dispatch::test_utils::CollectSender;

    fn two_column_batch(partitions: &[i64], keys: &[i64]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("partition", DataType::Int64, false),
            Field::new("key", DataType::Int64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(partitions.to_vec())),
                Arc::new(Int64Array::from(keys.to_vec())),
            ],
        )
        .unwrap()
    }

    fn keys_of(run: &SortedPartitionRun) -> Vec<i64> {
        run.batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect()
    }

    fn build_partition_sorter() -> PartitionSorter {
        PartitionSorterFactory::create_for_workers(
            vec![0],
            vec![OrderBy::new(1, false, true)],
            Topology::single_node(1),
        )
        .pop()
        .unwrap()
        .build_unary()
    }

    #[test]
    fn a_mixed_batch_produces_one_sorted_batch_per_partition() {
        init_test_free_pool(16);
        let mut partition_sorter = build_partition_sorter();
        let mut sender = CollectSender::new();

        partition_sorter
            .consume(two_column_batch(&[2, 1, 2, 1], &[9, 5, 3, 8]), &mut sender)
            .unwrap();

        let mut partition_runs: Vec<(Vec<i64>, usize)> = sender
            .items
            .iter()
            .map(|batch| {
                (
                    keys_of(batch),
                    batch.batches.iter().map(RecordBatch::num_rows).sum(),
                )
            })
            .collect();
        partition_runs.sort();
        assert_eq!(partition_runs, vec![(vec![3, 9], 2), (vec![5, 8], 2)]);
        assert!(sender.items.iter().all(|batch| batch.in_memory_bytes > 0));
    }

    #[test]
    fn output_does_not_retain_the_input_arrays() {
        init_test_free_pool(16);
        let mut partition_sorter = build_partition_sorter();
        let mut sender = CollectSender::new();
        let batch = two_column_batch(&[1, 1], &[1, 2]);

        partition_sorter
            .consume(batch.clone(), &mut sender)
            .unwrap();

        assert_eq!(sender.items.len(), 1);
        let output = &sender.items[0].batches[0];
        assert!(
            !Arc::ptr_eq(output.column(1), batch.column(1)),
            "an already sorted batch must still receive independent buffers"
        );
        assert_eq!(keys_of(&sender.items[0]), vec![1, 2]);
    }

    #[test]
    fn output_copies_string_view_payloads() {
        init_test_free_pool(16);
        let schema = Arc::new(Schema::new(vec![
            Field::new("partition", DataType::Int64, false),
            Field::new("key", DataType::Utf8View, false),
        ]));
        let input_strings = Arc::new(StringViewArray::from(vec![
            "a string long enough to use an external buffer",
            "another string long enough to use an external buffer",
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 1])),
                input_strings.clone(),
            ],
        )
        .unwrap();
        let mut partition_sorter = PartitionSorterFactory::create_for_workers(
            vec![0],
            vec![OrderBy::new(1, false, true)],
            Topology::single_node(1),
        )
        .pop()
        .unwrap()
        .build_unary();
        let mut sender = CollectSender::new();

        partition_sorter.consume(batch, &mut sender).unwrap();

        let output_strings = sender.items[0].batches[0].column(1).to_data();
        let input_strings = input_strings.to_data();
        assert_ne!(
            output_strings.buffers()[1].as_ptr(),
            input_strings.buffers()[1].as_ptr(),
            "retained string data must not point into the input batch"
        );
    }
}
