//! The streaming front of the Parquet write pipeline: split, sort, ship.
//!
//! [`PartitionerFactory`] builds one [`Partitioner`] per worker. Each batch
//! that arrives is split into one piece per distinct partition tuple it
//! holds, each piece's rows are put in key order, and every piece ships
//! downstream as its own [`SortedPiece`] the moment it is cut. The stage
//! buffers nothing: grouping pieces into files is entirely the consumer's
//! job (the write pipeline's collector), so the memory a write holds lives
//! in exactly one place, governed by one threshold.
//!
//! A shipped piece is copied into self-contained slab-backed arrays rather
//! than referencing the arriving batch: pieces are retained downstream while
//! files gather, and a batch is typically a slice into much larger decoded
//! buffers (a scan's whole row-group column, a view column's data buffers),
//! so holding thousands of such slices would pin an amplified working set.
//! The compact copy costs exactly the rows' own bytes and lets the source
//! memory recycle as soon as the batch is consumed.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_row::{OwnedRow, RowConverter, SortField};

use crate::arrays::take::{take, take_chunked};
use crate::memory::SlabAllocator;
use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::factory::UnaryFactory;
use crate::operations::unary::order_by_limit::OrderBy;

use super::keys::{KeyOrdering, RunRow, SelectedKeyOrdering, select_key_ordering};
use super::merge::{batch_arrives_sorted, sorted_row_indices};
use super::with_key_ordering;

/// One shipped piece: a batch's worth of rows of a single partition, in key
/// order, self-contained in its own memory. Pieces of one partition arrive in
/// no particular order relative to each other; each is sorted within itself.
pub struct SortedPiece {
    /// The partition's encoded tuple, comparable and hashable so a consumer
    /// can group pieces by it; `None` when the write has no partition keys
    /// and everything is one partition.
    pub tuple: Option<OwnedRow>,
    pub chunk: RecordBatch,
    /// The chunk's memory footprint, so a consumer gathering pieces toward a
    /// byte budget need not re-measure them.
    pub bytes: usize,
}

/// Builds one worker's [`Partitioner`]; the workers share nothing, every
/// instance splitting and sorting whatever batches reach it.
pub struct PartitionerFactory {
    partition_keys: Arc<[usize]>,
    order_by: Arc<[OrderBy]>,
}

impl PartitionerFactory {
    /// One factory per worker. Rows split by their `partition_keys` tuple and
    /// sort by `order_by` within each partition; every sorted piece ships the
    /// moment it is cut, so a worker buffers nothing between batches.
    pub fn create_for_workers(
        partition_keys: Vec<usize>,
        order_by: Vec<OrderBy>,
        worker_count: usize,
    ) -> Vec<PartitionerFactory> {
        let partition_keys: Arc<[usize]> = partition_keys.into();
        let order_by: Arc<[OrderBy]> = order_by.into();
        (0..worker_count)
            .map(|_| PartitionerFactory {
                partition_keys: partition_keys.clone(),
                order_by: order_by.clone(),
            })
            .collect()
    }
}

impl UnaryFactory<RecordBatch, SortedPiece> for PartitionerFactory {
    type Unary = Partitioner;

    fn build_unary(self) -> Partitioner {
        Partitioner {
            partition_keys: self.partition_keys,
            order_by: self.order_by,
            partition_tuple_converter: None,
            allocator: None,
        }
    }
}

/// One worker's streaming splitter-sorter: every piece ships as it is cut.
pub struct Partitioner {
    partition_keys: Arc<[usize]>,
    order_by: Arc<[OrderBy]>,
    /// Encodes a row's partition columns into its comparable tuple; built from
    /// the first batch's schema.
    partition_tuple_converter: Option<RowConverter>,
    /// Ring memory the self-contained copies land in. Taken on first use.
    allocator: Option<SlabAllocator>,
}

impl unary::Unary<RecordBatch, SortedPiece> for Partitioner {
    fn consume(
        &mut self,
        batch: RecordBatch,
        sender: &mut dyn Sender<SortedPiece>,
    ) -> unary::Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        for (tuple, piece) in self.split_by_partition(batch)? {
            for window in self.compact_sorted_windows(piece)? {
                sender.send(SortedPiece {
                    tuple: tuple.clone(),
                    bytes: window.get_array_memory_size(),
                    chunk: window,
                })?;
            }
        }
        Ok(())
    }
}

impl Partitioner {
    /// Cut `batch` into one single-partition piece per distinct tuple it
    /// holds. A batch of one tuple (always, when there are no partition keys)
    /// passes through whole; a mixed batch is clustered by its tuples first —
    /// stably, so each partition keeps its arrival order — and sliced.
    fn split_by_partition(
        &mut self,
        batch: RecordBatch,
    ) -> unary::Result<Vec<(Option<OwnedRow>, RecordBatch)>> {
        if self.partition_keys.is_empty() {
            return Ok(vec![(None, batch)]);
        }
        let partition_columns: Vec<ArrayRef> = self
            .partition_keys
            .iter()
            .map(|&column| batch.column(column).clone())
            .collect();
        let converter = match &self.partition_tuple_converter {
            Some(converter) => converter,
            None => self.partition_tuple_converter.insert(
                RowConverter::new(
                    partition_columns
                        .iter()
                        .map(|column| SortField::new(column.data_type().clone()))
                        .collect(),
                )
                .map_err(unary::Error::from)?,
            ),
        };
        let tuples = converter
            .convert_columns(&partition_columns)
            .map_err(unary::Error::from)?;

        let single_tuple = (1..batch.num_rows()).all(|row| tuples.row(row) == tuples.row(0));
        if single_tuple {
            return Ok(vec![(Some(tuples.row(0).owned()), batch)]);
        }

        // Cluster the rows by tuple, each tuple's rows keeping arrival order,
        // and slice one piece per tuple off the clustered copy.
        let mut clustered_order: Vec<u32> = (0..batch.num_rows() as u32).collect();
        clustered_order.sort_unstable_by(|&a, &b| {
            tuples
                .row(a as usize)
                .cmp(&tuples.row(b as usize))
                .then(a.cmp(&b))
        });
        let allocator = self
            .allocator
            .get_or_insert_with(|| SlabAllocator::new(false));
        let columns = batch
            .columns()
            .iter()
            .map(|column| take(allocator, column, &clustered_order))
            .collect::<Result<Vec<ArrayRef>, _>>()?;
        let clustered =
            RecordBatch::try_new(batch.schema(), columns).map_err(unary::Error::from)?;

        let mut pieces = Vec::new();
        let mut piece_start = 0;
        for row in 1..=clustered_order.len() {
            let piece_ends = row == clustered_order.len()
                || tuples.row(clustered_order[row] as usize)
                    != tuples.row(clustered_order[piece_start] as usize);
            if piece_ends {
                pieces.push((
                    Some(tuples.row(clustered_order[piece_start] as usize).owned()),
                    clustered.slice(piece_start, row - piece_start),
                ));
                piece_start = row;
            }
        }
        Ok(pieces)
    }

    /// The piece's rows in key order as batch-sized self-contained copies,
    /// sharing nothing with the piece (see the module docs for why). Bounding
    /// each window to a batch's worth of rows also keeps a copy's column
    /// within one slab, however large the arriving batch was.
    fn compact_sorted_windows(&mut self, piece: RecordBatch) -> unary::Result<Vec<RecordBatch>> {
        let chunks = std::slice::from_ref(&piece);
        let selected = select_key_ordering(&self.order_by, chunks, chunks)?;
        let arrives_sorted = with_key_ordering!(selected, |ordering| batch_arrives_sorted(
            &mut ordering,
            piece.num_rows()
        ));
        let mapping: Vec<(u32, u32)> = if arrives_sorted {
            (0..piece.num_rows() as u32).map(|row| (0, row)).collect()
        } else {
            let selected = select_key_ordering(&self.order_by, chunks, chunks)?;
            let sorted_indices = with_key_ordering!(selected, |ordering| sorted_row_indices(
                &mut ordering,
                piece.num_rows()
            ));
            sorted_indices.into_iter().map(|row| (0, row)).collect()
        };

        let allocator = self
            .allocator
            .get_or_insert_with(|| SlabAllocator::new(false));
        mapping
            .chunks(crate::RECORD_BATCH_SIZE)
            .map(|window| {
                let columns = piece
                    .columns()
                    .iter()
                    .map(|column| take_chunked(allocator, std::slice::from_ref(column), window))
                    .collect::<Result<Vec<ArrayRef>, _>>()?;
                RecordBatch::try_new(piece.schema(), columns).map_err(unary::Error::from)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use crate::operations::unary::Unary;
    use arrow_array::Int64Array;
    use arrow_schema::{DataType, Field, Schema};

    struct CollectingSender(Vec<SortedPiece>);

    impl Sender<SortedPiece> for CollectingSender {
        fn send(&mut self, item: SortedPiece) -> crate::operations::channels::Result<()> {
            self.0.push(item);
            Ok(())
        }
    }

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

    fn keys_of(piece: &SortedPiece) -> Vec<i64> {
        piece
            .chunk
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .to_vec()
    }

    fn build_partitioner() -> Partitioner {
        PartitionerFactory::create_for_workers(vec![0], vec![OrderBy::new(1, false, true)], 1)
            .pop()
            .unwrap()
            .build_unary()
    }

    #[test]
    fn a_mixed_batch_ships_one_sorted_piece_per_partition() {
        init_test_free_pool(16);
        let mut partitioner = build_partitioner();
        let mut sender = CollectingSender(Vec::new());

        partitioner
            .consume(two_column_batch(&[2, 1, 2, 1], &[9, 5, 3, 8]), &mut sender)
            .unwrap();

        let mut pieces: Vec<(Vec<i64>, usize)> = sender
            .0
            .iter()
            .map(|piece| (keys_of(piece), piece.chunk.num_rows()))
            .collect();
        pieces.sort();
        assert_eq!(pieces, vec![(vec![3, 9], 2), (vec![5, 8], 2)]);
        assert!(sender.0.iter().all(|piece| piece.bytes > 0));
    }

    #[test]
    fn a_shipped_piece_shares_nothing_with_the_arriving_batch() {
        init_test_free_pool(16);
        let mut partitioner = build_partitioner();
        let mut sender = CollectingSender(Vec::new());
        let batch = two_column_batch(&[1, 1], &[1, 2]);

        partitioner.consume(batch.clone(), &mut sender).unwrap();

        assert_eq!(sender.0.len(), 1, "each batch ships immediately");
        let shipped = &sender.0[0].chunk;
        assert!(
            !Arc::ptr_eq(shipped.column(1), batch.column(1)),
            "an already-sorted piece still copies into its own memory"
        );
        assert_eq!(keys_of(&sender.0[0]), vec![1, 2]);
    }
}
