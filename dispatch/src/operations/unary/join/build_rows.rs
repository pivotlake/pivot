//! The join's stored build rows, and the one `u64` id that addresses a row.
//!
//! Each build worker copies the rows it consumes out of the batches they
//! arrived in and into batches of its own, one set per hash partition, so
//! that a partition's rows are a unit of their own: the arriving batch is
//! released once copied, and nothing a partition holds is shared with another
//! partition. A worker's rows for one partition are batches of exactly
//! [`ROWS_PER_BATCH`] rows, plus a shorter last one. The merge lays every
//! worker's batches out partition by partition, so a partition's rows are a
//! contiguous run of the stored batches.
//!
//! A stored row's id packs which batch and which row within it into one
//! `u64`, so the build side is bounded by memory rather than by the id width;
//! a batch shorter than [`ROWS_PER_BATCH`] leaves a gap in the id space, which
//! costs nothing because ids are only gather addresses and are never required
//! to be dense.
//!
//! This module is the only place that knows how an id splits. Everything else
//! goes through the helpers below.

use crate::arrays::accumulator::{BatchAccumulator, ChunkedColumn};
use crate::memory::{MultiSlabBuffer, SlabAllocator};
use arrow_array::RecordBatch;
use arrow_schema::ArrowError;

/// Rows a stored batch holds at most.
const ROWS_PER_BATCH: usize = crate::RECORD_BATCH_SIZE;
/// A row id is `build_row_batch_idx << BATCH_SHIFT | row_in_batch`.
const BATCH_SHIFT: u32 = ROWS_PER_BATCH.trailing_zeros();
const _: () = assert!(ROWS_PER_BATCH.is_power_of_two());

/// One worker's stored build rows: per hash partition, the batches that
/// partition's rows were copied into.
pub(crate) type WorkerPartitionRows = Vec<Vec<RecordBatch>>;

/// Where one build worker copies its rows as they arrive, one accumulator
/// per hash partition. An accumulator is created the first time its
/// partition receives a row, so a small build reserves memory only for the
/// partitions it touches, and it is sealed into a stored batch each time it
/// fills to exactly [`ROWS_PER_BATCH`] rows.
pub(crate) struct WorkerRowStore {
    allocator: SlabAllocator,
    accumulators: Vec<Option<BatchAccumulator>>,
    rows: WorkerPartitionRows,
}

impl WorkerRowStore {
    pub(crate) fn new(partitions: usize) -> Self {
        Self {
            allocator: SlabAllocator::new(false),
            accumulators: (0..partitions).map(|_| None).collect(),
            rows: (0..partitions).map(|_| Vec::new()).collect(),
        }
    }

    /// Copy the rows of `batch` at `indices` (ascending positions) into
    /// `partition`. The rows go in as one or more runs, each ending where a
    /// stored batch fills; `stored` is told each run's indices along with the
    /// row id of its first row, the following rows of the run having the
    /// following ids. Ids are local to this worker; the merge assigns each
    /// worker and partition the base that globalizes them.
    pub(crate) fn store(
        &mut self,
        partition: usize,
        batch: &RecordBatch,
        mut indices: &[u32],
        mut stored: impl FnMut(&[u32], u64),
    ) -> Result<(), ArrowError> {
        let allocator = &mut self.allocator;
        let accumulator = self.accumulators[partition].get_or_insert_with(|| {
            BatchAccumulator::with_capacity(batch.schema(), ROWS_PER_BATCH, allocator)
        });
        let sealed = &mut self.rows[partition];
        while !indices.is_empty() {
            let room = ROWS_PER_BATCH - accumulator.len();
            let (run, rest) = indices.split_at(room.min(indices.len()));
            let first_row_id = (sealed.len() << BATCH_SHIFT | accumulator.len()) as u64;
            accumulator.append_batch_by_indices(batch, run, allocator);
            stored(run, first_row_id);
            if accumulator.len() == ROWS_PER_BATCH {
                sealed.push(accumulator.take_batch(allocator)?);
            }
            indices = rest;
        }
        Ok(())
    }

    /// Seal every partition's partial batch and hand the stored rows over.
    pub(crate) fn finish(mut self) -> Result<WorkerPartitionRows, ArrowError> {
        for (accumulator, sealed) in self.accumulators.iter_mut().zip(&mut self.rows) {
            if let Some(accumulator) = accumulator
                && !accumulator.is_empty()
            {
                sealed.push(accumulator.take_batch(&mut self.allocator)?);
            }
        }
        Ok(self.rows)
    }
}

/// The build-side rows after all workers have merged them, prepared once for
/// every probe worker to read.
pub(crate) struct BuildRows {
    /// Full build batches, partition by partition, used to verify keys whose
    /// stored representation is lossy.
    pub(crate) batches: Vec<RecordBatch>,
    /// The same batches restricted to the build columns the join emits, used
    /// when scanning unmatched rows one batch at a time.
    pub(crate) output_batches: Vec<RecordBatch>,
    /// Each output column prepared for gathering matched rows by row id.
    pub(crate) output_columns: Vec<ChunkedColumn>,
    /// One flag per row id for a build-side outer join.
    pub(crate) matched: MultiSlabBuffer<u8>,
}

impl BuildRows {
    /// Merge every worker's stored rows into the published whole, partition
    /// by partition and within a partition in worker order, and prepare them
    /// for probing. No bytes move in the merge. Alongside the rows, returns
    /// each worker's row id base per partition: the single `u64` added to
    /// that worker's local ids of that partition at scatter time.
    pub(crate) fn new<const TRACK_MATCHES: bool>(
        mut workers: Vec<WorkerPartitionRows>,
        partitions: usize,
        output_indices: &[usize],
    ) -> Result<(Self, Vec<Vec<u64>>), ArrowError> {
        let mut batches = Vec::new();
        let mut row_id_bases = vec![Vec::with_capacity(partitions); workers.len()];
        for partition in 0..partitions {
            for (worker_rows, bases) in workers.iter_mut().zip(&mut row_id_bases) {
                bases.push((batches.len() << BATCH_SHIFT) as u64);
                batches.append(&mut worker_rows[partition]);
            }
        }
        let output_batches: Vec<RecordBatch> = batches
            .iter()
            .map(|batch| batch.project(output_indices))
            .collect::<Result<_, _>>()?;
        let output_columns = (0..output_indices.len())
            .map(|column| {
                ChunkedColumn::new(
                    output_batches
                        .iter()
                        .map(|batch| batch.column(column).to_data())
                        .collect(),
                )
            })
            .collect();
        let matched = if TRACK_MATCHES {
            let mut allocator = SlabAllocator::new(false);
            allocator.create_multi_slab_buffer::<u8>(row_id_space(&batches).max(1), true)
        } else {
            MultiSlabBuffer::new(Vec::new())
        };
        Ok((
            Self {
                batches,
                output_batches,
                output_columns,
                matched,
            },
            row_id_bases,
        ))
    }

    pub(crate) fn empty() -> Self {
        Self {
            batches: Vec::new(),
            output_batches: Vec::new(),
            output_columns: Vec::new(),
            matched: MultiSlabBuffer::new(Vec::new()),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }
}

/// Split a row id into its batch index and the row within that batch.
#[inline(always)]
pub(crate) fn split_row_id(row_id: u64) -> (usize, usize) {
    (
        (row_id >> BATCH_SHIFT) as usize,
        (row_id as usize) & (ROWS_PER_BATCH - 1),
    )
}

/// The first row id of the batch at `build_row_batch_idx`.
#[inline(always)]
pub(crate) fn first_row_id(build_row_batch_idx: usize) -> usize {
    build_row_batch_idx << BATCH_SHIFT
}

/// How a row id splits, as the opaque shift the accumulator's gather takes.
#[inline(always)]
pub(crate) fn row_id_shift() -> u32 {
    BATCH_SHIFT
}

/// The id space the stored rows occupy, gaps included.
pub(crate) fn row_id_space(batches: &[RecordBatch]) -> usize {
    batches.len() << BATCH_SHIFT
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use arrow_array::Int64Array;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    use std::sync::Arc;

    fn int64_batch(values: impl Iterator<Item = i64>) -> RecordBatch {
        RecordBatch::try_from_iter([("v", Arc::new(Int64Array::from_iter_values(values)) as _)])
            .unwrap()
    }

    fn int64_values(batch: &RecordBatch) -> Vec<i64> {
        batch
            .column(0)
            .as_primitive::<Int64Type>()
            .values()
            .to_vec()
    }

    #[test]
    fn row_ids_split_past_the_u32_range() {
        let batch_idx = 1usize << 40;
        let row = 5usize;

        let row_id = (batch_idx << BATCH_SHIFT | row) as u64;

        assert_eq!(split_row_id(row_id), (batch_idx, row));
        assert_eq!(first_row_id(batch_idx), row_id as usize - row);
    }

    #[test]
    fn a_partition_fills_whole_batches_and_ids_follow_the_stored_order() {
        init_test_free_pool(8);
        let mut store = WorkerRowStore::new(2);
        let batch = int64_batch(0..(ROWS_PER_BATCH as i64 + 3));
        let odd_rows: Vec<u32> = (0..batch.num_rows() as u32)
            .filter(|i| i % 2 == 1)
            .collect();
        let even_rows: Vec<u32> = (0..batch.num_rows() as u32)
            .filter(|i| i % 2 == 0)
            .collect();
        let mut ids = Vec::new();

        store
            .store(1, &batch, &even_rows, |run, first_row_id| {
                ids.extend((0..run.len() as u64).map(|offset| first_row_id + offset))
            })
            .unwrap();
        store
            .store(1, &batch, &odd_rows, |run, first_row_id| {
                ids.extend((0..run.len() as u64).map(|offset| first_row_id + offset))
            })
            .unwrap();
        let rows = store.finish().unwrap();

        assert!(rows[0].is_empty());
        assert_eq!(rows[1][0].num_rows(), ROWS_PER_BATCH);
        assert_eq!(rows[1][1].num_rows(), 3);
        let stored: Vec<i64> = rows[1].iter().flat_map(int64_values).collect();
        let by_id: Vec<i64> = ids
            .iter()
            .map(|&id| {
                let (batch_idx, row) = split_row_id(id);
                int64_values(&rows[1][batch_idx])[row]
            })
            .collect();
        let expected: Vec<i64> = even_rows
            .iter()
            .chain(&odd_rows)
            .map(|&i| i as i64)
            .collect();
        assert_eq!(stored, expected);
        assert_eq!(by_id, expected);
    }

    #[test]
    fn the_merge_lays_partitions_out_contiguously_with_a_base_per_worker() {
        let workers = vec![
            vec![vec![int64_batch(0..2)], vec![]],
            vec![vec![int64_batch(10..11)], vec![int64_batch(20..23)]],
        ];

        let (rows, bases) = BuildRows::new::<false>(workers, 2, &[0]).unwrap();

        assert_eq!(rows.batches.len(), 3);
        assert_eq!(bases[0], vec![0, 2 << BATCH_SHIFT]);
        assert_eq!(bases[1], vec![1 << BATCH_SHIFT, 2 << BATCH_SHIFT]);
        let (batch_idx, row) = split_row_id(bases[1][1] + 2);
        assert_eq!(int64_values(&rows.batches[batch_idx])[row], 22);
    }
}
