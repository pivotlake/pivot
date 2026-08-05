//! The join's stored build rows, and the one `u32` id that addresses a row.
//!
//! The build side keeps its rows as the batches they arrived in, without
//! copying: a batch of at most [`ROWS_PER_BATCH`] rows is stored as it is, a
//! larger one as zero-copy slices. A stored row's id packs which batch and
//! which row within it into one `u32`; a batch shorter than
//! [`ROWS_PER_BATCH`] leaves a gap in the id space, which costs nothing
//! because ids are only gather addresses and are never required to be dense.
//!
//! This module is the only place that knows how an id splits. Everything else
//! goes through the helpers below.

use crate::memory::{MultiSlabBuffer, SlabAllocator};
use arrow::array::ArrayData;
use arrow_array::RecordBatch;
use arrow_schema::ArrowError;

/// Rows a stored batch holds at most.
const ROWS_PER_BATCH: usize = crate::RECORD_BATCH_SIZE;
/// A row id is `build_row_batch_idx << BATCH_SHIFT | row_in_batch`.
const BATCH_SHIFT: u32 = ROWS_PER_BATCH.trailing_zeros();
/// How many batches the `u32` id encoding addresses.
const MAX_BATCHES: usize = (u32::MAX as usize + 1) >> BATCH_SHIFT;
const _: () = assert!(ROWS_PER_BATCH.is_power_of_two());

/// The build-side rows after all workers have merged them, prepared once for
/// every probe worker to read.
pub(crate) struct BuildRows {
    /// Full build batches, used to verify keys whose stored representation is
    /// lossy.
    pub(crate) batches: Vec<RecordBatch>,
    /// The same batches restricted to the build columns the join emits, used
    /// when scanning unmatched rows one batch at a time.
    pub(crate) output_batches: Vec<RecordBatch>,
    /// Each output column's per-batch Arrow data, prepared for gathering
    /// matched rows by row id.
    pub(crate) output_columns: Vec<Vec<ArrayData>>,
    /// One flag per row id for a build-side outer join.
    pub(crate) matched: MultiSlabBuffer<u8>,
}

impl BuildRows {
    pub(crate) fn new<const TRACK_MATCHES: bool>(
        batches: Vec<RecordBatch>,
        output_indices: &[usize],
    ) -> Result<Self, ArrowError> {
        let output_batches: Vec<RecordBatch> = batches
            .iter()
            .map(|batch| batch.project(output_indices))
            .collect::<Result<_, _>>()?;
        let output_columns = (0..output_indices.len())
            .map(|column| {
                output_batches
                    .iter()
                    .map(|batch| batch.column(column).to_data())
                    .collect()
            })
            .collect();
        let matched = if TRACK_MATCHES {
            let mut allocator = SlabAllocator::new(false);
            allocator.create_multi_slab_buffer::<u8>(row_id_space(&batches).max(1), true)
        } else {
            MultiSlabBuffer::new(Vec::new())
        };
        Ok(Self {
            batches,
            output_batches,
            output_columns,
            matched,
        })
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
pub(crate) fn split_row_id(row_id: u32) -> (usize, usize) {
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

/// Store a batch's rows without copying and hand back each stored batch with
/// its first row's id, for the caller to generate tuples from. Ids are local
/// to `batches`; a worker's ids are globalized with the base [`merge`] assigns
/// it.
pub(crate) fn adopt(batches: &mut Vec<RecordBatch>, batch: RecordBatch) -> Vec<(u32, RecordBatch)> {
    let total = batch.num_rows();
    if total == 0 {
        return Vec::new();
    }
    let mut adopted = Vec::new();
    let mut offset = 0;
    while offset < total {
        let rows = (total - offset).min(ROWS_PER_BATCH);
        let stored = if rows == total {
            batch.clone()
        } else {
            batch.slice(offset, rows)
        };
        assert!(
            batches.len() < MAX_BATCHES,
            "join build side exceeds the row id space"
        );
        adopted.push(((batches.len() << BATCH_SHIFT) as u32, stored.clone()));
        batches.push(stored);
        offset += rows;
    }
    adopted
}

/// Merge every worker's stored rows, in the given order, into the published
/// whole. Returns the merged rows and each worker's row id base: the single
/// `u32` added to that worker's local ids at scatter time.
pub(crate) fn merge(
    workers: impl IntoIterator<Item = Vec<RecordBatch>>,
) -> (Vec<RecordBatch>, Vec<u32>) {
    let mut batches = Vec::new();
    let mut row_id_bases = Vec::new();
    for worker in workers {
        row_id_bases.push((batches.len() << BATCH_SHIFT) as u32);
        batches.extend(worker);
    }
    assert!(
        batches.len() <= MAX_BATCHES,
        "join build side exceeds the row id space"
    );
    (batches, row_id_bases)
}

/// The id space the stored rows occupy, gaps included.
fn row_id_space(batches: &[RecordBatch]) -> usize {
    batches.len() << BATCH_SHIFT
}
