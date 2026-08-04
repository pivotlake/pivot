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
//! goes through [`BuildRowBatches`], [`split_row_id`], or [`first_row_id`].

use crate::arrays::accumulator::ChunkedColumns;
use arrow_array::RecordBatch;
use arrow_schema::ArrowError;

/// Rows a stored batch holds at most.
const ROWS_PER_BATCH: usize = crate::RECORD_BATCH_SIZE;
/// A row id is `build_row_batch_idx << BATCH_SHIFT | row_in_batch`.
const BATCH_SHIFT: u32 = ROWS_PER_BATCH.trailing_zeros();
/// How many batches the `u32` id encoding addresses.
const MAX_BATCHES: usize = (u32::MAX as usize + 1) >> BATCH_SHIFT;
const _: () = assert!(ROWS_PER_BATCH.is_power_of_two());

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

/// The build rows one worker stores during its consume phase, and, after
/// [`merge`](Self::merge), every worker's rows as the probe reads them.
#[derive(Default)]
pub(crate) struct BuildRowBatches {
    batches: Vec<RecordBatch>,
}

impl BuildRowBatches {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Store a batch's rows without copying and hand back each stored batch
    /// with its first row's id, for the caller to generate tuples from. Ids
    /// are local to this instance; a worker's ids are globalized with the
    /// base [`merge`](Self::merge) assigns it.
    pub(crate) fn adopt(&mut self, batch: RecordBatch) -> Vec<(u32, RecordBatch)> {
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
                self.batches.len() < MAX_BATCHES,
                "join build side exceeds the row id space"
            );
            adopted.push(((self.batches.len() << BATCH_SHIFT) as u32, stored.clone()));
            self.batches.push(stored);
            offset += rows;
        }
        adopted
    }

    /// Merge every worker's stored rows, in the given order, into the
    /// published whole. Returns the merged rows and each worker's row id
    /// base: the single `u32` added to that worker's local ids at scatter
    /// time.
    pub(crate) fn merge(workers: impl IntoIterator<Item = BuildRowBatches>) -> (Self, Vec<u32>) {
        let mut batches = Vec::new();
        let mut row_id_bases = Vec::new();
        for worker in workers {
            row_id_bases.push((batches.len() << BATCH_SHIFT) as u32);
            batches.extend(worker.batches);
        }
        assert!(
            batches.len() <= MAX_BATCHES,
            "join build side exceeds the row id space"
        );
        (Self { batches }, row_id_bases)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }

    /// The stored batches; a row id's batch half indexes this list.
    pub(crate) fn batches(&self) -> &[RecordBatch] {
        &self.batches
    }

    /// The id space the stored rows occupy, gaps included: what sizes the
    /// outer join's matched-flag array, indexed by row id.
    pub(crate) fn row_id_space(&self) -> usize {
        self.batches.len() << BATCH_SHIFT
    }

    /// The stored batches restricted to `columns`, paired with a gather
    /// source over them prepared for reading rows back by id.
    pub(crate) fn gather_source(
        &self,
        columns: &[usize],
    ) -> Result<(Vec<RecordBatch>, ChunkedColumns), ArrowError> {
        let projected: Vec<RecordBatch> = self
            .batches
            .iter()
            .map(|batch| batch.project(columns))
            .collect::<Result<_, _>>()?;
        let gather = ChunkedColumns::prepare(&projected, BATCH_SHIFT);
        Ok((projected, gather))
    }
}
