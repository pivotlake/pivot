//! What the compacter, the gates and the cutters share about the reads.
//!
//! A *read* is one column of one input row group. The compacter requests
//! it once, and the pipeline fetches the column's chunk of that row group
//! whole. Its pages then wait, compressed, for a
//! [`DecompressGate`](super::decompress_gate::DecompressGate) to let them
//! on, and its rows wait, as decompressed pages, for a
//! [`ReleasedRowsCutter`](super::released_rows_cutter::ReleasedRowsCutter)
//! to cut them. Every worker has one of each, for what reached that
//! worker.
//!
//! The compacter lets a read's rows through by releasing them
//! ([`GatedReads::release_rows_before`]): every row before the released one
//! may be decompressed and decoded. The released row of each read is one
//! number here, which only the compacter writes and every gate and cutter
//! reads.
//!
//! A gate or cutter that holds something registers its worker as waiting
//! before it looks at the released rows (see [`WaitingWorkers`]). After
//! releasing rows, the compacter wakes the waiting workers, and they look
//! again at what they hold.
//!
//! The `DecompressGate`s report back how many compressed bytes they have
//! let through.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use dispatch::waker::WaitingWorkers;

use crate::reading::DECODE_RANGE_ROWS;
use crate::types::metadata::QueryRowGroupMetadata;

/// How many rows of a sort column are released, and cut for the decoders,
/// at a time. The merge compares the keys of the same rows of every sort
/// column, so the rows of each are cut into ranges of the same rows, in
/// whatever order their pages and releases arrive. Keys are held until
/// their rows are merged and sent on, which is long when many files are
/// merged together, so the ranges are short: each of their arrays keeps
/// the block of memory it was built in.
pub(crate) const SORT_KEY_RANGE_ROWS: u32 = 2048;

/// The reads of a compaction, as the compacter, the gates and the cutters
/// share them.
pub(crate) struct GatedReads {
    /// How many columns the table has.
    column_count: usize,
    /// For each column of the table, whether the files are sorted by it.
    is_sort_column: Vec<bool>,
    /// For each row group, how many of its reads have rows that are not
    /// cut for the decoders yet.
    uncut_reads_per_row_group: Vec<AtomicUsize>,
    /// For each read, the row of its row group before which every row is
    /// released. Only the compacter writes it.
    released_rows_end: Vec<AtomicU32>,
    /// The workers whose `DecompressGate` has a page waiting that is not
    /// released.
    pub(crate) waiting_decompress_gates: WaitingWorkers,
    /// The workers whose `ReleasedRowsCutter` has rows waiting that are
    /// not released.
    pub(crate) waiting_cutters: WaitingWorkers,
    /// The compressed size of every page that no longer waits compressed
    /// at a `DecompressGate`.
    let_through_compressed_bytes: AtomicUsize,
}

impl GatedReads {
    /// `sort_columns` are the table columns the files are sorted by.
    /// The reads of each of `column_count` columns of each of
    /// `row_group_count` row groups, none released yet, for gates on each of
    /// `worker_count` workers.
    pub(crate) fn new(
        row_group_count: usize,
        column_count: usize,
        worker_count: usize,
        sort_columns: &[usize],
    ) -> Arc<Self> {
        Arc::new(Self {
            column_count,
            is_sort_column: (0..column_count)
                .map(|column| sort_columns.contains(&column))
                .collect(),
            uncut_reads_per_row_group: (0..row_group_count)
                .map(|_| AtomicUsize::new(column_count))
                .collect(),
            released_rows_end: (0..row_group_count * column_count)
                .map(|_| AtomicU32::new(0))
                .collect(),
            waiting_decompress_gates: WaitingWorkers::new(worker_count),
            waiting_cutters: WaitingWorkers::new(worker_count),
            let_through_compressed_bytes: AtomicUsize::new(0),
        })
    }

    /// The number of the read of table column `column` of the row group
    /// at `row_group` in the table's list of row groups.
    pub(crate) fn index_read(&self, row_group: usize, column: usize) -> usize {
        row_group * self.column_count + column
    }

    /// The number of the read that `metadata` is carried by.
    pub(crate) fn find_read_index(&self, metadata: &QueryRowGroupMetadata) -> usize {
        let read_key = metadata.read_key();
        let column = read_key
            .column
            .expect("every read of a compaction names its column");
        self.index_read(read_key.row_group_index, column)
    }

    /// Let every row of `read` before `rows_end` be decompressed and
    /// decoded, now or once its pages have been fetched. The gates and
    /// cutters that hold some of those rows only look again once their
    /// workers are woken.
    pub(crate) fn release_rows_before(&self, read: usize, rows_end: u32) {
        self.released_rows_end[read].store(rows_end, Ordering::SeqCst);
    }

    /// The most rows of `read` that are cut into one range for the
    /// decoders: [`SORT_KEY_RANGE_ROWS`] of a sort column, and as many as a
    /// scan decodes in one go of any other.
    pub(crate) fn find_range_rows(&self, read: usize) -> u32 {
        match self.is_sort_column[read % self.column_count] {
            true => SORT_KEY_RANGE_ROWS,
            false => DECODE_RANGE_ROWS,
        }
    }

    /// Record that every row of `read` was cut for the decoders. Returns
    /// whether it was the last read of its row group of which that was
    /// still to happen: nothing will read the row group's bytes again.
    pub(crate) fn record_read_cut(&self, read: usize) -> bool {
        let uncut_reads = &self.uncut_reads_per_row_group[read / self.column_count];
        uncut_reads.fetch_sub(1, Ordering::AcqRel) == 1
    }

    /// The row of `read` before which every row is released.
    pub(crate) fn find_released_rows_end(&self, read: usize) -> u32 {
        self.released_rows_end[read].load(Ordering::SeqCst)
    }

    /// Whether row `row` of `read` is released.
    pub(crate) fn is_row_released(&self, read: usize, row: u32) -> bool {
        row < self.released_rows_end[read].load(Ordering::SeqCst)
    }

    /// Record that a page of `compressed_bytes` no longer waits at a
    /// `DecompressGate`: the gate let it through, or asked for it again
    /// because a row of it was released.
    pub(crate) fn record_page_let_through(&self, compressed_bytes: usize) {
        self.let_through_compressed_bytes
            .fetch_add(compressed_bytes, Ordering::Relaxed);
    }

    /// The compressed size of every page that no longer waits compressed
    /// at a `DecompressGate`.
    pub(crate) fn count_compressed_bytes_let_through(&self) -> usize {
        self.let_through_compressed_bytes.load(Ordering::Relaxed)
    }
}
