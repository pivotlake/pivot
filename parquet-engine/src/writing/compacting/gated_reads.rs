//! What the compacter and the gates share about the reads.
//!
//! A *read* is one column of one input row group. The compacter requests
//! it once, and the pipeline fetches the column's chunk of that row group
//! whole. Its pages then wait, compressed, at a
//! [`DecompressGate`](super::decompress_gate::DecompressGate), and its
//! ranges of rows wait, as decompressed pages, at a
//! [`DecodeGate`](super::decode_gate::DecodeGate). Every worker has a gate
//! of each kind, which holds what reached that worker.
//!
//! The compacter lets a read's rows through the gates by releasing them
//! ([`GatedReads::release_rows_before`]): every row before the released one
//! may be decompressed and decoded. The released row of each read is one
//! number here, which only the compacter writes and every gate reads.
//!
//! A gate that holds something registers its worker as waiting and then
//! looks at the released rows once more (see [`WaitingWorkers`]). After
//! releasing rows, the compacter wakes the waiting workers, and their gates
//! look again at what they hold.
//!
//! The `DecompressGate`s report back how many compressed bytes they have
//! let through.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use dispatch::waker::WaitingWorkers;

use crate::types::metadata::QueryRowGroupMetadata;

/// The reads of a compaction, as the compacter and the gates share them.
pub(super) struct GatedReads {
    /// How many columns the table has.
    column_count: usize,
    /// For each read, the row of its row group before which every row is
    /// released. Only the compacter writes it.
    released_rows_end: Vec<AtomicU32>,
    /// The workers whose `DecompressGate` holds a page that is not
    /// released.
    pub(super) waiting_decompress_gates: WaitingWorkers,
    /// The workers whose `DecodeGate` holds a range that is not released.
    pub(super) waiting_decode_gates: WaitingWorkers,
    /// The compressed size of every page that no longer waits compressed
    /// at a `DecompressGate`.
    let_through_compressed_bytes: AtomicUsize,
}

impl GatedReads {
    /// The reads of each of `column_count` columns of each of
    /// `row_group_count` row groups, none released yet, for gates on each of
    /// `worker_count` workers.
    pub(super) fn new(
        row_group_count: usize,
        column_count: usize,
        worker_count: usize,
    ) -> Arc<Self> {
        Arc::new(Self {
            column_count,
            released_rows_end: (0..row_group_count * column_count)
                .map(|_| AtomicU32::new(0))
                .collect(),
            waiting_decompress_gates: WaitingWorkers::new(worker_count),
            waiting_decode_gates: WaitingWorkers::new(worker_count),
            let_through_compressed_bytes: AtomicUsize::new(0),
        })
    }

    /// The number of the read of table column `column` of the row group
    /// at `row_group` in the table's list of row groups.
    pub(super) fn index_read(&self, row_group: usize, column: usize) -> usize {
        row_group * self.column_count + column
    }

    /// The number of the read that `metadata` is carried by.
    pub(super) fn find_read_index(&self, metadata: &QueryRowGroupMetadata) -> usize {
        let read_key = metadata.read_key();
        let column = read_key
            .column
            .expect("every read of a compaction names its column");
        self.index_read(read_key.row_group_index, column)
    }

    /// Let every row of `read` before `rows_end` be decompressed and
    /// decoded, now or once its pages have been fetched. The gates that
    /// hold some of those rows only look again once their workers are
    /// woken.
    pub(super) fn release_rows_before(&self, read: usize, rows_end: u32) {
        self.released_rows_end[read].store(rows_end, Ordering::SeqCst);
    }

    /// Whether row `row` of `read` is released.
    pub(super) fn is_row_released(&self, read: usize, row: u32) -> bool {
        row < self.released_rows_end[read].load(Ordering::SeqCst)
    }

    /// Record that a page of `compressed_bytes` no longer waits
    /// compressed at a `DecompressGate`: the gate let it through, or it
    /// arrived decompressed.
    pub(super) fn record_page_let_through(&self, compressed_bytes: usize) {
        self.let_through_compressed_bytes
            .fetch_add(compressed_bytes, Ordering::Relaxed);
    }

    /// The compressed size of every page that no longer waits compressed
    /// at a `DecompressGate`.
    pub(super) fn count_compressed_bytes_let_through(&self) -> usize {
        self.let_through_compressed_bytes.load(Ordering::Relaxed)
    }
}
