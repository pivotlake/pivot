//! What the input files of a compaction share to get their columns read.
//!
//! An [`InputFile`](super::input_file::InputFile) fetches, decompresses and
//! decodes nothing itself. It asks through the [`ReadRequests`]: a request
//! for a column chunk goes into the pipeline's request queue, and a release
//! of rows goes to the gates (see [`GatedReads`]). The memory those hold is
//! counted here against the compaction's one [`MemoryBudget`].

use std::sync::Arc;

use dispatch::ChannelInputSender;

use crate::ParquetTable;
use crate::types::metadata::QueryRowGroupMetadata;
use crate::writing::compacting::memory_budget::MemoryBudget;
use crate::writing::compacting::read::pipeline::gated_reads::GatedReads;

/// The requests and releases of every input file.
pub(crate) struct ReadRequests {
    table: Arc<ParquetTable>,
    /// The queue the requests for column chunks go into.
    request_sender: ChannelInputSender<QueryRowGroupMetadata>,
    gated_reads: Arc<GatedReads>,
    pub(crate) memory_budget: Arc<MemoryBudget>,
    /// The compressed size of every chunk requested so far.
    requested_compressed_bytes: usize,
    /// The most compressed bytes that may be requested and not yet let
    /// through to be decompressed, once a chunk fetched ahead is among
    /// them.
    fetch_ahead_bytes: usize,
    /// Whether rows were released since the gates' workers were last
    /// woken.
    has_releases_to_wake_gates_for: bool,
}

impl ReadRequests {
    /// Nothing requested yet of the files of `table`. Requests go to the
    /// pipeline through `request_sender`, which must have room for one
    /// request per column of every row group of the table. Chunks are
    /// fetched ahead within `fetch_ahead_bytes` (see
    /// [`has_room_to_fetch_ahead`](Self::has_room_to_fetch_ahead)).
    pub(crate) fn new(
        table: Arc<ParquetTable>,
        gated_reads: Arc<GatedReads>,
        memory_budget: Arc<MemoryBudget>,
        request_sender: ChannelInputSender<QueryRowGroupMetadata>,
        fetch_ahead_bytes: usize,
    ) -> Self {
        Self {
            requested_compressed_bytes: 0,
            fetch_ahead_bytes,
            has_releases_to_wake_gates_for: false,
            table,
            request_sender,
            gated_reads,
            memory_budget,
        }
    }

    /// Request the chunk of table column `column` of the row group at
    /// `row_group` in the table's list of row groups, which is
    /// `compressed_bytes` in the file. The pipeline fetches it whole, and
    /// its size is counted as held from here on.
    pub(crate) fn request_column(
        &mut self,
        row_group: usize,
        column: usize,
        compressed_bytes: usize,
    ) {
        self.requested_compressed_bytes += compressed_bytes;
        self.memory_budget.add(compressed_bytes);
        let request = QueryRowGroupMetadata::new_of_column(&self.table, row_group, column);
        self.request_sender
            .try_send(request)
            .unwrap_or_else(|_| panic!("the request queue has room for every read"));
    }

    /// Whether a chunk of `compressed_bytes` may be requested before its
    /// rows are to be decoded.
    ///
    /// It may while the memory budget has room for it and, with it, the
    /// compressed bytes that were requested and not yet let through to be
    /// decompressed stay within the limit on fetching ahead. Those bytes
    /// are on their way from disk or wait at the `DecompressGate`.
    pub(crate) fn has_room_to_fetch_ahead(&self, compressed_bytes: usize) -> bool {
        let waiting_bytes =
            self.requested_compressed_bytes - self.gated_reads.count_compressed_bytes_let_through();
        waiting_bytes + compressed_bytes <= self.fetch_ahead_bytes
            && self.memory_budget.has_room_for(compressed_bytes)
    }

    /// Let every row before `rows_end` of table column `column` of the row
    /// group at `row_group` be decompressed and decoded. The rows that this
    /// newly lets through are counted as held from here on, at
    /// `decoded_bytes`, the most their decoded values can own. The gates
    /// and cutters that hold some of those rows learn of it when
    /// [`wake_gates_after_releases`](Self::wake_gates_after_releases) is
    /// called.
    pub(crate) fn release_rows_before(
        &mut self,
        row_group: usize,
        column: usize,
        rows_end: u32,
        decoded_bytes: usize,
    ) {
        self.memory_budget.add(decoded_bytes);
        self.gated_reads
            .release_rows_before(self.gated_reads.index_read(row_group, column), rows_end);
        self.has_releases_to_wake_gates_for = true;
    }

    /// If rows were released since the last call, wake the workers whose
    /// gates and cutters hold pages, so that they look again at what is
    /// released.
    pub(crate) fn wake_gates_after_releases(&mut self) {
        if std::mem::take(&mut self.has_releases_to_wake_gates_for) {
            let waker_set = dispatch::waker::waker_set();
            self.gated_reads
                .waiting_decompress_gates
                .wake_all(waker_set);
            self.gated_reads.waiting_cutters.wake_all(waker_set);
        }
    }

    /// Tell the pipeline that no request follows, which lets the dataflow
    /// end.
    pub(crate) fn close(&self) {
        self.request_sender.close();
    }
}
