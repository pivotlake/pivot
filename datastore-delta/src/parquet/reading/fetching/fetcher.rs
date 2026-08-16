//! The [`RowGroupFetcher`]: turns claimed row groups into fully read buffers.
//!
//! It accepts [`RowGroupRequest`]s (logical claims - which row group, which
//! leaf columns) and stages each as one pending IO request on its node's
//! [`OperatorIO`]. The worker resolves the request against the caches, reads
//! whatever is missing (local files via io_uring disk reads, remote objects
//! via HTTP range reads on the same ring), and hands the resolved bytes back
//! through `process_io_response`, where the fetcher emits the
//! [`RowGroupBuffer`] downstream.
//!
//! All cache resolution, read deduplication, and in-flight bounding live in
//! the worker's request tracker; the fetcher's only jobs are claim
//! backpressure (don't hoard undecoded row groups) and assembling the buffer
//! from the response.

use crate::parquet::types::requests::{RowGroupBuffer, RowGroupRequest};
use dispatch::Sender;
use dispatch::Unary;
use dispatch::io::{CompletedIoRequest, OperatorIO};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Row groups a worker may hold "claimed but not yet fully decoded" before it
/// stops claiming more, when the table's files are local: the one it is
/// decoding plus one prefetched next, the minimum that keeps its pipeline
/// overlapped. A row group's page decode runs only on the worker that claimed
/// it (the decoder holds per-row-group state), so every claim beyond this
/// bound serializes decode work on one worker while others may sit idle. That
/// matters most when a scan has few row groups per worker: its wall time is
/// `max claims per worker x per-row-group cost`, and a single worker hoarding
/// a third row group while the queue runs dry sets the whole query's critical
/// path.
const MAX_PENDING_LOCAL_ROW_GROUPS: usize = 2;

/// The remote-file counterpart of [`MAX_PENDING_LOCAL_ROW_GROUPS`]. A remote
/// claim spends most of its life waiting on the network, not decoding, so the
/// hoarding concern above barely applies while read-ahead depth is the whole
/// throughput game: a claim's reads must be in flight long before its decode
/// is due, or every row group pays a full network round trip serially.
const MAX_PENDING_REMOTE_ROW_GROUPS: usize = 16;

/// How many undecoded row groups one worker may hold claims on while
/// scanning `table`. Local tables keep claims minimal for decode fairness;
/// tables with any remote file get read-ahead depth.
pub fn pending_claim_bound(table: &crate::parquet::ParquetTable) -> usize {
    let any_remote = table
        .row_groups()
        .iter()
        .any(|rg| matches!(rg.open_file, dispatch::io::OpenFile::Remote(_)));
    if any_remote {
        MAX_PENDING_REMOTE_ROW_GROUPS
    } else {
        MAX_PENDING_LOCAL_ROW_GROUPS
    }
}

pub struct RowGroupFetcher {
    /// Requests staged but not yet answered - claims whose bytes are still
    /// being resolved. `finish` waits for this to drain.
    unanswered_requests: usize,
    /// How many row groups this worker has claimed but not fully decoded.
    /// Incremented here per claim, decremented by the worker's `Decoder` when
    /// a row group finishes (or prunes), and consulted as claim backpressure.
    pending_row_groups: Arc<AtomicUsize>,
    /// The claim bound for this scan (see [`pending_claim_bound`]).
    max_pending_row_groups: usize,
}

impl RowGroupFetcher {
    pub fn new(pending_row_groups: Arc<AtomicUsize>, max_pending_row_groups: usize) -> Self {
        Self {
            unanswered_requests: 0,
            pending_row_groups,
            max_pending_row_groups,
        }
    }
}

impl Unary<RowGroupRequest, RowGroupBuffer> for RowGroupFetcher {
    fn consume(
        &mut self,
        request: RowGroupRequest,
        _sender: &mut dyn Sender<RowGroupBuffer>,
        io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        self.pending_row_groups.fetch_add(1, Ordering::Relaxed);
        self.unanswered_requests += 1;
        io.push(request.into_pending_io());
        Ok(())
    }

    fn process_io_response(
        &mut self,
        sender: &mut dyn Sender<RowGroupBuffer>,
        _io: &mut OperatorIO,
        response: CompletedIoRequest,
    ) -> dispatch::UnaryResult<()> {
        self.unanswered_requests -= 1;
        let metadata = *response
            .state
            .downcast()
            .expect("a row group request rides its metadata as state");
        sender.send(RowGroupBuffer::from_response(metadata, response.ranges))?;
        Ok(())
    }

    fn ready_for_more_work(&mut self) -> bool {
        self.pending_row_groups.load(Ordering::Relaxed) < self.max_pending_row_groups
    }

    fn finish(&mut self, _sender: &mut dyn Sender<RowGroupBuffer>) -> dispatch::UnaryResult<bool> {
        Ok(self.unanswered_requests == 0)
    }
}
