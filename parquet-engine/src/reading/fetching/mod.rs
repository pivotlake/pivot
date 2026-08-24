//! Row-group fetching pipeline: issuing IO requests and collecting raw column
//! buffers.
//!
//! This module turns [`RowGroupRequest`](crate::RowGroupRequest)s into fully
//! populated [`RowGroupBuffer`](crate::RowGroupBuffer)s by
//! reading column chunks from disk via io-uring, with file-cache lookups to
//! avoid redundant reads.
//!
//! - [`RowGroupInjectorFactory`] / `RowGroupInjector` — the root data source
//!   that enqueues every row group in a [`ParquetTable`](crate::ParquetTable)
//!   into a work-stealing [`Injector`](crossbeam_deque::Injector) so workers
//!   can pull row groups on demand.
//! - [`RowGroupFetcherFactory`] / [`fetcher::RowGroupFetcher`] — the unary
//!   operator that receives a `RowGroupRequest`, submits aligned IO requests
//!   for each projected column (disk or HTTP), collects completions, and emits
//!   the finished `RowGroupBuffer` downstream. It keeps disk- and HTTP-backed
//!   row groups outstanding under separate caps.

/// Builds one worker's [`RowGroupFetcher`], carrying that worker's
/// claimed-but-undecoded row-group count as claim backpressure, bounded by the
/// scan's claim bound (see [`fetcher::pending_claim_bound`]).
pub struct RowGroupFetcherFactory {
    pending_row_groups: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    max_pending_row_groups: usize,
}

impl RowGroupFetcherFactory {
    pub fn new(
        pending_row_groups: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        max_pending_row_groups: usize,
    ) -> Self {
        Self {
            pending_row_groups,
            max_pending_row_groups,
        }
    }
}

impl dispatch::UnaryFactory<crate::RowGroupRequest, crate::RowGroupBuffer>
    for RowGroupFetcherFactory
{
    type Unary = RowGroupFetcher;

    fn build_unary(self) -> RowGroupFetcher {
        RowGroupFetcher::new(self.pending_row_groups, self.max_pending_row_groups)
    }
}
mod fetcher;
pub use fetcher::pending_claim_bound;

mod table_source;
use crate::reading::fetching::fetcher::RowGroupFetcher;
pub use table_source::RowGroupInjectorFactory;
