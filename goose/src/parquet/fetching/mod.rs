//! Row-group fetching pipeline: issuing IO requests and collecting raw column
//! buffers.
//!
//! This module turns [`RowGroupRequest`](super::RowGroupRequest)s into fully
//! populated [`RowGroupBuffer`](super::types::requests::RowGroupBuffer)s by
//! reading column chunks from disk via io-uring, with file-cache lookups to
//! avoid redundant reads.
//!
//! - [`RowGroupInjectorFactory`] / `RowGroupInjector` — the root data source
//!   that enqueues every row group in a [`ParquetTable`](super::types::table::ParquetTable)
//!   into a work-stealing [`Injector`](crossbeam_deque::Injector) so workers
//!   can pull row groups on demand.
//! - [`RowGroupFetcherFactory`] / [`fetcher::RowGroupFetcher`] — the unary
//!   operator that receives a `RowGroupRequest`, submits aligned IO requests
//!   for each projected column, collects completions, and emits the finished
//!   `RowGroupBuffer` downstream. Used for **local** (disk) tables.
//! - [`RemoteRowGroupFetcherFactory`] / [`remote_fetcher::RemoteRowGroupFetcher`]
//!   — the same role for **remote** (HTTP) tables, but keeping many row groups
//!   in flight per worker to hide network latency.

/// Disk fetcher: one row group at a time per worker (io-uring depth).
pub type RowGroupFetcherFactory = DefaultUnaryFactory<RowGroupFetcher>;
/// HTTP fetcher: many row groups in flight per worker (latency hiding).
pub type RemoteRowGroupFetcherFactory = DefaultUnaryFactory<RemoteRowGroupFetcher>;
mod fetcher;
mod remote_fetcher;

mod table_source;
use crate::parquet::fetching::fetcher::RowGroupFetcher;
use crate::parquet::fetching::remote_fetcher::RemoteRowGroupFetcher;
use dispatch::DefaultUnaryFactory;
pub use table_source::RowGroupInjectorFactory;
