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
//!   for each projected column (disk or HTTP), collects completions, and emits
//!   the finished `RowGroupBuffer` downstream. It keeps disk- and HTTP-backed
//!   row groups outstanding under separate caps.

pub type RowGroupFetcherFactory = DefaultUnaryFactory<RowGroupFetcher>;
mod fetcher;

mod table_source;
use crate::parquet::reading::fetching::fetcher::RowGroupFetcher;
use dispatch::DefaultUnaryFactory;
pub use table_source::RowGroupInjectorFactory;
