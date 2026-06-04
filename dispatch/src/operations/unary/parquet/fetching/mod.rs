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
//!   `RowGroupBuffer` downstream.

pub type RowGroupFetcherFactory = DefaultUnaryFactory<RowGroupFetcher>;
mod fetcher;

mod table_source;
use crate::operations::DefaultUnaryFactory;
use crate::operations::unary::parquet::fetching::fetcher::RowGroupFetcher;
pub use table_source::{RowGroupFilter, RowGroupInjectorFactory};
