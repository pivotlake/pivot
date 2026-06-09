//! Parallel row-group **metadata** fetch — the table-load pipeline, run once at
//! `CREATE`/`ATTACH` (the reading pipeline scans the row groups it produces).
//! A small work-stealing dataflow with one module per stage:
//!
//! - [`injector`] — the source: hands out the input files (with their indices).
//! - [`fetcher`] — reads each file's footer (over the io_uring ring, through the
//!   file cache) on whatever worker steals it, emitting its row groups.
//! - [`writer`] — the terminal fan-in sink: gathers the row groups on one worker,
//!   assembles the [`ParquetTable`], and commits it.
//!
//! It is consumed two ways, both over the same fetch core ([`fetch_factories`] +
//! [`assemble`]): [`load_table`] collects on the coordinator and returns a
//! [`ParquetTable`] **value** (the catalog's reload and the test/bench
//! constructors), while [`create_load_and_commit_spec`] returns a
//! `RecordBatchOperatorSpec` ending in the [`writer`] sink that assembles the
//! table and hands it to a `commit` closure — the `CREATE TABLE` the server
//! executes.

mod fetcher;
mod injector;
mod writer;

use crate::parquet::types::metadata::RowGroupMetadata;
use crate::parquet::types::table::ParquetTable;
use crate::store::DataFileLocation;
use dispatch::{
    DataFlowDispatcher, DefaultUnaryFactory, OperatorSpec, RecordBatchOperatorSpec,
    RootUnaryOperatorFactory, fan_in,
};
use fetcher::RowGroupMetadataFetcher;
use injector::FileInjectorFactory;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use writer::TableBuildSinkFactory;

/// A file location tagged with its position in the input list, so the row groups
/// can be numbered in file order regardless of which worker reads which footer.
type IndexedFile = (usize, DataFileLocation);
/// A row group tagged with the index of the file it came from.
type IndexedRowGroup = (usize, RowGroupMetadata);

/// The per-worker source→fetch factories: each worker steals files from a shared
/// injector and reads their footers, emitting one [`IndexedRowGroup`] per group.
fn fetch_factories(
    files: &[DataFileLocation],
    workers: usize,
) -> Vec<
    RootUnaryOperatorFactory<
        IndexedFile,
        IndexedRowGroup,
        DefaultUnaryFactory<RowGroupMetadataFetcher>,
        FileInjectorFactory,
    >,
> {
    let injector = FileInjectorFactory::new(files);
    let siblings = Arc::new(AtomicUsize::new(workers));
    (0..workers)
        .map(|_| {
            RootUnaryOperatorFactory::new(
                DefaultUnaryFactory::<RowGroupMetadataFetcher>::new(),
                injector.clone(),
                siblings.clone(),
            )
        })
        .collect()
}

/// Order the gathered row groups by `(file order, file-internal order)` and
/// assign each its global index, so the result is identical regardless of how
/// the parallel fetch interleaved.
fn assemble(mut rows: Vec<IndexedRowGroup>) -> Vec<Arc<RowGroupMetadata>> {
    rows.sort_by_key(|(file_idx, rg)| (*file_idx, rg.file_row_group_idx));
    rows.into_iter()
        .enumerate()
        .map(|(global_idx, (_, mut rg))| {
            rg.global_row_group_idx = global_idx;
            Arc::new(rg)
        })
        .collect()
}

/// Materialize a list of data files into a [`ParquetTable`] value on the
/// coordinator: read every footer in parallel, gather, and assemble. A pipeline
/// breaker — it drives a dataflow, so it must run on the coordinator (a nested
/// dataflow would deadlock a worker). The value path, used by the catalog's
/// reload and the `ParquetTable` test/bench constructors; `CREATE TABLE` uses
/// [`create_load_and_commit_spec`] instead.
pub fn load_table(
    dispatcher: &DataFlowDispatcher,
    files: &[DataFileLocation],
) -> Result<ParquetTable, dispatch::DataFlowError> {
    if files.is_empty() {
        return Ok(ParquetTable::new(Vec::new()));
    }
    let n = dispatcher.worker_count().max(1);
    let collected = OperatorSpec::new(dispatcher.clone(), fetch_factories(files, n)).collect()?;
    Ok(ParquetTable::new(assemble(collected)))
}

/// A `RecordBatchOperatorSpec` that, when executed, reads every file's footer in
/// parallel and — at its terminal stage — assembles the [`ParquetTable`] and
/// hands it to `commit` (which runs once, on the worker that finishes last, and
/// returns an error to fail the statement). Emits no rows. This is `CREATE TABLE`
/// as a single dataflow: fetch, then commit (the `commit` records the table in
/// the manifest and the catalog map).
pub fn create_load_and_commit_spec<C>(
    dispatcher: &DataFlowDispatcher,
    files: &[DataFileLocation],
    commit: C,
) -> RecordBatchOperatorSpec
where
    C: FnOnce(
            Arc<ParquetTable>,
        ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>
        + Send
        + 'static,
{
    let n = dispatcher.worker_count().max(1);
    let fetch = OperatorSpec::new(dispatcher.clone(), fetch_factories(files, n));

    // `fan_in` funnels every worker's row groups to worker 0; only worker 0 gets
    // the commit, so it alone builds and writes the table (the rest no-op). No
    // shared state — the channel closing is the "all fetched" signal.
    let mut commit = Some(commit);
    let sinks: Vec<_> = (0..n)
        .map(|_| TableBuildSinkFactory::new(commit.take()))
        .collect();
    RecordBatchOperatorSpec::from_spec(fetch.chain(fan_in::<IndexedRowGroup>(n), sinks))
}
