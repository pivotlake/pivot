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
//! It is consumed two ways, both over the same fetch core ([`fetch_row_group_metadata_factories`]
//! and [`LoadedFiles::assemble`]): [`LoadedFiles::load`] collects on the
//! coordinator and returns the per-file row groups as a **value** (the
//! catalog's reload and registrations; flattened via
//! [`LoadedFiles::into_table`] for whole-table constructors), while
//! [`create_load_and_commit_spec`] returns a `RecordBatchOperatorSpec` ending
//! in the [`writer`] sink that hands the `LoadedFiles` to a `commit` closure —
//! the `CREATE TABLE` the server executes.

mod fetcher;
mod injector;
mod writer;

use crate::parquet::types::metadata::RowGroupMetadata;
use crate::parquet::types::table::ParquetTable;
use crate::store::DataFile;
use dispatch::{
    DataFlowDispatcher, DefaultUnaryFactory, OperatorSpec, RecordBatchOperatorSpec,
    RootUnaryOperatorFactory, fan_in,
};
use fetcher::TableFileMetadataFetcher;
use injector::FileInjectorFactory;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use writer::TableBuildSinkFactory;
use crate::catalog::TableFile;

/// A data file tagged with its position in the input list, so the row groups
/// can be regrouped in file order regardless of which worker reads which footer.
type IndexedFile = (usize, DataFile);
/// A row group tagged with the index of the file it came from.
type IndexedRowGroup = (usize, RowGroupMetadata);

/// The per-worker source→fetch factories: each worker steals files from a shared
/// injector and reads their footers, emitting one [`IndexedRowGroup`] per group.
fn fetch_row_group_metadata_factories(
    files: &[DataFile],
    workers: usize,
) -> Vec<
    RootUnaryOperatorFactory<
        IndexedFile,
        TableFile,
        DefaultUnaryFactory<TableFileMetadataFetcher>,
        FileInjectorFactory,
    >,
> {
    let injector = FileInjectorFactory::new(files);
    let siblings = Arc::new(AtomicUsize::new(workers));
    (0..workers)
        .map(|_| {
            RootUnaryOperatorFactory::new(
                DefaultUnaryFactory::<TableFileMetadataFetcher>::new(),
                injector.clone(),
                siblings.clone(),
            )
        })
        .collect()
}


/// A `RecordBatchOperatorSpec` that, when executed, reads every file's footer in
/// parallel and — at its terminal stage — regroups the [`LoadedFiles`] and
/// hands it to `commit` (which runs once, on the worker that finishes last, and
/// returns an error to fail the statement). Emits no rows. This is `CREATE TABLE`
/// as a single dataflow: fetch, then commit (the `commit` records the table in
/// the manifest, the table log, and the catalog map).
pub fn create_load_and_commit_spec<C>(
    dispatcher: &DataFlowDispatcher,
    files: &[DataFile],
    commit: C,
) -> RecordBatchOperatorSpec
where
    C: FnOnce(Vec<TableFile>) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>
        + Send
        + 'static,
{
    let file_count = files.len();
    let fetch = OperatorSpec::new(dispatcher.clone(), fetch_row_group_metadata_factories(files, dispatcher.worker_count()));

    // `fan_in` funnels every worker's row groups to worker 0; only worker 0 gets
    // the commit, so it alone builds and writes the table (the rest no-op). No
    // shared state — the channel closing is the "all fetched" signal.
    let mut commit = Some(commit);
    let sinks: Vec<_> = (0..dispatcher.worker_count())
        .map(|_| TableBuildSinkFactory::new(commit.take(), file_count))
        .collect();
    RecordBatchOperatorSpec::from_spec(fetch.chain(fan_in::<TableFile>(dispatcher.worker_count()), sinks))
}
