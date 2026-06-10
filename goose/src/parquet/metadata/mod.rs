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
//! It is consumed two ways, both over the same fetch core ([`fetch_factories`]
//! and [`LoadedTable::assemble`]): [`load`] collects on the coordinator and
//! returns a [`LoadedTable`] **value** of per-file row groups (the catalog's
//! reload and registrations; [`load_table`] flattens it for the test/bench
//! constructors), while [`create_load_and_commit_spec`] returns a
//! `RecordBatchOperatorSpec` ending in the [`writer`] sink that hands the
//! `LoadedTable` to a `commit` closure — the `CREATE TABLE` the server
//! executes.

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
use fetcher::RowGroupMetadataFetcher;
use injector::FileInjectorFactory;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use writer::TableBuildSinkFactory;

/// A data file tagged with its position in the input list, so the row groups
/// can be regrouped in file order regardless of which worker reads which footer.
type IndexedFile = (usize, DataFile);
/// A row group tagged with the index of the file it came from.
type IndexedRowGroup = (usize, RowGroupMetadata);

/// The per-worker source→fetch factories: each worker steals files from a shared
/// injector and reads their footers, emitting one [`IndexedRowGroup`] per group.
fn fetch_factories(
    files: &[DataFile],
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

/// The result of one metadata fetch over a list of data files: each input
/// file's row groups, in input-file order (file-internal order within), with
/// global indices not yet assigned. Callers that track files individually (the
/// catalog's versioned table entries) keep this shape;
/// [`into_table`](Self::into_table) flattens it for everyone else.
pub struct LoadedTable {
    files: Vec<Vec<RowGroupMetadata>>,
}

impl LoadedTable {
    /// Regroup the fetch's interleaved output by `(file order, file-internal
    /// order)`, so the result is identical regardless of how the parallel
    /// fetch raced.
    pub(super) fn assemble(mut rows: Vec<IndexedRowGroup>, file_count: usize) -> Self {
        rows.sort_by_key(|(file_idx, rg)| (*file_idx, rg.file_row_group_idx));
        let mut files = vec![Vec::new(); file_count];
        for (file_idx, rg) in rows {
            files[file_idx].push(rg);
        }
        Self { files }
    }

    /// Each input file's row groups, in input order.
    pub fn into_per_file(self) -> Vec<Vec<RowGroupMetadata>> {
        self.files
    }

    /// Flatten into a [`ParquetTable`], assigning global row-group indices in
    /// file order.
    pub fn into_table(self) -> ParquetTable {
        let rows = self
            .files
            .into_iter()
            .flatten()
            .enumerate()
            .map(|(global_idx, mut rg)| {
                rg.global_row_group_idx = global_idx;
                Arc::new(rg)
            })
            .collect();
        ParquetTable::new(rows)
    }
}

/// Materialize a list of data files into a [`LoadedTable`] value on the
/// coordinator: read every footer in parallel and regroup per file. A pipeline
/// breaker — it drives a dataflow, so it must run on the coordinator (a nested
/// dataflow would deadlock a worker). The value path, used by the catalog;
/// `CREATE TABLE` uses [`create_load_and_commit_spec`] instead.
pub fn load(
    dispatcher: &DataFlowDispatcher,
    files: &[DataFile],
) -> Result<LoadedTable, dispatch::DataFlowError> {
    if files.is_empty() {
        return Ok(LoadedTable { files: Vec::new() });
    }
    let n = dispatcher.worker_count().max(1);
    let collected = OperatorSpec::new(dispatcher.clone(), fetch_factories(files, n)).collect()?;
    Ok(LoadedTable::assemble(collected, files.len()))
}

/// [`load`] flattened into a [`ParquetTable`] — the test/bench constructors'
/// path.
pub fn load_table(
    dispatcher: &DataFlowDispatcher,
    files: &[DataFile],
) -> Result<ParquetTable, dispatch::DataFlowError> {
    Ok(load(dispatcher, files)?.into_table())
}

/// A `RecordBatchOperatorSpec` that, when executed, reads every file's footer in
/// parallel and — at its terminal stage — regroups the [`LoadedTable`] and
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
    C: FnOnce(LoadedTable) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>
        + Send
        + 'static,
{
    let n = dispatcher.worker_count().max(1);
    let file_count = files.len();
    let fetch = OperatorSpec::new(dispatcher.clone(), fetch_factories(files, n));

    // `fan_in` funnels every worker's row groups to worker 0; only worker 0 gets
    // the commit, so it alone builds and writes the table (the rest no-op). No
    // shared state — the channel closing is the "all fetched" signal.
    let mut commit = Some(commit);
    let sinks: Vec<_> = (0..n)
        .map(|_| TableBuildSinkFactory::new(commit.take(), file_count))
        .collect();
    RecordBatchOperatorSpec::from_spec(fetch.chain(fan_in::<IndexedRowGroup>(n), sinks))
}
