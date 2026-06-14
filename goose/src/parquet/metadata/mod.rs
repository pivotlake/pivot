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
//! It is consumed two ways, both over the same fetch core
//! ([`fetch_table_file_factories`]): [`load_table_files`] collects the
//! [`TableFile`]s on the coordinator and returns them as a **value** (the
//! `ParquetTable::from_*` constructors, which flatten them into a table), while
//! [`create_load_and_commit_spec`] returns a `RecordBatchOperatorSpec` ending
//! in the [`writer`] sink that hands the `Vec<TableFile>` to a `commit` closure
//! — the `CREATE TABLE` the server executes.

mod fetcher;
mod injector;
mod writer;

use crate::store::DataFile;
use dispatch::{
    DataFlowDispatcher, DefaultUnaryFactory, OperatorSpec, RecordBatchOperatorSpec,
    RootUnaryOperatorFactory, fan_in,
};
use fetcher::TableFileMetadataFetcher;
use injector::FileInjectorFactory;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use writer::TableBuildSinkFactory;
use crate::catalog::TableFile;

/// The per-worker source→fetch factories: each worker steals files from a shared
/// injector and reads their footers, emitting one [`TableFile`] per file (the
/// file's [`FileRef`](crate::store::FileRef) rides along on the [`DataFile`] and
/// lands on the `TableFile`).
fn fetch_table_file_factories(
    files: &[DataFile],
    workers: usize,
) -> Vec<
    RootUnaryOperatorFactory<
        DataFile,
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

/// Read every file's footer in parallel and collect the resulting
/// [`TableFile`]s on the coordinator (file order is not preserved — the table's
/// row groups are flattened across whichever order the workers finish in).
/// Drives the dataflow, so it must run on the **coordinator**, not inside a
/// `run_on_worker` closure.
pub(crate) fn load_table_files(
    dispatcher: &DataFlowDispatcher,
    files: &[DataFile],
) -> Result<Vec<TableFile>, dispatch::DataFlowError> {
    let collected = Arc::new(Mutex::new(Vec::new()));
    let sink = collected.clone();
    create_load_and_commit_spec(dispatcher, files, move |loaded| {
        *sink.lock().unwrap() = loaded;
        Ok(())
    })
    .collect()?;
    Ok(std::mem::take(&mut *collected.lock().unwrap()))
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
    let fetch = OperatorSpec::new(dispatcher.clone(), fetch_table_file_factories(files, dispatcher.worker_count()));

    // `fan_in` funnels every worker's row groups to worker 0; only worker 0 gets
    // the commit, so it alone builds and writes the table (the rest no-op). No
    // shared state — the channel closing is the "all fetched" signal.
    let mut commit = Some(commit);
    let sinks: Vec<_> = (0..dispatcher.worker_count())
        .map(|_| TableBuildSinkFactory::new(commit.take(), file_count))
        .collect();
    RecordBatchOperatorSpec::from_spec(fetch.chain(fan_in::<TableFile>(dispatcher.worker_count()), sinks))
}
