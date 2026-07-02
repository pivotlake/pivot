//! Parallel **footer metadata** fetch — the table-load pipeline, run once at
//! `CREATE`/`ATTACH` (the reading pipeline scans the row groups it produces).
//! A small work-stealing dataflow with one module per stage, carrying one
//! [`TableFile`] (a file's [`FileRef`](crate::store::FileRef) plus its row
//! groups) per file end to end:
//!
//! - [`injector`] — the source: hands out the input files.
//! - [`fetcher`] — reads each file's footer (over the io_uring ring, through the
//!   compressed cache) on whatever worker steals it, emitting one [`TableFile`].
//!
//! [`fetch_table_files_spec`] exposes that as an unlaunched spec, consumed two
//! ways: [`load_table_files`] collects the [`TableFile`]s on the coordinator
//! and returns them as a **value** (the `ParquetTable::from_*` constructors,
//! which flatten them into a table), while `CREATE TABLE` chains its own
//! commit stage onto the spec (a `fan_in` fold in the catalog) before
//! executing.

mod fetcher;
mod injector;

use crate::catalog::TableFile;
use crate::store::DataFile;
use dispatch::{
    DataFlowDispatcher, DefaultUnaryFactory, OperatorFactory, OperatorSpec,
    RootUnaryOperatorFactory,
};
use fetcher::TableFileMetadataFetcher;
use injector::FileInjectorFactory;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

/// A spec that reads every file's footer in parallel, streaming one
/// [`TableFile`] per file: each worker steals files from a shared injector
/// (the file's [`FileRef`](crate::store::FileRef) rides along on the
/// [`DataFile`] and lands on the `TableFile`). File order is not preserved -
/// files stream out in whichever order the workers finish.
pub(crate) fn fetch_table_files_spec(
    dispatcher: &DataFlowDispatcher,
    files: &[DataFile],
) -> OperatorSpec<TableFile, impl OperatorFactory<TableFile> + 'static> {
    let workers = dispatcher.worker_count();
    let injector = FileInjectorFactory::new(files);
    let siblings = Arc::new(AtomicUsize::new(workers));
    let factories: Vec<_> = (0..workers)
        .map(|_| {
            RootUnaryOperatorFactory::new(
                DefaultUnaryFactory::<TableFileMetadataFetcher>::new(),
                injector.clone(),
                siblings.clone(),
            )
        })
        .collect();
    OperatorSpec::new(dispatcher.clone(), factories)
}

/// Read every file's footer in parallel and collect the resulting
/// [`TableFile`]s on the coordinator. Drives the dataflow, so it must run on
/// the **coordinator**, not inside a `run_on_worker` closure.
pub(crate) fn load_table_files(
    dispatcher: &DataFlowDispatcher,
    files: &[DataFile],
) -> Result<Vec<TableFile>, dispatch::DataFlowError> {
    fetch_table_files_spec(dispatcher, files).collect()
}
