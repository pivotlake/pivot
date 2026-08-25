//! Parallel **footer metadata** fetch — the table-load pipeline, run once at
//! `CREATE`/`ATTACH` (the reading pipeline scans the row groups it produces).
//! A small work-stealing dataflow with one module per stage, carrying one
//! [`FileRowGroups`] (a file's [`FileRef`] plus the row groups read from its
//! footer) per file end to end. This layer is Parquet-specific and knows nothing
//! of any table log: the owning table format joins each result with its file
//! entry afterward.
//!
//! - [`injector`] — the source: hands out the input files.
//! - [`fetcher`] — reads each file's footer (over the io_uring ring, through the
//!   compressed cache) on whatever worker steals it, emitting one [`FileRowGroups`].
//! - [`writer`] — the terminal fan-in sink: gathers the [`FileRowGroups`] on one
//!   worker and hands them to the transaction-staging closure.
//!
//! It is consumed two ways, both over the same fetch core
//! ([`fetch_file_row_group_factories`]): [`load_file_row_groups`] collects the
//! [`FileRowGroups`] on the coordinator and returns them as a **value** (the
//! `ParquetTable::from_*` constructors, which flatten them into a table), while
//! [`create_load_and_stage_spec`] returns a `RecordBatchOperatorSpec` ending
//! in the [`writer`] sink that hands the `Vec<FileRowGroups>` to a staging closure
//! for the `CREATE TABLE` transaction.

mod fetcher;
mod injector;
mod writer;

use crate::types::metadata::RowGroupMetadata;
use dispatch::{
    DataFlowDispatcher, OperatorSpec, RecordBatchOperatorSpec, RootUnaryOperatorFactory,
    UnaryFactory, fan_in,
};
use fetcher::FileRowGroupsFetcher;
use injector::FileInjectorFactory;
use object_storage::{DataFile, DataFileLocation, FileRef};
use planner::catalog::Column;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use writer::FileRowGroupsSinkFactory;

/// A file's footer metadata: its store identity ([`FileRef`]) and the row groups
/// read from its footer, in file-local order. The Parquet-level result of the
/// metadata-fetch pipeline; a table format can join it with its own file entry.
pub struct FileRowGroups {
    pub file: FileRef,
    pub row_groups: Vec<Arc<RowGroupMetadata>>,
}

/// Builds each worker's [`FileRowGroupsFetcher`], carrying the table's
/// declared column types so every parsed footer is reconciled with them.
struct MetadataFetcherFactory {
    declared_columns: Arc<[Column]>,
}

impl UnaryFactory<DataFile, FileRowGroups> for MetadataFetcherFactory {
    type Unary = FileRowGroupsFetcher;

    fn build_unary(self) -> Self::Unary {
        FileRowGroupsFetcher::new(self.declared_columns)
    }
}

/// Build the scan metadata for a file a worker has just uploaded, from the footer
/// metadata the writer already produced. This skips both scheduling a nested
/// metadata dataflow from a worker and re-parsing a footer we just wrote; only
/// the open_file has to be bound now, since it names the stored file that future
/// scans read and exists only once the upload has landed.
pub fn file_row_groups_from_metadata(
    file: FileRef,
    source: DataFileLocation,
    metadata: crate::thrift::footer::FileMetaData,
    declared_columns: &[Column],
) -> Result<FileRowGroups, crate::ParquetTableError> {
    let open_file = source
        .open_read(file.size)
        .map_err(crate::ParquetTableError::IO)?;
    let row_groups =
        crate::types::table::row_groups_from_metadata(metadata, open_file, declared_columns)?
            .into_iter()
            .map(Arc::new)
            .collect();
    Ok(FileRowGroups { file, row_groups })
}

/// The per-worker source→fetch factories: each worker steals files from a shared
/// injector and reads their footers, emitting one [`FileRowGroups`] per file (the
/// file's [`FileRef`] rides along on the [`DataFile`] and lands on the result).
fn fetch_file_row_group_factories(
    files: &[DataFile],
    workers: usize,
    declared_columns: Arc<[Column]>,
) -> Vec<
    RootUnaryOperatorFactory<DataFile, FileRowGroups, MetadataFetcherFactory, FileInjectorFactory>,
> {
    let injector = FileInjectorFactory::new(files);
    let siblings = Arc::new(AtomicUsize::new(workers));
    (0..workers)
        .map(|_| {
            RootUnaryOperatorFactory::new(
                MetadataFetcherFactory {
                    declared_columns: declared_columns.clone(),
                },
                injector.clone(),
                siblings.clone(),
            )
        })
        .collect()
}

/// Read every file's footer in parallel and collect the resulting
/// [`FileRowGroups`] on the coordinator (file order is not preserved — the table's
/// row groups are flattened across whichever order the workers finish in). The
/// fetch stage already emits `FileRowGroups`, so the dataflow's typed `collect`
/// drains them directly — no terminal sink. Drives the dataflow, so it must run
/// on the **coordinator**, not inside a `run_on_worker` closure.
pub fn load_file_row_groups(
    dispatcher: &DataFlowDispatcher,
    files: &[DataFile],
    declared_columns: Arc<[Column]>,
) -> Result<Vec<FileRowGroups>, dispatch::DataFlowError> {
    // When there is nothing to fetch, skip the dataflow round-trip entirely.
    // Every query's compile resolves its table through here, and a warm
    // catalog has no missing footers, so this is the common case.
    if files.is_empty() {
        return Ok(Vec::new());
    }
    OperatorSpec::new(
        dispatcher.clone(),
        fetch_file_row_group_factories(files, dispatcher.worker_count(), declared_columns),
    )
    .collect()
}

/// A `RecordBatchOperatorSpec` that, when executed, reads every file's footer in
/// parallel and — at its terminal stage — regroups the [`FileRowGroups`] and
/// hands them to `stage` once on the terminal worker. Emits no rows. The stage
/// closure must only enqueue the loaded metadata; durable table creation belongs
/// to the transaction's commit path, outside the dispatch pool.
pub fn create_load_and_stage_spec<C>(
    dispatcher: &DataFlowDispatcher,
    files: &[DataFile],
    declared_columns: Arc<[Column]>,
    stage: C,
) -> RecordBatchOperatorSpec
where
    C: FnOnce(Vec<FileRowGroups>) + Send + 'static,
{
    let fetch = OperatorSpec::new(
        dispatcher.clone(),
        fetch_file_row_group_factories(files, dispatcher.worker_count(), declared_columns),
    );

    // One worker receives every row group and the staging closure; the channel
    // closing is the "all fetched" signal. Rotate that role across creations.
    let worker_count = dispatcher.worker_count();
    let target = dispatcher.next_worker();
    let mut stage = Some(stage);
    let sinks: Vec<_> = (0..worker_count)
        .map(|worker| {
            FileRowGroupsSinkFactory::new(if worker == target { stage.take() } else { None })
        })
        .collect();
    RecordBatchOperatorSpec::from_spec(
        fetch.chain(fan_in::<FileRowGroups>(worker_count, target), sinks),
    )
}
