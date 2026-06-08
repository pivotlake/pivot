//! Parallel row-group **metadata** fetch — the scan's first phase.
//!
//! A table is a list of data-file [`DataFileLocation`]s. Before the column-chunk
//! pipeline can run, each file's Parquet footer must be read and parsed into
//! [`RowGroupMetadata`]. Doing that serially (a loop over files) wastes the
//! worker pool, especially for remote files where each footer is a network
//! round trip. So this runs the footer reads as a small dispatch dataflow: a
//! work-stealing source hands out files, a [`MetadataFetcher`] reads each one's
//! footer on whatever worker steals it, and [`materialize_metadata`] collects every row
//! group (a pipeline breaker) and assigns global indices in file order. The
//! materialized [`ParquetTable`] then feeds the existing scan pipeline.

use crate::parquet::types::metadata::RowGroupMetadata;
use crate::parquet::types::table::{
    DataFileLocation, Error, ParquetTable, Result, parse_file_metadatas,
};
use crossbeam_deque::{Injector, Steal};
use dispatch::{
    DataFlowDispatcher, DefaultUnaryFactory, OperatorSpec, Receiver, RootChannelFactory,
    RootUnaryOperatorFactory, Sender, Unary,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use url::Url;

/// A table's data files, identified only by **location** — the lazy counterpart
/// to a [`ParquetTable`]. The catalog holds one of these; the scan's first phase
/// ([`ParquetSource::materialize`]) reads every footer in parallel to produce
/// the [`ParquetTable`] (row groups) the rest of the pipeline consumes.
#[derive(Clone, Debug)]
pub struct ParquetSource {
    files: Vec<DataFileLocation>,
}

impl ParquetSource {
    /// Every Parquet file in `path` (a local directory), by location only — no
    /// footer is read here, so this needs no dispatch worker.
    pub fn from_directory(path: &Path) -> Result<Self> {
        let files = fs::read_dir(path)?
            .flatten()
            .filter(|entry| entry.path().is_file())
            .map(|entry| DataFileLocation::Local(entry.path()))
            .collect();
        Ok(Self { files })
    }

    /// An explicit list of local files.
    pub fn from_files<P: AsRef<Path>>(paths: &[P]) -> Self {
        Self {
            files: paths
                .iter()
                .map(|p| DataFileLocation::Local(PathBuf::from(p.as_ref())))
                .collect(),
        }
    }

    /// An explicit list of remote files (concrete fetchable URLs).
    pub fn from_remote_files(urls: &[Url]) -> Self {
        Self {
            files: urls.iter().cloned().map(DataFileLocation::Remote).collect(),
        }
    }

    /// The file locations.
    pub fn files(&self) -> &[DataFileLocation] {
        &self.files
    }

    /// Read every file's footer in parallel and assemble the [`ParquetTable`].
    pub fn materialize(&self, dispatcher: &DataFlowDispatcher) -> Result<ParquetTable> {
        materialize_metadata(dispatcher, &self.files).map_err(|e| Error::Materialize(e.to_string()))
    }
}

/// An indexed file location flowing through the metadata-fetch dataflow. The
/// index records the file's position in the source list so the materialized row
/// groups can be ordered deterministically (file order) regardless of which
/// worker read which footer.
type IndexedFile = (usize, DataFileLocation);
/// A row group tagged with the index of the file it came from.
type IndexedRowGroup = (usize, RowGroupMetadata);

/// Materialize a table's file locations into row-group metadata, reading every
/// footer in parallel across workers. A pipeline breaker: it collects all row
/// groups before returning. Global row-group indices are assigned in file order
/// (then file-internal order) so the result is identical regardless of fetch
/// interleaving.
pub fn materialize_metadata(
    dispatcher: &DataFlowDispatcher,
    files: &[DataFileLocation],
) -> Result<ParquetTable, dispatch::DataFlowError> {
    if files.is_empty() {
        return Ok(ParquetTable::new(Vec::new()));
    }
    let n = dispatcher.worker_count().max(1);
    let injector = FileInjectorFactory::new(files);
    let siblings = Arc::new(AtomicUsize::new(n));
    let factories: Vec<_> = (0..n)
        .map(|_| {
            RootUnaryOperatorFactory::new(
                DefaultUnaryFactory::<MetadataFetcher>::new(),
                injector.clone(),
                siblings.clone(),
            )
        })
        .collect();
    let spec = OperatorSpec::new(dispatcher.clone(), factories);
    let mut collected: Vec<IndexedRowGroup> = spec.collect()?;

    // Order by (file index, file-internal row-group index), then number the row
    // groups globally.
    collected.sort_by_key(|(file_idx, rg)| (*file_idx, rg.file_row_group_idx));
    let row_groups = collected
        .into_iter()
        .enumerate()
        .map(|(global_idx, (_, mut rg))| {
            rg.global_row_group_idx = global_idx;
            Arc::new(rg)
        })
        .collect();
    Ok(ParquetTable::new(row_groups))
}

/// Reads one file's footer per input, emitting each of its row groups tagged
/// with the file index.
#[derive(Default)]
struct MetadataFetcher;

impl Unary<IndexedFile, IndexedRowGroup> for MetadataFetcher {
    fn consume<S: Sender<IndexedRowGroup>>(
        &mut self,
        (file_idx, location): IndexedFile,
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        let row_groups = parse_file_metadatas(&location).map_err(crate::parquet::op_err)?;
        for rg in row_groups {
            sender.send((file_idx, rg))?;
        }
        Ok(())
    }
}

/// Work-stealing source that hands out every file (with its index) once.
#[derive(Clone)]
struct FileInjectorFactory {
    files: Arc<Injector<IndexedFile>>,
}

impl FileInjectorFactory {
    fn new(files: &[DataFileLocation]) -> Self {
        let injector = Injector::new();
        for (idx, location) in files.iter().enumerate() {
            injector.push((idx, location.clone()));
        }
        Self {
            files: Arc::new(injector),
        }
    }
}

impl RootChannelFactory<IndexedFile> for FileInjectorFactory {
    type Receiver = FileInjector;

    fn build(self) -> FileInjector {
        FileInjector { files: self.files }
    }
}

struct FileInjector {
    files: Arc<Injector<IndexedFile>>,
}

impl Receiver<IndexedFile> for FileInjector {
    fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    fn try_recv(&self) -> Option<IndexedFile> {
        None
    }

    fn steal(&self) -> Option<IndexedFile> {
        loop {
            match self.files.steal() {
                Steal::Empty => return None,
                Steal::Retry => continue,
                Steal::Success(file) => return Some(file),
            }
        }
    }
}
