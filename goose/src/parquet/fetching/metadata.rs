//! Parallel row-group **metadata** fetch — the scan's first phase.
//!
//! A table is a list of data-file [`DataFileLocation`]s. Before the column-chunk
//! pipeline can run, each file's Parquet footer must be read and parsed into
//! [`RowGroupMetadata`]. Doing that serially (a loop over files) wastes the
//! worker pool, especially for remote files where each footer is a network
//! round trip. So this runs the footer reads as a small dispatch dataflow: a
//! work-stealing source hands out files, a [`FooterFetcher`] reads each one's
//! footer on whatever worker steals it, and [`materialize_metadata`] collects every row
//! group (a pipeline breaker) and assigns global indices in file order. The
//! materialized [`ParquetTable`] then feeds the existing scan pipeline.

use crate::parquet::types::metadata::{FileSource, RowGroupMetadata};
use crate::parquet::types::table::{
    DataFileLocation, Error, FOOTER_PROBE_BYTES, ParquetTable, Result, footer_len_from_tail,
    row_groups_from_footer,
};
use crossbeam_deque::{Injector, Steal};
use dispatch::io::{FileLocation, FsRequest, HttpRequest, IORequest, RemoteFile, open_direct_read};
use dispatch::memory::{CacheLookup, memory_ctx};
use dispatch::{
    DataFlowDispatcher, DefaultUnaryFactory, OperatorSpec, Receiver, RootChannelFactory,
    RootUnaryOperatorFactory, Sender, Unary,
};
use std::fs;
use std::os::fd::AsRawFd;
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

    /// An explicit list of remote files: concrete fetchable URLs paired with
    /// their total size (from the catalog snapshot), which locates each footer.
    pub fn from_remote_files(files: &[(Url, u64)]) -> Self {
        Self {
            files: files
                .iter()
                .cloned()
                .map(|(url, size)| DataFileLocation::Remote { url, size })
                .collect(),
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
                DefaultUnaryFactory::<FooterFetcher>::new(),
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

/// Reads one file's footer per input — through the io_uring ring and the file
/// cache, exactly like a column-chunk read — and emits the file's row groups.
///
/// One file at a time per worker (the work-stealing source already spreads files
/// across workers): it fetches the file's tail probe window through the cache,
/// parses the footer length, and — if the footer didn't fit the probe — does an
/// exact second cache read, then parses the footer. The same file handle is
/// carried into the emitted [`RowGroupMetadata`], so the later column-chunk
/// reads hit the same cache entry.
///
/// The file size is known up front — from `stat` for a local file, from the
/// catalog snapshot for a remote one — so the footer's tail window is at a known
/// offset and there's no HEAD or suffix probe to pay for.
#[derive(Default)]
struct FooterFetcher {
    current: Option<InFlight>,
}

/// The footer read in progress for one file.
struct InFlight {
    file_idx: usize,
    location: FileLocation,
    /// Keeps the file handle (fd / remote) alive and travels into the row groups.
    source: FileSource,
    /// Total file size, known when the read starts.
    size: usize,
    /// Cache lookups pinning the region currently being read.
    lookups: Vec<CacheLookup>,
    pending_fs: Vec<FsRequest>,
    pending_http: Vec<HttpRequest>,
    /// Outstanding blocks for the current region.
    remaining: usize,
    /// `false` while reading the tail probe window; `true` once we know the
    /// footer overflowed the probe and are reading it exactly.
    reading_exact: bool,
}

impl InFlight {
    /// Look up `[offset, offset+len)` in the cache and queue any missing blocks.
    fn read_region(&mut self, offset: usize, len: usize) {
        self.lookups = memory_ctx().file_cache().get(&self.location, offset, len);
        self.remaining = 0;
        for lookup in &self.lookups {
            for block in lookup.missing() {
                match &self.location {
                    FileLocation::Local(fd) => self.pending_fs.push(FsRequest {
                        fd: *fd,
                        block: block.clone(),
                    }),
                    FileLocation::Remote(remote) => self.pending_http.push(HttpRequest {
                        remote: remote.clone(),
                        block: block.clone(),
                    }),
                }
                self.remaining += 1;
            }
        }
    }

    /// The fetched region's bytes (concatenated), consuming the lookups.
    fn region_bytes(&mut self) -> Vec<u8> {
        let mut out = Vec::new();
        for lookup in std::mem::take(&mut self.lookups) {
            out.extend_from_slice(&lookup.into_data());
        }
        out
    }
}

impl FooterFetcher {
    /// Start reading one file's footer: register the file in the cache and issue
    /// the tail probe window `[size - probe, size)`. `location`/`source` select
    /// the transport (local fd or remote URL); `size` is already known by the
    /// caller (`stat` or the catalog snapshot). If the window was already cached,
    /// completes inline.
    fn begin_footer_read<S: Sender<IndexedRowGroup>>(
        &mut self,
        file_idx: usize,
        location: FileLocation,
        source: FileSource,
        size: usize,
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        memory_ctx().file_cache().open_entry(location.clone());
        let mut inflight = InFlight {
            file_idx,
            location,
            source,
            size,
            lookups: Vec::new(),
            pending_fs: Vec::new(),
            pending_http: Vec::new(),
            remaining: 0,
            reading_exact: false,
        };
        let probe = size.min(FOOTER_PROBE_BYTES);
        inflight.read_region(size - probe, probe);
        let cached = inflight.remaining == 0;
        self.current = Some(inflight);
        if cached {
            self.on_region_complete(sender)?;
        }
        Ok(())
    }

    /// The current region's reads have all landed: parse it. Either emit the row
    /// groups (footer in hand) or issue the exact-footer read.
    fn on_region_complete<S: Sender<IndexedRowGroup>>(
        &mut self,
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        let mut inflight = self.current.take().unwrap();
        let bytes = inflight.region_bytes();

        let footer: &[u8] = if inflight.reading_exact {
            // The region is exactly the footer.
            &bytes
        } else {
            let footer_len = footer_len_from_tail(&bytes).map_err(crate::parquet::op_err)?;
            if footer_len + 8 > bytes.len() {
                // Footer overflowed the probe window — read it exactly, then retry.
                let exact_offset = inflight.size - 8 - footer_len;
                inflight.reading_exact = true;
                inflight.read_region(exact_offset, footer_len);
                let cached = inflight.remaining == 0;
                self.current = Some(inflight);
                if cached {
                    return self.on_region_complete(sender);
                }
                return Ok(());
            }
            let start = bytes.len() - 8 - footer_len;
            &bytes[start..bytes.len() - 8]
        };

        let row_groups =
            row_groups_from_footer(footer, inflight.source).map_err(crate::parquet::op_err)?;
        for rg in row_groups {
            sender.send((inflight.file_idx, rg))?;
        }
        Ok(())
    }
}

impl Unary<IndexedFile, IndexedRowGroup> for FooterFetcher {
    fn consume<S: Sender<IndexedRowGroup>>(
        &mut self,
        (file_idx, location): IndexedFile,
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        // Resolve the location to its transport, byte source, and size. Local
        // files `stat` for the size; remote files carry it from the catalog
        // snapshot — either way the footer offset is known with no probe.
        let (location, source, size) = match location {
            DataFileLocation::Local(path) => {
                let file = open_direct_read(&path).map_err(crate::parquet::op_err)?;
                let size = fs::metadata(&path).map_err(crate::parquet::op_err)?.len() as usize;
                let location = FileLocation::Local(file.as_raw_fd());
                (location, FileSource::Local(Arc::new(file)), size)
            }
            DataFileLocation::Remote { url, size } => {
                let remote = Arc::new(RemoteFile::open(url).map_err(crate::parquet::op_err)?);
                let location = FileLocation::Remote(remote.clone());
                (location, FileSource::Remote(remote), size as usize)
            }
        };
        self.begin_footer_read(file_idx, location, source, size, sender)
    }

    fn next_fs_requests(&mut self) -> dispatch::UnaryResult<Vec<FsRequest>> {
        Ok(self
            .current
            .as_mut()
            .map(|i| std::mem::take(&mut i.pending_fs))
            .unwrap_or_default())
    }

    fn next_http_requests(&mut self) -> dispatch::UnaryResult<Vec<HttpRequest>> {
        Ok(self
            .current
            .as_mut()
            .map(|i| std::mem::take(&mut i.pending_http))
            .unwrap_or_default())
    }

    fn ready_for_more_work(&mut self) -> bool {
        self.current.is_none()
    }

    fn process_io_response<S: Sender<IndexedRowGroup>>(
        &mut self,
        sender: &mut S,
        _request: IORequest,
    ) -> dispatch::UnaryResult<()> {
        // Single file in flight, so every completion is for the current region.
        if let Some(inflight) = self.current.as_mut() {
            inflight.remaining -= 1;
            if inflight.remaining == 0 {
                self.on_region_complete(sender)?;
            }
        }
        Ok(())
    }

    fn finish<S: Sender<IndexedRowGroup>>(
        &mut self,
        _sender: &mut S,
    ) -> dispatch::UnaryResult<bool> {
        Ok(self.current.is_none())
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
