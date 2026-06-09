//! Parallel row-group **metadata** fetch — the scan's first phase.
//!
//! Before the column-chunk pipeline can run, each data file's Parquet footer
//! must be read and parsed into [`RowGroupMetadata`]. Reading them serially (a
//! loop over files) wastes the worker pool — especially for remote files, where
//! each footer is a network round trip. So [`materialize_metadata`] runs the
//! footer reads as a small work-stealing dataflow: a [`FileInjector`] hands out
//! the files, a [`RowGroupMetadataFetcher`] reads each one's footer (over the
//! io_uring ring, through the file cache) on whatever worker steals it, and the
//! collected row groups are numbered in file order (a pipeline breaker). The
//! [`ParquetTable`](crate::parquet::ParquetTable) constructors (`from_directory`
//! and friends) drive this once, at `CREATE`/`ATTACH` time; the resulting row
//! groups are reused by every query that scans the table.

use crate::parquet::types::metadata::{FileSource, RowGroupMetadata};
use crate::parquet::types::table::{
    DataFileLocation, Error, FOOTER_PROBE_BYTES, ParquetTable, Result, row_groups_from_footer,
};
use crossbeam_deque::{Injector, Steal};
use dispatch::io::{FileLocation, FsRequest, HttpRequest, IORequest, RemoteFile, open_direct_read};
use dispatch::memory::{CacheLookup, memory_ctx};
use dispatch::{
    DataFlowDispatcher, DefaultUnaryFactory, OperatorSpec, Receiver, RootChannelFactory,
    RootUnaryOperatorFactory, Sender, Unary,
};
use std::collections::{HashMap, VecDeque};
use std::os::fd::AsRawFd;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

/// A file location tagged with its position in the input list, so the row groups
/// can be numbered in file order regardless of which worker reads which footer.
type IndexedFile = (usize, DataFileLocation);
/// A row group tagged with the index of the file it came from.
type IndexedRowGroup = (usize, RowGroupMetadata);

const PARQUET_MAGIC: [u8; 4] = [b'P', b'A', b'R', b'1'];

/// Parse the 4-byte footer length from a file's `tail` (whose final 8 bytes are
/// `[footer_len][PAR1]`). The reader fetches the tail through the cache, so it
/// has the bytes in hand rather than `seek`+`read`ing them.
fn footer_len_from_tail(tail: &[u8]) -> Result<usize> {
    if tail.len() < 8 || tail[tail.len() - 4..] != PARQUET_MAGIC {
        return Err(Error::InvalidFooter("missing PAR1 magic".to_string()));
    }
    let len = &tail[tail.len() - 8..tail.len() - 4];
    Ok(u32::from_le_bytes(len.try_into().unwrap()) as usize)
}

/// Materialize a list of data files into a [`ParquetTable`], reading every
/// footer in parallel across the worker pool.
///
/// A pipeline breaker: it collects all row groups before returning. Global
/// row-group indices are assigned in `(file order, file-internal order)`, so the
/// result is identical regardless of how the parallel fetch interleaves.
///
/// Must run on the coordinator (the thread holding `dispatcher`): it drives a
/// dataflow, which would deadlock if nested inside a worker.
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
                DefaultUnaryFactory::<RowGroupMetadataFetcher>::new(),
                injector.clone(),
                siblings.clone(),
            )
        })
        .collect();
    let mut collected: Vec<IndexedRowGroup> =
        OperatorSpec::new(dispatcher.clone(), factories).collect()?;

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

/// Disk-backed footer reads outstanding per worker. One at a time — like the
/// column-chunk [`RowGroupFetcher`](super::fetcher), io_uring gives a single
/// read ample depth.
const MAX_DISK_IN_FLIGHT: usize = 1;
/// Remote-backed footer reads outstanding per worker. Larger, to hide the HTTP
/// round trip each footer costs.
const MAX_HTTP_IN_FLIGHT: usize = 32;

/// Reads files' footers — through the io_uring ring and the file cache, exactly
/// like a column-chunk read — and emits each file's row groups.
///
/// Mirrors the column-chunk [`RowGroupFetcher`](super::fetcher): rather than one
/// file at a time, it keeps many footer reads in flight, bounded independently by
/// medium ([`MAX_DISK_IN_FLIGHT`] disk-backed, [`MAX_HTTP_IN_FLIGHT`]
/// remote-backed), so an all-remote directory's footers load in parallel instead
/// of one network round trip after another. Each [`RowGroupMetadataRequest`]
/// reads its file's tail probe window, parses the footer, and — if the footer
/// overflowed the probe — reads it exactly before parsing. Completions route back
/// to the owning request by their `(location, file offset)`.
#[derive(Default)]
struct RowGroupMetadataFetcher {
    /// In-flight footer reads, slot-indexed (`None` = free slot).
    in_flight: Vec<Option<RowGroupMetadataRequest>>,
    /// Reusable freed slot indices.
    free_slots: Vec<usize>,
    /// `(location, file offset)` → slots awaiting a completion at that key.
    routing: HashMap<(FileLocation, usize), VecDeque<usize>>,
    /// Outstanding disk-backed footer reads (capped at [`MAX_DISK_IN_FLIGHT`]).
    disk_in_flight: usize,
    /// Outstanding remote-backed footer reads (capped at [`MAX_HTTP_IN_FLIGHT`]).
    http_in_flight: usize,
}

impl RowGroupMetadataFetcher {
    fn alloc_slot(&mut self, request: RowGroupMetadataRequest) -> usize {
        if request.is_remote() {
            self.http_in_flight += 1;
        } else {
            self.disk_in_flight += 1;
        }
        if let Some(slot) = self.free_slots.pop() {
            self.in_flight[slot] = Some(request);
            slot
        } else {
            self.in_flight.push(Some(request));
            self.in_flight.len() - 1
        }
    }

    /// Route each currently-pending read of the request in `slot` back to that
    /// slot. Called after a region's reads are queued (`start`, or an exact
    /// re-read) and before [`next_fs_requests`](Self::next_fs_requests) drains
    /// them.
    fn register_routes(&mut self, slot: usize) {
        let request = self.in_flight[slot].as_ref().unwrap();
        let mut keys: Vec<(FileLocation, usize)> = Vec::new();
        for fs in &request.pending_fs {
            keys.push((FileLocation::Local(fs.fd), fs.block.file_offset()));
        }
        for http in &request.pending_http {
            keys.push((
                FileLocation::Remote(http.remote.clone()),
                http.block.file_offset(),
            ));
        }
        for key in keys {
            self.routing.entry(key).or_default().push_back(slot);
        }
    }

    /// Advance the in-flight read in `slot` as far as it can without blocking on
    /// IO. While its current region is fully present, parse the region and either
    /// emit the file's row groups (done) or issue the next region's read (the
    /// exact-footer re-read); stop once a region has reads outstanding (await
    /// their completions) or the file is finished.
    ///
    /// Called from `consume` (after issuing the probe) and `process_io_response`
    /// (after a block lands) — both just "make a read happen, then advance" — so
    /// the pending-vs-already-cached decision lives only here.
    fn advance<S: Sender<IndexedRowGroup>>(
        &mut self,
        slot: usize,
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        while !self.in_flight[slot].as_ref().unwrap().is_pending() {
            let parsed = self.in_flight[slot]
                .as_mut()
                .unwrap()
                .parse_region()
                .map_err(crate::parquet::op_err)?;
            match parsed {
                // The footer overflowed the probe; an exact read was issued. Route
                // it; the loop condition re-checks whether it needs IO or was
                // already cached (and should be parsed right away).
                None => self.register_routes(slot),
                Some(row_groups) => {
                    let request = self.in_flight[slot].take().unwrap();
                    if request.is_remote() {
                        self.http_in_flight -= 1;
                    } else {
                        self.disk_in_flight -= 1;
                    }
                    self.free_slots.push(slot);
                    for rg in row_groups {
                        sender.send((request.file_idx, rg))?;
                    }
                    return Ok(());
                }
            }
        }
        Ok(())
    }
}

impl Unary<IndexedFile, IndexedRowGroup> for RowGroupMetadataFetcher {
    fn consume<S: Sender<IndexedRowGroup>>(
        &mut self,
        (file_idx, location): IndexedFile,
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        // Resolve the location to its transport and byte source. The size — which
        // locates the footer's tail window with no HEAD/suffix probe — is already
        // carried by the location (`stat`ed at table-build time for a local file,
        // from the catalog snapshot for a remote one).
        let (location, source, size) = match location {
            DataFileLocation::Local { path, size } => {
                let file = open_direct_read(&path).map_err(crate::parquet::op_err)?;
                let location = FileLocation::Local(file.as_raw_fd());
                (location, FileSource::Local(Arc::new(file)), size as usize)
            }
            DataFileLocation::Remote { url, size } => {
                let remote = Arc::new(RemoteFile::open(url).map_err(crate::parquet::op_err)?);
                let location = FileLocation::Remote(remote.clone());
                (location, FileSource::Remote(remote), size as usize)
            }
        };

        let slot = self.alloc_slot(RowGroupMetadataRequest::start(
            file_idx, location, source, size,
        ));
        self.register_routes(slot);
        self.advance(slot, sender)
    }

    fn next_fs_requests(&mut self) -> dispatch::UnaryResult<Vec<FsRequest>> {
        let mut all = Vec::new();
        for request in self.in_flight.iter_mut().flatten() {
            all.append(&mut request.pending_fs);
        }
        Ok(all)
    }

    fn next_http_requests(&mut self) -> dispatch::UnaryResult<Vec<HttpRequest>> {
        let mut all = Vec::new();
        for request in self.in_flight.iter_mut().flatten() {
            all.append(&mut request.pending_http);
        }
        Ok(all)
    }

    fn ready_for_more_work(&mut self) -> bool {
        self.disk_in_flight < MAX_DISK_IN_FLIGHT && self.http_in_flight < MAX_HTTP_IN_FLIGHT
    }

    fn process_io_response<S: Sender<IndexedRowGroup>>(
        &mut self,
        sender: &mut S,
        request: IORequest,
    ) -> dispatch::UnaryResult<()> {
        // The requester already committed this block's bytes; route the
        // completion to the owning request and count it off its current region.
        let key = (request.location, request.block.file_offset());
        let slot = match self.routing.get_mut(&key) {
            Some(waiters) => {
                let slot = waiters.pop_front();
                if waiters.is_empty() {
                    self.routing.remove(&key);
                }
                slot
            }
            None => None,
        };
        if let Some(slot) = slot {
            self.in_flight[slot].as_mut().unwrap().record_block();
            self.advance(slot, sender)?;
        }
        Ok(())
    }

    fn finish<S: Sender<IndexedRowGroup>>(
        &mut self,
        _sender: &mut S,
    ) -> dispatch::UnaryResult<bool> {
        Ok(self.disk_in_flight == 0 && self.http_in_flight == 0)
    }
}

/// One file's in-flight footer read.
///
/// It reads a region of the file (first the tail probe window, then — only if
/// the footer overflows it — the exact footer) into the file cache, tracking the
/// blocks still outstanding for the current region. Once a region's blocks have
/// all landed, [`parse_region`](Self::parse_region) turns it into the file's row
/// groups (or issues the exact re-read).
struct RowGroupMetadataRequest {
    file_idx: usize,
    location: FileLocation,
    /// Keeps the file handle (fd / remote) alive; travels into the row groups.
    source: FileSource,
    /// Total file size, known when the read starts.
    size: usize,
    /// Cache lookups pinning the region currently being read.
    lookups: Vec<CacheLookup>,
    pending_fs: Vec<FsRequest>,
    pending_http: Vec<HttpRequest>,
    /// Blocks still outstanding for the current region.
    remaining: usize,
    /// `false` while reading the tail probe window; `true` once the footer was
    /// found to overflow the probe and the exact footer is being read.
    reading_exact: bool,
}

impl RowGroupMetadataRequest {
    /// Register the file in the cache and issue the tail probe read
    /// `[size - probe, size)`.
    fn start(file_idx: usize, location: FileLocation, source: FileSource, size: usize) -> Self {
        memory_ctx().file_cache().open_entry(location.clone());
        let mut request = Self {
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
        request.read_region(size - probe, probe);
        request
    }

    /// Whether the file is read over HTTP (vs a local disk read).
    fn is_remote(&self) -> bool {
        matches!(self.location, FileLocation::Remote(_))
    }

    /// Whether the current region still has blocks in flight.
    fn is_pending(&self) -> bool {
        self.remaining > 0
    }

    /// Record one block of the current region as landed.
    fn record_block(&mut self) {
        debug_assert!(
            self.remaining > 0,
            "completion for a region with no pending blocks"
        );
        self.remaining -= 1;
    }

    /// Look up `[offset, offset + len)` in the cache and queue any missing blocks
    /// as the current region.
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

    /// The current region's bytes (concatenated), consuming the lookups.
    fn region_bytes(&mut self) -> Vec<u8> {
        let mut out = Vec::new();
        for lookup in std::mem::take(&mut self.lookups) {
            out.extend_from_slice(&lookup.into_data());
        }
        out
    }

    /// Parse the just-completed region. Returns the file's row groups, or `None`
    /// if the footer overflowed the probe window and an exact re-read was issued
    /// (its completion will call back here).
    fn parse_region(&mut self) -> Result<Option<Vec<RowGroupMetadata>>> {
        let bytes = self.region_bytes();

        let footer: &[u8] = if self.reading_exact {
            // The exact read returned precisely the footer.
            &bytes
        } else {
            let footer_len = footer_len_from_tail(&bytes)?;
            // A footer that can't fit within the file is corrupt — reject it
            // before the offset math below can underflow.
            if footer_len + 8 > self.size {
                return Err(Error::InvalidFooter(format!(
                    "footer length {footer_len} exceeds file size {}",
                    self.size
                )));
            }
            if footer_len + 8 > bytes.len() {
                // Footer overflowed the probe window — read it exactly, then wait.
                let exact_offset = self.size - 8 - footer_len;
                self.reading_exact = true;
                self.read_region(exact_offset, footer_len);
                return Ok(None);
            }
            let start = bytes.len() - 8 - footer_len;
            &bytes[start..bytes.len() - 8]
        };

        Ok(Some(row_groups_from_footer(footer, self.source.clone())?))
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
