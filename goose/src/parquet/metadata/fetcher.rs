//! Fetch stage: reads each file's Parquet footer — through the io_uring ring and
//! the file cache, exactly like a column-chunk read — and emits the file's row
//! groups. Mirrors the column-chunk `RowGroupFetcher` (the scan's fetch stage):
//! many footer reads in flight, bounded per medium, so an all-remote directory's
//! footers load in parallel instead of one network round trip after another.

use super::{IndexedFile, IndexedRowGroup};
use crate::parquet::types::metadata::RowGroupMetadata;
use crate::parquet::types::table::{Error, FOOTER_PROBE_BYTES, Result, row_groups_from_footer};
use crate::store::{DataFile, DataFileSource};
use dispatch::io::{FileLocation, FsRequest, HttpRequest, RemoteFile, open_direct_read};
use dispatch::memory::{CacheLookup, memory_ctx};
use dispatch::{Sender, Unary};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

const PARQUET_MAGIC: [u8; 4] = [b'P', b'A', b'R', b'1'];

/// Disk-backed footer reads outstanding per worker. One at a time — like the
/// column-chunk fetcher, io_uring gives a single read ample depth.
const MAX_DISK_IN_FLIGHT: usize = 1;
/// Remote-backed footer reads outstanding per worker. Larger, to hide the HTTP
/// round trip each footer costs.
const MAX_HTTP_IN_FLIGHT: usize = 32;

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

/// Reads files' footers and emits each file's row groups. It keeps many footer
/// reads in flight, bounded independently by medium ([`MAX_DISK_IN_FLIGHT`]
/// disk-backed, [`MAX_HTTP_IN_FLIGHT`] remote-backed). Each [`RowGroupMetadataRequest`]
/// reads its file's tail probe window, parses the footer, and — if the footer
/// overflowed the probe — reads it exactly before parsing. Completions route back
/// to the owning request by their `(location, file offset)`.
#[derive(Default)]
pub(super) struct RowGroupMetadataFetcher {
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
            keys.push((FileLocation::Local(fs.file.clone()), fs.block.file_offset()));
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
    /// A completed read (fs or http): the requester already committed its bytes;
    /// route it to the owning request and count it off its current region.
    fn process_completion<S: Sender<IndexedRowGroup>>(
        &mut self,
        key: (FileLocation, usize),
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
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
        (file_idx, file): IndexedFile,
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        // Open the file's transport. The size — which locates the footer's tail
        // window with no HEAD/suffix probe — is already carried by the data
        // file (`stat`ed at listing time for a local file, from the store
        // listing for a remote one).
        let DataFile { name, size, source } = file;
        let location = match source {
            DataFileSource::Local(path) => {
                let file = open_direct_read(&path).map_err(crate::parquet::op_err)?;
                FileLocation::Local(Arc::new(file))
            }
            DataFileSource::Remote(url) => {
                let remote = Arc::new(RemoteFile::open(url).map_err(crate::parquet::op_err)?);
                FileLocation::Remote(remote)
            }
        };

        let slot = self.alloc_slot(RowGroupMetadataRequest::start(
            file_idx,
            location,
            Arc::from(name),
            size as usize,
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

    fn process_fs_response<S: Sender<IndexedRowGroup>>(
        &mut self,
        sender: &mut S,
        request: FsRequest,
    ) -> dispatch::UnaryResult<()> {
        let key = (
            FileLocation::Local(request.file),
            request.block.file_offset(),
        );
        self.process_completion(key, sender)
    }

    fn process_http_response<S: Sender<IndexedRowGroup>>(
        &mut self,
        sender: &mut S,
        request: HttpRequest,
    ) -> dispatch::UnaryResult<()> {
        let key = (
            FileLocation::Remote(request.remote),
            request.block.file_offset(),
        );
        self.process_completion(key, sender)
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
    /// The open file (it keeps the handle alive, and travels into the row
    /// groups as their location).
    location: FileLocation,
    /// The file's name — the table log identity stamped onto its row groups.
    file_name: Arc<str>,
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
    fn start(file_idx: usize, location: FileLocation, file_name: Arc<str>, size: usize) -> Self {
        memory_ctx().file_cache().open_entry(location.clone());
        let mut request = Self {
            file_idx,
            location,
            file_name,
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
                    FileLocation::Local(file) => self.pending_fs.push(FsRequest {
                        file: file.clone(),
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

        Ok(Some(row_groups_from_footer(
            footer,
            self.location.clone(),
            self.file_name.clone(),
        )?))
    }
}
