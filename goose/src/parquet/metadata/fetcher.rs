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
use std::collections::HashMap;
use std::sync::Arc;

const PARQUET_MAGIC: [u8; 4] = [b'P', b'A', b'R', b'1'];

/// Disk-backed footer-read *blocks* outstanding per worker before admitting
/// another file. A soft cap: a file already admitted may push over it (its
/// blocks still complete) — we just stop pulling in new files. One keeps disk
/// reads serial, since io_uring gives a single read ample depth.
const MAX_DISK_IN_FLIGHT: usize = 1;
/// Remote-backed footer-read *blocks* outstanding per worker before admitting
/// another file. Larger, to hide the HTTP round trip each read costs.
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
    /// `(location, file offset)` → the slot awaiting that block's completion.
    /// Exactly one waiter per key: each request reads its own file (a distinct
    /// fd, which is how a [`FileLocation`] is identified), so no two in-flight
    /// reads ever target the same key.
    routing: HashMap<(FileLocation, usize), usize>,
    /// Outstanding disk-backed footer-read blocks (soft-capped at
    /// [`MAX_DISK_IN_FLIGHT`] when admitting new files).
    disk_in_flight: usize,
    /// Outstanding remote-backed footer-read blocks (soft-capped at
    /// [`MAX_HTTP_IN_FLIGHT`]).
    http_in_flight: usize,
}

impl RowGroupMetadataFetcher {
    fn alloc_slot(&mut self, request: RowGroupMetadataRequest) -> usize {
        if let Some(slot) = self.free_slots.pop() {
            self.in_flight[slot] = Some(request);
            slot
        } else {
            self.in_flight.push(Some(request));
            self.in_flight.len() - 1
        }
    }

    /// Stage the request in `slot`'s freshly-generated reads: count each missing
    /// block toward the in-flight cap, and route its completion back to `slot`.
    /// Called after a region's reads are queued (`start`, or an exact re-read)
    /// and before [`next_fs_requests`](Self::next_fs_requests) drains them.
    ///
    /// Counting per block (not per file) makes the cap a soft one: a request can
    /// stage more blocks than the cap, but [`ready_for_more_work`] then stops
    /// admitting the *next* file until they drain. A request is local xor remote,
    /// so only one of the two counters moves.
    ///
    /// [`ready_for_more_work`]: Self::ready_for_more_work
    fn stage_reads(&mut self, slot: usize) {
        let request = self.in_flight[slot].as_ref().unwrap();
        let fs_keys: Vec<(FileLocation, usize)> = request
            .pending_fs
            .iter()
            .map(|fs| (FileLocation::Local(fs.file.clone()), fs.block.file_offset()))
            .collect();
        let http_keys: Vec<(FileLocation, usize)> = request
            .pending_http
            .iter()
            .map(|http| (FileLocation::Remote(http.remote.clone()), http.block.file_offset()))
            .collect();
        self.disk_in_flight += fs_keys.len();
        self.http_in_flight += http_keys.len();
        for key in fs_keys.into_iter().chain(http_keys) {
            self.routing.insert(key, slot);
        }
    }

    /// A completed read (fs or http): find the slot waiting on its
    /// `(location, file offset)` key, count the landed block off its current
    /// region, and advance the request.
    fn process_completion<S: Sender<IndexedRowGroup>>(
        &mut self,
        key: (FileLocation, usize),
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        if let Some(slot) = self.routing.remove(&key) {
            self.in_flight[slot].as_mut().unwrap().record_block();
            self.advance(slot, sender)?;
        }
        Ok(())
    }

    /// Advance the in-flight read in `slot` as far as it can without blocking on
    /// IO. While its current region is fully present, parse the region and either
    /// emit the file's row groups (done) or issue the next region's read (the
    /// exact-footer re-read); stop once a region has reads outstanding (await
    /// their completions) or the file is finished.
    ///
    /// Called from `consume` (after issuing the probe) and `process_completion`
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
                // The footer overflowed the probe; an exact read was issued. Stage
                // it; the loop condition re-checks whether it needs IO or was
                // already cached (and should be parsed right away).
                None => self.stage_reads(slot),
                Some(row_groups) => {
                    let request = self.in_flight[slot].take().unwrap();
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
        let DataFile { size, source, .. } = file;
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

        let slot =
            self.alloc_slot(RowGroupMetadataRequest::start(file_idx, location, size as usize));
        self.stage_reads(slot);
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
        self.disk_in_flight -= 1;
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
        self.http_in_flight -= 1;
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
    fn start(file_idx: usize, location: FileLocation, size: usize) -> Self {
        memory_ctx().file_cache().open_entry(location.clone());
        let mut request = Self {
            file_idx,
            location,
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
    /// as the current region. The previous region's blocks must already be
    /// drained (submitted) and completed — the fetcher counts `pending_*` as it
    /// stages them, so a leftover would be double-counted.
    fn read_region(&mut self, offset: usize, len: usize) {
        debug_assert!(
            self.pending_fs.is_empty() && self.pending_http.is_empty(),
            "read_region over un-drained pending reads"
        );
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

        Ok(Some(row_groups_from_footer(footer, self.location.clone())?))
    }
}
