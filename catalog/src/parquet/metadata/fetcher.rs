//! Fetch stage: reads each file's Parquet footer — through the io_uring ring and
//! the compressed cache, exactly like a column-chunk read — and emits one [`TableFile`]
//! per file (its [`FileRef`] paired with its row groups). The footer-reading
//! analog of the column-chunk scan fetcher: it keeps many footer reads in flight,
//! bounded per medium, and shares the same slot/routing/in-flight bookkeeping
//! ([`RequestTracker`]). The footer-specific part is the per-file state machine
//! ([`FooterRead`]): read the tail probe window, parse the footer, and — if it
//! overflowed the probe — read it exactly before parsing.

use crate::catalog::TableFile;
use crate::parquet::request_tracker::{PendingRequest, ReadRequest, RequestTracker};
use crate::parquet::types::metadata::RowGroupMetadata;
use crate::parquet::types::table::{Error, FOOTER_PROBE_BYTES, Result, row_groups_from_footer};
use crate::store::{DataFile, DataFileSource, FileRef};
use dispatch::io::{FileLocation, FsRequest, HttpRequest, RemoteFile, open_direct_read};
use dispatch::memory::{CacheLookup, memory_ctx};
use dispatch::{Sender, Unary};
use std::sync::Arc;

const PARQUET_MAGIC: [u8; 4] = [b'P', b'A', b'R', b'1'];

/// Disk-backed footer-read blocks outstanding per worker before admitting
/// another file. Larger than a column-chunk read's depth: footers are small and
/// scattered, so keeping many in flight hides per-file read latency.
const MAX_DISK_IN_FLIGHT: usize = 32;

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

/// Reads files' footers and emits one [`TableFile`] per file, keeping many reads
/// in flight (bounded by `MAX_DISK_IN_FLIGHT`/`MAX_HTTP_IN_FLIGHT` via the
/// shared [`RequestTracker`]).
#[derive(Default)]
pub(super) struct TableFileMetadataFetcher {
    tracker: RequestTracker<FooterRead>,
}

impl TableFileMetadataFetcher {
    /// Advance the read in `slot` as far as it can without blocking on IO: while
    /// its current region is fully present, parse it and either emit the file's
    /// [`TableFile`] (done) or issue the next region's read (the exact-footer
    /// re-read). Stop once a region has reads outstanding or the file is finished.
    ///
    /// Called from `consume` (after the probe) and on each completion — both just
    /// "make a read happen, then advance" — so the pending-vs-cached decision
    /// lives only here.
    fn advance<S: Sender<TableFile>>(
        &mut self,
        slot: usize,
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        while !self.tracker.request_for_slot(slot).unwrap().is_pending() {
            let parsed = self
                .tracker
                .request_for_slot(slot)
                .unwrap()
                .parse_region()
                .map_err(crate::parquet::op_err)?;
            match parsed {
                // The footer overflowed the probe; an exact read was queued. Stage
                // it; the loop re-checks whether it needs IO or was already cached.
                None => self.tracker.stage_reads_for_slot(slot),
                Some(row_groups) => {
                    let request = self.tracker.take_request_at_slot(slot);
                    let row_groups = row_groups.into_iter().map(Arc::new).collect();
                    sender.send(TableFile::new(request.file, row_groups))?;
                    return Ok(());
                }
            }
        }
        Ok(())
    }
}

impl Unary<DataFile, TableFile> for TableFileMetadataFetcher {
    fn consume<S: Sender<TableFile>>(
        &mut self,
        file: DataFile,
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        // Open the file's transport. The size — which locates the footer's tail
        // window with no HEAD/suffix probe — is already carried by the data
        // file (`stat`ed at listing time for a local file, from the store
        // listing for a remote one). The `file_ref` rides through to the emitted
        // `TableFile`.
        let DataFile {
            file: file_ref,
            source,
        } = file;
        let size = file_ref.size as usize;
        let location = match source {
            DataFileSource::Local(path) => {
                let fd = open_direct_read(&path).map_err(crate::parquet::op_err)?;
                FileLocation::Local(Arc::new(fd))
            }
            DataFileSource::Remote { url, auth } => {
                let remote = Arc::new(
                    RemoteFile::open(url, auth, file_ref.size).map_err(crate::parquet::op_err)?,
                );
                FileLocation::Remote(remote)
            }
        };

        let slot = self
            .tracker
            .admit_request(FooterRead::start(file_ref, location, size));
        self.advance(slot, sender)
    }

    fn next_fs_requests(&mut self) -> dispatch::UnaryResult<Vec<FsRequest>> {
        Ok(self.tracker.take_fs_requests())
    }

    fn next_http_requests(&mut self) -> dispatch::UnaryResult<Vec<HttpRequest>> {
        Ok(self.tracker.take_http_requests())
    }

    fn ready_for_more_work(&mut self) -> bool {
        self.tracker.disk_in_flight() < MAX_DISK_IN_FLIGHT
            && self.tracker.http_in_flight() < crate::parquet::http_readahead()
    }

    fn process_fs_response<S: Sender<TableFile>>(
        &mut self,
        sender: &mut S,
        request: FsRequest,
    ) -> dispatch::UnaryResult<()> {
        for slot in self.tracker.complete(&ReadRequest::of_fs(&request)) {
            self.tracker.request_for_slot(slot).unwrap().record_block();
            self.advance(slot, sender)?;
        }
        Ok(())
    }

    fn process_http_response<S: Sender<TableFile>>(
        &mut self,
        sender: &mut S,
        request: HttpRequest,
    ) -> dispatch::UnaryResult<()> {
        for slot in self.tracker.complete(&ReadRequest::of_http(&request)) {
            self.tracker.request_for_slot(slot).unwrap().record_block();
            self.advance(slot, sender)?;
        }
        Ok(())
    }

    fn finish<S: Sender<TableFile>>(&mut self, _sender: &mut S) -> dispatch::UnaryResult<bool> {
        Ok(self.tracker.is_idle())
    }
}

/// One file's in-flight footer read.
///
/// It reads a region of the file (first the tail probe window, then — only if
/// the footer overflows it — the exact footer) into the compressed cache, tracking the
/// blocks still outstanding for the current region. Once a region's blocks have
/// all landed, [`parse_region`](Self::parse_region) turns it into the file's row
/// groups (or issues the exact re-read).
struct FooterRead {
    /// The file's durable identity, stamped onto the emitted [`TableFile`].
    file: FileRef,
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

impl PendingRequest for FooterRead {
    fn pending_fs(&mut self) -> &mut Vec<FsRequest> {
        &mut self.pending_fs
    }

    fn pending_http(&mut self) -> &mut Vec<HttpRequest> {
        &mut self.pending_http
    }
}

impl FooterRead {
    /// Register the file in the cache and issue the tail probe read
    /// `[size - probe, size)`.
    fn start(file: FileRef, location: FileLocation, size: usize) -> Self {
        memory_ctx().compressed_cache().open_entry(location.clone());
        let mut request = Self {
            file,
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
    /// drained (staged) and completed — the tracker counts staged reads, so a
    /// leftover would be double-counted.
    fn read_region(&mut self, offset: usize, len: usize) {
        debug_assert!(
            self.pending_fs.is_empty() && self.pending_http.is_empty(),
            "read_region over un-drained pending reads"
        );
        self.lookups = memory_ctx()
            .compressed_cache()
            .get(&self.location, offset, len);
        self.remaining = 0;
        for lookup in &self.lookups {
            if let Some(block) = lookup.missing() {
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
