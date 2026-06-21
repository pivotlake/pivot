//! HTTP(S) range reads backed by an optional on-disk cache.
//!
//! Every remote read becomes one [`RequestedRead`], satisfied by one or more
//! pieces and completed once every piece has landed:
//!
//! * With a [`DiskCache`], the requested run is split against what's on disk —
//!   resident pieces are read from the local cache file, holes are fetched over
//!   HTTP and written back.
//! * With no cache, it's a single piece fetched from the network.
//!
//! The original request is held by its [`RequestedRead`] and handed back
//! unchanged once the last piece lands, so the operator sees exactly the request
//! it submitted regardless of how it was satisfied.
//!
//! Like [`HttpEngine`], this is **ring-less**: the owning [`IORequester`] lends
//! it the worker's shared [`IOBackend`] (and the disk-op id counter) at submit
//! and drain time, so cache-file reads and write-backs ride the same per-core
//! io_uring as fs reads and HTTP sockets. The requester drains that ring and
//! routes each completion back here.
//!
//! [`IORequester`]: super::IORequester

use crate::Identifier;
use crate::io::IORequesterError as Error;
use crate::io::backend::IOBackend;
use crate::io::disk_cache::{DiskCache, Object, Segment};
use crate::io::http::{HttpEngine, RemoteRead, default_client_config};
use crate::io::{Completion, DataFlowRequest, FailedRead, HttpRequest};
use crate::memory::file_cache::MissingBlock;
use std::collections::HashMap;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Error>;

/// One drained read result: a completion or a terminal failure, matching
/// [`IORequester::completions`](super::IORequester::completions)'s element type.
type ReadResult = std::result::Result<Completion, FailedRead>;

/// HTTP range reads with an optional disk cache in front of the network.
pub(crate) struct CachedHttpEngine {
    http: HttpEngine,
    /// The shared disk cache, or `None` to read straight from the network.
    disk_cache: Option<Arc<DiskCache>>,

    /// One per in-flight read; holds the original request until every piece lands.
    requested_reads: HashMap<Identifier, RequestedRead>,
    next_read_id: Identifier,
    /// Pieces being read from a local cache file (backend disk-id space).
    cache_reads: HashMap<Identifier, CacheRead>,
    /// Pieces being fetched over HTTP (engine request-id space).
    http_reads: HashMap<Identifier, HttpRead>,
    next_http_id: Identifier,
    /// Write-backs populating the cache file after an HTTP piece landed (backend
    /// disk-id space). Each holds the slot pin until the write has read it.
    cache_writes: HashMap<Identifier, CacheWrite>,
}

/// The overarching record for one remote read an operator requested, in flight as
/// one or more pieces (cache-file reads and/or HTTP fetches). It holds the
/// original request and yields it unchanged once the last piece has landed.
struct RequestedRead {
    remaining: usize,
    request: DataFlowRequest<HttpRequest>,
}

/// One piece read from the local cache file.
struct CacheRead {
    read: Identifier,
    block: MissingBlock,
    /// Keeps the cache file open for the read's lifetime (`Some` whenever there's
    /// a resident block to read).
    _object: Option<Arc<Object>>,
}

/// One piece fetched over HTTP. `object` is `Some` when it should be written back
/// to that cache file once it lands, `None` for an uncached read.
struct HttpRead {
    read: Identifier,
    block: MissingBlock,
    object: Option<Arc<Object>>,
}

/// A write-back populating the cache file after an HTTP piece landed.
struct CacheWrite {
    object: Arc<Object>,
    file_offset: usize,
    len: usize,
    /// Keeps the source slot pinned until the write has read it.
    _block: MissingBlock,
}

impl CachedHttpEngine {
    pub fn new(
        http_config: Arc<rustls::ClientConfig>,
        disk_cache: Option<Arc<DiskCache>>,
    ) -> Result<Self> {
        Ok(Self {
            http: HttpEngine::new(http_config)?,
            disk_cache,
            requested_reads: HashMap::new(),
            next_read_id: 0,
            cache_reads: HashMap::new(),
            http_reads: HashMap::new(),
            next_http_id: 0,
            cache_writes: HashMap::new(),
        })
    }

    pub fn with_default_config(disk_cache: Option<Arc<DiskCache>>) -> Result<Self> {
        Self::new(default_client_config(), disk_cache)
    }

    /// Submit a read for a remote region as one [`RequestedRead`]: resident pieces
    /// are served from the cache file and holes fetched over HTTP, so only the
    /// missing ranges hit the network. With no cache the whole block is a single
    /// fetched piece. `disk_id` allocates ids for cache-file ops on the backend.
    pub fn request(
        &mut self,
        backend: &mut IOBackend,
        disk_id: &mut Identifier,
        request: DataFlowRequest<HttpRequest>,
    ) -> Result<()> {
        // The cache file for this object (its fd serves resident pieces), or `None`
        // to read straight from the network. `fd` is only used for resident pieces,
        // which only exist when `object` is `Some`, so the `None` default is dead.
        let object = self
            .disk_cache
            .as_deref()
            .and_then(|dc| dc.open_object(&request.request.remote));
        let fd = object.as_ref().map(|o| o.fd()).unwrap_or_default();
        let segments = match &object {
            Some(obj) => obj.split_into_segments(
                request.request.block.file_offset(),
                request.request.block.len(),
            ),
            // No cache: the whole block is one hole to fetch.
            None => vec![Segment {
                rel_offset: 0,
                len: request.request.block.len(),
                present: false,
            }],
        };

        let read = self.next_read_id;
        self.next_read_id += 1;

        for seg in &segments {
            let block = request
                .request
                .block
                .carve_sub_block(seg.rel_offset, seg.len);
            if seg.present {
                // Resident on disk: read it from the cache file into the slot.
                let id = *disk_id;
                *disk_id += 1;
                backend.submit_read(
                    fd,
                    block.file_offset() as u64,
                    block.dest(),
                    block.len(),
                    id,
                )?;
                self.cache_reads.insert(
                    id,
                    CacheRead {
                        read,
                        block,
                        _object: object.clone(),
                    },
                );
            } else {
                // A hole (or an uncached read): fetch over HTTP, written back once
                // it lands iff there's a cache object.
                let id = self.next_http_id;
                self.next_http_id += 1;
                let remote_read = RemoteRead {
                    remote: request.request.remote.clone(),
                    offset: block.file_offset() as u64,
                    len: block.len(),
                    dest: block.dest(),
                };
                self.start_http(backend, id, remote_read)?;
                self.http_reads.insert(
                    id,
                    HttpRead {
                        read,
                        block,
                        object: object.clone(),
                    },
                );
            }
            // Flush each segment as we go: one block can split into more pieces
            // than the ring's submission queue holds, so batching all the pushes
            // before a single submit could overflow it.
            backend.submit()?;
        }

        // Register the read once every piece is in flight. If a submit above
        // failed we never reach here: `request` drops and the pieces already
        // issued complete harmlessly (their read is simply absent).
        self.requested_reads.insert(
            read,
            RequestedRead {
                remaining: segments.len(),
                request,
            },
        );
        Ok(())
    }

    /// Hand a remote read to the engine (ring-driven on Linux, synchronous else).
    fn start_http(
        &mut self,
        backend: &mut IOBackend,
        id: Identifier,
        read: RemoteRead,
    ) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            self.http.start(&mut backend.ring, id, read)?;
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = backend;
            self.http.start(id, read)?;
        }
        Ok(())
    }

    /// Feed an HTTP socket completion to the engine (it may submit follow-up SQEs,
    /// including transparent reconnect-and-retry). Linux only — elsewhere the
    /// engine runs synchronously and never produces ring completions.
    #[cfg(target_os = "linux")]
    pub fn on_socket_completion(
        &mut self,
        backend: &mut IOBackend,
        user_data: u64,
        result: i32,
    ) -> Result<()> {
        self.http.on_cqe(&mut backend.ring, user_data, result)?;
        Ok(())
    }

    /// Handle a finished backend disk op that belongs to this engine — a
    /// cache-file read (advances its read) or a write-back (records the bytes
    /// resident). The caller has already ruled out its own fs reads.
    pub fn complete_disk(&mut self, id: Identifier, result: i32, out: &mut Vec<ReadResult>) {
        if let Some(cache_read) = self.cache_reads.remove(&id) {
            self.complete_cache_read(cache_read, result, out);
        } else if let Some(write) = self.cache_writes.remove(&id) {
            // Only record the bytes resident if the *whole* range landed — a
            // short/failed write must not mark blocks present that a later read
            // would then serve as garbage. A grown resident set may need eviction.
            if result >= 0 && result as usize == write.len {
                let added = write.object.mark_present(write.file_offset, write.len);
                if added > 0
                    && let Some(dc) = self.disk_cache.as_deref()
                {
                    dc.enforce_budget();
                }
            }
        } else {
            // The requester only delegates ids that aren't its fs reads, so an id
            // unknown to both maps means a completion was tracked nowhere — a bug.
            debug_assert!(false, "disk completion id {id} belongs to no in-flight op");
        }
    }

    /// Drain HTTP reads the engine finished this pass: each piece commits, gets
    /// written back if cached, and advances its read; the original request is
    /// yielded once its last piece lands. Terminal failures cancel the dataflow.
    pub fn drain(
        &mut self,
        backend: &mut IOBackend,
        disk_id: &mut Identifier,
        out: &mut Vec<ReadResult>,
    ) -> Result<()> {
        let mut queued_writeback = false;
        for id in self.http.take_completed() {
            if let Some(http_read) = self.http_reads.remove(&id) {
                queued_writeback |= self.complete_http_read(backend, disk_id, http_read, out)?;
            }
        }
        for (id, error) in self.http.take_failed() {
            if let Some(http_read) = self.http_reads.remove(&id)
                && let Some((data_flow_id, operator_idx)) = self.fail_read(http_read.read)
            {
                out.push(Err(FailedRead {
                    data_flow_id,
                    operator_idx,
                    error: error.into(),
                }));
            }
        }
        // Flush only if we queued write-backs (executing them synchronously on the
        // non-Linux pread backend; queueing SQEs on Linux).
        if queued_writeback {
            backend.submit()?;
        }
        Ok(())
    }

    /// `true` while any HTTP read is in flight.
    pub fn has_network_pending(&self) -> bool {
        self.http.has_active() || !self.http_reads.is_empty()
    }

    /// `true` while any cache-file read or write-back is in flight on the backend.
    pub fn has_disk_pending(&self) -> bool {
        !self.cache_reads.is_empty() || !self.cache_writes.is_empty()
    }

    /// Number of HTTP reads issued but not yet completed (the read-ahead depth).
    /// With no cache this is one per request; with a cache it counts the holes
    /// being fetched.
    pub fn network_in_flight(&self) -> usize {
        self.http_reads.len()
    }

    /// A cache-file piece landed: commit its sub-blocks and, if it was the last
    /// piece of its read, yield the original request. A failed read fails the
    /// whole requested read.
    fn complete_cache_read(
        &mut self,
        cache_read: CacheRead,
        result: i32,
        out: &mut Vec<ReadResult>,
    ) {
        if result < 0 {
            if let Some((data_flow_id, operator_idx)) = self.fail_read(cache_read.read) {
                out.push(Err(FailedRead {
                    data_flow_id,
                    operator_idx,
                    error: std::io::Error::from_raw_os_error(-result).into(),
                }));
            }
            return;
        }
        cache_read.block.commit();
        if let Some(request) = self.record_piece(cache_read.read) {
            out.push(Ok(Completion::Http(request)));
        }
    }

    /// An HTTP piece landed: commit it, queue a write-back into the cache file if
    /// it came from a cached read, and advance its read. Returns whether a
    /// write-back was queued (so the caller knows to flush the backend).
    fn complete_http_read(
        &mut self,
        backend: &mut IOBackend,
        disk_id: &mut Identifier,
        http_read: HttpRead,
        out: &mut Vec<ReadResult>,
    ) -> Result<bool> {
        http_read.block.commit();

        let queued = if let Some(object) = http_read.object {
            let id = *disk_id;
            *disk_id += 1;
            backend.submit_write(
                object.fd(),
                http_read.block.file_offset() as u64,
                http_read.block.dest(),
                http_read.block.len(),
                id,
            )?;
            self.cache_writes.insert(
                id,
                CacheWrite {
                    object,
                    file_offset: http_read.block.file_offset(),
                    len: http_read.block.len(),
                    _block: http_read.block,
                },
            );
            true
        } else {
            false
        };

        if let Some(request) = self.record_piece(http_read.read) {
            out.push(Ok(Completion::Http(request)));
        }
        Ok(queued)
    }

    /// Record one piece of requested read `read` as landed; returns the original
    /// request once the last piece lands (the pieces have already committed every
    /// sub-block). Yields `None` if the read already failed.
    fn record_piece(&mut self, read: Identifier) -> Option<DataFlowRequest<HttpRequest>> {
        let r = self.requested_reads.get_mut(&read)?;
        r.remaining -= 1;
        if r.remaining == 0 {
            Some(self.requested_reads.remove(&read).unwrap().request)
        } else {
            None
        }
    }

    /// Tear down a requested read whose piece failed, returning the dataflow to
    /// cancel (or `None` if an earlier failed piece already tore it down).
    fn fail_read(&mut self, read: Identifier) -> Option<(Identifier, Identifier)> {
        self.requested_reads
            .remove(&read)
            .map(|r| (r.request.data_flow_id, r.request.operator_idx))
    }
}
