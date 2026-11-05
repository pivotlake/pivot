//! HTTP(S) reads as part of the per-core io_uring, driven by
//! [`IORequester`](super::IORequester).
//!
//! Reads of [`Remote`](super::OpenFile::Remote) cache regions are served by
//! issuing HTTP `Range` requests whose body lands directly in the pinned cache
//! slot — exactly like a disk read, just over TLS instead of `pread`. The
//! transport ([`HttpEngine`]) is ring-less: on Linux it submits its socket SQEs
//! onto the *same* per-core io_uring the file reads use (see [`backend`]); on
//! other platforms it falls back to synchronous `std::net` + rustls.
//!
//! Unlike a disk read (one SQE → one CQE), an HTTPS range request is a multi-step
//! exchange (connect → TLS handshake → request → response). That state machine,
//! plus per-host keep-alive pooling, lives in [`backend`]; the requester only
//! tracks which dataflow each in-flight request belongs to and commits the block
//! once its body has fully landed.
//!
//! A whole-object download is the one request whose destination does not
//! exist when it is issued: its body lands in a [`DownloadBody`] allocated the
//! moment the response head says how long the body is.

use crate::io::RemoteFile;
use crate::memory::{BUFFER_SIZE, FileBytes, WriteBuffer};
use bytes::Bytes;
use std::sync::Arc;
use thiserror::Error;

mod backend;
mod http1;
mod proto;
mod tls;

pub use tls::default_client_config;

#[cfg(target_os = "linux")]
pub(crate) use backend::HTTP_TAG;
#[cfg(all(unix, not(target_os = "linux")))]
pub(crate) use backend::HttpCompletion;
pub(crate) use backend::HttpEngine;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Proto(#[from] proto::ProtoError),
    #[error("{0}")]
    Http1(#[from] http1::Http1Error),
    #[error("tls error: {0}")]
    Tls(#[from] rustls::Error),
    #[error("invalid dns name: {0}")]
    InvalidDnsName(#[from] rustls::pki_types::InvalidDnsNameError),
}

pub(crate) type Result<T, E = Error> = std::result::Result<T, E>;

/// A single remote range read: fetch `[offset, offset + len)` of `remote` into
/// `dest` (a pointer into the pinned cache slot, kept alive by the issuing
/// [`MissingExtent`](crate::memory::compressed_cache::MissingExtent)'s pin).
///
/// `Clone` so a transient transport failure can re-issue the same read on a
/// fresh connection (see the engine's retry path); the clone aliases the same
/// pinned `dest`, which is sound because only one attempt is ever in flight.
#[derive(Clone)]
pub(crate) struct RemoteRead {
    pub remote: Arc<RemoteFile>,
    pub offset: u64,
    pub len: usize,
    pub dest: *mut u8,
}

impl RemoteRead {
    /// Expected response-body length for this block-aligned range, based on the
    /// object size recorded when `RemoteFile` was opened.
    ///
    /// A range wholly inside the object should return `len` bytes. A range
    /// overlapping EOF should return only the bytes remaining after `offset`.
    /// An offset at or beyond EOF yields zero, though a conforming server will
    /// normally reject that range with `416`.
    ///
    /// The response parser requires this exact length. A mismatch indicates a
    /// truncated response or that the remote object no longer matches its recorded
    /// size.
    pub fn expected_body_len(&self) -> usize {
        (self.len as u64).min(self.remote.size().saturating_sub(self.offset)) as usize
    }
}

// SAFETY: `dest` points into a cache slot kept alive for the read's whole
// lifetime by the originating block's `Arc<ReadBuffer>` pin; only this read
// writes to `[dest, dest+len)` (currently-invalid sub-blocks), so moving the
// descriptor into the engine is sound. Mirrors the file path's PendingRead.
unsafe impl Send for RemoteRead {}

/// A whole-object upload. The body is reference counted so the operator-owned
/// bytes remain stable while socket SQEs point into them, and is held as runs so
/// it can live on the ring; the transport sends one run at a time.
#[derive(Clone)]
pub(crate) struct RemoteUpload {
    pub remote: Arc<RemoteFile>,
    pub data: Arc<FileBytes>,
}

/// A whole-object download: one `GET` of `remote` with no `Range`, its size
/// unknown until the response head carries it. The body's destination is not
/// part of the request: the exchange allocates it when the head parses, so a
/// retry on a fresh connection starts from nothing again.
#[derive(Clone)]
pub(crate) struct RemoteDownload {
    pub remote: Arc<RemoteFile>,
}

/// Where a download's body lands: runs of memory allocated once the response
/// head says how long the body is, in body order. One run per
/// [`BUFFER_SIZE`] of body when the transport runs on a worker, whose ring
/// hands out memory in slots of that size; one heap run when it runs on a
/// thread with no ring (the blocking pool of a non-Linux host).
pub(crate) struct DownloadBody {
    runs: Vec<BodyRun>,
    len: usize,
}

enum BodyRun {
    /// A ring slot, of which the first `len` bytes are body. Held as the
    /// write buffer it was taken as, so the slot goes back to the free pool
    /// with the last view of it, the way the caches hand out a transient
    /// buffer.
    Ring { buffer: WriteBuffer, len: usize },
    #[cfg(all(unix, not(target_os = "linux")))]
    Heap(Vec<u8>),
}

impl BodyRun {
    fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            Self::Ring { buffer, len } => &mut buffer.as_mut_slice()[..*len],
            #[cfg(all(unix, not(target_os = "linux")))]
            Self::Heap(bytes) => bytes,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Ring { len, .. } => *len,
            #[cfg(all(unix, not(target_os = "linux")))]
            Self::Heap(bytes) => bytes.len(),
        }
    }

    fn into_bytes(self) -> Bytes {
        match self {
            Self::Ring { buffer, len } => Bytes::from_owner(buffer).slice(..len),
            #[cfg(all(unix, not(target_os = "linux")))]
            Self::Heap(bytes) => Bytes::from(bytes),
        }
    }
}

impl DownloadBody {
    /// Room for `len` body bytes on the ring of the calling worker: as many
    /// slots as the body fills, taken now, so a pool too full to hold the
    /// whole body is found out before any byte of it lands.
    pub(crate) fn on_ring(len: u64) -> Self {
        let len = usize::try_from(len).expect("a response body longer than the address space");
        let mut runs = Vec::with_capacity(len.div_ceil(BUFFER_SIZE));
        let mut remaining = len;
        while remaining > 0 {
            let run_len = remaining.min(BUFFER_SIZE);
            runs.push(BodyRun::Ring {
                buffer: crate::memory::memory_ctx().get_write_buffer(false),
                len: run_len,
            });
            remaining -= run_len;
        }
        Self { runs, len }
    }

    /// Room for `len` body bytes on the heap, for a transport thread that has
    /// no ring.
    #[cfg(all(unix, not(target_os = "linux")))]
    pub(crate) fn on_heap(len: u64) -> Self {
        let len = usize::try_from(len).expect("a response body longer than the address space");
        let runs = if len == 0 {
            Vec::new()
        } else {
            vec![BodyRun::Heap(vec![0u8; len])]
        };
        Self { runs, len }
    }

    /// Total body length.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The rest of the run that body offset `offset` falls in: where the next
    /// bytes of a body `offset` bytes of which have landed go. Empty once the
    /// whole body has.
    pub(crate) fn region_at(&mut self, offset: usize) -> &mut [u8] {
        assert!(
            offset <= self.len,
            "offset {offset} past a {} byte body",
            self.len
        );
        let mut start = 0;
        for run in &mut self.runs {
            let len = run.len();
            if offset < start + len {
                return &mut run.as_mut_slice()[offset - start..];
            }
            start += len;
        }
        &mut []
    }

    /// The body as its runs, in order. Ring runs stay on the ring for as long
    /// as the bytes are held, and their slots are freed with them.
    pub(crate) fn into_bytes(self) -> Vec<Bytes> {
        self.runs.into_iter().map(BodyRun::into_bytes).collect()
    }
}

#[derive(Clone)]
pub(crate) enum RemoteRequest {
    Read(RemoteRead),
    Download(RemoteDownload),
    Upload(RemoteUpload),
}

impl RemoteRequest {
    pub fn remote(&self) -> &Arc<RemoteFile> {
        match self {
            Self::Read(r) => &r.remote,
            Self::Download(r) => &r.remote,
            Self::Upload(r) => &r.remote,
        }
    }

    pub fn request_head(&self) -> Vec<u8> {
        let remote = self.remote();
        let auth = remote.auth_header();
        match self {
            Self::Read(read) => proto::build_range_get(
                remote.host_header(),
                remote.request_target(),
                read.offset,
                read.len,
                auth.as_deref(),
            ),
            Self::Download(_) => proto::build_whole_object_get(
                remote.host_header(),
                remote.request_target(),
                auth.as_deref(),
            ),
            Self::Upload(upload) => proto::build_upload(
                remote.host_header(),
                remote.request_target(),
                upload.data.len(),
                auth.as_deref(),
            ),
        }
    }
}

// The read pointer has the safety argument above; downloads and uploads contain
// only Arc-owned data and are Send without an unsafe implementation of their own.
unsafe impl Send for RemoteRequest {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{init_test_free_pool, memory_ctx};

    /// A body longer than one slot is as many slots as it fills, filled run
    /// by run, and reads back as the bytes that landed in it.
    #[test]
    fn a_body_spans_runs_and_reads_back_in_order() {
        init_test_free_pool(4);
        let len = BUFFER_SIZE + 3;
        let mut body = DownloadBody::on_ring(len as u64);

        let mut written = 0;
        while written < len {
            let region = body.region_at(written);
            let n = region.len();
            for (i, byte) in region.iter_mut().enumerate() {
                *byte = ((written + i) % 251) as u8;
            }
            written += n;
        }
        let runs = body.into_bytes();

        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].len(), BUFFER_SIZE);
        assert_eq!(runs[1].len(), 3);
        let landed: Vec<u8> = runs.iter().flatten().copied().collect();
        assert!(
            landed
                .iter()
                .enumerate()
                .all(|(i, &b)| b == (i % 251) as u8)
        );
    }

    /// The slots a body took go back to the free pool with the last view of
    /// its bytes, not to a cache: nothing else would ever reclaim them.
    #[test]
    fn a_dropped_body_returns_its_slots_to_the_pool() {
        init_test_free_pool(2);
        let body = DownloadBody::on_ring(BUFFER_SIZE as u64 + 1);
        let pool_while_held = memory_ctx().pop_free_idx(false);

        drop(body.into_bytes());

        assert!(pool_while_held.is_none(), "the body holds both slots");
        assert!(memory_ctx().pop_free_idx(false).is_some());
    }

    #[test]
    fn an_empty_body_takes_no_slot() {
        init_test_free_pool(1);
        let body = DownloadBody::on_ring(0);

        let runs = body.into_bytes();

        assert!(runs.is_empty());
        assert!(memory_ctx().pop_free_idx(false).is_some());
    }
}
