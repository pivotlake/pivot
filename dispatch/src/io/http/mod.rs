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
//! A whole-object download is the same `GET` without a `Range`, and the one
//! request whose destination does not exist when it is issued: its body lands
//! in a [`DownloadBody`] allocated the moment the response head says how long
//! the body is (see [`download`]).

use crate::io::RemoteFile;
use crate::memory::FileBytes;
use std::sync::Arc;
use thiserror::Error;

mod backend;
mod download;
mod http1;
mod proto;
mod tls;

pub(crate) use download::DownloadBody;
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

/// A single `GET` of `remote`: a sized range into a pinned cache slot, or the
/// whole object, whose size the response head declares.
///
/// `Clone` so a transient transport failure can re-issue the same request on
/// a fresh connection (see the engine's retry path); the clone aliases the
/// same pinned `dest`, which is sound because only one attempt is ever in
/// flight.
#[derive(Clone)]
pub(crate) struct RemoteGet {
    pub remote: Arc<RemoteFile>,
    pub body: GetBody,
}

/// What a [`RemoteGet`] fetches, and where its body lands.
#[derive(Clone, Copy)]
pub(crate) enum GetBody {
    /// `[offset, offset + len)` of an object opened with its size, into `dest`
    /// (a pointer into the pinned cache slot, kept alive by the issuing
    /// [`MissingExtent`](crate::memory::compressed_cache::MissingExtent)'s
    /// pin). The response must be a `206` of exactly the bytes the range
    /// covers (see [`RemoteFile::range_body_len`]).
    Range {
        offset: u64,
        len: usize,
        dest: *mut u8,
    },
    /// The whole object, with no `Range`: the response must be a `200` that
    /// declares its length, and the body is allocated then (see
    /// [`download`]).
    Whole,
}

// SAFETY: a range's `dest` points into a cache slot kept alive for the
// request's whole lifetime by the originating block's `Arc<ReadBuffer>` pin;
// only this request writes to `[dest, dest+len)` (currently-invalid
// sub-blocks), so moving the descriptor into the engine is sound. Mirrors the
// file path's PendingRead.
unsafe impl Send for RemoteGet {}

/// A whole-object upload. The body is reference counted so the operator-owned
/// bytes remain stable while socket SQEs point into them, and is held as runs so
/// it can live on the ring; the transport sends one run at a time.
#[derive(Clone)]
pub(crate) struct RemoteUpload {
    pub remote: Arc<RemoteFile>,
    pub data: Arc<FileBytes>,
}

#[derive(Clone)]
pub(crate) enum RemoteRequest {
    Get(RemoteGet),
    Upload(RemoteUpload),
}

impl RemoteRequest {
    pub fn remote(&self) -> &Arc<RemoteFile> {
        match self {
            Self::Get(get) => &get.remote,
            Self::Upload(upload) => &upload.remote,
        }
    }

    pub fn request_head(&self) -> Vec<u8> {
        let remote = self.remote();
        let auth = remote.auth_header();
        match self {
            Self::Get(get) => {
                let range = match get.body {
                    GetBody::Range { offset, len, .. } => Some((offset, len)),
                    GetBody::Whole => None,
                };
                proto::build_get(
                    remote.host_header(),
                    remote.request_target(),
                    range,
                    auth.as_deref(),
                )
            }
            Self::Upload(upload) => proto::build_upload(
                remote.host_header(),
                remote.request_target(),
                upload.data.len(),
                auth.as_deref(),
            ),
        }
    }
}

// A GET's pointer has the safety argument above; an upload contains only
// Arc-owned data and is Send without an unsafe implementation of its own.
unsafe impl Send for RemoteRequest {}
