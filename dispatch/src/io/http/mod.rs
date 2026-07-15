//! HTTP(S) reads as part of the per-core io_uring, driven by
//! [`IORequester`](super::IORequester).
//!
//! Reads of [`Remote`](super::FileLocation::Remote) cache regions are served by
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

use crate::io::{HttpMethod, RemoteFile};
use std::sync::Arc;
use thiserror::Error;

mod backend;
mod http1;
mod proto;
mod tls;
mod upload;

pub use tls::default_client_config;

#[cfg(target_os = "linux")]
pub(crate) use backend::HTTP_TAG;
#[cfg(all(unix, not(target_os = "linux")))]
pub(crate) use backend::HttpCompletion;
pub(crate) use backend::HttpEngine;

#[cfg(target_os = "linux")]
pub(crate) use upload::UPLOAD_TAG;
#[cfg(all(unix, not(target_os = "linux")))]
pub(crate) use upload::UploadCompletion;
pub(crate) use upload::UploadEngine;

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

// SAFETY: `dest` points into a cache slot kept alive for the read's whole
// lifetime by the originating block's `Arc<ReadBuffer>` pin; only this read
// writes to `[dest, dest+len)` (currently-invalid sub-blocks), so moving the
// descriptor into the engine is sound. Mirrors the file path's PendingRead.
unsafe impl Send for RemoteRead {}

/// A single object upload: send `data` to `remote` with `method` and the
/// backend-signed `headers`. `Clone` so a transient transport failure can
/// re-issue it on a fresh connection — a PUT to a unique object key is
/// idempotent, so a retry either creates the object once or overwrites it with
/// the identical bytes. `data` is reference-counted, so a clone shares the buffer
/// rather than copying the (multi-megabyte) file.
#[derive(Clone)]
pub(crate) struct RemoteWrite {
    pub remote: Arc<RemoteFile>,
    pub method: HttpMethod,
    pub headers: Vec<(String, String)>,
    pub data: Arc<Vec<u8>>,
}
