//! I/O subsystem.
//!
//! We use direct I/O (bypassing the kernel page cache) for a few reasons:
//!
//! 1. We need to distinguish real disk reads from memory-satisfied reads.
//!    Our execution engine pipelines IO and CPU work: while the disk is
//!    serving the next batch of pages, workers process already-resident data.
//!    If a "read" silently hits the page cache and returns instantly, the
//!    scheduler sees it as completed IO and eagerly submits more, flooding
//!    the pipeline with new record batches instead of letting workers drain
//!    the ones already in flight (hot in cache).
//!
//! 2. We want explicit control over memory residency and eviction policy,
//!    rather than competing with the OS for the same physical pages.
//!
//! Provides direct I/O reads with platform-specific backends (io_uring on Linux,
//! pread on other Unix), reading straight into the file cache's 4 KB-aligned slot
//! regions.
//!
//! # Architecture
//!
//! - [`IORequester`] — submits and completes read requests on behalf of dataflows.
//! - [`backend::IOBackend`] — platform-specific submission/completion engine.

mod backend;

use crate::Identifier;
use crate::memory::file_cache::MissingBlock;
use std::fmt::{Debug, Formatter};
use std::fs::{File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::net::{SocketAddr, ToSocketAddrs};
use std::os::fd::RawFd;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use url::Url;

mod requester;
pub use requester::{Error as IORequesterError, IORequester};

pub mod http;

/// Where the bytes behind a cached region live: a local file descriptor, or a
/// remote HTTP(S) object fetched via range requests.
///
/// This is the key the [`FileCache`](crate::memory::file_cache::FileCache) uses
/// to bucket 2 MB regions, so it must be cheap to `Hash`/`Eq` — the cache's
/// pin re-check runs on every hit. `Local` compares an `i32`; `Remote` carries
/// an [`Arc<RemoteFile>`] whose `Hash`/`Eq` delegate to a single interned id
/// (never the URL string).
#[derive(Clone)]
pub enum FileLocation {
    Local(RawFd),
    Remote(Arc<RemoteFile>),
}

impl FileLocation {
    /// The local fd, or `None` for a remote location.
    pub fn as_raw_fd(&self) -> Option<RawFd> {
        match self {
            FileLocation::Local(fd) => Some(*fd),
            FileLocation::Remote(_) => None,
        }
    }
}

impl PartialEq for FileLocation {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (FileLocation::Local(a), FileLocation::Local(b)) => a == b,
            // RemoteFile's Eq compares the interned id, not the URL.
            (FileLocation::Remote(a), FileLocation::Remote(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for FileLocation {}

impl Hash for FileLocation {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            FileLocation::Local(fd) => {
                0u8.hash(state);
                fd.hash(state);
            }
            FileLocation::Remote(remote) => {
                1u8.hash(state);
                remote.hash(state);
            }
        }
    }
}

impl Debug for FileLocation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            FileLocation::Local(fd) => write!(f, "Local({fd})"),
            FileLocation::Remote(r) => write!(f, "Remote({})", r.url()),
        }
    }
}

static NEXT_REMOTE_FILE_ID: AtomicU32 = AtomicU32::new(0);

/// A remote HTTP(S) object, resolved once and reused for every range request
/// against it.
///
/// `Hash`/`Eq` delegate solely to the interned `id`, so using a `RemoteFile` as
/// (part of) a cache key is as cheap as comparing a `u32` — the URL/host/addr
/// are only read by the HTTP transport when actually issuing a request.
pub struct RemoteFile {
    /// Process-unique id; the only thing `Hash`/`Eq` look at.
    id: u32,
    url: Url,
    host: String,
    port: u16,
    /// `host:port` resolved at construction time.
    addr: SocketAddr,
    /// Origin-form request target (path + query) for the HTTP request line.
    request_target: String,
    is_https: bool,
}

impl RemoteFile {
    /// Parse `url`, resolve its host to a [`SocketAddr`], and intern it. The DNS
    /// lookup happens here (once) so the per-request hot path never blocks on
    /// resolution.
    pub fn open(url: Url) -> std::io::Result<Self> {
        let host = url
            .host_str()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "url has no host"))?
            .to_string();
        let is_https = match url.scheme() {
            "https" => true,
            "http" => false,
            other => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("unsupported url scheme: {other}"),
                ));
            }
        };
        let port = url
            .port_or_known_default()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "url has no port"))?;
        let addr = (host.as_str(), port)
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::NotFound, "host resolved to no addresses")
            })?;

        let mut request_target = url.path().to_string();
        if let Some(query) = url.query() {
            request_target.push('?');
            request_target.push_str(query);
        }

        Ok(Self {
            id: NEXT_REMOTE_FILE_ID.fetch_add(1, Ordering::Relaxed),
            url,
            host,
            port,
            addr,
            request_target,
            is_https,
        })
    }

    pub fn url(&self) -> &Url {
        &self.url
    }
    pub fn host(&self) -> &str {
        &self.host
    }
    pub fn port(&self) -> u16 {
        self.port
    }
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }
    pub fn request_target(&self) -> &str {
        &self.request_target
    }
    pub fn is_https(&self) -> bool {
        self.is_https
    }
}

impl PartialEq for RemoteFile {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for RemoteFile {}

impl Hash for RemoteFile {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

/// A filesystem read request: read `block` from local descriptor `fd`. This is
/// what an operator's [`next_fs_requests`](crate::operations::Operator::next_fs_requests)
/// yields — by construction it can only describe a local read, never a remote
/// one. Converted to an [`IORequest`] (via `From`) when handed to the requester.
pub struct FsRequest {
    pub fd: RawFd,
    pub block: MissingBlock,
}

/// An HTTP(S) read request: read `block` (a byte range) from `remote`. This is
/// what an operator's [`next_http_requests`](crate::operations::Operator::next_http_requests)
/// yields — by construction it can only describe a remote read, never a local
/// one. Converted to an [`IORequest`] (via `From`) when handed to the requester.
pub struct HttpRequest {
    pub remote: Arc<RemoteFile>,
    pub block: MissingBlock,
}

/// A read request for one `MissingBlock`: read `block.len` bytes at
/// `block.file_offset` from `location` straight into the block's `dest` (its
/// pinned cache slot), then `commit` it. The block carries everything needed to
/// issue and commit the read; `location` selects the transport (a local fd read
/// or an HTTP range request, both on the per-core uring).
///
/// Operators never build this directly — they yield the kind-specific
/// [`FsRequest`] / [`HttpRequest`], which convert here. That keeps the transport
/// kind a type-level guarantee at the operator boundary.
pub struct IORequest {
    pub location: FileLocation,
    pub block: MissingBlock,
}

impl From<FsRequest> for IORequest {
    fn from(r: FsRequest) -> Self {
        IORequest {
            location: FileLocation::Local(r.fd),
            block: r.block,
        }
    }
}

impl From<HttpRequest> for IORequest {
    fn from(r: HttpRequest) -> Self {
        IORequest {
            location: FileLocation::Remote(r.remote),
            block: r.block,
        }
    }
}

impl Debug for IORequest {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IORequest")
            .field("location", &self.location)
            .field("file_offset", &self.block.file_offset())
            .field("len", &self.block.len())
            .finish_non_exhaustive()
    }
}

/// Associates an [`IORequest`] with the dataflow and operator that issued it,
/// so completed reads can be routed back to the correct operator.
pub struct DataFlowRequest {
    pub data_flow_id: Identifier,
    pub operator_idx: Identifier,
    pub request: IORequest,
}

impl DataFlowRequest {
    pub fn new(data_flow_id: Identifier, operator_idx: Identifier, request: IORequest) -> Self {
        Self {
            data_flow_id,
            operator_idx,
            request,
        }
    }
}

/// Opens a file for direct (uncached) reads.
///
/// - **Linux**: uses `O_DIRECT`.
/// - **macOS**: uses `F_NOCACHE` (best-effort, not true direct I/O).
/// - **Other**: falls back to normal cached reads.
pub fn open_direct_read(path: &Path) -> std::io::Result<File> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)
    }

    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        let file = OpenOptions::new().read(true).open(path)?;
        let rc = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
        if rc == -1 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(file)
    }

    #[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
    {
        return OpenOptions::new().read(true).open(path);
    }

    #[cfg(not(unix))]
    {
        return OpenOptions::new().read(true).open(path);
    }
}
