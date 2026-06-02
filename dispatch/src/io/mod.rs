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
use std::os::fd::RawFd;
use std::path::Path;

mod requester;
pub use requester::{Error as IORequesterError, IORequester};

/// A read request for one `MissingBlock`: read `block.len` bytes from `fd` at
/// `block.file_offset` straight into the block's `dest` (its pinned cache slot),
/// then `commit` it. The block carries everything needed to issue and commit
/// the read.
pub struct IORequest {
    pub fd: RawFd,
    pub block: MissingBlock,
}

impl Debug for IORequest {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IORequest")
            .field("fd", &self.fd)
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
