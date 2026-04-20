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
//! pread on other Unix). Reads are aligned to the device's direct I/O alignment
//! and routed through a buffer pool managed by the memory subsystem.
//!
//! # Architecture
//!
//! - [`IORequester`] — submits and completes read requests on behalf of dataflows.
//! - [`backend::IOBackend`] — platform-specific submission/completion engine.
//! - [`alignment`] — detects direct I/O alignment requirements at runtime.

mod alignment;
mod backend;

use crate::Identifier;
use crate::memory::BUFFER_SIZE;
use alignment::DIO_ALIGNMENT;
use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::fs::{File, OpenOptions};
use std::os::fd::RawFd;
use std::path::Path;

mod requester;
pub use requester::{Error as IORequesterError, IORequester};

/// A file descriptor and byte offset identifying a single buffer-sized block on disk.
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct IOLocation {
    pub raw_fd: RawFd,
    pub offset: usize,
}

/// A read request targeting an `IOLocation`, carrying an opaque context that is
/// returned alongside the completed read buffer.
pub struct IORequest {
    pub location: IOLocation,
    pub ctx: Box<dyn Any + Send>,
}

impl Debug for IORequest {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IORequest")
            .field("location", &self.location)
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

/// A set of aligned [`IOLocation`]s that cover a byte range, plus the offsets within
/// the first and last buffers where the requested data actually begins and ends (because the reads
/// are aligned, there could be extraneous data in the beginning and end).
#[derive(Debug)]
pub struct AlignedRead {
    pub locations: Vec<IOLocation>,
    /// Byte offset into the first buffer where the requested range starts.
    pub first_offset: usize,
    /// Byte offset into the last buffer where the requested range ends.
    pub end_offset: usize,
}

/// Computes the aligned buffer-sized reads needed to cover the byte range `[start, end)`.
///
/// The start is rounded down to the nearest direct I/O alignment boundary, then
/// buffer-sized chunks are generated until `end` is covered.
pub fn create_aligned_read_from_start_end(raw_fd: RawFd, start: usize, end: usize) -> AlignedRead {
    let aligned_offset = start & !(*DIO_ALIGNMENT - 1);

    let mut locations = Vec::new();
    let mut offset = aligned_offset;
    while offset < end {
        locations.push(IOLocation { raw_fd, offset });
        offset += BUFFER_SIZE;
    }

    AlignedRead {
        end_offset: end - locations.last().unwrap().offset,
        locations,
        first_offset: start - aligned_offset,
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
