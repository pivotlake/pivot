//! Platform-specific I/O backends.
//!
//! Both backends expose the same API so [`super::requester::IORequester`] can
//! treat them identically:
//!
//! - **Linux** — `uring_backend::IOBackend` wraps `io_uring` for truly
//!   asynchronous, kernel-managed reads. Submissions are batched in the SQ and
//!   completions are drained from the CQ.
//! - **Other Unix** — `pread_backend::IOBackend` executes reads synchronously
//!   via `pread(2)` at `submit` time, so "completions" are always immediately
//!   available.

use crate::Identifier;

use std::io;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("submission queue full")]
    SubmissionQueueFull,
    #[error("io-uring operation failed with code {0}")]
    UringOperationFailed(i32),
    #[error("IO Uring error: {0}")]
    IOUring(i32),
}

type Result<T, E = Error> = std::result::Result<T, E>;

// ============================================================================
// Linux: io_uring backend
// ============================================================================

#[cfg(target_os = "linux")]
mod uring_backend {
    use super::*;
    use io_uring::{IoUring, opcode, types};
    use std::os::unix::io::RawFd;

    /// io_uring based backend for Linux.
    pub struct IOBackend {
        pub ring: IoUring,
    }

    impl IOBackend {
        pub fn new(ring_size: u32) -> io::Result<Self> {
            // `IORING_SETUP_COOP_TASKRUN` (kernel >= 5.19) lets the kernel skip an
            // IPI when completing work on the submitting task — a latency win. On
            // older kernels `io_uring_setup` rejects the unknown flag with EINVAL;
            // fall back to a plain ring rather than failing to start the engine.
            let ring = match IoUring::builder().setup_coop_taskrun().build(ring_size) {
                Ok(ring) => ring,
                Err(e) if e.raw_os_error() == Some(libc::EINVAL) => {
                    IoUring::builder().build(ring_size)?
                }
                Err(e) => return Err(e),
            };
            Ok(Self { ring })
        }

        /// Pushes a read of `length` bytes into `dest` onto the submission queue
        /// (does not flush). `dest` points into a pinned cache slot.
        pub fn submit_read(
            &mut self,
            fd: RawFd,
            offset: u64,
            dest: *mut u8,
            length: usize,
            request_id: Identifier,
        ) -> Result<()> {
            let read_op = opcode::Read::new(types::Fd(fd), dest, length as u32)
                .offset(offset)
                .build()
                .user_data(request_id as u64);

            unsafe {
                self.ring
                    .submission()
                    .push(&read_op)
                    .map_err(|_| Error::SubmissionQueueFull)?;
            }

            Ok(())
        }

        /// Flushes the submission queue to the kernel.
        pub fn submit(&mut self) -> io::Result<usize> {
            self.ring.submit()
        }

        /// Flushes and blocks until at least `want` completions are ready.
        pub fn submit_and_wait(&mut self, want: usize) -> io::Result<usize> {
            self.ring.submit_and_wait(want)
        }

        /// Drains the completion queue, returning the raw `(result, request_id)`
        /// pairs. `result` is the kernel's signed CQE value: `>= 0` is the byte
        /// count, `< 0` is `-errno`. We deliberately do **not** collapse a
        /// negative result into an error here — that would discard the
        /// `request_id`, so a transient socket error (e.g. `ECONNRESET` on an
        /// HTTP range read) could not be mapped back to its connection and
        /// retried. The caller interprets the sign per request instead.
        pub fn completions(&mut self) -> io::Result<Vec<(i32, Identifier)>> {
            Ok(self
                .ring
                .completion()
                .map(|cqe| (cqe.result(), cqe.user_data() as Identifier))
                .collect())
        }
    }
}

#[cfg(all(unix, not(target_os = "linux")))]
mod pread_backend {
    use super::*;
    use std::collections::VecDeque;
    use std::os::fd::BorrowedFd;
    use std::os::unix::io::RawFd;

    struct PendingRead {
        fd: RawFd,
        offset: u64,
        length: usize,
        request_id: Identifier,
        buffer_ptr: *mut u8,
    }

    unsafe impl Send for PendingRead {}

    /// Synchronous pread-based fallback for non-Linux Unix.
    ///
    /// Reads are queued in [`submit_read`](Self::submit_read) and executed
    /// synchronously when [`submit`](Self::submit) is called. Completions are
    /// therefore always available immediately after submission.
    pub struct IOBackend {
        pending: VecDeque<PendingRead>,
        completed: VecDeque<Identifier>,
    }

    impl IOBackend {
        pub fn new(_ring_size: u32) -> io::Result<Self> {
            Ok(Self {
                pending: VecDeque::new(),
                completed: VecDeque::new(),
            })
        }

        /// Queues a read into `dest`; the actual `pread` happens at
        /// [`submit`](Self::submit) time.
        pub fn submit_read(
            &mut self,
            fd: RawFd,
            offset: u64,
            dest: *mut u8,
            length: usize,
            request_id: Identifier,
        ) -> Result<()> {
            self.pending.push_back(PendingRead {
                fd,
                offset,
                length,
                request_id,
                buffer_ptr: dest,
            });
            Ok(())
        }

        /// Executes all pending reads synchronously via `pread(2)`.
        pub fn submit(&mut self) -> io::Result<usize> {
            self.execute_pending()
        }

        pub fn submit_and_wait(&mut self, _want: usize) -> io::Result<usize> {
            self.execute_pending()
        }

        fn execute_pending(&mut self) -> io::Result<usize> {
            let count = self.pending.len();

            while let Some(req) = self.pending.pop_front() {
                // SAFETY: The fd is valid because it's from an open file managed by the reader
                unsafe {
                    let buf = std::slice::from_raw_parts_mut(req.buffer_ptr, req.length);
                    let borrowed_fd = BorrowedFd::borrow_raw(req.fd);
                    nix::sys::uio::pread(borrowed_fd, buf, req.offset as i64)
                }?;

                self.completed.push_back(req.request_id);
            }

            Ok(count)
        }

        pub fn completions(&mut self) -> io::Result<Vec<(i32, Identifier)>> {
            Ok(self.completed.drain(..).map(|i| (0, i)).collect())
        }
    }
}

#[cfg(target_os = "linux")]
pub(crate) use uring_backend::IOBackend;

#[cfg(all(unix, not(target_os = "linux")))]
pub(crate) use pread_backend::IOBackend;
