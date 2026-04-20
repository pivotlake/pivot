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
    use crate::memory::WriteBuffer;
    use io_uring::{IoUring, opcode, types};
    use std::os::unix::io::RawFd;

    /// io_uring based backend for Linux.
    pub struct IOBackend {
        pub ring: IoUring,
    }

    impl IOBackend {
        pub fn new(ring_size: u32) -> io::Result<Self> {
            Ok(Self {
                ring: IoUring::builder().setup_coop_taskrun().build(ring_size)?,
            })
        }

        /// Pushes a read operation onto the submission queue (does not flush).
        pub fn submit_read(
            &mut self,
            fd: RawFd,
            offset: u64,
            buffer: &mut WriteBuffer,
            length: usize,
            request_id: Identifier,
        ) -> Result<()> {
            let read_op = opcode::Read::new(types::Fd(fd), buffer.as_mut_ptr(), length as u32)
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

        /// Drains the completion queue, returning `(bytes_read, request_id)` pairs.
        pub fn completions(&mut self) -> io::Result<Vec<(usize, Identifier)>> {
            self.ring
                .completion()
                .map(|cqe| {
                    let res = cqe.result();
                    if res < 0 {
                        Err(std::io::Error::from_raw_os_error(-res))
                    } else {
                        Ok((cqe.result() as usize, cqe.user_data() as Identifier))
                    }
                })
                .collect::<Result<Vec<_>, _>>()
        }
    }
}

#[cfg(all(unix, not(target_os = "linux")))]
mod pread_backend {
    use super::*;
    use crate::memory::WriteBuffer;
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

        /// Queues a read; the actual `pread` happens at [`submit`](Self::submit) time.
        pub fn submit_read(
            &mut self,
            fd: RawFd,
            offset: u64,
            buffer: &mut WriteBuffer,
            length: usize,
            request_id: Identifier,
        ) -> Result<()> {
            self.pending.push_back(PendingRead {
                fd,
                offset,
                length,
                request_id,
                buffer_ptr: buffer.as_mut_ptr(),
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

        pub fn completions(&mut self) -> io::Result<Vec<(usize, Identifier)>> {
            Ok(self.completed.drain(..).map(|i| (0, i)).collect())
        }
    }
}

#[cfg(target_os = "linux")]
pub(crate) use uring_backend::IOBackend;

#[cfg(all(unix, not(target_os = "linux")))]
pub(crate) use pread_backend::IOBackend;
