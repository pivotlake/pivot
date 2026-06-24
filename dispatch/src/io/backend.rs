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

        /// Pushes a write of `length` bytes from `src` onto the submission queue
        /// (does not flush). `src` points into a pinned cache slot. Used to fill
        /// the on-disk cache after a remote (HTTP) read has landed.
        pub fn submit_write(
            &mut self,
            fd: RawFd,
            offset: u64,
            src: *const u8,
            length: usize,
            request_id: Identifier,
        ) -> Result<()> {
            let write_op = opcode::Write::new(types::Fd(fd), src, length as u32)
                .offset(offset)
                .build()
                .user_data(request_id as u64);

            unsafe {
                self.ring
                    .submission()
                    .push(&write_op)
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
            let mut out = Vec::new();
            loop {
                let dropped = {
                    let mut cq = self.ring.completion();
                    for cqe in &mut cq {
                        out.push((cqe.result(), cqe.user_data() as Identifier));
                    }
                    cq.overflow()
                    // `cq` drops here, publishing the consumed head so the kernel can
                    // refill the ring (and accept the backlog flushed below).
                };
                // A kernel without `IORING_FEAT_NODROP` silently *drops* completions it
                // can't fit. They're gone, so surface it loudly rather than losing reads.
                if dropped > 0 {
                    return Err(io::Error::other(format!(
                        "io_uring dropped {dropped} completions: ring oversubscribed"
                    )));
                }
                // A NODROP kernel instead parks the surplus in an overflow backlog
                // (flagged by `IORING_SQ_CQ_OVERFLOW`) and only moves it into the CQ on
                // the next `io_uring_enter`. We've just drained the CQ, so flush the
                // backlog and reap it. Without this, a worker that has stopped
                // submitting (e.g. spinning on a pipeline breaker's finish) would never
                // enter the ring again and would stall forever on reads that landed.
                if !self.ring.submission().cq_overflow() {
                    break;
                }
                self.ring.submit()?;
            }
            Ok(out)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::io::Write;
        use std::os::unix::io::AsRawFd;
        use std::time::Duration;

        /// Submitting more reads than the completion queue can hold makes the kernel
        /// park the surplus in its overflow backlog; a single drain must still recover
        /// every completion (by flushing the backlog), or a worker that has stopped
        /// submitting would stall forever on reads that already landed. Regression for
        /// the `SELECT *` row-group-fetcher livelock.
        #[test]
        fn completions_recover_an_overflowed_backlog() {
            // SQ of 4 → CQ of 8 (io_uring's 2x default).
            let mut backend = IOBackend::new(4).unwrap();
            let mut tmp = tempfile::NamedTempFile::new().unwrap();
            tmp.write_all(&vec![7u8; 256 * 1024]).unwrap();
            let fd = tmp.as_file().as_raw_fd();

            // More reads than the CQ holds, so the rest must overflow into the backlog.
            const N: usize = 20;
            let mut bufs: Vec<Vec<u8>> = (0..N).map(|_| vec![0u8; 4096]).collect();
            let mut id = 0;
            while id < N {
                let dest = bufs[id].as_mut_ptr();
                match backend.submit_read(fd, (id * 4096) as u64, dest, 4096, id as Identifier) {
                    Ok(()) => id += 1,
                    // SQ (depth 4) full: flush it to the kernel and retry the push.
                    Err(_) => {
                        backend.submit().unwrap();
                    }
                }
            }
            // Fill the CQ, then let the remaining reads complete into the backlog.
            backend.submit_and_wait(8).unwrap();
            std::thread::sleep(Duration::from_millis(50));
            assert!(
                backend.ring.submission().cq_overflow(),
                "test did not actually overflow the completion queue"
            );

            let recovered = backend.completions().unwrap();

            assert_eq!(recovered.len(), N, "overflow backlog was not fully recovered");
            assert!(recovered.iter().all(|&(res, _)| res == 4096));
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
        /// `true` for a `pwrite` (disk-cache fill), `false` for a `pread`.
        is_write: bool,
    }

    unsafe impl Send for PendingRead {}

    /// Synchronous pread-based fallback for non-Linux Unix.
    ///
    /// Reads are queued in [`submit_read`](Self::submit_read) and executed
    /// synchronously when [`submit`](Self::submit) is called. Completions are
    /// therefore always available immediately after submission.
    pub struct IOBackend {
        pending: VecDeque<PendingRead>,
        /// Completed ops as `(bytes_transferred, request_id)`, mirroring the
        /// io_uring backend's `(result, id)` so callers can check the byte count.
        completed: VecDeque<(i32, Identifier)>,
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
                is_write: false,
            });
            Ok(())
        }

        /// Queues a write from `src`; the actual `pwrite` happens at
        /// [`submit`](Self::submit) time. Mirrors [`submit_read`](Self::submit_read).
        pub fn submit_write(
            &mut self,
            fd: RawFd,
            offset: u64,
            src: *const u8,
            length: usize,
            request_id: Identifier,
        ) -> Result<()> {
            self.pending.push_back(PendingRead {
                fd,
                offset,
                length,
                request_id,
                buffer_ptr: src as *mut u8,
                is_write: true,
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
                let bytes = unsafe {
                    let borrowed_fd = BorrowedFd::borrow_raw(req.fd);
                    if req.is_write {
                        let buf = std::slice::from_raw_parts(req.buffer_ptr, req.length);
                        nix::sys::uio::pwrite(borrowed_fd, buf, req.offset as i64)?
                    } else {
                        let buf = std::slice::from_raw_parts_mut(req.buffer_ptr, req.length);
                        nix::sys::uio::pread(borrowed_fd, buf, req.offset as i64)?
                    }
                };

                self.completed.push_back((bytes as i32, req.request_id));
            }

            Ok(count)
        }

        pub fn completions(&mut self) -> io::Result<Vec<(i32, Identifier)>> {
            Ok(self.completed.drain(..).collect())
        }
    }
}

#[cfg(target_os = "linux")]
pub(crate) use uring_backend::IOBackend;

#[cfg(all(unix, not(target_os = "linux")))]
pub(crate) use pread_backend::IOBackend;
