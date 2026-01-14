//! Platform-specific I/O backends
//!
//! - Linux: Uses io_uring for high-performance async I/O
//! - Other Unix: Uses pread for synchronous positioned reads

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

#[cfg(target_os = "linux")]
mod uring_backend {
    use super::*;
    use crate::io::disk_buffer::DiskBuffer;
    use io_uring::{IoUring, opcode, types};
    use std::os::unix::io::RawFd;

    /// io_uring based backend for Linux
    pub struct IOBackend {
        pub ring: IoUring,
    }

    impl IOBackend {
        pub fn new(ring_size: u32) -> io::Result<Self> {
            Ok(Self {
                // ring: IoUring::builder().setup_coop_taskrun().build(ring_size)?,
                ring: IoUring::builder().build(ring_size)?,
            })
        }

        pub fn register_buffers(&self, buffers: impl Iterator<Item = (*mut libc::c_void, usize)>) {
            unsafe {
                self.ring
                    .submitter()
                    .register_buffers(
                        &buffers
                            .map(|(ptr, size)| libc::iovec {
                                iov_base: ptr as *mut libc::c_void,
                                iov_len: size,
                            })
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
            }
        }

        pub fn submit_read(
            &mut self,
            fd: RawFd,
            offset: u64,
            buffer: &mut DiskBuffer,
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

        pub fn submit(&mut self) -> io::Result<usize> {
            self.ring.submit()
        }

        pub fn submit_and_wait(&mut self, want: usize) -> io::Result<usize> {
            self.ring.submit_and_wait(want)
        }

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

        pub fn has_completions(&mut self) -> bool {
            !self.ring.completion().is_empty()
        }

        pub fn has_pending_submissions(&mut self) -> bool {
            !self.ring.submission().is_empty()
        }

        pub fn is_submission_full(&mut self) -> bool {
            self.ring.submission().is_full()
        }
    }
}

// ============================================================================
// Non-Linux Unix: pread fallback backend
// ============================================================================

#[cfg(all(unix, not(target_os = "linux")))]
mod pread_backend {
    use super::*;
    use crate::io::disk_buffer::DiskBuffer;
    use std::collections::VecDeque;
    use std::os::fd::BorrowedFd;
    use std::os::unix::io::RawFd;

    /// Pending read request for pread backend
    struct PendingRead {
        fd: RawFd,
        offset: u64,
        length: usize,
        request_id: Identifier,
        buffer_ptr: *mut u8,
    }

    unsafe impl Send for PendingRead {}

    /// pread-based fallback backend for non-Linux Unix systems
    pub struct IOBackend {
        pending: VecDeque<PendingRead>,
        completed: VecDeque<Identifier>,
        available: usize,
    }

    impl IOBackend {
        pub fn new(ring_size: u32) -> io::Result<Self> {
            Ok(Self {
                pending: VecDeque::new(),
                completed: VecDeque::new(),
                available: ring_size as usize,
            })
        }

        pub fn submit_read(
            &mut self,
            fd: RawFd,
            offset: u64,
            buffer: &mut DiskBuffer,
            length: usize,
            request_id: Identifier,
        ) -> Result<()> {
            self.available -= 1;
            self.pending.push_back(PendingRead {
                fd,
                offset,
                length,
                request_id,
                buffer_ptr: buffer.as_mut_ptr(),
            });
            Ok(())
        }

        pub fn register_buffers(&self, _buffers: impl Iterator<Item = (*mut libc::c_void, usize)>) {
        }

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
            self.available += self.completed.len();
            Ok(self
                .completed
                .drain(..)
                .into_iter()
                .map(|i| (0, i))
                .collect())
        }

        pub fn has_pending_submissions(&mut self) -> bool {
            !self.pending.is_empty()
        }

        pub fn has_completions(&mut self) -> bool {
            !self.completed.is_empty() || !self.pending.is_empty()
        }

        pub fn is_submission_full(&self) -> bool {
            self.available == 0
        }
    }
}

#[cfg(target_os = "linux")]
pub(crate) use uring_backend::IOBackend;

use crate::identified::Identifier;
#[cfg(all(unix, not(target_os = "linux")))]
pub(crate) use pread_backend::IOBackend;
