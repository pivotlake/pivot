//! Platform-specific I/O backends.
//!
//! Both backends expose the same API so [`super::requester::IORequester`] can
//! treat them identically:
//!
//! - **Linux** — `uring_backend::IOBackend` wraps `io_uring` for truly
//!   asynchronous, kernel-managed reads. Submissions are batched in the SQ and
//!   completions are drained from the CQ.
//! - **Other Unix (macOS)**: `pread_pool_backend::IOBackend` hands each read to
//!   a shared pool of blocking `pread(2)`/`pwrite(2)` threads, so submission
//!   never blocks the worker and reads run concurrently (filling device queue
//!   depth and overlapping with compute). Each pool thread delivers its result
//!   back over the issuing worker's MPSC channel; `completions` `try_recv`-drains
//!   it and `submit_and_wait` `recv`-parks on it. There is no userspace lock on
//!   the hot path.

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

/// Largest byte count a single io_uring read/write/send may carry. The opcode
/// length field is 32-bit, so a larger `usize` would truncate when cast; this is
/// also the kernel's own per-syscall transfer cap (`MAX_RW_COUNT`). A transfer
/// bigger than this is submitted in successive ops, each resuming where the last
/// left off, so the caller must already handle a short completion (it does).
pub(crate) const MAX_IO_OP_LEN: usize = 0x7fff_f000;

/// The `user_data` of a worker wake-up event, distinct from every disk id (small
/// counters) and every HTTP id (the `HTTP_TAG` bit plus a counter, which never
/// reaches all-ones). Both backends filter it out of `completions`, so a wake
/// never surfaces as an IO completion.
pub(crate) const WAKE_UD: u64 = u64::MAX;

/// Wakes a worker blocked in its backend's completion wait, from any thread.
///
/// On Linux the handle is the worker ring's eventfd: writing it completes the
/// standing poll the backend keeps armed, so `submit_and_wait` returns. On
/// other platforms it is a clone of the completion channel's sender, and the
/// wake is a sentinel message. Either way the woken worker sees an empty-handed
/// wait and re-runs its loop, which is where it finds the work the waker was
/// signalling about. The handle stays valid for the worker's lifetime (workers
/// live until process shutdown).
#[derive(Clone)]
pub enum RingWakeHandle {
    #[cfg(target_os = "linux")]
    EventFd(std::os::unix::io::RawFd),
    #[cfg(not(target_os = "linux"))]
    Channel(crossbeam_channel::Sender<(i32, Identifier)>),
}

impl RingWakeHandle {
    pub fn wake(&self) {
        match self {
            #[cfg(target_os = "linux")]
            RingWakeHandle::EventFd(fd) => {
                let one: u64 = 1;
                // A full eventfd counter (EAGAIN) already means a wake is
                // pending, which is all this write is for.
                unsafe {
                    libc::write(*fd, (&one as *const u64).cast(), size_of::<u64>());
                }
            }
            #[cfg(not(target_os = "linux"))]
            RingWakeHandle::Channel(sender) => {
                let _ = sender.send((0, WAKE_UD as Identifier));
            }
        }
    }
}

// ============================================================================
// Linux: io_uring backend
// ============================================================================

#[cfg(target_os = "linux")]
mod uring_backend {
    use super::*;
    use io_uring::{IoUring, opcode, types};
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::io::RawFd;

    /// io_uring based backend for Linux.
    pub struct IOBackend {
        pub ring: IoUring,
        /// The worker's wake-up line: [`RingWakeHandle::wake`] writes it, the
        /// standing poll below turns the write into a CQE, and a blocked
        /// `submit_and_wait` returns.
        wake_eventfd: std::os::fd::OwnedFd,
        /// Whether the [`WAKE_UD`] poll on the eventfd is currently in flight.
        /// Re-armed after every firing (and after a full submission queue made
        /// arming fail).
        wake_armed: bool,
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
            let raw = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
            if raw < 0 {
                return Err(io::Error::last_os_error());
            }
            let wake_eventfd = unsafe { std::os::fd::OwnedFd::from_raw_fd(raw) };
            let mut backend = Self {
                ring,
                wake_eventfd,
                wake_armed: false,
            };
            backend.arm_wake();
            Ok(backend)
        }

        /// The handle a waker uses to interrupt this ring's completion wait.
        pub fn wake_handle(&self) -> RingWakeHandle {
            RingWakeHandle::EventFd(self.wake_eventfd.as_raw_fd())
        }

        /// Keep one poll on the wake eventfd in flight. A full submission
        /// queue just defers the re-arm to the next submit/completion pass;
        /// a wake written in the meantime stays in the eventfd counter and
        /// fires the moment the poll lands.
        fn arm_wake(&mut self) {
            if self.wake_armed {
                return;
            }
            let poll = opcode::PollAdd::new(
                types::Fd(self.wake_eventfd.as_raw_fd()),
                libc::POLLIN as u32,
            )
            .build()
            .user_data(WAKE_UD);
            if unsafe { self.ring.submission().push(&poll) }.is_ok() {
                self.wake_armed = true;
            }
        }

        /// Reset the eventfd counter after its poll fired.
        fn drain_wake(&mut self) {
            let mut counter = 0u64;
            unsafe {
                libc::read(
                    self.wake_eventfd.as_raw_fd(),
                    (&mut counter as *mut u64).cast(),
                    size_of::<u64>(),
                );
            }
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
            let length = length.min(MAX_IO_OP_LEN);
            // Reads execute inline in `io_uring_enter`: the kernel issues them
            // with nowait semantics and punts a would-block request to io-wq
            // itself, so submission does not sleep in the block layer's
            // request allocation even when the device queue is full. An
            // unconditional `Flags::ASYNC` punt would avoid even the inline
            // attempt, but pays an io-wq dispatch and a kernel-worker wakeup
            // per read, which costs latency-bound queries several percent of
            // cold time. The requester's in-flight cap additionally bounds how
            // hard one worker can push the device queue.
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

        /// Pushes a write from `src` onto the submission queue (does not flush).
        /// `src` is the request-owned buffer to write out - a pinned cache slot for
        /// a cache write-back (filling the on-disk cache after a remote read lands),
        /// or an operator's own buffer for a filesystem write. At most
        /// [`MAX_IO_OP_LEN`] bytes go out per op, so a larger `length` is written
        /// over successive calls resuming at `offset` - the caller drives the
        /// resubmit on a short completion.
        pub fn submit_write(
            &mut self,
            fd: RawFd,
            offset: u64,
            src: *const u8,
            length: usize,
            request_id: Identifier,
        ) -> Result<()> {
            let length = length.min(MAX_IO_OP_LEN);
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
            self.arm_wake();
            self.ring.submit()
        }

        /// Flushes and blocks until at least `want` completions are ready. A
        /// worker-wake write to the eventfd counts as a completion (its poll
        /// CQE), so a waker can cut the wait short: see [`RingWakeHandle`].
        pub fn submit_and_wait(&mut self, want: usize) -> io::Result<usize> {
            self.arm_wake();
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
                let mut wake_fired = false;
                let dropped = {
                    let mut cq = self.ring.completion();
                    for cqe in &mut cq {
                        if cqe.user_data() == WAKE_UD {
                            wake_fired = true;
                            continue;
                        }
                        out.push((cqe.result(), cqe.user_data() as Identifier));
                    }
                    cq.overflow()
                    // `cq` drops here, publishing the consumed head so the kernel can
                    // refill the ring (and accept the backlog flushed below).
                };
                if wake_fired {
                    self.drain_wake();
                    self.wake_armed = false;
                    self.arm_wake();
                }
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
        /// submitting would stall forever on reads that already landed.
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

            assert_eq!(
                recovered.len(),
                N,
                "overflow backlog was not fully recovered"
            );
            assert!(recovered.iter().all(|&(res, _)| res == 4096));
        }
    }
}

#[cfg(all(unix, not(target_os = "linux")))]
mod pread_pool_backend {
    use super::*;
    use crossbeam_channel::{Receiver, Sender, unbounded};
    use std::collections::VecDeque;
    use std::os::fd::BorrowedFd;
    use std::os::unix::io::RawFd;
    use std::sync::OnceLock;

    /// One read or write for a pool thread to perform: a `pread`/`pwrite` into or
    /// out of a pinned cache slot, plus the channel its result is delivered on.
    struct Job {
        fd: RawFd,
        offset: u64,
        /// Destination of a read or source of a write; points into a pinned
        /// cache slot kept alive by an `Arc` on the issuing block until commit.
        buffer_ptr: *mut u8,
        length: usize,
        request_id: Identifier,
        /// `true` for a `pwrite` (disk-cache fill), `false` for a `pread`.
        is_write: bool,
        /// The issuing backend's completion channel; the pool thread sends the
        /// `(result, request_id)` here once the op finishes.
        sink: Sender<(i32, Identifier)>,
    }

    // SAFETY: `buffer_ptr` addresses a pinned cache slot whose `Arc` the issuing
    // block holds for the whole flight of the op; only the one pool thread that
    // takes this job touches that region, and only until it sends the completion.
    unsafe impl Send for Job {}

    /// The process-wide pool of blocking-I/O threads. Created once, lazily, and
    /// shared by every worker's backend so the threads load-balance: any pool
    /// thread can serve any worker's read, which is what keeps the device queue
    /// full under a burst from a single worker.
    struct IoPool {
        jobs: Sender<Job>,
    }

    fn io_pool() -> &'static IoPool {
        static POOL: OnceLock<IoPool> = OnceLock::new();
        POOL.get_or_init(|| {
            let (jobs_tx, jobs_rx) = unbounded::<Job>();
            for _ in 0..io_pool_thread_count() {
                let jobs_rx = jobs_rx.clone();
                std::thread::Builder::new()
                    .name("pivot-io".to_string())
                    .spawn(move || run_pool_thread(&jobs_rx))
                    .expect("failed to spawn io pool thread");
            }
            IoPool { jobs: jobs_tx }
        })
    }

    /// Pool size: four blocking threads per core (override with `PIVOT_IO_THREADS`).
    /// The threads spend their time parked in `pread`, not on the CPU, so
    /// oversubscribing keeps the device's queue deep (many reads in flight) without
    /// starving the dataflow workers.
    fn io_pool_thread_count() -> usize {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        crate::env::get_env_var_with_default("PIVOT_IO_THREADS", cores * 4)
    }

    fn run_pool_thread(jobs: &Receiver<Job>) {
        // The channel closes only when the (static) pool's sender is dropped,
        // i.e. never; this loop runs for the life of the process.
        while let Ok(job) = jobs.recv() {
            // Every job MUST yield exactly one completion: a worker parked in
            // `submit_and_wait` blocks until its read reports back, and this
            // thread must survive to serve the next job. `perform` is a raw
            // syscall and shouldn't panic, but if it ever did (e.g. a bad
            // pointer), an unguarded unwind would kill this thread, strand that
            // worker forever, and shrink the pool. Catch it and fail just that
            // read instead.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| perform(&job)))
                .unwrap_or(-libc::EIO);
            // The issuing backend may have been dropped while this op was in
            // flight (its receiver gone); its completion then has nowhere to go,
            // so a failed send is expected and ignored.
            let _ = job.sink.send((result, job.request_id));
        }
    }

    /// Runs the job's `pread`/`pwrite`, looping until the whole slot has moved,
    /// and returns the bytes transferred or `-errno`, mirroring the signed CQE
    /// result the io_uring backend reports so the requester interprets failures
    /// by sign rather than by type.
    ///
    /// A blocking `pread`/`pwrite` may move fewer bytes than asked (a partial
    /// transfer) or be interrupted by a signal before moving any (`EINTR`).
    /// io_uring surfaces neither for a regular-file op, so this retries rather
    /// than hand back a half-filled slot or cancel the query: it loops on a
    /// partial transfer, retries on `EINTR`, and stops on `0` (end of file).
    fn perform(job: &Job) -> i32 {
        let mut moved = 0usize;
        while moved < job.length {
            // SAFETY: `fd` belongs to a file the issuing request keeps open, and
            // `buffer_ptr`/`length` describe its pinned slot, live for this op
            // (see the `Send` note on `Job`). `moved < length`, so the offset
            // pointer and remaining length stay inside that slot.
            let outcome = unsafe {
                let fd = BorrowedFd::borrow_raw(job.fd);
                let ptr = job.buffer_ptr.add(moved);
                let remaining = job.length - moved;
                let at = (job.offset + moved as u64) as i64;
                if job.is_write {
                    let buf = std::slice::from_raw_parts(ptr as *const u8, remaining);
                    nix::sys::uio::pwrite(fd, buf, at)
                } else {
                    let buf = std::slice::from_raw_parts_mut(ptr, remaining);
                    nix::sys::uio::pread(fd, buf, at)
                }
            };
            match outcome {
                // End of file (read) or a write that made no progress: stop and
                // report what landed, leaving the shortfall for the caller's
                // `result == len` check to treat as a failure.
                Ok(0) => break,
                Ok(bytes) => moved += bytes,
                Err(nix::errno::Errno::EINTR) => continue,
                // Negate to match io_uring's `-errno`, guarding the zero hole: an
                // errno nix cannot classify is `UnknownErrno` (value 0), and a
                // bare `-0` would be misread as a 0-byte success.
                Err(errno) => {
                    let code = errno as i32;
                    return if code > 0 { -code } else { -libc::EIO };
                }
            }
        }
        moved as i32
    }

    /// Non-Linux backend: a thin handle onto the shared blocking-I/O pool.
    ///
    /// Reads and writes are staged in [`submit_read`](Self::submit_read) /
    /// [`submit_write`](Self::submit_write) and dispatched to the pool in
    /// [`submit`](Self::submit) (mirroring io_uring's SQ-push-then-flush). The
    /// pool runs them concurrently and delivers each result back over `rx`;
    /// [`completions`](Self::completions) `try_recv`-drains it, and
    /// [`submit_and_wait`](Self::submit_and_wait) `recv`-parks on it.
    pub struct IOBackend {
        /// This worker's job sender into the shared pool.
        pool: Sender<Job>,
        /// Cloned onto each dispatched [`Job`] so the pool thread can deliver the
        /// completion back here.
        sink: Sender<(i32, Identifier)>,
        /// Receives completions from the pool threads. MPSC: many pool threads
        /// send, this one backend receives.
        rx: Receiver<(i32, Identifier)>,
        /// Completions pulled off `rx` but not yet handed to the caller; lets
        /// `submit_and_wait` block for `want` of them while leaving them for
        /// `completions` to drain.
        ready: VecDeque<(i32, Identifier)>,
        /// Ops dispatched to the pool that have not yet reported back. Bounds how
        /// long `submit_and_wait` will park (never past what's outstanding) and
        /// gates the drain in `Drop`.
        in_flight: usize,
        /// Staged ops not yet handed to the pool; flushed by [`submit`](Self::submit).
        staged: Vec<Job>,
    }

    impl IOBackend {
        /// `_ring_size` is accepted for parity with the io_uring backend's
        /// constructor but ignored here: the pool's concurrency is set by
        /// `io_pool_thread_count()` and its backlog is an unbounded channel.
        pub fn new(_ring_size: u32) -> io::Result<Self> {
            let (sink, rx) = unbounded();
            Ok(Self {
                pool: io_pool().jobs.clone(),
                sink,
                rx,
                ready: VecDeque::new(),
                in_flight: 0,
                staged: Vec::new(),
            })
        }

        /// Stages a read into `dest`; dispatched to the pool at
        /// [`submit`](Self::submit) time.
        pub fn submit_read(
            &mut self,
            fd: RawFd,
            offset: u64,
            dest: *mut u8,
            length: usize,
            request_id: Identifier,
        ) -> Result<()> {
            self.staged.push(Job {
                fd,
                offset,
                buffer_ptr: dest,
                length,
                request_id,
                is_write: false,
                sink: self.sink.clone(),
            });
            Ok(())
        }

        /// Stages a write from `src`; dispatched at [`submit`](Self::submit) time.
        /// Mirrors [`submit_read`](Self::submit_read).
        pub fn submit_write(
            &mut self,
            fd: RawFd,
            offset: u64,
            src: *const u8,
            length: usize,
            request_id: Identifier,
        ) -> Result<()> {
            self.staged.push(Job {
                fd,
                offset,
                buffer_ptr: src as *mut u8,
                length,
                request_id,
                is_write: true,
                sink: self.sink.clone(),
            });
            Ok(())
        }

        /// Hands every staged op to the pool (non-blocking) and returns how many
        /// were dispatched. The pool starts them immediately and concurrently.
        pub fn submit(&mut self) -> io::Result<usize> {
            let dispatched = self.staged.len();
            for job in self.staged.drain(..) {
                // The pool's receivers are the static pool threads, alive for the
                // life of the process, so the send never fails.
                self.pool.send(job).expect("io pool thread gone");
                self.in_flight += 1;
            }
            Ok(dispatched)
        }

        /// Flushes staged ops, then blocks until at least `want` completions are
        /// ready to drain (or nothing is left in flight). Returns the count of
        /// ops dispatched this call (callers must not read bytes from it).
        ///
        /// The requester parks on the disk and HTTP channels together (via
        /// [`completion_receiver`](Self::completion_receiver)), so this self-contained
        /// blocking wait is used only by the backend's own tests; kept for parity
        /// with the io_uring backend's `submit_and_wait`.
        #[cfg_attr(not(test), allow(dead_code))]
        pub fn submit_and_wait(&mut self, want: usize) -> io::Result<usize> {
            let dispatched = self.submit()?;
            self.block_until_ready(want);
            Ok(dispatched)
        }

        /// The handle a waker uses to interrupt this backend's completion wait:
        /// a sender onto the completion channel, carrying the [`WAKE_UD`]
        /// sentinel the drains below filter out.
        pub fn wake_handle(&self) -> RingWakeHandle {
            RingWakeHandle::Channel(self.sink.clone())
        }

        /// Moves every completion the pool has delivered so far off the channel
        /// into `ready` (non-blocking). Wake sentinels are dropped: they exist
        /// only to end a blocking wait, and are not in-flight ops.
        fn drain_ready(&mut self) {
            while let Ok(completion) = self.rx.try_recv() {
                if completion.1 == WAKE_UD as Identifier {
                    continue;
                }
                self.ready.push_back(completion);
                self.in_flight -= 1;
            }
        }

        /// Buffers completions into `ready`, blocking, until at least `want` are
        /// held, nothing is left in flight, or a wake sentinel arrives. `self.sink`
        /// keeps a sender alive, so `recv` only ever blocks; it never errors on a
        /// closed channel.
        fn block_until_ready(&mut self, want: usize) {
            self.drain_ready();
            while self.ready.len() < want && self.in_flight > 0 {
                let completion = self.rx.recv().expect("backend holds a sender");
                if completion.1 == WAKE_UD as Identifier {
                    return;
                }
                self.ready.push_back(completion);
                self.in_flight -= 1;
            }
        }

        /// Blocks until every dispatched op has reported back, ignoring wake
        /// sentinels. [`block_until_ready`](Self::block_until_ready) treats a
        /// sentinel as "stop waiting", which is right for a worker's park but
        /// wrong for teardown: a sentinel sent moments before shutdown (a
        /// notifier claims the slot first and sends second, so one can arrive
        /// after the worker already decided to exit) must not end this wait
        /// while a pool thread still holds a pointer into a cache slot.
        fn drain_all_in_flight(&mut self) {
            while self.in_flight > 0 {
                let completion = self.rx.recv().expect("backend holds a sender");
                if completion.1 == WAKE_UD as Identifier {
                    continue;
                }
                self.ready.push_back(completion);
                self.in_flight -= 1;
            }
        }

        /// Drains finished ops as `(bytes_transferred, request_id)`, mirroring the
        /// io_uring backend's `(result, id)` (negative is `-errno`).
        pub fn completions(&mut self) -> io::Result<Vec<(i32, Identifier)>> {
            self.drain_ready();
            Ok(self.ready.drain(..).collect())
        }

        /// The completion channel, so the requester can park on it alongside the
        /// HTTP channel (the two pools deliver independently, with no shared ring
        /// here to provide a single wake point).
        pub fn completion_receiver(&self) -> &Receiver<(i32, Identifier)> {
            &self.rx
        }

        /// `true` if a disk completion is already in hand, so the worker need not
        /// park.
        pub fn has_ready_completion(&self) -> bool {
            !self.ready.is_empty() || !self.rx.is_empty()
        }
    }

    impl Drop for IOBackend {
        /// Block until every dispatched op has reported back before tearing down,
        /// so no pool thread writes into a cache slot after the issuing request
        /// (and its `Arc` pin) has been dropped. The backend is declared before
        /// the requester's pending-request map, so it drops first, meaning the
        /// slots are still pinned here. Ops target local files and always finish.
        fn drop(&mut self) {
            self.drain_all_in_flight();
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::io::Write;
        use std::os::unix::io::AsRawFd;

        /// File byte at absolute offset `off`, so a read's bytes are predictable.
        fn pattern(off: usize) -> u8 {
            (off % 251) as u8
        }

        /// A temp file of `len` bytes filled with [`pattern`], kept alive by the
        /// returned handle so its fd stays valid for the read.
        fn patterned_file(len: usize) -> tempfile::NamedTempFile {
            let mut tmp = tempfile::NamedTempFile::new().unwrap();
            let data: Vec<u8> = (0..len).map(pattern).collect();
            tmp.write_all(&data).unwrap();
            tmp.flush().unwrap();
            tmp
        }

        /// More reads than pool threads, issued at once, must all run and each
        /// land in its own buffer with its own id, the core concurrency contract
        /// the old serial backend couldn't make.
        #[test]
        fn many_concurrent_reads_each_land_in_their_own_buffer() {
            const N: usize = 64;
            const BLK: usize = 4096;
            let file = patterned_file(N * BLK);
            let fd = file.as_file().as_raw_fd();
            let mut backend = IOBackend::new(64).unwrap();
            let mut bufs: Vec<Vec<u8>> = (0..N).map(|_| vec![0u8; BLK]).collect();

            for (id, buf) in bufs.iter_mut().enumerate() {
                backend
                    .submit_read(
                        fd,
                        (id * BLK) as u64,
                        buf.as_mut_ptr(),
                        BLK,
                        id as Identifier,
                    )
                    .unwrap();
            }
            backend.submit().unwrap();

            let mut seen = [false; N];
            let mut remaining = N;
            while remaining > 0 {
                backend.submit_and_wait(1).unwrap();
                for (result, id) in backend.completions().unwrap() {
                    assert_eq!(result, BLK as i32);
                    assert!(!seen[id], "id {id} completed twice");
                    seen[id] = true;
                    remaining -= 1;
                }
            }

            for (i, buf) in bufs.iter().enumerate() {
                for (j, &byte) in buf.iter().enumerate() {
                    assert_eq!(byte, pattern(i * BLK + j), "block {i} byte {j}");
                }
            }
        }

        /// An op that fails syscall-side comes back as a negative result carrying
        /// its id (mirroring io_uring's `-errno`), never a panic or a hang, so
        /// the requester can fail just the owning dataflow.
        #[test]
        fn a_failed_op_reports_a_negative_result_with_its_id() {
            let file = patterned_file(64);
            // A read-only descriptor: a `pwrite` to it fails with EBADF, a real
            // syscall error from a valid fd (no invalid-descriptor edge cases).
            let read_only = std::fs::File::open(file.path()).unwrap();
            let fd = read_only.as_raw_fd();
            let mut backend = IOBackend::new(8).unwrap();
            let src = [0u8; 64];

            backend
                .submit_write(fd, 0, src.as_ptr(), src.len(), 7)
                .unwrap();
            backend.submit_and_wait(1).unwrap();

            let completions = backend.completions().unwrap();
            assert_eq!(completions.len(), 1);
            let (result, id) = completions[0];
            assert!(result < 0, "expected -errno, got {result}");
            assert_eq!(id, 7);
        }

        /// A `submit_write` then `submit_read` of the same range round-trips the
        /// bytes: the disk-cache fill path the requester drives on a remote miss.
        #[test]
        fn a_write_then_read_round_trips() {
            const LEN: usize = 4096;
            let file = tempfile::NamedTempFile::new().unwrap();
            let fd = file.as_file().as_raw_fd();
            let mut backend = IOBackend::new(8).unwrap();
            let src: Vec<u8> = (0..LEN).map(pattern).collect();

            backend.submit_write(fd, 0, src.as_ptr(), LEN, 1).unwrap();
            backend.submit_and_wait(1).unwrap();
            assert_eq!(backend.completions().unwrap(), vec![(LEN as i32, 1)]);

            let mut readback = vec![0u8; LEN];
            backend
                .submit_read(fd, 0, readback.as_mut_ptr(), LEN, 2)
                .unwrap();
            backend.submit_and_wait(1).unwrap();
            assert_eq!(backend.completions().unwrap(), vec![(LEN as i32, 2)]);

            assert_eq!(readback, src);
        }

        /// A read that runs past end of file fills what is there and stops at the
        /// terminal zero-byte `pread`, reporting the short count rather than
        /// spinning on EOF or erroring.
        #[test]
        fn a_read_past_eof_returns_the_available_bytes() {
            const FILE_LEN: usize = 100;
            const REQ_LEN: usize = 4096;
            let file = patterned_file(FILE_LEN);
            let fd = file.as_file().as_raw_fd();
            let mut backend = IOBackend::new(8).unwrap();
            let mut buf = vec![0u8; REQ_LEN];

            backend
                .submit_read(fd, 0, buf.as_mut_ptr(), REQ_LEN, 3)
                .unwrap();
            backend.submit_and_wait(1).unwrap();

            assert_eq!(backend.completions().unwrap(), vec![(FILE_LEN as i32, 3)]);
        }

        /// `submit_and_wait(want)` parks until at least `want` completions are
        /// ready, so a single follow-up drain yields all of them.
        #[test]
        fn submit_and_wait_blocks_for_the_requested_count() {
            const N: usize = 8;
            const BLK: usize = 4096;
            let file = patterned_file(N * BLK);
            let fd = file.as_file().as_raw_fd();
            let mut backend = IOBackend::new(64).unwrap();
            let mut bufs: Vec<Vec<u8>> = (0..N).map(|_| vec![0u8; BLK]).collect();

            for (id, buf) in bufs.iter_mut().enumerate() {
                backend
                    .submit_read(
                        fd,
                        (id * BLK) as u64,
                        buf.as_mut_ptr(),
                        BLK,
                        id as Identifier,
                    )
                    .unwrap();
            }
            backend.submit_and_wait(N).unwrap();

            assert_eq!(backend.completions().unwrap().len(), N);
        }

        /// Dropping a backend with reads still dispatched must block until the
        /// pool finishes them, so no thread writes into `bufs` after they (would)
        /// be freed. The test reaching its end at all means `Drop` didn't hang and
        /// the in-flight drain branch ran.
        #[test]
        fn dropping_with_reads_in_flight_drains_before_returning() {
            const N: usize = 32;
            const BLK: usize = 4096;
            let file = patterned_file(N * BLK);
            let fd = file.as_file().as_raw_fd();
            let mut bufs: Vec<Vec<u8>> = (0..N).map(|_| vec![0u8; BLK]).collect();

            {
                let mut backend = IOBackend::new(64).unwrap();
                for (id, buf) in bufs.iter_mut().enumerate() {
                    backend
                        .submit_read(
                            fd,
                            (id * BLK) as u64,
                            buf.as_mut_ptr(),
                            BLK,
                            id as Identifier,
                        )
                        .unwrap();
                }
                backend.submit().unwrap();
                // Drop without draining: the `Drop` impl must wait out the pool.
            }

            drop(bufs);
        }
    }
}

#[cfg(target_os = "linux")]
pub(crate) use uring_backend::IOBackend;

#[cfg(all(unix, not(target_os = "linux")))]
pub(crate) use pread_pool_backend::IOBackend;
