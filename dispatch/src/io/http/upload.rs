//! Platform-specific HTTP(S) object-upload transport for
//! [`IORequester`](crate::io::IORequester).
//!
//! Uploads are the write-side mirror of the range reads in [`super::backend`]:
//! on Linux each upload submits its socket `connect`/`send`/`recv` SQEs onto the
//! worker's existing file io_uring (the same ring disk reads and writes use),
//! disambiguated by the [`UPLOAD_TAG`] bit; on other Unix platforms the engine
//! hands each upload to a shared pool of blocking `std::net` + rustls threads.
//!
//! The exchange is connect → (TLS handshake) → send request head + body → read
//! the response head. A `2xx` status is success; the response body is discarded
//! (the store's ack carries no bytes the engine needs). Each upload uses its own
//! `Connection: close` socket rather than a keep-alive pool: uploads are far
//! rarer than the hot column-chunk reads, so the pooling machinery isn't worth
//! its complexity here.

use super::proto::build_object_write;
use super::{Error, RemoteWrite, Result};

#[cfg(target_os = "linux")]
pub(crate) use uring_engine::UploadEngine;

#[cfg(all(unix, not(target_os = "linux")))]
pub(crate) use blocking_engine::{UploadCompletion, UploadEngine};

/// High bit of an SQE `user_data` marking a completion as belonging to the upload
/// engine (distinct from a file op — a small counter with no high bit — and from
/// a range-read socket op, which sets [`HTTP_TAG`](super::backend::HTTP_TAG)).
#[cfg(target_os = "linux")]
pub(crate) const UPLOAD_TAG: u64 = 1 << 62;

/// How many bytes of the body to hand the transport at a time. Bounds the socket
/// send buffer (and rustls's plaintext staging) so a multi-megabyte file doesn't
/// balloon a single buffer to the whole file size.
const BODY_CHUNK: usize = 256 * 1024;

/// How many times a single upload is re-issued on a fresh connection before it's
/// reported as failed — covers a transient reconnect without spinning on a real
/// outage. A PUT to a unique key is idempotent, so a retry is safe.
const MAX_UPLOAD_RETRIES: u32 = 3;

/// Whether a transport error is worth retrying on a fresh connection. Socket
/// teardown is transient; a TLS/protocol error is not.
fn is_retryable(err: &Error) -> bool {
    use std::io::ErrorKind::*;
    match err {
        Error::Io(e) => matches!(
            e.kind(),
            UnexpectedEof
                | ConnectionReset
                | ConnectionAborted
                | ConnectionRefused
                | BrokenPipe
                | NotConnected
                | TimedOut
                | Interrupted
        ),
        _ => false,
    }
}

/// Inspect a fully-parsed response head: `Ok(())` for a `2xx`, otherwise an error
/// carrying the status and a snippet of the response so a rejection is
/// diagnosable (S3/GCS explain the refusal in the body).
fn check_status(response: &[u8]) -> Result<StatusOutcome> {
    use super::http1::{HeadStatus, parse_response_head};
    match parse_response_head(response)? {
        HeadStatus::Incomplete => Ok(StatusOutcome::NeedMore),
        HeadStatus::Complete(head) => {
            if (200..300).contains(&head.status) {
                Ok(StatusOutcome::Ok)
            } else {
                let snippet = String::from_utf8_lossy(&response[..response.len().min(800)]);
                Err(Error::Io(std::io::Error::other(format!(
                    "object upload returned HTTP {}: {snippet}",
                    head.status
                ))))
            }
        }
    }
}

enum StatusOutcome {
    /// The head hasn't fully arrived; recv more.
    NeedMore,
    /// A `2xx`: the upload is done.
    Ok,
}

// ============================================================================
// Other Unix (macOS): blocking std::net + rustls thread pool
// ============================================================================

#[cfg(all(unix, not(target_os = "linux")))]
mod blocking_engine {
    use super::*;
    use crate::Identifier;
    use crossbeam_channel::{Receiver, Sender, unbounded};
    use rustls::{ClientConnection, StreamOwned};
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::sync::{Arc, OnceLock};
    use std::time::Duration;

    const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
    const IO_TIMEOUT: Duration = Duration::from_secs(60);

    /// A finished upload handed back from a pool thread: `Ok` once the store
    /// accepted it, `Err` for a terminal transport/status failure.
    pub(crate) type UploadCompletion = (Identifier, Result<()>);

    struct UploadJob {
        id: Identifier,
        write: RemoteWrite,
        client_config: Arc<rustls::ClientConfig>,
        sink: Sender<UploadCompletion>,
    }

    enum Conn {
        Tls(Box<StreamOwned<ClientConnection, TcpStream>>),
        Plain(TcpStream),
    }

    impl Read for Conn {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            match self {
                Conn::Tls(s) => s.read(buf),
                Conn::Plain(s) => s.read(buf),
            }
        }
    }

    impl Write for Conn {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            match self {
                Conn::Tls(s) => s.write(buf),
                Conn::Plain(s) => s.write(buf),
            }
        }
        fn flush(&mut self) -> std::io::Result<()> {
            match self {
                Conn::Tls(s) => s.flush(),
                Conn::Plain(s) => s.flush(),
            }
        }
    }

    struct UploadPool {
        jobs: Sender<UploadJob>,
    }

    fn upload_pool() -> &'static UploadPool {
        static POOL: OnceLock<UploadPool> = OnceLock::new();
        POOL.get_or_init(|| {
            let (jobs_tx, jobs_rx) = unbounded::<UploadJob>();
            for _ in 0..upload_pool_thread_count() {
                let jobs_rx = jobs_rx.clone();
                std::thread::Builder::new()
                    .name("pivot-upload".to_string())
                    .spawn(move || {
                        while let Ok(job) = jobs_rx.recv() {
                            let outcome =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    run_upload(&job.client_config, &job.write)
                                }))
                                .unwrap_or_else(|_| {
                                    Err(Error::Io(std::io::Error::other(
                                        "upload pool thread panicked",
                                    )))
                                });
                            let _ = job.sink.send((job.id, outcome));
                        }
                    })
                    .expect("failed to spawn upload pool thread");
            }
            UploadPool { jobs: jobs_tx }
        })
    }

    fn upload_pool_thread_count() -> usize {
        crate::env::get_env_var_with_default("PIVOT_UPLOAD_THREADS", 64)
    }

    fn connect(client_config: &Arc<rustls::ClientConfig>, write: &RemoteWrite) -> Result<Conn> {
        let tcp = TcpStream::connect_timeout(&write.remote.addr(), CONNECT_TIMEOUT)?;
        tcp.set_nodelay(true).ok();
        tcp.set_read_timeout(Some(IO_TIMEOUT))?;
        tcp.set_write_timeout(Some(IO_TIMEOUT))?;
        if write.remote.is_https() {
            let server_name =
                rustls::pki_types::ServerName::try_from(write.remote.host().to_string())?;
            let client = ClientConnection::new(client_config.clone(), server_name)?;
            Ok(Conn::Tls(Box::new(StreamOwned::new(client, tcp))))
        } else {
            Ok(Conn::Plain(tcp))
        }
    }

    fn run_upload(client_config: &Arc<rustls::ClientConfig>, write: &RemoteWrite) -> Result<()> {
        let mut attempt = 0;
        loop {
            match try_once(client_config, write) {
                Ok(()) => return Ok(()),
                Err(e) if is_retryable(&e) && attempt < MAX_UPLOAD_RETRIES => attempt += 1,
                Err(e) => return Err(e),
            }
        }
    }

    fn try_once(client_config: &Arc<rustls::ClientConfig>, write: &RemoteWrite) -> Result<()> {
        let mut conn = connect(client_config, write)?;
        let head = build_object_write(write);
        conn.write_all(&head)?;
        conn.write_all(&write.data)?;
        conn.flush()?;

        let mut acc = Vec::new();
        let mut chunk = vec![0u8; 16 * 1024];
        loop {
            let n = conn.read(&mut chunk)?;
            if n == 0 {
                return Err(Error::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "connection closed before response head",
                )));
            }
            acc.extend_from_slice(&chunk[..n]);
            match check_status(&acc)? {
                StatusOutcome::Ok => return Ok(()),
                StatusOutcome::NeedMore => continue,
            }
        }
    }

    /// A per-worker handle onto the shared blocking-upload pool. Mirrors the read
    /// path's [`HttpEngine`](super::super::backend::HttpEngine).
    pub(crate) struct UploadEngine {
        client_config: Arc<rustls::ClientConfig>,
        pool: Sender<UploadJob>,
        sink: Sender<UploadCompletion>,
        rx: Receiver<UploadCompletion>,
        active: usize,
        completed: Vec<Identifier>,
        failed: Vec<(Identifier, Error)>,
    }

    impl UploadEngine {
        pub fn new(client_config: Arc<rustls::ClientConfig>) -> Result<Self> {
            let (sink, rx) = unbounded();
            Ok(Self {
                client_config,
                pool: upload_pool().jobs.clone(),
                sink,
                rx,
                active: 0,
                completed: Vec::new(),
                failed: Vec::new(),
            })
        }

        pub fn start(&mut self, id: Identifier, write: RemoteWrite) -> Result<()> {
            self.pool
                .send(UploadJob {
                    id,
                    write,
                    client_config: self.client_config.clone(),
                    sink: self.sink.clone(),
                })
                .expect("upload pool thread gone");
            self.active += 1;
            Ok(())
        }

        fn drain_rx(&mut self) {
            while let Ok((id, outcome)) = self.rx.try_recv() {
                self.active -= 1;
                match outcome {
                    Ok(()) => self.completed.push(id),
                    Err(error) => self.failed.push((id, error)),
                }
            }
        }

        pub fn take_completed(&mut self) -> Vec<Identifier> {
            self.drain_rx();
            std::mem::take(&mut self.completed)
        }

        pub fn take_failed(&mut self) -> Vec<(Identifier, Error)> {
            self.drain_rx();
            std::mem::take(&mut self.failed)
        }

        pub fn has_active(&self) -> bool {
            self.active > 0
        }

        pub fn has_ready_completion(&self) -> bool {
            !self.rx.is_empty() || !self.completed.is_empty() || !self.failed.is_empty()
        }

        pub fn completion_receiver(&self) -> &Receiver<UploadCompletion> {
            &self.rx
        }
    }

    impl Drop for UploadEngine {
        fn drop(&mut self) {
            self.drain_rx();
            while self.active > 0 {
                match self.rx.recv() {
                    Ok(_) => self.active -= 1,
                    Err(_) => break,
                }
            }
        }
    }
}

// ============================================================================
// Linux: io_uring socket engine (shares the worker's file ring)
// ============================================================================

#[cfg(target_os = "linux")]
mod uring_engine {
    use super::*;
    use crate::Identifier;
    use io_uring::{IoUring, opcode, squeue, types};
    use rustls::ClientConnection;
    use std::io::{Read, Write};
    use std::net::SocketAddr;
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::sync::Arc;

    const RECV_CHUNK: usize = 16 * 1024;

    /// What the connection's last in-flight SQE was, so its CQE can be interpreted.
    enum State {
        Connecting,
        Sending,
        Receiving,
    }

    enum Transport {
        Plain,
        Tls(Box<ClientConnection>),
    }

    /// One upload's socket + (optional) TLS session and progress cursors.
    struct Conn {
        fd: OwnedFd,
        transport: Transport,
        sockaddr: Box<libc::sockaddr_storage>,
        sockaddr_len: libc::socklen_t,

        id: Identifier,
        /// The upload, retained so a transient failure can rebuild a fresh
        /// connection and re-issue it (a keyed PUT is idempotent).
        write: RemoteWrite,
        retries: u32,

        /// Pre-built request head (request line + headers), sent before the body.
        request_head: Vec<u8>,
        head_queued: bool,
        /// Bytes of the body handed to the transport so far.
        body_fed: usize,

        /// Bytes pending send (ciphertext for TLS, head/body chunk for plain).
        out_buf: Vec<u8>,
        out_pos: usize,
        /// Reusable recv scratch (TLS ciphertext, or plaintext response bytes).
        in_buf: Vec<u8>,
        /// Accumulated response head bytes until it parses.
        response: Vec<u8>,

        state: State,
    }

    impl Conn {
        /// Ensure `out_buf` holds the next bytes to send: drive the TLS handshake,
        /// queue the request head, then feed the body in bounded chunks.
        fn fill_out(&mut self) {
            if self.out_pos < self.out_buf.len() {
                return; // still have unsent bytes
            }
            self.out_buf.clear();
            self.out_pos = 0;

            let data_len = self.write.data.len();
            match &mut self.transport {
                Transport::Plain => {
                    if !self.head_queued {
                        self.out_buf.extend_from_slice(&self.request_head);
                        self.head_queued = true;
                    } else if self.body_fed < data_len {
                        let end = (self.body_fed + BODY_CHUNK).min(data_len);
                        self.out_buf
                            .extend_from_slice(&self.write.data[self.body_fed..end]);
                        self.body_fed = end;
                    }
                }
                Transport::Tls(tls) => {
                    if !tls.is_handshaking() {
                        if !self.head_queued {
                            tls.writer()
                                .write_all(&self.request_head)
                                .expect("rustls writer is infallible into its buffer");
                            self.head_queued = true;
                        } else if self.body_fed < data_len {
                            // rustls's plaintext buffer is bounded, so feed only
                            // what it accepts now (a `write_all` of a full chunk can
                            // overflow it); the remainder is fed once these records
                            // drain and free buffer space.
                            let end = (self.body_fed + BODY_CHUNK).min(data_len);
                            let fed = tls
                                .writer()
                                .write(&self.write.data[self.body_fed..end])
                                .expect("rustls writer never errors on a partial write");
                            self.body_fed += fed;
                        }
                    }
                    while tls.wants_write() {
                        tls.write_tls(&mut self.out_buf)
                            .expect("write_tls into a Vec is infallible");
                    }
                }
            }
        }

        /// Process `n` freshly received bytes from `in_buf`: for plaintext, append
        /// to `response`; for TLS, decrypt and append the plaintext. Returns
        /// whether a complete `2xx` response head has now arrived.
        fn consume_received(&mut self, n: usize) -> Result<bool> {
            match &mut self.transport {
                Transport::Plain => {
                    self.response.extend_from_slice(&self.in_buf[..n]);
                }
                Transport::Tls(tls) => {
                    let mut cursor = &self.in_buf[..n];
                    let mut plaintext = [0u8; 8192];
                    while !cursor.is_empty() {
                        let fed = tls.read_tls(&mut cursor)?;
                        tls.process_new_packets().map_err(Error::Tls)?;
                        loop {
                            match tls.reader().read(&mut plaintext) {
                                Ok(0) => break,
                                Ok(m) => self.response.extend_from_slice(&plaintext[..m]),
                                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                                Err(e) => return Err(Error::Io(e)),
                            }
                        }
                        if fed == 0 {
                            break;
                        }
                    }
                }
            }
            Ok(matches!(check_status(&self.response)?, StatusOutcome::Ok))
        }
    }

    /// Ring-less upload state machine. Submits onto the worker's borrowed file
    /// [`IoUring`] and is driven by the CQEs the requester routes to it by the
    /// [`UPLOAD_TAG`] bit.
    pub(crate) struct UploadEngine {
        client_config: Arc<rustls::ClientConfig>,
        conns: Vec<Option<Conn>>,
        free_slots: Vec<usize>,
        active: usize,
        completed: Vec<Identifier>,
        failed: Vec<(Identifier, Error)>,
    }

    impl UploadEngine {
        pub fn new(client_config: Arc<rustls::ClientConfig>) -> Result<Self> {
            Ok(Self {
                client_config,
                conns: Vec::new(),
                free_slots: Vec::new(),
                active: 0,
                completed: Vec::new(),
                failed: Vec::new(),
            })
        }

        pub fn has_active(&self) -> bool {
            self.active > 0
        }

        pub fn take_completed(&mut self) -> Vec<Identifier> {
            std::mem::take(&mut self.completed)
        }

        pub fn take_failed(&mut self) -> Vec<(Identifier, Error)> {
            std::mem::take(&mut self.failed)
        }

        /// Begin an upload: build a fresh connection and submit its connect SQE.
        pub fn start(
            &mut self,
            ring: &mut IoUring,
            id: Identifier,
            write: RemoteWrite,
        ) -> Result<()> {
            self.active += 1;
            let conn = self.new_conn(&write, id)?;
            let idx = self.alloc_slot(conn);
            self.start_connect(ring, idx)
        }

        /// Advance the connection identified by a tagged `user_data` after its SQE
        /// completed with kernel result `result`.
        pub fn on_cqe(&mut self, ring: &mut IoUring, user_data: u64, result: i32) -> Result<()> {
            let idx = (user_data & !UPLOAD_TAG) as usize;

            if result < 0 {
                let err = Error::Io(std::io::Error::from_raw_os_error(-result));
                return self.retry_or_fail(ring, idx, err);
            }

            let finished = match self.advance(idx, result as usize) {
                Ok(f) => f,
                Err(e) => return self.retry_or_fail(ring, idx, e),
            };

            if finished {
                let id = self.conns[idx].as_ref().unwrap().id;
                self.conns[idx] = None;
                self.free_slots.push(idx);
                self.active -= 1;
                self.completed.push(id);
                return Ok(());
            }

            self.pump(ring, idx)
        }

        /// Fold `size` freshly transferred bytes into the connection's state,
        /// returning whether the upload is now complete (a `2xx` head received).
        fn advance(&mut self, idx: usize, size: usize) -> Result<bool> {
            let conn = self.conns[idx].as_mut().unwrap();
            match conn.state {
                State::Connecting => Ok(false),
                State::Sending => {
                    conn.out_pos += size;
                    Ok(false)
                }
                State::Receiving => {
                    if size == 0 {
                        return Err(Error::Io(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "connection closed before response head",
                        )));
                    }
                    conn.consume_received(size)
                }
            }
        }

        fn retry_or_fail(&mut self, ring: &mut IoUring, idx: usize, err: Error) -> Result<()> {
            let dead = self.conns[idx].take().expect("cqe for a live connection");
            self.free_slots.push(idx);
            let id = dead.id;
            let retries = dead.retries;
            let write = dead.write.clone();
            drop(dead);

            if is_retryable(&err) && retries < MAX_UPLOAD_RETRIES {
                match self.new_conn(&write, id) {
                    Ok(mut fresh) => {
                        fresh.retries = retries + 1;
                        let new_idx = self.alloc_slot(fresh);
                        return self.start_connect(ring, new_idx);
                    }
                    Err(e) => {
                        self.active -= 1;
                        self.failed.push((id, e));
                        return Ok(());
                    }
                }
            }

            self.active -= 1;
            self.failed.push((id, err));
            Ok(())
        }

        fn new_conn(&self, write: &RemoteWrite, id: Identifier) -> Result<Conn> {
            let addr = write.remote.addr();
            let domain = match addr {
                SocketAddr::V4(_) => libc::AF_INET,
                SocketAddr::V6(_) => libc::AF_INET6,
            };
            let raw = unsafe { libc::socket(domain, libc::SOCK_STREAM, 0) };
            if raw < 0 {
                return Err(Error::Io(std::io::Error::last_os_error()));
            }
            let fd = unsafe { OwnedFd::from_raw_fd(raw) };
            let (sockaddr, sockaddr_len) = to_sockaddr(addr);

            let transport = if write.remote.is_https() {
                let server_name =
                    rustls::pki_types::ServerName::try_from(write.remote.host().to_string())?;
                Transport::Tls(Box::new(ClientConnection::new(
                    self.client_config.clone(),
                    server_name,
                )?))
            } else {
                Transport::Plain
            };

            Ok(Conn {
                fd,
                transport,
                sockaddr,
                sockaddr_len,
                id,
                request_head: build_object_write(write),
                head_queued: false,
                body_fed: 0,
                write: write.clone(),
                retries: 0,
                out_buf: Vec::new(),
                out_pos: 0,
                in_buf: vec![0u8; RECV_CHUNK],
                response: Vec::new(),
                state: State::Connecting,
            })
        }

        fn alloc_slot(&mut self, conn: Conn) -> usize {
            if let Some(idx) = self.free_slots.pop() {
                self.conns[idx] = Some(conn);
                idx
            } else {
                self.conns.push(Some(conn));
                self.conns.len() - 1
            }
        }

        fn start_connect(&mut self, ring: &mut IoUring, idx: usize) -> Result<()> {
            let conn = self.conns[idx].as_mut().unwrap();
            conn.state = State::Connecting;
            let fd = conn.fd.as_raw_fd();
            let addr_ptr =
                conn.sockaddr.as_ref() as *const libc::sockaddr_storage as *const libc::sockaddr;
            let len = conn.sockaddr_len;
            let entry = opcode::Connect::new(types::Fd(fd), addr_ptr, len)
                .build()
                .user_data(UPLOAD_TAG | idx as u64);
            push(ring, &entry)
        }

        /// Submit the next op (send or recv) for the connection. Sends while there
        /// are request bytes still to hand over (head, then body chunks, then any
        /// TLS records); once the whole request is out, receives the response.
        fn pump(&mut self, ring: &mut IoUring, idx: usize) -> Result<()> {
            let (is_send, ptr, len, fd) = {
                let conn = self.conns[idx].as_mut().unwrap();
                conn.fill_out();
                let has_unsent = conn.out_pos < conn.out_buf.len();
                // Keep sending until the body is fully handed over and drained; a
                // TLS handshake may momentarily have nothing to send and must recv
                // the peer's next flight, so gate purely on unsent bytes.
                if has_unsent {
                    conn.state = State::Sending;
                    let ptr = conn.out_buf[conn.out_pos..].as_ptr();
                    let len = conn.out_buf.len() - conn.out_pos;
                    (true, ptr, len, conn.fd.as_raw_fd())
                } else {
                    conn.state = State::Receiving;
                    let ptr = conn.in_buf.as_ptr();
                    (false, ptr, RECV_CHUNK, conn.fd.as_raw_fd())
                }
            };

            let entry = if is_send {
                opcode::Send::new(types::Fd(fd), ptr, len as u32)
                    .build()
                    .user_data(UPLOAD_TAG | idx as u64)
            } else {
                opcode::Recv::new(types::Fd(fd), ptr as *mut u8, len as u32)
                    .build()
                    .user_data(UPLOAD_TAG | idx as u64)
            };
            push(ring, &entry)
        }
    }

    fn push(ring: &mut IoUring, entry: &squeue::Entry) -> Result<()> {
        unsafe {
            ring.submission()
                .push(entry)
                .map_err(|_| Error::Io(std::io::Error::other("upload submission queue full")))?;
        }
        ring.submit()?;
        Ok(())
    }

    fn to_sockaddr(addr: SocketAddr) -> (Box<libc::sockaddr_storage>, libc::socklen_t) {
        let mut storage: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
        let len = match addr {
            SocketAddr::V4(a) => {
                let sin = &mut storage as *mut _ as *mut libc::sockaddr_in;
                unsafe {
                    (*sin).sin_family = libc::AF_INET as libc::sa_family_t;
                    (*sin).sin_port = a.port().to_be();
                    (*sin).sin_addr = libc::in_addr {
                        s_addr: u32::from_ne_bytes(a.ip().octets()),
                    };
                }
                std::mem::size_of::<libc::sockaddr_in>()
            }
            SocketAddr::V6(a) => {
                let sin6 = &mut storage as *mut _ as *mut libc::sockaddr_in6;
                unsafe {
                    (*sin6).sin6_family = libc::AF_INET6 as libc::sa_family_t;
                    (*sin6).sin6_port = a.port().to_be();
                    (*sin6).sin6_addr = libc::in6_addr {
                        s6_addr: a.ip().octets(),
                    };
                    (*sin6).sin6_flowinfo = a.flowinfo();
                    (*sin6).sin6_scope_id = a.scope_id();
                }
                std::mem::size_of::<libc::sockaddr_in6>()
            }
        };
        (Box::new(storage), len as libc::socklen_t)
    }
}
