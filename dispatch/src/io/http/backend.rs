//! Platform-specific HTTP(S) transport for [`IORequester`](crate::io::IORequester).
//!
//! The engine is **ring-less**: on Linux it submits its socket `connect`/`send`/
//! `recv` SQEs onto the worker's *existing* file io_uring (the same one disk reads
//! use) and is fed back the CQEs whose `user_data` carries the `HTTP_TAG` bit.
//! Sharing one ring per core is the standard io_uring pattern and means a single
//! submit/wait point — no separate network ring to coordinate. rustls is sans-IO,
//! so TLS is just shuttling ciphertext between the socket and rustls's buffers;
//! one SQE is in flight per connection at a time and each CQE re-pumps that
//! connection's state machine (connect → handshake → request → response). Idle
//! keep-alive connections are pooled per host and reused.
//!
//! On non-Linux (macOS) there is no ring: the engine hands each range read to a
//! shared pool of blocking `std::net` + `rustls::StreamOwned` threads (mirroring
//! the file path's `pread` pool) and collects their completions over a channel,
//! so a fetch never blocks the dataflow worker and many run concurrently.

use super::proto;
use super::{RemoteRead, RemoteRequest, RemoteUpload};

const IO_CHUNK_SIZE: usize = 16 * 1024;

#[cfg(target_os = "linux")]
pub(crate) use uring_engine::HttpEngine;

#[cfg(all(unix, not(target_os = "linux")))]
pub(crate) use blocking_engine::{HttpCompletion, HttpEngine};

/// High bit of an SQE `user_data` marking a completion as belonging to the HTTP
/// engine rather than a file read. File-read ids are small monotonic counters,
/// so they never collide with this. (Only meaningful on the Linux shared ring.)
#[cfg(target_os = "linux")]
pub(crate) const HTTP_TAG: u64 = 1 << 63;

/// Per-host pool key.
fn host_key(remote: &crate::io::RemoteFile) -> String {
    format!("{}:{}", remote.host(), remote.port())
}

// ============================================================================
// Other Unix (macOS): blocking std::net + rustls thread pool
// ============================================================================

#[cfg(all(unix, not(target_os = "linux")))]
mod blocking_engine {
    use super::*;
    use crate::Identifier;
    use crate::io::http::http1;
    use crate::io::http::{Error, Result};
    use crossbeam_channel::{Receiver, Sender, unbounded};
    use rustls::{ClientConnection, StreamOwned};
    use std::collections::HashMap;
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::Duration;

    /// Bound how long a single connect / read / write may block, so a hung server
    /// can't pin a pool thread (and stall the engine's `Drop`) forever. Generous
    /// for a range read against object storage; well above any healthy latency.
    const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
    const IO_TIMEOUT: Duration = Duration::from_secs(30);

    /// A finished fetch handed back from a pool thread: `Ok` once the body landed
    /// in the slot, `Err` for a terminal transport failure (the engine surfaces it
    /// to the owning dataflow). The id is the read's engine-side request id.
    pub(crate) type HttpCompletion = (Identifier, Result<()>);

    /// One range read for a pool thread to perform: fetch `read` into its pinned
    /// slot and report `(id, outcome)` on `sink`. `client_config` builds any fresh
    /// connection the fetch needs.
    struct HttpJob {
        id: Identifier,
        request: RemoteRequest,
        client_config: Arc<rustls::ClientConfig>,
        sink: Sender<HttpCompletion>,
    }

    /// A pooled connection: TLS-wrapped or plain TCP.
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

    /// The keep-alive connection pool, keyed by host. Shared across all pool
    /// threads (behind a mutex held only to pop or push a connection, never during
    /// the network exchange) so reuse is global, exactly as the single engine did:
    /// a sequential pair of reads to one host rides one connection.
    type ConnPool = Mutex<HashMap<String, Vec<Conn>>>;

    /// The process-wide pool of blocking-HTTP threads, created once and shared by
    /// every worker's engine.
    struct HttpPool {
        jobs: Sender<HttpJob>,
    }

    fn http_pool() -> &'static HttpPool {
        static POOL: OnceLock<HttpPool> = OnceLock::new();
        POOL.get_or_init(|| {
            let (jobs_tx, jobs_rx) = unbounded::<HttpJob>();
            let conns: Arc<ConnPool> = Arc::new(Mutex::new(HashMap::new()));
            for _ in 0..http_pool_thread_count() {
                let jobs_rx = jobs_rx.clone();
                let conns = conns.clone();
                std::thread::Builder::new()
                    .name("pivot-http".to_string())
                    .spawn(move || run_http_thread(&jobs_rx, &conns))
                    .expect("failed to spawn http pool thread");
            }
            HttpPool { jobs: jobs_tx }
        })
    }

    /// Pool size: 512 threads by default (override with `PIVOT_HTTP_THREADS`).
    /// HTTP fetches are latency-bound (a thread parks on the network, not the
    /// CPU), so a deep pool overlaps many in-flight range reads against object
    /// storage to hide per-request round trips. 512 pairs with the default
    /// per-worker read-ahead to keep the pool fed without exceeding a laptop's
    /// cross-region connection budget; raise it on a host close to the store.
    fn http_pool_thread_count() -> usize {
        crate::env::get_env_var_with_default("PIVOT_HTTP_THREADS", 512)
    }

    fn run_http_thread(jobs: &Receiver<HttpJob>, conns: &ConnPool) {
        // The channel closes only when the (static) pool's sender is dropped,
        // i.e. never; this loop runs for the life of the process.
        while let Ok(job) = jobs.recv() {
            let HttpJob {
                id,
                request,
                client_config,
                sink,
            } = job;
            // Every job MUST yield one completion (a worker parks until its read
            // reports back) and this thread must survive to serve the next job, so
            // a panic in the fetch fails just this read rather than stranding the
            // worker or shrinking the pool.
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_fetch(conns, &client_config, &request)
            }))
            .unwrap_or_else(|_| {
                Err(Error::Io(std::io::Error::other(
                    "http pool thread panicked",
                )))
            });
            // Let go of the request before reporting. An upload body is ring
            // memory, which is released through the owning worker's memory
            // context, and a pool thread has none. The issuing engine holds the
            // request until it takes this completion, so dropping here only ever
            // decrements the refcount.
            drop(request);
            // The issuing engine may have been dropped mid-flight (its receiver
            // gone); a failed send is then expected and ignored.
            let _ = sink.send((id, outcome));
        }
    }

    /// Fetch `read` into its slot, reusing a pooled keep-alive connection when one
    /// is idle for the host and reconnecting once if it turns out to be stale.
    fn run_fetch(
        conns: &ConnPool,
        client_config: &Arc<rustls::ClientConfig>,
        request: &RemoteRequest,
    ) -> Result<()> {
        let key = host_key(request.remote());
        // Recover a poisoned lock rather than propagating the panic: the guarded
        // region is only a pop/push, so a poisoned map is still usable, and the
        // pool is process-wide, so a panic-unwrap would brick every worker's HTTP
        // reads for the life of the process rather than failing one read.
        let pooled = conns
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get_mut(&key)
            .and_then(|v| v.pop());
        let conn = match pooled {
            // A pooled connection may have been closed by the server's keep-alive
            // timeout; on any error, reconnect once and retry.
            Some(c) => match do_request(c, request) {
                Ok(c) => c,
                Err(_) => do_request(connect(client_config, request)?, request)?,
            },
            None => do_request(connect(client_config, request)?, request)?,
        };
        // `None` means the request finished but its connection can't be pooled
        // (a response body of unknown framing); dropping it closes the socket.
        if let Some(conn) = conn {
            conns
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .entry(key)
                .or_default()
                .push(conn);
        }
        Ok(())
    }

    fn connect(client_config: &Arc<rustls::ClientConfig>, request: &RemoteRequest) -> Result<Conn> {
        let remote = request.remote();
        let tcp = TcpStream::connect_timeout(&remote.addr(), CONNECT_TIMEOUT)?;
        tcp.set_nodelay(true).ok();
        // These bound every read/write so a hung server can't pin a pool thread
        // (and stall Drop) forever, so a failure to set them must fail the
        // connection rather than be ignored.
        tcp.set_read_timeout(Some(IO_TIMEOUT))?;
        tcp.set_write_timeout(Some(IO_TIMEOUT))?;
        if remote.is_https() {
            let server_name = rustls::pki_types::ServerName::try_from(remote.host().to_string())?;
            let client = ClientConnection::new(client_config.clone(), server_name)?;
            Ok(Conn::Tls(Box::new(StreamOwned::new(client, tcp))))
        } else {
            Ok(Conn::Plain(tcp))
        }
    }

    /// Run one request on `conn`, returning the connection for re-pooling, or
    /// `None` when the request succeeded but the connection must be closed.
    fn do_request(conn: Conn, request: &RemoteRequest) -> Result<Option<Conn>> {
        match request {
            // A range GET response is always identity-framed with a known
            // Content-Length that is fully drained, so `do_read` can never leave
            // the connection in an unreusable state: on success it always hands
            // the connection back for pooling, hence the unconditional `Some`.
            // Upload responses can have unknown framing, so `do_upload` decides
            // per-response whether the connection may be reused.
            RemoteRequest::Read(read) => do_read(conn, read).map(Some),
            RemoteRequest::Upload(upload) => do_upload(conn, upload),
        }
    }

    /// Issue the range GET on `conn` and read its body into `read.dest`, returning
    /// the connection for re-pooling on success.
    fn do_read(mut conn: Conn, read: &RemoteRead) -> Result<Conn> {
        let auth = read.remote.auth_header();
        let request = proto::build_range_get(
            read.remote.host_header(),
            read.remote.request_target(),
            read.offset,
            read.len,
            auth.as_deref(),
        );
        conn.write_all(&request)?;
        conn.flush()?;

        // Accumulate until the response head parses.
        let mut chunk = vec![0u8; IO_CHUNK_SIZE];
        let mut acc: Vec<u8> = Vec::new();
        let head = loop {
            let n = conn.read(&mut chunk)?;
            if n == 0 {
                return Err(Error::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "connection closed before response head",
                )));
            }
            acc.extend_from_slice(&chunk[..n]);
            match proto::parse_get_response_head(&acc, read.len) {
                Ok(proto::HeadParse::Complete(h)) => break h,
                Ok(proto::HeadParse::Incomplete) => continue,
                Err(e) => {
                    // A rejected request (e.g. a GCS 404/403) carries an error
                    // body explaining why; pull a little more so it lands in the
                    // message, since a bare status is hard to diagnose.
                    if let Ok(extra) = conn.read(&mut chunk) {
                        acc.extend_from_slice(&chunk[..extra]);
                    }
                    let response = String::from_utf8_lossy(&acc);
                    let response = &response[..response.len().min(800)];
                    return Err(Error::Io(std::io::Error::other(format!(
                        "{e}; {} {} -> {response}",
                        read.remote.host_header(),
                        read.remote.request_target(),
                    ))));
                }
            }
        };

        let body_len = head.content_length as usize;
        // SAFETY: dest points into the pinned cache slot for exactly this block's
        // currently-invalid sub-blocks; body_len <= read.len <= the block length
        // (validated in parse_get_response_head).
        let dest = unsafe { std::slice::from_raw_parts_mut(read.dest, body_len) };

        // Drive the body through the sans-IO decoder. The range policy guarantees
        // an identity (`Content-Length`) body, so the Direct path applies: read
        // straight into the cache slot and report the count, no intermediate
        // buffer. `decode` first absorbs any body bytes that arrived alongside the
        // head, and would transparently handle a chunked body too were the range
        // policy ever relaxed.
        let mut body = http1::BodyDecoder::new(head.content_length);
        let mut written = 0usize;
        let leftover = &acc[head.head_len..];
        body.decode(leftover, |bytes| {
            dest[written..written + bytes.len()].copy_from_slice(bytes);
            written += bytes.len();
        });

        while !body.is_complete() {
            match body.read_plan() {
                http1::ReadPlan::Done => break,
                http1::ReadPlan::Direct { max } => {
                    let cap = (max as usize).min(dest.len() - written);
                    let n = conn.read(&mut dest[written..written + cap])?;
                    if n == 0 {
                        return Err(Error::Io(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "connection closed before response body completed",
                        )));
                    }
                    written += n;
                    body.consumed(n as u64)?;
                }
            }
        }

        Ok(conn)
    }

    fn do_upload(mut conn: Conn, upload: &RemoteUpload) -> Result<Option<Conn>> {
        let request = RemoteRequest::Upload(upload.clone()).request_head();
        conn.write_all(&request)?;
        for run in upload.data.runs() {
            conn.write_all(run)?;
        }
        conn.flush()?;

        let mut chunk = [0u8; IO_CHUNK_SIZE];
        let mut acc = Vec::new();
        let head = loop {
            let n = conn.read(&mut chunk)?;
            if n == 0 {
                return Err(Error::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "connection closed before upload response",
                )));
            }
            acc.extend_from_slice(&chunk[..n]);
            match proto::parse_upload_response_head(&acc)? {
                proto::HeadParse::Complete(head) => break head,
                proto::HeadParse::Incomplete => continue,
            }
        };
        // The upload itself succeeded (2xx), but a body of unknown framing
        // can't be drained; close the connection instead of pooling it with
        // unread bytes.
        if !head.reuse_connection {
            return Ok(None);
        }
        // Nothing currently consumes the upload response body (an object store
        // may return an ETag or a small JSON result); the `|_| {}` discards each
        // chunk. We only decode it to drain the body off the socket so the
        // connection can return to the keep-alive pool with no unread bytes.
        let mut body = http1::BodyDecoder::new(head.content_length);
        body.decode(&acc[head.head_len..], |_| {});
        while !body.is_complete() {
            let n = conn.read(&mut chunk)?;
            if n == 0 {
                return Err(Error::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "connection closed before upload response body completed",
                )));
            }
            body.decode(&chunk[..n], |_| {});
        }
        Ok(Some(conn))
    }

    /// A thin per-worker handle onto the shared blocking-HTTP pool. Stages each
    /// range read onto the pool and collects completions over `rx`; the requester
    /// drains them via [`take_completed`](Self::take_completed) /
    /// [`take_failed`](Self::take_failed) and parks on `rx` (alongside the disk
    /// channel) through [`completion_receiver`](Self::completion_receiver).
    pub(crate) struct HttpEngine {
        client_config: Arc<rustls::ClientConfig>,
        /// This worker's job sender into the shared pool.
        pool: Sender<HttpJob>,
        /// Cloned onto each dispatched [`HttpJob`] so the pool thread reports back.
        sink: Sender<HttpCompletion>,
        /// Receives completions from the pool threads (MPSC: many threads, one
        /// engine).
        rx: Receiver<HttpCompletion>,
        /// Fetches dispatched but not yet drained from `rx`.
        active: usize,
        /// Drained successes, awaiting [`take_completed`](Self::take_completed).
        completed: Vec<Identifier>,
        /// Drained failures with their error, awaiting [`take_failed`](Self::take_failed).
        failed: Vec<(Identifier, Error)>,
    }

    impl HttpEngine {
        pub fn new(client_config: Arc<rustls::ClientConfig>) -> Result<Self> {
            let (sink, rx) = unbounded();
            Ok(Self {
                client_config,
                pool: http_pool().jobs.clone(),
                sink,
                rx,
                active: 0,
                completed: Vec::new(),
                failed: Vec::new(),
            })
        }

        /// Dispatch a range read onto the pool (non-blocking); it completes later
        /// on `rx`.
        pub fn start_get(&mut self, id: Identifier, read: RemoteRead) -> Result<()> {
            self.start_request(id, RemoteRequest::Read(read))
        }

        pub fn start_upload(&mut self, id: Identifier, upload: RemoteUpload) -> Result<()> {
            self.start_request(id, RemoteRequest::Upload(upload))
        }

        fn start_request(&mut self, id: Identifier, request: RemoteRequest) -> Result<()> {
            self.pool
                .send(HttpJob {
                    id,
                    request,
                    client_config: self.client_config.clone(),
                    sink: self.sink.clone(),
                })
                .expect("http pool thread gone");
            self.active += 1;
            Ok(())
        }

        /// Move every completion the pool has delivered into the success/failure
        /// buckets (non-blocking).
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

        /// `true` if a completion is already in hand, so the worker need not park.
        pub fn has_ready_completion(&self) -> bool {
            !self.rx.is_empty() || !self.completed.is_empty() || !self.failed.is_empty()
        }

        /// The completion channel, so the requester can park on it alongside the
        /// disk channel (the two pools deliver independently, with no shared ring
        /// here to provide a single wake point).
        pub fn completion_receiver(&self) -> &Receiver<HttpCompletion> {
            &self.rx
        }
    }

    impl Drop for HttpEngine {
        /// Block until every dispatched fetch reports back, so no pool thread
        /// writes into a cache slot after the issuing read's pin is dropped. The
        /// socket timeouts bound how long this can take.
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
    use crate::io::http::http1;
    use crate::io::http::{Error, Result};
    use io_uring::{IoUring, opcode, squeue, types};
    use rustls::ClientConnection;
    use std::collections::HashMap;
    use std::io::{Read, Write};
    use std::net::SocketAddr;
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::sync::Arc;

    /// How many times a single range read is re-issued on a fresh connection
    /// before it's reported as failed. Covers the common case — a pooled
    /// keep-alive connection the server closed on its idle timeout — plus a few
    /// genuinely flaky reconnects, without spinning forever on a real outage.
    const MAX_HTTP_RETRIES: u32 = 3;

    /// Whether a transport error is worth retrying on a fresh connection. Socket
    /// teardown (a closed keep-alive, a reset, a refused reconnect) is transient;
    /// a TLS/protocol/DNS error is not — retrying would just fail the same way.
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

    /// Reusable socket/TLS resources. Only this value enters the keep-alive pool;
    /// request and response state lives in [`HttpExchange`] and is dropped when
    /// that exchange completes.
    struct Conn {
        fd: OwnedFd,
        transport: Transport,
        host_key: String,
        /// Connect target; boxed so the SQE can hold a stable pointer to it.
        sockaddr: Box<libc::sockaddr_storage>,
        sockaddr_len: libc::socklen_t,

        // Reusable send buffer for plaintext request heads and TLS ciphertext.
        out_buf: Vec<u8>,
        // Reusable recv scratch (TLS ciphertext, or plaintext headers). Allocated
        // once at `IO_CHUNK_SIZE` and never re-zeroed — recv overwrites `[..n]`
        // and we only read that. Plaintext bodies skip it entirely (recv'd into
        // `dest`).
        in_buf: Vec<u8>,
    }

    /// One complete HTTP request/response exchange, owning a connection while it
    /// is active. Dropping this value releases all request-specific resources;
    /// successful completion first moves `conn` back into the keep-alive pool.
    struct HttpExchange {
        conn: Conn,
        id: Identifier,
        dest: *mut u8,
        req_len: usize,
        request_bytes: Vec<u8>,
        request_queued: bool,
        /// The originating operation, retained for transparent reconnects and
        /// retries and for choosing read-vs-upload response handling.
        request: RemoteRequest,
        /// How many times this request has already been retried on a fresh
        /// connection; bounded by [`MAX_HTTP_RETRIES`].
        retries: u32,

        // Cursor into the connection's pending send buffer (TLS ciphertext, or
        // the request head for plain HTTP).
        out_pos: usize,
        /// Bytes of an upload body already handed to the transport. Plain HTTP
        /// sends directly from the Arc; TLS feeds bounded chunks to rustls.
        upload_pos: usize,
        /// The in-flight send SQE points directly into the upload Arc rather
        /// than `out_buf`.
        sending_upload_direct: bool,

        // Response assembly. `head_acc` accumulates header bytes until the head
        // parses; `body` is then the sans-IO decoder that tracks body progress and
        // completion (`None` until the head is in). `body_written` is the write
        // cursor into `dest`.
        head_acc: Vec<u8>,
        body: Option<http1::BodyDecoder>,
        body_written: usize,
        /// True when the last recv was posted to read straight into `dest` (a
        /// plaintext body, zero-copy) rather than into `in_buf`.
        recv_in_dest: bool,
        /// Whether the connection may return to the keep-alive pool when the
        /// exchange finishes (false for a response body of unknown framing).
        conn_reusable: bool,

        state: State,
    }

    impl HttpExchange {
        fn new(conn: Conn, id: Identifier, request: RemoteRequest, state: State) -> Self {
            let (dest, req_len) = match &request {
                RemoteRequest::Read(read) => (read.dest, read.len),
                RemoteRequest::Upload(_) => (std::ptr::null_mut(), 0),
            };
            let request_bytes = request.request_head();
            Self {
                conn,
                id,
                dest,
                req_len,
                request_bytes,
                request_queued: false,
                request,
                retries: 0,
                out_pos: 0,
                upload_pos: 0,
                sending_upload_direct: false,
                head_acc: Vec::new(),
                body: None,
                body_written: 0,
                recv_in_dest: false,
                conn_reusable: true,
                state,
            }
        }

        /// Finish this exchange and return only its reusable connection. Every
        /// request/response field, including an upload's `Arc<[u8]>`, is dropped.
        fn into_connection(mut self) -> (Identifier, Conn) {
            self.conn.out_buf.clear();
            (self.id, self.conn)
        }

        /// Ensure `out_buf` holds the next bytes to send: queue the HTTP request
        /// once it's allowed, then drain rustls's outgoing records.
        fn fill_out(&mut self) {
            if self.out_pos < self.conn.out_buf.len() {
                return; // still have unsent bytes
            }
            self.conn.out_buf.clear();
            self.out_pos = 0;
            match &mut self.conn.transport {
                Transport::Plain => {
                    if !self.request_queued {
                        self.conn.out_buf.extend_from_slice(&self.request_bytes);
                        self.request_queued = true;
                    }
                }
                Transport::Tls(tls) => {
                    if !tls.is_handshaking() && !self.request_queued {
                        tls.writer()
                            .write_all(&self.request_bytes)
                            .expect("rustls writer is infallible into its buffer");
                        self.request_queued = true;
                    }
                    if !tls.is_handshaking()
                        && self.request_queued
                        && let RemoteRequest::Upload(upload) = &self.request
                        && self.upload_pos < upload.data.len()
                    {
                        // A chunk never crosses a run boundary: take what is
                        // left of the run at the current position, capped at the
                        // chunk size.
                        let run = upload.data.run_at(self.upload_pos);
                        let chunk = &run[..run.len().min(IO_CHUNK_SIZE)];
                        tls.writer()
                            .write_all(chunk)
                            .expect("rustls writer is infallible into its buffer");
                        self.upload_pos += chunk.len();
                    }
                    while tls.wants_write() {
                        tls.write_tls(&mut self.conn.out_buf)
                            .expect("write_tls into a Vec is infallible");
                    }
                }
            }
        }

        /// Process `n` freshly received bytes from `in_buf`.
        ///
        /// Only called when the recv landed in `in_buf` (TLS ciphertext, or — for
        /// plaintext — the header phase before the body is recv'd straight into
        /// `dest`). For TLS this decrypts the body **directly into the cache slot**:
        /// the single copy TLS requires, since rustls AEAD-decrypts into its own
        /// buffer and `read` moves the plaintext out (no decrypt-into-user-buffer
        /// API exists).
        fn consume_received(&mut self, n: usize) -> Result<()> {
            let HttpExchange {
                conn,
                head_acc,
                body,
                request,
                dest,
                req_len,
                body_written,
                conn_reusable,
                ..
            } = self;
            let Conn {
                transport, in_buf, ..
            } = conn;
            let dest = *dest;
            let req_len = *req_len;

            match transport {
                // Reached only in the header phase; once the head parses, `pump`
                // recvs the body straight into `dest`, so plaintext bodies never
                // pass through here.
                Transport::Plain => {
                    // A plaintext GET body is consumed either from the bytes
                    // accompanying its head in `parse_head`, or directly into
                    // `dest` when it arrives in a later recv, so it never reaches
                    // this branch. Nothing currently consumes the upload response
                    // body (e.g. an ETag or JSON result); the `|_| {}` discards
                    // it - we only drain it before pooling the connection.
                    if let Some(decoder) = body.as_mut()
                        && matches!(request, RemoteRequest::Upload(_))
                    {
                        decoder.decode(&in_buf[..n], |_| {});
                        Ok(())
                    } else {
                        parse_head(
                            head_acc,
                            body,
                            request,
                            dest,
                            req_len,
                            body_written,
                            conn_reusable,
                            &in_buf[..n],
                        )
                    }
                }
                Transport::Tls(tls) => {
                    // `read_tls` only accepts as much ciphertext as fits rustls's
                    // bounded buffer, so one call may not consume all of `in_buf`.
                    // Loop: feed, process, drain (draining frees buffer space) until
                    // the chunk is consumed — dropping the remainder would silently
                    // lose body bytes and hang the read.
                    let mut cursor = &in_buf[..n];
                    let mut header_scratch = [0u8; 8192];
                    while !cursor.is_empty() {
                        let fed = tls.read_tls(&mut cursor)?;
                        tls.process_new_packets().map_err(Error::Tls)?;
                        let mut drained = 0usize;
                        loop {
                            if let Some(decoder) = body.as_mut() {
                                if matches!(request, RemoteRequest::Upload(_)) {
                                    // An upload endpoint may return a response body (for
                                    // example, GCS object metadata). Uploads have no cache
                                    // destination, but we must drain and discard that body
                                    // before returning the connection to the keep-alive pool.
                                    match tls.reader().read(&mut header_scratch) {
                                        Ok(0) => break,
                                        Ok(m) => {
                                            decoder.decode(&header_scratch[..m], |_| {});
                                            drained += m;
                                        }
                                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                                            break;
                                        }
                                        Err(e) => return Err(Error::Io(e)),
                                    }
                                    continue;
                                }
                                // Body phase: decrypt straight into the cache slot.
                                // A range body is identity-framed (the `proto`
                                // policy enforces `Content-Length`), so the decoder
                                // hands back a Direct plan whose `max` is the bytes
                                // still expected — exactly the room to decrypt into.
                                let remaining = match decoder.read_plan() {
                                    http1::ReadPlan::Direct { max } => max as usize,
                                    http1::ReadPlan::Done => 0,
                                };
                                if remaining == 0 {
                                    break;
                                }
                                // SAFETY: writing this block's currently-invalid,
                                // pinned slot region; content_length <= req_len ==
                                // block length (validated when parsing the head).
                                let dst = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        dest.add(*body_written),
                                        remaining,
                                    )
                                };
                                match tls.reader().read(dst) {
                                    Ok(0) => break,
                                    Ok(m) => {
                                        decoder.consumed(m as u64)?;
                                        *body_written += m;
                                        drained += m;
                                    }
                                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                                    Err(e) => return Err(Error::Io(e)),
                                }
                            } else {
                                // Header phase: small scratch until the head parses.
                                match tls.reader().read(&mut header_scratch) {
                                    Ok(0) => break,
                                    Ok(m) => {
                                        drained += m;
                                        parse_head(
                                            head_acc,
                                            body,
                                            request,
                                            dest,
                                            req_len,
                                            body_written,
                                            conn_reusable,
                                            &header_scratch[..m],
                                        )?;
                                    }
                                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                                    Err(e) => return Err(Error::Io(e)),
                                }
                            }
                        }
                        // No ciphertext accepted and no plaintext produced: rustls
                        // can't make progress on the rest (e.g. a partial trailing
                        // record). Stop; the next recv delivers it.
                        if fed == 0 && drained == 0 {
                            break;
                        }
                    }
                    Ok(())
                }
            }
        }

        fn request_complete(&self) -> bool {
            self.body.as_ref().is_some_and(|b| b.is_complete())
        }
    }

    /// Feed a header-phase (plaintext) chunk: accumulate into `head_acc` until the
    /// response head parses, then install the body decoder and copy any body bytes
    /// that arrived in the same packet as the head into `dest`. (Subsequent body
    /// bytes are recv'd/decrypted straight into `dest`, zero-copy.)
    #[allow(clippy::too_many_arguments)] // one cursor per response-assembly field
    fn parse_head(
        head_acc: &mut Vec<u8>,
        body: &mut Option<http1::BodyDecoder>,
        request: &RemoteRequest,
        dest: *mut u8,
        req_len: usize,
        body_written: &mut usize,
        conn_reusable: &mut bool,
        chunk: &[u8],
    ) -> Result<()> {
        head_acc.extend_from_slice(chunk);
        let parsed = match request {
            RemoteRequest::Read(_) => proto::parse_get_response_head(head_acc, req_len)?,
            RemoteRequest::Upload(_) => proto::parse_upload_response_head(head_acc)?,
        };
        if let proto::HeadParse::Complete(h) = parsed {
            *conn_reusable = h.reuse_connection;
            let mut decoder = http1::BodyDecoder::new(h.content_length);
            // The body bytes that arrived alongside the head — the one copy (out of
            // the shared recv scratch into the slot) the identity path needs; the
            // rest of the body never passes through scratch.
            let leftover = &head_acc[h.head_len..];
            decoder.decode(leftover, |bytes| {
                // Upload response body bytes are currently discarded.
                if matches!(request, RemoteRequest::Upload(_)) {
                    return;
                }
                // SAFETY: as in consume_received — pinned, currently-invalid slot
                // region; total body (content_length) <= req_len == block length.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        dest.add(*body_written),
                        bytes.len(),
                    );
                }
                *body_written += bytes.len();
            });
            *body = Some(decoder);
        }
        Ok(())
    }

    /// Ring-less HTTP state machine. Submits onto a borrowed [`IoUring`] (the
    /// worker's file ring) and is driven by the CQEs the requester routes to it.
    pub(crate) struct HttpEngine {
        client_config: Arc<rustls::ClientConfig>,
        /// Stable-index slab of active exchanges; the index (tagged with
        /// [`HTTP_TAG`]) is the SQE `user_data`.
        exchanges: Vec<Option<HttpExchange>>,
        free_slots: Vec<usize>,
        /// host_key -> idle reusable connections.
        pool: HashMap<String, Vec<Conn>>,
        /// Requests in flight (not counting idle pooled connections).
        active: usize,
        /// Ids whose body fully landed during the last completion routing pass.
        completed: Vec<Identifier>,
        /// Ids whose transport failed terminally (retries exhausted or a
        /// non-retryable error), paired with the error to report to the dataflow.
        failed: Vec<(Identifier, Error)>,
    }

    impl HttpEngine {
        pub fn new(client_config: Arc<rustls::ClientConfig>) -> Result<Self> {
            Ok(Self {
                client_config,
                exchanges: Vec::new(),
                free_slots: Vec::new(),
                pool: HashMap::new(),
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

        /// Begin a range read: bind it to a pooled or fresh connection and submit
        /// the first SQE onto `ring`.
        pub fn start_get(
            &mut self,
            ring: &mut IoUring,
            id: Identifier,
            read: RemoteRead,
        ) -> Result<()> {
            self.start_request(ring, id, RemoteRequest::Read(read))
        }

        pub fn start_upload(
            &mut self,
            ring: &mut IoUring,
            id: Identifier,
            upload: RemoteUpload,
        ) -> Result<()> {
            self.start_request(ring, id, RemoteRequest::Upload(upload))
        }

        fn start_request(
            &mut self,
            ring: &mut IoUring,
            id: Identifier,
            request: RemoteRequest,
        ) -> Result<()> {
            let key = host_key(request.remote());
            if let Some(conn) = self.pool.get_mut(&key).and_then(Vec::pop) {
                // Reuse a pooled keep-alive connection. The new exchange owns it
                // until the response completes or the connection fails.
                let exchange = HttpExchange::new(conn, id, request, State::Sending);
                let idx = self.alloc_slot(exchange);
                self.active += 1;
                self.pump(ring, idx)
            } else {
                let conn = self.new_conn(key, &request)?;
                let exchange = HttpExchange::new(conn, id, request, State::Connecting);
                let idx = self.alloc_slot(exchange);
                self.active += 1;
                self.start_connect(ring, idx)
            }
        }

        /// Advance the connection identified by a tagged `user_data` after its SQE
        /// completed with kernel result `result` (`>= 0` byte count, `< 0` is
        /// `-errno`).
        ///
        /// A transport error — a negative result, an EOF mid-response, or an
        /// error surfaced while parsing — does not propagate out (which would
        /// panic the worker). Instead the read is retried on a fresh connection
        /// or, once that's exhausted, recorded in `failed` so the requester can
        /// fail just the owning dataflow.
        pub fn on_cqe(&mut self, ring: &mut IoUring, user_data: u64, result: i32) -> Result<()> {
            let idx = (user_data & !HTTP_TAG) as usize;

            // Negative result = socket-level error (connect refused, reset,
            // broken pipe, ...). The connection is dead; retry or fail.
            if result < 0 {
                let err = Error::Io(std::io::Error::from_raw_os_error(-result));
                return self.retry_or_fail(ring, idx, err);
            }

            let finished = match self.advance(idx, result as usize) {
                Ok(f) => f,
                Err(e) => return self.retry_or_fail(ring, idx, e),
            };

            if finished {
                let exchange = self.exchanges[idx]
                    .take()
                    .expect("cqe for an active HTTP exchange");
                self.free_slots.push(idx);
                let reusable = exchange.conn_reusable;
                let (id, conn) = exchange.into_connection();
                // A connection whose response body had unknown framing may
                // still hold unread bytes; dropping it closes the socket
                // instead of poisoning the pool.
                if reusable {
                    let key = conn.host_key.clone();
                    self.pool.entry(key).or_default().push(conn);
                }
                self.active -= 1;
                self.completed.push(id);
                return Ok(());
            }

            self.pump(ring, idx)
        }

        /// Fold `size` freshly transferred bytes into the connection's request
        /// state, returning whether the response body is now complete. Errors
        /// (an EOF mid-body, a parse failure) are surfaced to [`on_cqe`](Self::on_cqe), which
        /// turns them into a retry or a recorded failure.
        fn advance(&mut self, idx: usize, size: usize) -> Result<bool> {
            let exchange = self.exchanges[idx].as_mut().unwrap();
            Ok(match exchange.state {
                State::Connecting => false, // connected; pump starts the exchange
                State::Sending => {
                    if exchange.sending_upload_direct {
                        exchange.upload_pos += size;
                    } else {
                        exchange.out_pos += size;
                    }
                    false
                }
                State::Receiving => {
                    if size == 0 {
                        return Err(Error::Io(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "connection closed mid-response",
                        )));
                    }
                    if exchange.recv_in_dest {
                        // Plaintext body recv'd straight into the cache slot
                        // (zero-copy) — nothing to copy or parse, just advance
                        // the decoder and the write cursor.
                        exchange
                            .body
                            .as_mut()
                            .expect("body decoder set before a Direct recv")
                            .consumed(size as u64)?;
                        exchange.body_written += size;
                    } else {
                        exchange.consume_received(size)?;
                    }
                    exchange.request_complete()
                }
            })
        }

        /// Handle a dead connection for the request at slot `idx`: tear it down
        /// and either re-issue the (idempotent) read on a fresh connection — the
        /// common stale-keep-alive case — or, if the error isn't retryable or the
        /// retry budget is spent, record the failure for the requester to surface.
        fn retry_or_fail(&mut self, ring: &mut IoUring, idx: usize, err: Error) -> Result<()> {
            // Take the exchange out of the slab. Its connection was already
            // removed from the pool at `start`, so dropping it cannot leave a
            // stale pool entry behind.
            let dead = self.exchanges[idx]
                .take()
                .expect("cqe for an active HTTP exchange");
            self.free_slots.push(idx);
            let HttpExchange {
                conn,
                id,
                retries,
                request,
                ..
            } = dead;
            let key = conn.host_key.clone();
            drop(conn); // close the failed socket before reconnecting

            if is_retryable(&err) && retries < MAX_HTTP_RETRIES {
                match self.new_conn(key, &request) {
                    Ok(conn) => {
                        let mut fresh = HttpExchange::new(conn, id, request, State::Connecting);
                        fresh.retries = retries + 1;
                        let new_idx = self.alloc_slot(fresh);
                        return self.start_connect(ring, new_idx);
                    }
                    // Couldn't even build the socket — treat as a terminal failure.
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

        fn new_conn(&self, key: String, request: &RemoteRequest) -> Result<Conn> {
            let remote = request.remote();
            let addr = remote.addr();
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

            let transport = if remote.is_https() {
                let server_name =
                    rustls::pki_types::ServerName::try_from(remote.host().to_string())?;
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
                host_key: key,
                sockaddr,
                sockaddr_len,
                out_buf: Vec::new(),
                // Allocated once; reused (never re-zeroed) for every recv.
                in_buf: vec![0u8; IO_CHUNK_SIZE],
            })
        }

        fn alloc_slot(&mut self, exchange: HttpExchange) -> usize {
            if let Some(idx) = self.free_slots.pop() {
                self.exchanges[idx] = Some(exchange);
                idx
            } else {
                self.exchanges.push(Some(exchange));
                self.exchanges.len() - 1
            }
        }

        fn start_connect(&mut self, ring: &mut IoUring, idx: usize) -> Result<()> {
            let exchange = self.exchanges[idx].as_mut().unwrap();
            exchange.state = State::Connecting;
            let fd = exchange.conn.fd.as_raw_fd();
            let addr_ptr = exchange.conn.sockaddr.as_ref() as *const libc::sockaddr_storage
                as *const libc::sockaddr;
            let len = exchange.conn.sockaddr_len;
            let entry = opcode::Connect::new(types::Fd(fd), addr_ptr, len)
                .build()
                .user_data(HTTP_TAG | idx as u64);
            push(ring, &entry)
        }

        /// Submit the next op (send or recv) for the connection based on its
        /// current send buffer.
        fn pump(&mut self, ring: &mut IoUring, idx: usize) -> Result<()> {
            let (is_send, addr, len, fd) = {
                let exchange = self.exchanges[idx].as_mut().unwrap();
                exchange.fill_out();
                if exchange.out_pos < exchange.conn.out_buf.len() {
                    exchange.sending_upload_direct = false;
                    exchange.state = State::Sending;
                    let l = exchange.conn.out_buf.len() - exchange.out_pos;
                    let a = exchange.conn.out_buf[exchange.out_pos..].as_ptr() as usize;
                    (true, a, l, exchange.conn.fd.as_raw_fd())
                } else if matches!(exchange.conn.transport, Transport::Plain)
                    && let RemoteRequest::Upload(upload) = &exchange.request
                    && exchange.upload_pos < upload.data.len()
                {
                    exchange.sending_upload_direct = true;
                    exchange.state = State::Sending;
                    // One send covers one run; `advance` moves `upload_pos` on by
                    // what went out and `pump` picks up the next, exactly as it
                    // does after a short send.
                    let run = upload.data.run_at(exchange.upload_pos);
                    (
                        true,
                        run.as_ptr() as usize,
                        run.len(),
                        exchange.conn.fd.as_raw_fd(),
                    )
                } else {
                    exchange.sending_upload_direct = false;
                    exchange.state = State::Receiving;
                    let fd = exchange.conn.fd.as_raw_fd();
                    // Plaintext body phase: recv straight into the cache slot
                    // (zero-copy). Otherwise recv into the scratch `in_buf` — TLS
                    // ciphertext, or the response headers (for either scheme).
                    let plain_body = matches!(exchange.conn.transport, Transport::Plain)
                        && exchange.body.is_some()
                        && matches!(exchange.request, RemoteRequest::Read(_));
                    if plain_body {
                        // Identity body: the decoder's Direct plan gives the bytes
                        // still expected. `pump` only runs while the request is
                        // unfinished, so this is always > 0 here.
                        let remaining = match exchange.body.as_ref().unwrap().read_plan() {
                            http1::ReadPlan::Direct { max } => max as usize,
                            http1::ReadPlan::Done => 0,
                        };
                        exchange.recv_in_dest = true;
                        // SAFETY: pinned, currently-invalid slot region; remaining
                        // bytes are within the block (content_length <= req_len).
                        let a = unsafe { exchange.dest.add(exchange.body_written) } as usize;
                        (false, a, remaining, fd)
                    } else {
                        exchange.recv_in_dest = false;
                        let a = exchange.conn.in_buf.as_mut_ptr() as usize;
                        (false, a, IO_CHUNK_SIZE, fd)
                    }
                }
            };

            // Cap at the io_uring op length so a body larger than the 32-bit
            // length field goes out over successive sends; `advance` resumes each
            // one at `upload_pos`/`out_pos`.
            let len = len.min(crate::io::backend::MAX_IO_OP_LEN);
            let entry = if is_send {
                opcode::Send::new(types::Fd(fd), addr as *const u8, len as u32)
                    .build()
                    .user_data(HTTP_TAG | idx as u64)
            } else {
                opcode::Recv::new(types::Fd(fd), addr as *mut u8, len as u32)
                    .build()
                    .user_data(HTTP_TAG | idx as u64)
            };
            push(ring, &entry)
        }
    }

    /// Push one SQE onto the (shared) ring and flush. The buffers it references
    /// live in the active exchange slab (not moved until the SQE completes), so
    /// they outlive the op.
    fn push(ring: &mut IoUring, entry: &squeue::Entry) -> Result<()> {
        unsafe {
            ring.submission()
                .push(entry)
                .map_err(|_| Error::Io(std::io::Error::other("http submission queue full")))?;
        }
        ring.submit()?;
        Ok(())
    }

    /// Convert a [`SocketAddr`] into a heap `sockaddr_storage` for io_uring connect.
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
