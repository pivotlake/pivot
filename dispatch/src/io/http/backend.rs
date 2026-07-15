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

use super::RemoteRead;
use super::RemoteWrite;
use super::proto;

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

    /// What a pool thread should do: fetch a range into a pinned slot, or upload
    /// a file's bytes to the object store. Both report `(id, outcome)` on the
    /// job's `sink`.
    enum JobKind {
        Read(RemoteRead),
        Write(RemoteWrite),
    }

    /// One job for a pool thread to perform, reporting `(id, outcome)` on `sink`.
    /// `client_config` builds any fresh connection it needs.
    struct HttpJob {
        id: Identifier,
        kind: JobKind,
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
            // Every job MUST yield one completion (a worker parks until its read
            // reports back) and this thread must survive to serve the next job, so
            // a panic in the fetch fails just this read rather than stranding the
            // worker or shrinking the pool.
            let outcome =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match &job.kind {
                    JobKind::Read(read) => run_fetch(conns, &job.client_config, read),
                    JobKind::Write(write) => run_upload(&job.client_config, write),
                }))
                .unwrap_or_else(|_| {
                    Err(Error::Io(std::io::Error::other(
                        "http pool thread panicked",
                    )))
                });
            // The issuing engine may have been dropped mid-flight (its receiver
            // gone); a failed send is then expected and ignored.
            let _ = job.sink.send((job.id, outcome));
        }
    }

    /// Fetch `read` into its slot, reusing a pooled keep-alive connection when one
    /// is idle for the host and reconnecting once if it turns out to be stale.
    fn run_fetch(
        conns: &ConnPool,
        client_config: &Arc<rustls::ClientConfig>,
        read: &RemoteRead,
    ) -> Result<()> {
        let key = host_key(&read.remote);
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
            Some(c) => match do_request(c, read) {
                Ok(c) => c,
                Err(_) => do_request(connect(client_config, read)?, read)?,
            },
            None => do_request(connect(client_config, read)?, read)?,
        };
        conns
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(key)
            .or_default()
            .push(conn);
        Ok(())
    }

    fn connect(client_config: &Arc<rustls::ClientConfig>, read: &RemoteRead) -> Result<Conn> {
        connect_endpoint(
            client_config,
            read.remote.addr(),
            read.remote.host(),
            read.remote.is_https(),
        )
    }

    /// Open a (optionally TLS-wrapped) connection to `addr`, used by both reads
    /// and uploads.
    fn connect_endpoint(
        client_config: &Arc<rustls::ClientConfig>,
        addr: std::net::SocketAddr,
        host: &str,
        is_https: bool,
    ) -> Result<Conn> {
        let tcp = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT)?;
        tcp.set_nodelay(true).ok();
        // These bound every read/write so a hung server can't pin a pool thread
        // (and stall Drop) forever, so a failure to set them must fail the
        // connection rather than be ignored.
        tcp.set_read_timeout(Some(IO_TIMEOUT))?;
        tcp.set_write_timeout(Some(IO_TIMEOUT))?;
        if is_https {
            let server_name = rustls::pki_types::ServerName::try_from(host.to_string())?;
            let client = ClientConnection::new(client_config.clone(), server_name)?;
            Ok(Conn::Tls(Box::new(StreamOwned::new(client, tcp))))
        } else {
            Ok(Conn::Plain(tcp))
        }
    }

    /// Upload `write.bytes` to its object with one PUT/POST on a fresh connection
    /// (upload connections send `Connection: close` and are never pooled).
    fn run_upload(client_config: &Arc<rustls::ClientConfig>, write: &RemoteWrite) -> Result<()> {
        let conn = connect_endpoint(
            client_config,
            write.remote.addr(),
            write.remote.host(),
            write.remote.is_https(),
        )?;
        do_upload(conn, write)
    }

    /// Send the upload request (head + body) on `conn` and read its response,
    /// succeeding on a 2xx status.
    fn do_upload(mut conn: Conn, write: &RemoteWrite) -> Result<()> {
        let auth = write.remote.auth_header();
        let head = proto::build_put(
            write.remote.method().as_str(),
            write.remote.host_header(),
            write.remote.request_target(),
            write.bytes.len(),
            write.remote.content_type(),
            auth.as_deref(),
        );
        conn.write_all(&head)?;
        conn.write_all(&write.bytes)?;
        conn.flush()?;

        // Read until the response head parses; any 2xx is success.
        let mut chunk = vec![0u8; 16 * 1024];
        let mut acc: Vec<u8> = Vec::new();
        loop {
            let n = conn.read(&mut chunk)?;
            if n == 0 {
                return Err(Error::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "connection closed before upload response",
                )));
            }
            acc.extend_from_slice(&chunk[..n]);
            // Any 2xx is success; a rejected status carries the server's error body
            // in the `ProtoError` snippet.
            match proto::parse_response(&acc, (200, 299), None) {
                Ok(proto::RespParse::Ready { .. }) => return Ok(()),
                Ok(proto::RespParse::Incomplete) => continue,
                Err(e) => {
                    return Err(Error::Io(std::io::Error::other(format!(
                        "{e}; {} {}",
                        write.remote.host_header(),
                        write.remote.request_target(),
                    ))));
                }
            }
        }
    }

    /// Issue the range GET on `conn` and read its body into `read.dest`, returning
    /// the connection for re-pooling on success.
    fn do_request(mut conn: Conn, read: &RemoteRead) -> Result<Conn> {
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

        // Accumulate until the response head parses. A range read requires (and
        // `parse_response` enforces) a `Content-Length` that fits the slot; a
        // rejected status carries the server's error body in the snippet.
        let mut chunk = vec![0u8; 16 * 1024];
        let mut acc: Vec<u8> = Vec::new();
        let (head_len, content_length) = loop {
            let n = conn.read(&mut chunk)?;
            if n == 0 {
                return Err(Error::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "connection closed before response head",
                )));
            }
            acc.extend_from_slice(&chunk[..n]);
            match proto::parse_response(&acc, (206, 206), Some(read.len as u64)) {
                Ok(proto::RespParse::Ready {
                    head_len,
                    content_length,
                }) => {
                    break (
                        head_len,
                        content_length.expect("read requires Content-Length"),
                    );
                }
                Ok(proto::RespParse::Incomplete) => continue,
                Err(e) => {
                    return Err(Error::Io(std::io::Error::other(format!(
                        "{e}; {} {}",
                        read.remote.host_header(),
                        read.remote.request_target(),
                    ))));
                }
            }
        };

        let body_len = content_length as usize;
        // SAFETY: dest points into the pinned cache slot for exactly this block's
        // currently-invalid sub-blocks; body_len <= read.len <= the block length
        // (validated in parse_response).
        let dest = unsafe { std::slice::from_raw_parts_mut(read.dest, body_len) };

        // Drive the body through the sans-IO decoder. The range policy guarantees
        // an identity (`Content-Length`) body, so the Direct path applies: read
        // straight into the cache slot and report the count, no intermediate
        // buffer. `decode` first absorbs any body bytes that arrived alongside the
        // head, and would transparently handle a chunked body too were the range
        // policy ever relaxed.
        let mut body = http1::BodyDecoder::new(content_length);
        let mut written = 0usize;
        let leftover = &acc[head_len..];
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
        pub fn start(&mut self, id: Identifier, read: RemoteRead) -> Result<()> {
            self.dispatch(id, JobKind::Read(read))
        }

        /// Dispatch an upload onto the pool (non-blocking); it completes later on
        /// `rx`, exactly like a read.
        pub fn start_upload(&mut self, id: Identifier, write: RemoteWrite) -> Result<()> {
            self.dispatch(id, JobKind::Write(write))
        }

        fn dispatch(&mut self, id: Identifier, kind: JobKind) -> Result<()> {
            self.pool
                .send(HttpJob {
                    id,
                    kind,
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

    const RECV_CHUNK: usize = 16 * 1024;

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

    /// How many body bytes to hand the transport per pump. Bounds the ciphertext
    /// (or plaintext) buffered for one in-flight `send`, so a large upload streams
    /// in bounded memory rather than one giant buffered write.
    const SEND_CHUNK: usize = 256 * 1024;

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

    /// A read's response-body destination: a pointer into the pinned cache slot and
    /// its capacity. An upload has no slot (its response is drained, not kept).
    #[derive(Clone, Copy)]
    struct Slot {
        ptr: *mut u8,
        cap: usize,
    }

    /// A prepared HTTP request, self-contained so a transient transport failure can
    /// rebuild a fresh connection and re-issue it (a range GET and an object
    /// PUT/POST are both idempotent).
    ///
    /// It is **generic over method**: the difference between a read and an upload is
    /// entirely *data* — whether there is a `send_body`, where the response body
    /// goes (`slot`), and the acceptance policy — not a separate type. Both always
    /// receive and parse a response, so a rejected upload's body is available for
    /// the error message just like a read's.
    #[derive(Clone)]
    struct PreparedOp {
        addr: SocketAddr,
        /// TLS SNI host name.
        server_name: String,
        /// Per-host pool key — reads and uploads to the same host share connections.
        host_key: String,
        is_https: bool,
        /// The request line + headers, prebuilt.
        request_head: Vec<u8>,
        /// The request body to stream out (an upload); `None` for a GET.
        send_body: Option<Arc<[u8]>>,
        /// Where the response body is written: a read's pinned cache slot, or `None`
        /// to drain it (an upload only needs the status).
        slot: Option<Slot>,
        /// Accepted status codes, inclusive: `(206, 206)` for a range read, `(200,
        /// 299)` for an upload. Whether the whole body must land is implied by
        /// `slot`: a read (`Some`) needs its payload; an upload (`None`) does not.
        status_range: (u16, u16),
    }

    impl PreparedOp {
        /// A range read into `read`'s pinned cache slot.
        fn for_read(read: &RemoteRead) -> Self {
            PreparedOp {
                addr: read.remote.addr(),
                server_name: read.remote.host().to_string(),
                host_key: host_key(&read.remote),
                is_https: read.remote.is_https(),
                request_head: proto::build_range_get(
                    read.remote.host_header(),
                    read.remote.request_target(),
                    read.offset,
                    read.len,
                    read.remote.auth_header().as_deref(),
                ),
                send_body: None,
                slot: Some(Slot {
                    ptr: read.dest,
                    cap: read.len,
                }),
                status_range: (206, 206),
            }
        }

        /// An object upload (PUT/POST) streaming `write`'s bytes; its response is
        /// drained, not kept.
        fn for_upload(write: &RemoteWrite) -> Self {
            PreparedOp {
                addr: write.remote.addr(),
                server_name: write.remote.host().to_string(),
                host_key: format!("{}:{}", write.remote.host(), write.remote.addr().port()),
                is_https: write.remote.is_https(),
                request_head: proto::build_put(
                    write.remote.method().as_str(),
                    write.remote.host_header(),
                    write.remote.request_target(),
                    write.bytes.len(),
                    write.remote.content_type(),
                    write.remote.auth_header().as_deref(),
                ),
                send_body: Some(write.bytes.clone()),
                slot: None,
                status_range: (200, 299),
            }
        }
    }

    /// One socket + its (optional) TLS session, reused across requests — reads *or*
    /// uploads — via the keep-alive pool. The read/upload difference lives entirely
    /// in [`op`](Conn::op)'s data, not in the connection's shape.
    struct Conn {
        fd: OwnedFd,
        transport: Transport,
        /// Connect target; boxed so the SQE can hold a stable pointer to it.
        sockaddr: Box<libc::sockaddr_storage>,
        sockaddr_len: libc::socklen_t,

        // --- current request ---
        id: Identifier,
        /// The request being performed, kept whole so a failure can re-issue it.
        op: PreparedOp,
        /// How many times this request has been retried; bounded by [`MAX_HTTP_RETRIES`].
        retries: u32,

        // --- send progress ---
        /// Whether the request head has been handed to the transport yet.
        head_queued: bool,
        /// How many `send_body` bytes have been handed to the transport.
        body_sent: usize,

        // --- receive progress ---
        /// Accumulates the response head; on a rejected status it also holds the
        /// leading body bytes for the error message.
        resp_acc: Vec<u8>,
        /// The response body decoder, `None` until the head parses.
        resp_body: Option<http1::BodyDecoder>,
        /// Bytes of the response body written into the read slot.
        resp_written: usize,
        /// True when the last recv was posted straight into the slot (a plaintext
        /// read body, zero-copy) rather than into `in_buf`.
        recv_in_dest: bool,
        /// True once the request succeeded (an accepted status, and — for a read —
        /// the full body landed).
        complete: bool,
        /// True when the connection is clean and can be returned to the pool.
        poolable: bool,

        // Bytes pending send (ciphertext for TLS, plaintext for plain).
        out_buf: Vec<u8>,
        out_pos: usize,
        // Reusable recv scratch (TLS ciphertext, or plaintext headers / response).
        in_buf: Vec<u8>,

        state: State,
    }

    impl Conn {
        /// Reset request state for pooling, keeping the live socket/TLS session.
        /// `bind_request` fully re-installs the request on reuse; here we wipe the
        /// slot pointer so no stale pointer lingers in an idle connection.
        fn clear_request(&mut self) {
            self.id = 0;
            self.op.request_head.clear();
            self.op.send_body = None;
            self.op.slot = None;
            self.head_queued = false;
            self.body_sent = 0;
            self.out_buf.clear();
            self.out_pos = 0;
            // `in_buf` is reused as-is (not cleared); recv overwrites what it needs.
            self.resp_acc.clear();
            self.resp_body = None;
            self.resp_written = 0;
            self.recv_in_dest = false;
            self.complete = false;
            self.poolable = false;
        }

        /// Ensure `out_buf` holds the next bytes to send: queue the request head
        /// once, then stream any `send_body` in [`SEND_CHUNK`] pieces, then drain
        /// rustls's outgoing records.
        fn fill_out(&mut self) {
            if self.out_pos < self.out_buf.len() {
                return; // still have unsent bytes
            }
            self.out_buf.clear();
            self.out_pos = 0;
            match &mut self.transport {
                Transport::Plain => {
                    if !self.head_queued {
                        self.out_buf.extend_from_slice(&self.op.request_head);
                        self.head_queued = true;
                    } else if let Some(body) = &self.op.send_body
                        && self.body_sent < body.len()
                    {
                        let end = (self.body_sent + SEND_CHUNK).min(body.len());
                        self.out_buf.extend_from_slice(&body[self.body_sent..end]);
                        self.body_sent = end;
                    }
                }
                Transport::Tls(tls) => {
                    if !tls.is_handshaking() {
                        if !self.head_queued {
                            tls.writer()
                                .write_all(&self.op.request_head)
                                .expect("rustls writer is infallible into its buffer");
                            self.head_queued = true;
                        } else if let Some(body) = &self.op.send_body
                            && self.body_sent < body.len()
                        {
                            // rustls buffers bounded plaintext, so feed only what it
                            // accepts (a short `write` when its buffer fills); the
                            // drain below flushes it before we feed the rest.
                            let end = (self.body_sent + SEND_CHUNK).min(body.len());
                            let n = tls
                                .writer()
                                .write(&body[self.body_sent..end])
                                .expect("rustls writer is infallible into its buffer");
                            self.body_sent += n;
                        }
                    }
                    while tls.wants_write() {
                        tls.write_tls(&mut self.out_buf)
                            .expect("write_tls into a Vec is infallible");
                    }
                }
            }
        }

        /// Process `n` freshly received bytes from `in_buf` (recv landed in
        /// scratch): parse the response head, then feed the body to its
        /// destination — copied into the read slot, or drained for an upload.
        ///
        /// A read's plaintext body never reaches here (`pump` recvs it straight
        /// into the slot); a read's TLS body is decrypted straight into the slot
        /// below (the single copy TLS requires). Header bytes and an upload's
        /// response always go through here.
        fn consume_received(&mut self, n: usize) -> Result<()> {
            let Conn {
                transport,
                in_buf,
                op,
                resp_acc,
                resp_body,
                resp_written,
                complete,
                poolable,
                ..
            } = self;

            match transport {
                Transport::Plain => feed_response(
                    op,
                    resp_acc,
                    resp_body,
                    resp_written,
                    complete,
                    poolable,
                    &in_buf[..n],
                ),
                Transport::Tls(tls) => {
                    // `read_tls` accepts only as much ciphertext as fits rustls's
                    // bounded buffer, so loop: feed, process, drain until consumed.
                    let mut cursor = &in_buf[..n];
                    let mut scratch = [0u8; 8192];
                    while !cursor.is_empty() {
                        let fed = tls.read_tls(&mut cursor)?;
                        tls.process_new_packets().map_err(Error::Tls)?;
                        let mut drained = 0usize;
                        loop {
                            // Once the head is in, a read decrypts its body straight
                            // into the slot (one copy); header bytes and an upload's
                            // drained body go through the scratch + `feed_response`.
                            if let (Some(decoder), Some(slot)) = (resp_body.as_mut(), op.slot) {
                                let remaining = match decoder.read_plan() {
                                    http1::ReadPlan::Direct { max } => max as usize,
                                    http1::ReadPlan::Done => 0,
                                };
                                if remaining == 0 {
                                    break;
                                }
                                // SAFETY: pinned slot region; content_length <=
                                // slot.cap (validated when the head parsed).
                                let dst = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        slot.ptr.add(*resp_written),
                                        remaining,
                                    )
                                };
                                match tls.reader().read(dst) {
                                    Ok(0) => break,
                                    Ok(m) => {
                                        decoder.consumed(m as u64)?;
                                        *resp_written += m;
                                        drained += m;
                                        if decoder.is_complete() {
                                            *complete = true;
                                            *poolable = true;
                                        }
                                    }
                                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                                    Err(e) => return Err(Error::Io(e)),
                                }
                            } else {
                                match tls.reader().read(&mut scratch) {
                                    Ok(0) => break,
                                    Ok(m) => {
                                        drained += m;
                                        feed_response(
                                            op,
                                            resp_acc,
                                            resp_body,
                                            resp_written,
                                            complete,
                                            poolable,
                                            &scratch[..m],
                                        )?;
                                    }
                                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                                    Err(e) => return Err(Error::Io(e)),
                                }
                            }
                        }
                        if fed == 0 && drained == 0 {
                            break;
                        }
                    }
                    Ok(())
                }
            }
        }
    }

    /// Feed a chunk of received response bytes: accumulate and parse the head, then
    /// decode the body to its destination via [`write_body`] — copied into the read
    /// slot, or drained for an upload. A status outside `op.status_range` is a
    /// terminal error carrying the captured head + leading body.
    ///
    /// Once the head is in, a plaintext read's body is recv'd straight into the slot
    /// by `pump` and a TLS read's is decrypted straight into it by `consume_received`
    /// — so for a read this only sees header bytes. An upload's whole response comes
    /// through here (and is drained).
    fn feed_response(
        op: &PreparedOp,
        resp_acc: &mut Vec<u8>,
        resp_body: &mut Option<http1::BodyDecoder>,
        resp_written: &mut usize,
        complete: &mut bool,
        poolable: &mut bool,
        chunk: &[u8],
    ) -> Result<()> {
        // Body phase.
        if let Some(decoder) = resp_body.as_mut() {
            write_body(op, decoder, resp_written, chunk);
            if decoder.is_complete() {
                *complete = true;
                *poolable = true;
            }
            return Ok(());
        }

        // Head phase. `parse_response` checks the status range and (for a read's
        // fixed slot) that the body fits, so a rejected status surfaces here with
        // the server's explanation.
        resp_acc.extend_from_slice(chunk);
        let (head_len, content_length) = match proto::parse_response(
            resp_acc,
            op.status_range,
            op.slot.map(|s| s.cap as u64),
        )? {
            proto::RespParse::Incomplete => return Ok(()),
            proto::RespParse::Ready {
                head_len,
                content_length,
            } => (head_len, content_length),
        };
        let content_length = match content_length {
            Some(len) => len,
            // No Content-Length on an accepted response: only an upload reaches here
            // (a read required one via `max_body`). The write is done, but the
            // connection can't be drained, so it isn't pooled.
            None => {
                *complete = true;
                *poolable = false;
                return Ok(());
            }
        };
        let mut decoder = http1::BodyDecoder::new(content_length);
        // Body bytes that arrived alongside the head.
        write_body(op, &mut decoder, resp_written, &resp_acc[head_len..]);
        if decoder.is_complete() {
            *complete = true;
            *poolable = true;
        }
        *resp_body = Some(decoder);
        Ok(())
    }

    /// Write decoded response-body bytes to their destination: copy into the read
    /// slot (advancing `resp_written`), or discard for an upload. The range policy
    /// guarantees a read's body fits its slot, so no truncation.
    fn write_body(
        op: &PreparedOp,
        decoder: &mut http1::BodyDecoder,
        resp_written: &mut usize,
        chunk: &[u8],
    ) {
        match op.slot {
            Some(slot) => decoder.decode(chunk, |bytes| {
                // SAFETY: pinned slot region; total body <= slot.cap (validated when
                // the head parsed), so writes stay within the slot.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        slot.ptr.add(*resp_written),
                        bytes.len(),
                    );
                }
                *resp_written += bytes.len();
            }),
            None => decoder.decode(chunk, |_| {}),
        };
    }

    /// Ring-less HTTP state machine. Submits onto a borrowed [`IoUring`] (the
    /// worker's file ring) and is driven by the CQEs the requester routes to it.
    pub(crate) struct HttpEngine {
        client_config: Arc<rustls::ClientConfig>,
        /// Stable-index slab of connections; the index (tagged with [`HTTP_TAG`])
        /// is the SQE `user_data`.
        conns: Vec<Option<Conn>>,
        free_slots: Vec<usize>,
        /// host_key -> idle connection slab indices. Reads and uploads to the same
        /// host share this pool, so a PUT can reuse a connection a GET left warm.
        pool: HashMap<String, Vec<usize>>,
        /// Requests in flight (reads and uploads; not counting idle pooled ones).
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
                conns: Vec::new(),
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
        pub fn start(
            &mut self,
            ring: &mut IoUring,
            id: Identifier,
            read: RemoteRead,
        ) -> Result<()> {
            self.begin(ring, id, PreparedOp::for_read(&read))
        }

        /// Begin an upload, the write-side twin of [`start`](Self::start). Shares
        /// the same pool, so it can reuse a connection a read left warm.
        pub fn start_upload(
            &mut self,
            ring: &mut IoUring,
            id: Identifier,
            write: RemoteWrite,
        ) -> Result<()> {
            self.begin(ring, id, PreparedOp::for_upload(&write))
        }

        /// Bind a prepared request to a pooled or fresh connection and submit its
        /// first SQE. Generic over read/upload — the difference is `op`'s data.
        fn begin(&mut self, ring: &mut IoUring, id: Identifier, op: PreparedOp) -> Result<()> {
            let key = op.host_key.clone();
            self.active += 1;

            if let Some(idx) = self.pool.get_mut(&key).and_then(|v| v.pop()) {
                // Reuse a pooled keep-alive connection: bind and pump straight to
                // sending (handshake already done).
                bind_request(self.conns[idx].as_mut().unwrap(), id, op);
                self.pump(ring, idx)
            } else {
                let conn = self.new_conn(op, id)?;
                let idx = self.alloc_slot(conn);
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
                let conn = self.conns[idx].as_mut().unwrap();
                let id = conn.id;
                if conn.poolable {
                    // Clean (a read, or an upload with a fully-drained response):
                    // return it to the pool for a later read or upload to this host.
                    let key = conn.op.host_key.clone();
                    conn.clear_request();
                    self.pool.entry(key).or_default().push(idx);
                } else {
                    // An upload whose response body length was unknown: the write
                    // succeeded, but the connection can't be safely reused. Drop it.
                    self.conns[idx] = None;
                    self.free_slots.push(idx);
                }
                self.active -= 1;
                self.completed.push(id);
                return Ok(());
            }

            self.pump(ring, idx)
        }

        /// Fold `size` freshly transferred bytes into the connection's request
        /// state, returning whether the request is now complete. Errors (an EOF
        /// mid-body, a parse failure) are surfaced to [`on_cqe`](Self::on_cqe),
        /// which turns them into a retry or a recorded failure.
        fn advance(&mut self, idx: usize, size: usize) -> Result<bool> {
            let conn = self.conns[idx].as_mut().unwrap();
            match conn.state {
                State::Connecting => Ok(false), // connected; pump starts the exchange
                State::Sending => {
                    conn.out_pos += size;
                    Ok(false)
                }
                State::Receiving => {
                    if size == 0 {
                        // An upload (no slot) whose accepted head already parsed is
                        // durable — a truncated/absent response body just means we
                        // can't reuse the connection. A read (has a slot), or a
                        // head-not-yet-in, closing mid-response is a transport error.
                        if conn.op.slot.is_none() && (conn.resp_body.is_some() || conn.complete) {
                            conn.complete = true;
                            conn.poolable = false;
                            return Ok(true);
                        }
                        return Err(Error::Io(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "connection closed mid-response",
                        )));
                    }
                    // A read's plaintext body recv'd straight into the slot advances
                    // its decoder here; every other recv (TLS, headers, an upload's
                    // response) is parsed by `consume_received`.
                    if conn.recv_in_dest {
                        conn.resp_written += size;
                        let decoder = conn
                            .resp_body
                            .as_mut()
                            .expect("body decoder set before a Direct recv");
                        decoder.consumed(size as u64)?;
                        if decoder.is_complete() {
                            conn.complete = true;
                            conn.poolable = true;
                        }
                    } else {
                        conn.consume_received(size)?;
                    }
                    Ok(conn.complete)
                }
            }
        }

        /// Handle a dead connection for the request at slot `idx`: tear it down
        /// and either re-issue the (idempotent) read on a fresh connection — the
        /// common stale-keep-alive case — or, if the error isn't retryable or the
        /// retry budget is spent, record the failure for the requester to surface.
        fn retry_or_fail(&mut self, ring: &mut IoUring, idx: usize, err: Error) -> Result<()> {
            // Take the dead connection out of the slab; dropping it closes the
            // socket. It was in-flight (popped from the pool at `begin`), so it
            // leaves no stale pool entry behind.
            let dead = self.conns[idx].take().expect("cqe for a live connection");
            self.free_slots.push(idx);
            let id = dead.id;
            let retries = dead.retries;
            let op = dead.op.clone();
            drop(dead);

            if is_retryable(&err) && retries < MAX_HTTP_RETRIES {
                match self.new_conn(op, id) {
                    Ok(mut fresh) => {
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

        fn new_conn(&self, op: PreparedOp, id: Identifier) -> Result<Conn> {
            let domain = match op.addr {
                SocketAddr::V4(_) => libc::AF_INET,
                SocketAddr::V6(_) => libc::AF_INET6,
            };
            let raw = unsafe { libc::socket(domain, libc::SOCK_STREAM, 0) };
            if raw < 0 {
                return Err(Error::Io(std::io::Error::last_os_error()));
            }
            let fd = unsafe { OwnedFd::from_raw_fd(raw) };
            let (sockaddr, sockaddr_len) = to_sockaddr(op.addr);

            let transport = if op.is_https {
                let server_name = rustls::pki_types::ServerName::try_from(op.server_name.clone())?;
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
                op,
                retries: 0,
                head_queued: false,
                body_sent: 0,
                resp_acc: Vec::new(),
                resp_body: None,
                resp_written: 0,
                recv_in_dest: false,
                complete: false,
                poolable: false,
                out_buf: Vec::new(),
                out_pos: 0,
                // Allocated once; reused (never re-zeroed) for every recv.
                in_buf: vec![0u8; RECV_CHUNK],
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
                .user_data(HTTP_TAG | idx as u64);
            push(ring, &entry)
        }

        /// Submit the next op (send or recv) for the connection based on its
        /// current send buffer.
        fn pump(&mut self, ring: &mut IoUring, idx: usize) -> Result<()> {
            let (is_send, addr, len, fd) = {
                let conn = self.conns[idx].as_mut().unwrap();
                conn.fill_out();
                if conn.out_pos < conn.out_buf.len() {
                    conn.state = State::Sending;
                    let l = conn.out_buf.len() - conn.out_pos;
                    let a = conn.out_buf[conn.out_pos..].as_ptr() as usize;
                    (true, a, l, conn.fd.as_raw_fd())
                } else {
                    conn.state = State::Receiving;
                    let fd = conn.fd.as_raw_fd();
                    // A read's plaintext body phase recvs straight into the cache
                    // slot (zero-copy). Everything else (TLS ciphertext, response
                    // headers, an upload's response) recvs into the scratch `in_buf`.
                    let plain_read_body = matches!(conn.transport, Transport::Plain)
                        && conn.op.slot.is_some()
                        && conn.resp_body.is_some();
                    if plain_read_body {
                        let slot = conn.op.slot.expect("plain_read_body implies a slot");
                        // Identity body: the decoder's Direct plan gives the bytes
                        // still expected. `pump` only runs while unfinished, so > 0.
                        let remaining = match conn.resp_body.as_ref().unwrap().read_plan() {
                            http1::ReadPlan::Direct { max } => max as usize,
                            http1::ReadPlan::Done => 0,
                        };
                        conn.recv_in_dest = true;
                        // SAFETY: pinned, currently-invalid slot region; remaining
                        // bytes are within the slot (content_length <= slot.cap).
                        let a = unsafe { slot.ptr.add(conn.resp_written) } as usize;
                        (false, a, remaining, fd)
                    } else {
                        conn.recv_in_dest = false;
                        let a = conn.in_buf.as_mut_ptr() as usize;
                        (false, a, RECV_CHUNK, fd)
                    }
                }
            };

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
    /// live in the connection slab (not moved until the request completes), so
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

    /// Bind a new request onto an already-connected (pooled) connection, resetting
    /// all per-request state for a fresh exchange on the reused socket.
    fn bind_request(conn: &mut Conn, id: Identifier, op: PreparedOp) {
        conn.id = id;
        conn.op = op;
        conn.retries = 0;
        conn.head_queued = false;
        conn.body_sent = 0;
        conn.out_buf.clear();
        conn.out_pos = 0;
        // `in_buf` is reused as-is (not cleared); recv overwrites what it needs.
        conn.resp_acc.clear();
        conn.resp_body = None;
        conn.resp_written = 0;
        conn.recv_in_dest = false;
        conn.complete = false;
        conn.poolable = false;
        conn.state = State::Sending;
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
