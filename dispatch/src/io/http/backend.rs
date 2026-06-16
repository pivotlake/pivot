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
//! On non-Linux (macOS dev builds) there is no ring: the engine performs the whole
//! exchange synchronously at `start` time via `std::net` + `rustls::StreamOwned`,
//! mirroring the file path's `pread` fallback.

use super::RemoteRead;
use super::proto;

#[cfg(target_os = "linux")]
pub(crate) use uring_engine::HttpEngine;

#[cfg(all(unix, not(target_os = "linux")))]
pub(crate) use blocking_engine::HttpEngine;

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
// Other Unix: synchronous std::net + rustls fallback
// ============================================================================

#[cfg(all(unix, not(target_os = "linux")))]
mod blocking_engine {
    use super::*;
    use crate::Identifier;
    use crate::io::http::http1;
    use crate::io::http::{Error, Result};
    use rustls::{ClientConnection, StreamOwned};
    use std::collections::HashMap;
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::sync::Arc;

    /// A pooled connection — TLS-wrapped or plain TCP.
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

    pub(crate) struct HttpEngine {
        client_config: Arc<rustls::ClientConfig>,
        pool: HashMap<String, Vec<Conn>>,
        completed: Vec<Identifier>,
    }

    impl HttpEngine {
        pub fn new(client_config: Arc<rustls::ClientConfig>) -> Result<Self> {
            Ok(Self {
                client_config,
                pool: HashMap::new(),
                completed: Vec::new(),
            })
        }

        /// Perform the whole range read synchronously and queue its id as completed.
        pub fn start(&mut self, id: Identifier, read: RemoteRead) -> Result<()> {
            let key = host_key(&read.remote);
            let pooled = self.pool.get_mut(&key).and_then(|v| v.pop());
            let conn = match pooled {
                // A pooled connection may have been closed by the server's
                // keep-alive timeout; on any error, reconnect once and retry.
                Some(c) => match Self::do_request(c, &read) {
                    Ok(c) => c,
                    Err(_) => {
                        let fresh = self.connect(&read)?;
                        Self::do_request(fresh, &read)?
                    }
                },
                None => {
                    let fresh = self.connect(&read)?;
                    Self::do_request(fresh, &read)?
                }
            };
            self.pool.entry(key).or_default().push(conn);
            self.completed.push(id);
            Ok(())
        }

        pub fn take_completed(&mut self) -> Vec<Identifier> {
            std::mem::take(&mut self.completed)
        }

        pub fn has_active(&self) -> bool {
            !self.completed.is_empty()
        }

        fn connect(&self, read: &RemoteRead) -> Result<Conn> {
            let tcp = TcpStream::connect(read.remote.addr())?;
            tcp.set_nodelay(true).ok();
            if read.remote.is_https() {
                let server_name =
                    rustls::pki_types::ServerName::try_from(read.remote.host().to_string())?;
                let client = ClientConnection::new(self.client_config.clone(), server_name)?;
                Ok(Conn::Tls(Box::new(StreamOwned::new(client, tcp))))
            } else {
                Ok(Conn::Plain(tcp))
            }
        }

        /// Issue the range GET on `conn` and read its body into `read.dest`,
        /// returning the connection for re-pooling on success.
        fn do_request(mut conn: Conn, read: &RemoteRead) -> Result<Conn> {
            let request = proto::build_range_get(
                read.remote.host_header(),
                read.remote.request_target(),
                read.offset,
                read.len,
            );
            conn.write_all(&request)?;
            conn.flush()?;

            // Accumulate until the response head parses.
            let mut chunk = vec![0u8; 16 * 1024];
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
                match proto::parse_response_head(&acc, read.len)? {
                    proto::HeadParse::Complete(h) => break h,
                    proto::HeadParse::Incomplete => continue,
                }
            };

            let body_len = head.content_length as usize;
            // SAFETY: dest points into the pinned cache slot for exactly this
            // block's currently-invalid sub-blocks; body_len <= read.len <= the
            // block length (validated in parse_response_head).
            let dest = unsafe { std::slice::from_raw_parts_mut(read.dest, body_len) };

            // Drive the body through the sans-IO decoder. The range policy
            // guarantees an identity (`Content-Length`) body, so the Direct path
            // applies: read straight into the cache slot and report the count —
            // no intermediate buffer. `decode` first absorbs any body bytes that
            // arrived alongside the head, and would transparently handle a chunked
            // body too were the range policy ever relaxed.
            let mut body = http1::BodyDecoder::new(http1::Framing::Length(head.content_length));
            let mut written = 0usize;
            let leftover = &acc[head.head_len..];
            body.decode(leftover, |bytes| {
                dest[written..written + bytes.len()].copy_from_slice(bytes);
                written += bytes.len();
            })?;

            while !body.is_complete() {
                let eof = || {
                    Error::Io(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "connection closed before response body completed",
                    ))
                };
                match body.read_plan() {
                    http1::ReadPlan::Done => break,
                    http1::ReadPlan::Direct { max } => {
                        let cap = (max as usize).min(dest.len() - written);
                        let n = conn.read(&mut dest[written..written + cap])?;
                        if n == 0 {
                            return Err(eof());
                        }
                        written += n;
                        body.consumed(n as u64)?;
                    }
                    http1::ReadPlan::Chunked => {
                        let n = conn.read(&mut chunk)?;
                        if n == 0 {
                            return Err(eof());
                        }
                        body.decode(&chunk[..n], |bytes| {
                            dest[written..written + bytes.len()].copy_from_slice(bytes);
                            written += bytes.len();
                        })?;
                    }
                }
            }

            Ok(conn)
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

    /// One socket + its (optional) TLS session, reused across requests via the
    /// keep-alive pool.
    struct Conn {
        fd: OwnedFd,
        transport: Transport,
        host_key: String,
        /// Connect target; boxed so the SQE can hold a stable pointer to it.
        sockaddr: Box<libc::sockaddr_storage>,
        sockaddr_len: libc::socklen_t,

        // --- current request ---
        id: Identifier,
        dest: *mut u8,
        req_len: usize,
        request_bytes: Vec<u8>,
        request_queued: bool,

        // Bytes pending send (ciphertext for TLS, the request for plain).
        out_buf: Vec<u8>,
        out_pos: usize,
        // Reusable recv scratch (TLS ciphertext, or plaintext headers). Allocated
        // once at `RECV_CHUNK` and never re-zeroed — recv overwrites `[..n]` and we
        // only read that. Plaintext bodies skip it entirely (recv'd into `dest`).
        in_buf: Vec<u8>,

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

        state: State,
    }

    impl Conn {
        /// Reset the request-specific fields, keeping the live socket/TLS session
        /// so it can be returned to the pool for reuse.
        fn clear_request(&mut self) {
            self.id = 0;
            self.dest = std::ptr::null_mut();
            self.req_len = 0;
            self.request_bytes.clear();
            self.request_queued = false;
            self.out_buf.clear();
            self.out_pos = 0;
            // `in_buf` is reused as-is (not cleared); recv overwrites what it needs.
            self.head_acc.clear();
            self.body = None;
            self.body_written = 0;
            self.recv_in_dest = false;
        }

        /// Ensure `out_buf` holds the next bytes to send: queue the HTTP request
        /// once it's allowed, then drain rustls's outgoing records.
        fn fill_out(&mut self) {
            if self.out_pos < self.out_buf.len() {
                return; // still have unsent bytes
            }
            self.out_buf.clear();
            self.out_pos = 0;
            match &mut self.transport {
                Transport::Plain => {
                    if !self.request_queued {
                        self.out_buf.extend_from_slice(&self.request_bytes);
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
                    while tls.wants_write() {
                        tls.write_tls(&mut self.out_buf)
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
            let Conn {
                transport,
                in_buf,
                head_acc,
                body,
                dest,
                req_len,
                body_written,
                ..
            } = self;
            let dest = *dest;
            let req_len = *req_len;

            match transport {
                // Reached only in the header phase; once the head parses, `pump`
                // recvs the body straight into `dest`, so plaintext bodies never
                // pass through here.
                Transport::Plain => {
                    parse_head(head_acc, body, dest, req_len, body_written, &in_buf[..n])
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
                                // Body phase: decrypt straight into the cache slot.
                                // A range body is identity-framed (the `proto`
                                // policy enforces `Content-Length`), so the decoder
                                // hands back a Direct plan whose `max` is the bytes
                                // still expected — exactly the room to decrypt into.
                                let remaining = match decoder.read_plan() {
                                    http1::ReadPlan::Direct { max } => max as usize,
                                    http1::ReadPlan::Done => 0,
                                    http1::ReadPlan::Chunked => {
                                        unreachable!("range body is identity-framed")
                                    }
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
                                            dest,
                                            req_len,
                                            body_written,
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
    fn parse_head(
        head_acc: &mut Vec<u8>,
        body: &mut Option<http1::BodyDecoder>,
        dest: *mut u8,
        req_len: usize,
        body_written: &mut usize,
        chunk: &[u8],
    ) -> Result<()> {
        head_acc.extend_from_slice(chunk);
        if let proto::HeadParse::Complete(h) = proto::parse_response_head(head_acc, req_len)? {
            let mut decoder = http1::BodyDecoder::new(http1::Framing::Length(h.content_length));
            // The body bytes that arrived alongside the head — the one copy (out of
            // the shared recv scratch into the slot) the identity path needs; the
            // rest of the body never passes through scratch.
            let leftover = &head_acc[h.head_len..];
            decoder.decode(leftover, |bytes| {
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
            })?;
            *body = Some(decoder);
        }
        Ok(())
    }

    /// Ring-less HTTP state machine. Submits onto a borrowed [`IoUring`] (the
    /// worker's file ring) and is driven by the CQEs the requester routes to it.
    pub(crate) struct HttpEngine {
        client_config: Arc<rustls::ClientConfig>,
        /// Stable-index slab of connections; the index (tagged with [`HTTP_TAG`])
        /// is the SQE `user_data`.
        conns: Vec<Option<Conn>>,
        free_slots: Vec<usize>,
        /// host_key -> idle connection slab indices.
        pool: HashMap<String, Vec<usize>>,
        /// Requests in flight (not counting idle pooled connections).
        active: usize,
        /// Ids whose body fully landed during the last completion routing pass.
        completed: Vec<Identifier>,
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
            })
        }

        pub fn has_active(&self) -> bool {
            self.active > 0
        }

        pub fn take_completed(&mut self) -> Vec<Identifier> {
            std::mem::take(&mut self.completed)
        }

        /// Begin a range read: bind it to a pooled or fresh connection and submit
        /// the first SQE onto `ring`.
        pub fn start(
            &mut self,
            ring: &mut IoUring,
            id: Identifier,
            read: RemoteRead,
        ) -> Result<()> {
            let key = host_key(&read.remote);
            self.active += 1;

            if let Some(idx) = self.pool.get_mut(&key).and_then(|v| v.pop()) {
                // Reuse a pooled keep-alive connection: bind and pump straight to
                // sending (handshake already done).
                bind_request(self.conns[idx].as_mut().unwrap(), id, &read);
                self.pump(ring, idx)
            } else {
                let conn = self.new_conn(key, &read, id)?;
                let idx = self.alloc_slot(conn);
                self.start_connect(ring, idx)
            }
        }

        /// Advance the connection identified by a tagged `user_data` after its SQE
        /// completed with `size` bytes.
        pub fn on_cqe(&mut self, ring: &mut IoUring, user_data: u64, size: usize) -> Result<()> {
            let idx = (user_data & !HTTP_TAG) as usize;

            let finished = {
                let conn = self.conns[idx].as_mut().unwrap();
                match conn.state {
                    State::Connecting => false, // connected; pump below starts the exchange
                    State::Sending => {
                        conn.out_pos += size;
                        false
                    }
                    State::Receiving => {
                        if size == 0 {
                            return Err(Error::Io(std::io::Error::new(
                                std::io::ErrorKind::UnexpectedEof,
                                "connection closed mid-response",
                            )));
                        }
                        if conn.recv_in_dest {
                            // Plaintext body recv'd straight into the cache slot
                            // (zero-copy) — nothing to copy or parse, just advance
                            // the decoder and the write cursor.
                            conn.body
                                .as_mut()
                                .expect("body decoder set before a Direct recv")
                                .consumed(size as u64)?;
                            conn.body_written += size;
                        } else {
                            conn.consume_received(size)?;
                        }
                        conn.request_complete()
                    }
                }
            };

            if finished {
                let conn = self.conns[idx].as_mut().unwrap();
                let id = conn.id;
                let key = conn.host_key.clone();
                conn.clear_request();
                self.pool.entry(key).or_default().push(idx);
                self.active -= 1;
                self.completed.push(id);
                return Ok(());
            }

            self.pump(ring, idx)
        }

        fn new_conn(&self, key: String, read: &RemoteRead, id: Identifier) -> Result<Conn> {
            let addr = read.remote.addr();
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

            let transport = if read.remote.is_https() {
                let server_name =
                    rustls::pki_types::ServerName::try_from(read.remote.host().to_string())?;
                Transport::Tls(Box::new(ClientConnection::new(
                    self.client_config.clone(),
                    server_name,
                )?))
            } else {
                Transport::Plain
            };

            let request_bytes = proto::build_range_get(
                read.remote.host_header(),
                read.remote.request_target(),
                read.offset,
                read.len,
            );

            Ok(Conn {
                fd,
                transport,
                host_key: key,
                sockaddr,
                sockaddr_len,
                id,
                dest: read.dest,
                req_len: read.len,
                request_bytes,
                request_queued: false,
                out_buf: Vec::new(),
                out_pos: 0,
                // Allocated once; reused (never re-zeroed) for every recv.
                in_buf: vec![0u8; RECV_CHUNK],
                head_acc: Vec::new(),
                body: None,
                body_written: 0,
                recv_in_dest: false,
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
                    // Plaintext body phase: recv straight into the cache slot
                    // (zero-copy). Otherwise recv into the scratch `in_buf` — TLS
                    // ciphertext, or the response headers (for either scheme).
                    let plain_body =
                        matches!(conn.transport, Transport::Plain) && conn.body.is_some();
                    if plain_body {
                        // Identity body: the decoder's Direct plan gives the bytes
                        // still expected. `pump` only runs while the request is
                        // unfinished, so this is always > 0 here.
                        let remaining = match conn.body.as_ref().unwrap().read_plan() {
                            http1::ReadPlan::Direct { max } => max as usize,
                            _ => 0,
                        };
                        conn.recv_in_dest = true;
                        // SAFETY: pinned, currently-invalid slot region; remaining
                        // bytes are within the block (content_length <= req_len).
                        let a = unsafe { conn.dest.add(conn.body_written) } as usize;
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

    /// Bind a new request onto an already-connected (pooled) connection.
    fn bind_request(conn: &mut Conn, id: Identifier, read: &RemoteRead) {
        conn.id = id;
        conn.dest = read.dest;
        conn.req_len = read.len;
        conn.request_bytes = proto::build_range_get(
            read.remote.host_header(),
            read.remote.request_target(),
            read.offset,
            read.len,
        );
        conn.request_queued = false;
        conn.out_buf.clear();
        conn.out_pos = 0;
        // `in_buf` is reused as-is (not cleared); recv overwrites what it needs.
        conn.head_acc.clear();
        conn.body = None;
        conn.body_written = 0;
        conn.recv_in_dest = false;
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
