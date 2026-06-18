use crate::Identifier;
use crate::io::backend::IOBackend;
use crate::io::http::{HttpEngine, RemoteRead, default_client_config};
use crate::io::{Completion, DataFlowRequest, FailedRead, FsRequest, HttpRequest};
use std::collections::HashMap;
use std::os::fd::AsRawFd;
use std::sync::Arc;
use thiserror::Error;

#[cfg(target_os = "linux")]
use crate::io::http::HTTP_TAG;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Backend(#[from] super::backend::Error),
    #[error("{0}")]
    IO(#[from] std::io::Error),
    #[error("{0}")]
    Http(#[from] crate::io::http::Error),
}

type Result<T, E = Error> = std::result::Result<T, E>;

const RING_SIZE: u32 = 64;

/// Bridges dataflow operators and the I/O backend, holding state on outstanding
/// requests and returning responses together with the original requested context.
///
/// A single per-core io_uring serves **both** disk reads and HTTP(S) range reads
/// (the standard io_uring pattern): file reads are one SQE→one CQE, while HTTP
/// reads are driven by the ring-less [`HttpEngine`], which submits its socket SQEs
/// onto this same ring. Completions are disambiguated by the `HTTP_TAG` bit in
/// `user_data`.
///
/// Held one per worker.
pub struct IORequester {
    backend: IOBackend,
    /// In-flight disk reads, keyed by their `user_data` id.
    pending_io_requests: HashMap<Identifier, DataFlowRequest<FsRequest>>,
    next_id: Identifier,

    /// HTTP transport (shares `backend`'s ring on Linux).
    http: HttpEngine,
    /// In-flight HTTP reads, keyed by their engine request id.
    http_pending: HashMap<Identifier, DataFlowRequest<HttpRequest>>,
    next_http_id: Identifier,
}

impl Default for IORequester {
    fn default() -> Self {
        Self::new()
    }
}

impl IORequester {
    pub fn new() -> Self {
        Self::with_http_config(default_client_config())
    }

    /// Build a requester with a specific rustls client config for the HTTP engine
    /// (tests inject one trusting a loopback test server).
    pub fn with_http_config(http_config: Arc<rustls::ClientConfig>) -> Self {
        Self {
            backend: IOBackend::new(RING_SIZE).expect("Unable to create backend"),
            pending_io_requests: Default::default(),
            next_id: 0,
            http: HttpEngine::new(http_config).expect("Unable to create http engine"),
            http_pending: Default::default(),
            next_http_id: 0,
        }
    }

    /// Submits the block's read straight into its (pinned) cache slot and
    /// flushes immediately. No intermediate buffer: the slot region is the read
    /// target. The block holds an `Arc` on the slot pin, so it stays alive for
    /// the read even if the issuing query is cancelled meanwhile.
    pub fn request(&mut self, request: DataFlowRequest<FsRequest>) -> Result<()> {
        self.backend.submit_read(
            request.request.file.as_raw_fd(),
            request.request.block.file_offset() as u64,
            request.request.block.dest(),
            request.request.block.len(),
            self.next_id,
        )?;
        self.pending_io_requests.insert(self.next_id, request);
        self.next_id += 1;
        self.backend.submit()?;

        Ok(())
    }

    /// Submit an HTTP(S) range read for a remote region. The engine drives
    /// connect/handshake/request/response on the shared ring (Linux) or
    /// synchronously (other platforms); the body lands in the block's pinned
    /// cache slot just like a disk read.
    pub fn request_http(&mut self, request: DataFlowRequest<HttpRequest>) -> Result<()> {
        let block = &request.request.block;
        let read = RemoteRead {
            remote: request.request.remote.clone(),
            offset: block.file_offset() as u64,
            len: block.len(),
            dest: block.dest(),
        };
        let id = self.next_http_id;
        #[cfg(target_os = "linux")]
        self.http.start(&mut self.backend.ring, id, read)?;
        #[cfg(not(target_os = "linux"))]
        self.http.start(id, read)?;
        self.http_pending.insert(id, request);
        self.next_http_id += 1;
        Ok(())
    }

    /// Returns `true` if any read (disk or HTTP) has not yet completed.
    pub fn has_pending(&self) -> bool {
        self.has_file_pending() || self.has_http_pending()
    }

    /// Returns `true` if any disk read is in flight.
    pub fn has_file_pending(&self) -> bool {
        !self.pending_io_requests.is_empty()
    }

    /// Returns `true` if any HTTP read is in flight.
    pub fn has_http_pending(&self) -> bool {
        self.http.has_active()
    }

    /// Number of HTTP reads issued but not yet completed — the current read-ahead
    /// depth a worker uses to decide whether to submit more.
    pub fn http_in_flight(&self) -> usize {
        self.http_pending.len()
    }

    /// Drain finished reads, one per-read result each. The outer `Result` is for
    /// genuine ring-machinery failures; each inner result is `Ok` for a read
    /// whose bytes landed (block committed, request yielded as a [`Completion`]
    /// with the transport kind preserved) or `Err` for one that failed terminally
    /// (a [`FailedRead`] carrying the issuing dataflow and the error). A single
    /// failed read therefore never aborts the drain or tears down the worker —
    /// the worker cancels just the owning dataflow.
    ///
    /// Disk completions come straight off the ring; HTTP completions are routed
    /// through the [`HttpEngine`] (which may submit follow-up SQEs, including
    /// transparent reconnect-and-retry) before the finished ones are harvested.
    pub fn completions(&mut self) -> Result<Vec<std::result::Result<Completion, FailedRead>>> {
        let raw = self.backend.completions()?;

        // HTTP socket CQEs drive the engine first: they may submit follow-up
        // SQEs and populate the engine's completed/failed lists. A transient
        // transport error is retried inside the engine; a terminal one lands in
        // `take_failed` below. On non-Linux the ring is disk-only, so none here.
        #[cfg(target_os = "linux")]
        for &(result, ud) in &raw {
            if (ud as u64) & HTTP_TAG != 0 {
                self.http
                    .on_cqe(&mut self.backend.ring, ud as u64, result)?;
            }
        }

        let mut out = Vec::new();

        // Disk reads: one CQE each. A negative result is a failed read — surface
        // it as an `Err` for just that dataflow rather than aborting the drain.
        for &(result, ud) in &raw {
            if !disk_completion(ud) {
                continue;
            }
            let request = self.pending_io_requests.remove(&ud).unwrap();
            if result < 0 {
                out.push(Err(FailedRead {
                    data_flow_id: request.data_flow_id,
                    operator_idx: request.operator_idx,
                    error: std::io::Error::from_raw_os_error(-result).into(),
                }));
            } else {
                request.request.block.commit();
                out.push(Ok(Completion::Fs(request)));
            }
        }

        // HTTP reads whose body fully landed this pass.
        for id in self.http.take_completed() {
            let request = self.http_pending.remove(&id).unwrap();
            request.request.block.commit();
            out.push(Ok(Completion::Http(request)));
        }
        // HTTP reads that failed terminally (retries exhausted / non-retryable).
        // The block is left uncommitted.
        for (id, error) in self.http.take_failed() {
            let request = self.http_pending.remove(&id).unwrap();
            out.push(Err(FailedRead {
                data_flow_id: request.data_flow_id,
                operator_idx: request.operator_idx,
                error: error.into(),
            }));
        }

        Ok(out)
    }

    /// Blocks until at least one pending read (disk or HTTP) makes progress.
    pub fn wait(&mut self) -> Result<()> {
        self.backend.submit_and_wait(1)?;
        Ok(())
    }
}

/// Whether a CQE's `user_data` is a disk read rather than an HTTP socket op. The
/// HTTP engine sets the `HTTP_TAG` high bit on its SQEs (Linux only); disk ids
/// are small counters that never reach it.
#[cfg(target_os = "linux")]
fn disk_completion(ud: Identifier) -> bool {
    (ud as u64) & HTTP_TAG == 0
}
#[cfg(not(target_os = "linux"))]
fn disk_completion(_ud: Identifier) -> bool {
    true
}

#[cfg(test)]
mod tests {
    //! End-to-end HTTPS plumbing test: a loopback TLS server serves `206` range
    //! responses, and we drive a remote read through [`IORequester`] into a real
    //! cache slot, asserting the bytes land and `commit` makes the region a hit.
    //!
    //! On macOS this exercises the blocking engine; on Linux the same flow runs
    //! over the shared io_uring. It also covers per-host keep-alive reuse by
    //! issuing a second read for a different region of the same object.

    use super::*;
    use crate::io::{FileLocation, RemoteFile};
    use crate::memory::{init_test_free_pool, memory_ctx};
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::thread;
    use url::Url;

    /// Body byte for absolute file offset `off`, so a served range is predictable.
    fn pattern(off: usize) -> u8 {
        (off % 251) as u8
    }

    /// A client cert verifier that accepts anything — the test server uses a
    /// self-signed cert and we only care about the transport, not validation.
    #[derive(Debug)]
    struct NoVerify;

    impl rustls::client::danger::ServerCertVerifier for NoVerify {
        fn verify_server_cert(
            &self,
            _end_entity: &rustls::pki_types::CertificateDer<'_>,
            _intermediates: &[rustls::pki_types::CertificateDer<'_>],
            _server_name: &rustls::pki_types::ServerName<'_>,
            _ocsp: &[u8],
            _now: rustls::pki_types::UnixTime,
        ) -> std::result::Result<rustls::client::danger::ServerCertVerified, rustls::Error>
        {
            Ok(rustls::client::danger::ServerCertVerified::assertion())
        }

        fn verify_tls12_signature(
            &self,
            _message: &[u8],
            _cert: &rustls::pki_types::CertificateDer<'_>,
            _dss: &rustls::DigitallySignedStruct,
        ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error>
        {
            Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
        }

        fn verify_tls13_signature(
            &self,
            _message: &[u8],
            _cert: &rustls::pki_types::CertificateDer<'_>,
            _dss: &rustls::DigitallySignedStruct,
        ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error>
        {
            Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
        }

        fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
            rustls::crypto::ring::default_provider()
                .signature_verification_algorithms
                .supported_schemes()
        }
    }

    fn client_config() -> Arc<rustls::ClientConfig> {
        let config = rustls::ClientConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .unwrap()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(NoVerify))
        .with_no_client_auth();
        Arc::new(config)
    }

    /// A self-signed TLS server config for the loopback test server.
    fn server_tls_config() -> Arc<rustls::ServerConfig> {
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
        let cert_der = cert.cert.der().clone();
        let key_der = rustls::pki_types::PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der());
        let config = rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_single_cert(
            vec![cert_der],
            rustls::pki_types::PrivateKeyDer::Pkcs8(key_der),
        )
        .unwrap();
        Arc::new(config)
    }

    /// Read one range request and write back its `206` response body.
    fn serve_one_range<S: Read + Write>(tls: &mut S) {
        let head = read_head(tls);
        let (start, end) = parse_range(&head);
        let len = end - start + 1;
        let body: Vec<u8> = (0..len).map(|i| pattern(start + i)).collect();
        let resp = format!(
            "HTTP/1.1 206 Partial Content\r\n\
             Content-Length: {len}\r\n\
             Content-Range: bytes {start}-{end}/1000000\r\n\
             Connection: keep-alive\r\n\r\n"
        );
        tls.write_all(resp.as_bytes()).unwrap();
        tls.write_all(&body).unwrap();
        tls.flush().unwrap();
    }

    /// Spawn a loopback HTTPS server that serves `num_requests` range GETs on a
    /// single keep-alive connection, returning the bound port.
    fn spawn_server(num_requests: usize) -> u16 {
        let server_config = server_tls_config();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        thread::spawn(move || {
            let (tcp, _) = listener.accept().unwrap();
            let conn = rustls::ServerConnection::new(server_config).unwrap();
            let mut tls = rustls::StreamOwned::new(conn, tcp);
            for _ in 0..num_requests {
                serve_one_range(&mut tls);
            }
        });

        port
    }

    /// Spawn a server that serves one range GET, drops the connection (as a
    /// keep-alive idle timeout would), then accepts a second connection and
    /// serves one more. The pooled connection the client cached from the first
    /// read is therefore dead by the time it's reused.
    fn spawn_stale_pool_server() -> u16 {
        let server_config = server_tls_config();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        thread::spawn(move || {
            for _ in 0..2 {
                let (tcp, _) = listener.accept().unwrap();
                let conn = rustls::ServerConnection::new(server_config.clone()).unwrap();
                let mut tls = rustls::StreamOwned::new(conn, tcp);
                serve_one_range(&mut tls);
                // `tls` drops here: close_notify + FIN retire the connection, so
                // the client's pooled copy is stale on the next read.
            }
        });

        port
    }

    /// Read a request head (up to and including the blank-line terminator).
    fn read_head<R: Read>(r: &mut R) -> Vec<u8> {
        let mut buf = Vec::new();
        let mut byte = [0u8; 1];
        loop {
            let n = r.read(&mut byte).unwrap();
            if n == 0 {
                break;
            }
            buf.push(byte[0]);
            if buf.ends_with(b"\r\n\r\n") {
                break;
            }
        }
        buf
    }

    /// Parse the inclusive `[start, end]` from a `Range: bytes=START-END` header.
    fn parse_range(head: &[u8]) -> (usize, usize) {
        let text = String::from_utf8_lossy(head);
        let line = text
            .lines()
            .find(|l| l.to_ascii_lowercase().starts_with("range:"))
            .expect("request has a Range header");
        let spec = line.split('=').nth(1).unwrap().trim();
        let mut parts = spec.split('-');
        let start = parts.next().unwrap().parse().unwrap();
        let end = parts.next().unwrap().parse().unwrap();
        (start, end)
    }

    /// Fetch `[offset, offset + len)` of `loc` via the requester and block until
    /// it completes.
    fn fetch(requester: &mut IORequester, loc: &FileLocation, offset: usize, len: usize) {
        let FileLocation::Remote(remote) = loc else {
            panic!("test fetches over http")
        };
        let lookups = memory_ctx().file_cache().get(loc, offset, len);
        let mut submitted = 0;
        for lookup in &lookups {
            for block in lookup.missing() {
                let req = HttpRequest {
                    remote: remote.clone(),
                    block: block.clone(),
                };
                requester
                    .request_http(DataFlowRequest::new(0, 0, req))
                    .unwrap();
                submitted += 1;
            }
        }
        assert!(submitted > 0, "expected a cache miss to drive over http");

        // Drive completions until everything submitted has landed.
        let mut completed = 0;
        while completed < submitted {
            if requester.has_pending() {
                requester.wait().unwrap();
            }
            completed += requester.completions().unwrap().len();
        }
        // Keep the lookups' pins alive until after the reads committed.
        drop(lookups);
    }

    /// Assert `[offset, offset + len)` of `loc` is now a full cache hit holding
    /// the expected pattern.
    fn assert_cached(loc: &FileLocation, offset: usize, len: usize) {
        let hit = memory_ctx().file_cache().get(loc, offset, len);
        assert_eq!(hit.len(), 1);
        assert!(hit[0].missing().is_empty(), "expected a cache hit");
        let bytes = hit.into_iter().next().unwrap().into_data();
        assert_eq!(bytes.len(), len);
        for (i, &b) in bytes.iter().enumerate() {
            assert_eq!(b, pattern(offset + i), "byte {i}");
        }
    }

    #[test]
    fn https_range_read_lands_in_cache_and_reuses_connection() {
        init_test_free_pool(16);
        let port = spawn_server(2);

        let url = Url::parse(&format!("https://127.0.0.1:{port}/obj")).unwrap();
        let remote = Arc::new(RemoteFile::open(url).unwrap());
        let loc = FileLocation::Remote(remote);
        memory_ctx().file_cache().open_entry(loc.clone());

        let mut requester = IORequester::with_http_config(client_config());

        // First read: connect + TLS handshake + range GET into the cache slot.
        fetch(&mut requester, &loc, 0, 4096);
        assert_cached(&loc, 0, 4096);

        // Second read of a different region: must reuse the pooled keep-alive
        // connection (the server only accepts one TCP connection).
        fetch(&mut requester, &loc, 4096, 2 * 4096);
        assert_cached(&loc, 4096, 2 * 4096);
    }

    #[test]
    fn stale_pooled_connection_is_retried_on_a_fresh_one() {
        init_test_free_pool(16);
        let port = spawn_stale_pool_server();

        let url = Url::parse(&format!("https://127.0.0.1:{port}/obj")).unwrap();
        let remote = Arc::new(RemoteFile::open(url).unwrap());
        let loc = FileLocation::Remote(remote);
        memory_ctx().file_cache().open_entry(loc.clone());

        let mut requester = IORequester::with_http_config(client_config());

        // First read primes the keep-alive pool with connection #1.
        fetch(&mut requester, &loc, 0, 4096);
        assert_cached(&loc, 0, 4096);

        // The server has since closed connection #1. Reusing the now-stale
        // pooled connection fails mid-exchange; the engine must reconnect and
        // re-issue the (idempotent) range read rather than surfacing an error.
        fetch(&mut requester, &loc, 4096, 4096);
        assert_cached(&loc, 4096, 4096);
    }
}
