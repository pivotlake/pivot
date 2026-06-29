use crate::Identifier;
use crate::io::backend::IOBackend;
use crate::io::cached_http::CachedHttpEngine;
use crate::io::disk_cache::DiskCache;
use crate::io::{Completion, DataFlowRequest, FailedRead, FsRequest, HttpRequest, RemoteReadSplit};
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

/// Number of submission-queue entries in each worker's io_uring. The completion
/// queue is twice this (the io_uring default), which bounds how many reads may be
/// outstanding before completions overflow it - see callers that batch submissions.
pub const RING_SIZE: u32 = 64;

/// Bridges dataflow operators and the I/O backend, holding state on outstanding
/// requests and returning responses together with the original requested context.
///
/// A single per-core io_uring serves **both** disk reads and HTTP(S) range reads
/// (the standard io_uring pattern): file reads are one SQE→one CQE, while HTTP
/// reads are driven by the ring-less `HttpEngine`, which submits its socket SQEs
/// onto this same ring. Completions are disambiguated by the `HTTP_TAG` bit in
/// `user_data`.
///
/// Held one per worker.
pub struct IORequester {
    backend: IOBackend,
    /// In-flight local-file reads, keyed by their backend `user_data` id.
    pending_io_requests: HashMap<Identifier, DataFlowRequest<FsRequest>>,
    /// Allocates backend disk-op ids - shared by fs reads here and the engine's
    /// cache-file reads/write-backs, so a completion routes by which map holds it.
    next_id: Identifier,
    /// Remote reads, optionally served from the on-disk cache. Ring-less like the
    /// underlying `HttpEngine`: it borrows `backend` (and `next_id`) to submit.
    http: CachedHttpEngine,
}

impl Default for IORequester {
    fn default() -> Self {
        Self::new(None)
    }
}

impl IORequester {
    /// Build a requester sharing `disk_cache` (or `None` to disable disk caching).
    pub fn new(disk_cache: Option<Arc<DiskCache>>) -> Self {
        Self {
            backend: IOBackend::new(RING_SIZE).expect("Unable to create backend"),
            pending_io_requests: Default::default(),
            next_id: 0,
            http: CachedHttpEngine::with_default_config(disk_cache)
                .expect("Unable to create http engine"),
        }
    }

    /// Build a requester with a specific rustls client config for the HTTP engine
    /// (tests inject one trusting a loopback test server) and an optional cache.
    pub fn with_config(
        http_config: Arc<rustls::ClientConfig>,
        disk_cache: Option<Arc<DiskCache>>,
    ) -> Self {
        Self {
            backend: IOBackend::new(RING_SIZE).expect("Unable to create backend"),
            pending_io_requests: Default::default(),
            next_id: 0,
            http: CachedHttpEngine::new(http_config, disk_cache)
                .expect("Unable to create http engine"),
        }
    }

    /// Build a requester with a specific HTTP client config and no disk cache.
    pub fn with_http_config(http_config: Arc<rustls::ClientConfig>) -> Self {
        Self::with_config(http_config, None)
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

    /// Submit a read for a remote region, served from the on-disk cache where
    /// possible (only the missing ranges hit the network). Delegates to the
    /// [`CachedHttpEngine`], lending it the shared backend and disk-id counter.
    pub fn request_http(
        &mut self,
        request: DataFlowRequest<HttpRequest>,
    ) -> Result<RemoteReadSplit> {
        self.http
            .request(&mut self.backend, &mut self.next_id, request)
    }

    /// Returns `true` if any read (disk or HTTP) has not yet completed.
    pub fn has_pending(&self) -> bool {
        self.has_file_pending() || self.has_http_pending()
    }

    /// Returns `true` if any disk read is in flight - operator reads here, plus
    /// the engine's cache-file reads / write-backs (all on the shared backend).
    pub fn has_file_pending(&self) -> bool {
        !self.pending_io_requests.is_empty() || self.http.has_disk_pending()
    }

    /// Returns `true` if any HTTP read is in flight.
    pub fn has_http_pending(&self) -> bool {
        self.http.has_network_pending()
    }

    /// Number of HTTP reads issued but not yet completed — the current read-ahead
    /// depth a worker uses to decide whether to submit more.
    pub fn http_in_flight(&self) -> usize {
        self.http.network_in_flight()
    }

    /// Drain finished reads, one per-read result each. The outer `Result` is for
    /// genuine ring-machinery failures; each inner result is `Ok` for a read
    /// whose bytes landed (block committed, request yielded as a [`Completion`]
    /// with the transport kind preserved) or `Err` for one that failed terminally
    /// (a [`FailedRead`] carrying the issuing dataflow and the error). A single
    /// failed read therefore never aborts the drain or tears down the worker —
    /// the worker cancels just the owning dataflow.
    ///
    /// The single per-core ring carries everything: this requester's fs reads,
    /// the [`CachedHttpEngine`]'s cache-file reads / write-backs, and its HTTP
    /// sockets. We drain it once and route each completion to its owner.
    pub fn completions(&mut self) -> Result<Vec<std::result::Result<Completion, FailedRead>>> {
        let raw = self.backend.completions()?;

        // HTTP socket CQEs drive the engine first: they may submit follow-up SQEs
        // and populate its completed/failed lists. On non-Linux the ring is
        // disk-only (the engine runs synchronously), so there are none here.
        #[cfg(target_os = "linux")]
        for &(result, ud) in &raw {
            if (ud as u64) & HTTP_TAG != 0 {
                self.http
                    .on_socket_completion(&mut self.backend, ud as u64, result)?;
            }
        }

        let mut out = Vec::new();

        // Backend disk CQEs: this requester's fs reads, or the engine's cache-file
        // reads / write-backs. Disjoint id spaces, so the id's owning map sorts
        // them out. A negative result is a failure surfaced to just that dataflow.
        for &(result, ud) in &raw {
            if !disk_completion(ud) {
                continue; // HTTP socket op, already routed above
            }
            if let Some(request) = self.pending_io_requests.remove(&ud) {
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
            } else {
                // Not one of ours → a cache-file read or write-back.
                self.http.complete_disk(ud, result, &mut out);
            }
        }

        // HTTP reads the engine finished this pass (and any write-backs they queue).
        self.http
            .drain(&mut self.backend, &mut self.next_id, &mut out)?;

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
    use std::sync::{Arc, Mutex};
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

    /// Shared log of the `[start, end]` ranges a recording server was asked for.
    type RangeLog = Arc<Mutex<Vec<(usize, usize)>>>;

    /// Spawn a server that records every range it's asked for (so a test can
    /// assert exactly which bytes went to the network) and serves each on one
    /// keep-alive connection. Returns the port and the shared request log.
    fn spawn_recording_server() -> (u16, RangeLog) {
        let server_config = server_tls_config();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let log = Arc::new(Mutex::new(Vec::new()));
        let log_for_thread = log.clone();

        thread::spawn(move || {
            let (tcp, _) = listener.accept().unwrap();
            let conn = rustls::ServerConnection::new(server_config).unwrap();
            let mut tls = rustls::StreamOwned::new(conn, tcp);
            loop {
                let head = read_head(&mut tls);
                if head.is_empty() {
                    break; // client closed the connection
                }
                let (start, end) = parse_range(&head);
                log_for_thread.lock().unwrap().push((start, end));
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
        });

        (port, log)
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
        fetch_split(requester, loc, offset, len);
    }

    /// Fetch `[offset, offset + len)` of `loc` and block until it completes,
    /// returning the per-tier read split summed over the read's blocks (disk-cache
    /// hits versus network fetches).
    fn fetch_split(
        requester: &mut IORequester,
        loc: &FileLocation,
        offset: usize,
        len: usize,
    ) -> RemoteReadSplit {
        let FileLocation::Remote(remote) = loc else {
            panic!("test fetches over http")
        };
        let lookups = memory_ctx().compressed_cache().get(loc, offset, len);
        let mut split = RemoteReadSplit::default();
        let mut submitted = 0;
        for lookup in &lookups {
            for block in lookup.missing() {
                let req = HttpRequest {
                    remote: remote.clone(),
                    block: block.clone(),
                };
                let piece = requester
                    .request_http(DataFlowRequest::new(0, 0, req))
                    .unwrap();
                split.disk_cache_requests += piece.disk_cache_requests;
                split.disk_cache_bytes += piece.disk_cache_bytes;
                split.http_requests += piece.http_requests;
                split.http_bytes += piece.http_bytes;
                submitted += 1;
            }
        }
        assert!(submitted > 0, "expected blocks to read");

        let mut completed = 0;
        while completed < submitted {
            if requester.has_pending() {
                requester.wait().unwrap();
            }
            completed += requester.completions().unwrap().len();
        }
        drop(lookups);
        split
    }

    /// Assert `[offset, offset + len)` of `loc` is now a full cache hit holding
    /// the expected pattern.
    fn assert_cached(loc: &FileLocation, offset: usize, len: usize) {
        let hit = memory_ctx().compressed_cache().get(loc, offset, len);
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
        let remote = Arc::new(RemoteFile::open(url, None, 1 << 20).unwrap());
        let loc = FileLocation::Remote(remote);
        memory_ctx().compressed_cache().open_entry(loc.clone());

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
        let remote = Arc::new(RemoteFile::open(url, None, 1 << 20).unwrap());
        let loc = FileLocation::Remote(remote);
        memory_ctx().compressed_cache().open_entry(loc.clone());

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

    /// Drive every pending read *and* write-back to completion - `fetch` only
    /// waits for the reads, but the disk-cache test must let the asynchronous
    /// write-backs land before it drops the in-memory cache.
    fn settle(requester: &mut IORequester) {
        while requester.has_pending() {
            requester.wait().unwrap();
            for c in requester.completions().unwrap() {
                if let Err(e) = c {
                    panic!("a read failed: {}", e.error);
                }
            }
        }
    }

    /// A disk cache over `dir`. A second cache over the same dir models a restart.
    fn cache_in(dir: &tempfile::TempDir) -> Arc<DiskCache> {
        Arc::new(DiskCache::open(dir.path().to_path_buf(), 1 << 30, 1 << 20).unwrap())
    }

    /// Register a remote object served by the loopback server on `port`.
    fn remote_loc(port: u16) -> FileLocation {
        let url = Url::parse(&format!("https://127.0.0.1:{port}/obj")).unwrap();
        let loc = FileLocation::Remote(Arc::new(RemoteFile::open(url, None, 1 << 20).unwrap()));
        memory_ctx().compressed_cache().open_entry(loc.clone());
        loc
    }

    fn requester_with(cache: Arc<DiskCache>) -> IORequester {
        IORequester::with_config(client_config(), Some(cache))
    }

    /// A read that missed memory but whose bytes are already on disk is served
    /// from the cache file. The server allows exactly one request, so the second
    /// read would fail outright if it touched the network.
    #[test]
    fn a_read_already_on_disk_skips_the_network() {
        init_test_free_pool(16);
        let dir = tempfile::tempdir().unwrap();
        let mut requester = requester_with(cache_in(&dir));
        let loc = remote_loc(spawn_server(1));
        fetch(&mut requester, &loc, 0, 4096);
        settle(&mut requester);
        memory_ctx().compressed_cache().clear();

        fetch(&mut requester, &loc, 0, 4096);
        settle(&mut requester);

        assert_cached(&loc, 0, 4096);
    }

    /// The read split bills a cold read to the network and the same range, once
    /// resident on disk, to the disk cache with nothing over the network.
    #[test]
    fn the_read_split_attributes_each_tier() {
        const SB: usize = 4096;
        init_test_free_pool(16);
        let dir = tempfile::tempdir().unwrap();
        let mut requester = requester_with(cache_in(&dir));
        let loc = remote_loc(spawn_server(1));

        let cold = fetch_split(&mut requester, &loc, 0, SB);
        settle(&mut requester);
        memory_ctx().compressed_cache().clear();
        let warm = fetch_split(&mut requester, &loc, 0, SB);
        settle(&mut requester);

        assert_eq!((cold.http_requests, cold.http_bytes), (1, SB as u64));
        assert_eq!(cold.disk_cache_requests, 0);
        assert_eq!(
            (warm.disk_cache_requests, warm.disk_cache_bytes),
            (1, SB as u64)
        );
        assert_eq!(warm.http_requests, 0);
    }

    /// A read split across cached blocks and a hole counts each piece in its own
    /// tier: the two resident blocks as disk-cache reads, the hole as one network
    /// fetch, with bytes to match. Neither tier's work is hidden behind the other.
    #[test]
    fn a_split_read_counts_each_piece_in_its_tier() {
        const SB: usize = 4096;
        init_test_free_pool(16);
        let dir = tempfile::tempdir().unwrap();
        let mut requester = requester_with(cache_in(&dir));
        let (port, _requested) = spawn_recording_server();
        let loc = remote_loc(port);
        fetch(&mut requester, &loc, 0, SB); // prime block 0
        settle(&mut requester);
        fetch(&mut requester, &loc, 2 * SB, SB); // prime block 2, leaving 1 a hole
        settle(&mut requester);
        memory_ctx().compressed_cache().clear();

        let split = fetch_split(&mut requester, &loc, 0, 3 * SB);
        settle(&mut requester);

        assert_eq!(
            (split.disk_cache_requests, split.disk_cache_bytes),
            (2, 2 * SB as u64)
        );
        assert_eq!((split.http_requests, split.http_bytes), (1, SB as u64));
    }

    /// A partially-cached run fetches only the missing blocks: block 0 is primed
    /// onto disk, so reading blocks 0–1 asks the network for block 1 alone.
    #[test]
    fn a_partial_disk_hit_fetches_only_the_missing_blocks() {
        const SB: usize = 4096;
        init_test_free_pool(16);
        let dir = tempfile::tempdir().unwrap();
        let mut requester = requester_with(cache_in(&dir));
        let (port, requested) = spawn_recording_server();
        let loc = remote_loc(port);
        fetch(&mut requester, &loc, 0, SB);
        settle(&mut requester);
        memory_ctx().compressed_cache().clear();

        fetch(&mut requester, &loc, 0, 2 * SB);
        settle(&mut requester);

        assert_cached(&loc, 0, 2 * SB);
        assert_eq!(
            *requested.lock().unwrap(),
            vec![(0, SB - 1), (SB, 2 * SB - 1)]
        );
    }

    /// A run with cached blocks on both sides of a hole fetches only the interior
    /// block and reassembles all three correctly - exercising a multi-piece group
    /// (two cache reads + one HTTP fetch) and disk↔network byte stitching.
    #[test]
    fn a_split_read_fetches_only_the_interior_hole() {
        const SB: usize = 4096;
        init_test_free_pool(16);
        let dir = tempfile::tempdir().unwrap();
        let mut requester = requester_with(cache_in(&dir));
        let (port, requested) = spawn_recording_server();
        let loc = remote_loc(port);
        fetch(&mut requester, &loc, 0, SB); // prime block 0
        settle(&mut requester);
        fetch(&mut requester, &loc, 2 * SB, SB); // prime block 2, leaving 1 a hole
        settle(&mut requester);
        memory_ctx().compressed_cache().clear();

        fetch(&mut requester, &loc, 0, 3 * SB);
        settle(&mut requester);

        assert_cached(&loc, 0, 3 * SB);
        assert_eq!(
            *requested.lock().unwrap(),
            vec![(0, SB - 1), (2 * SB, 3 * SB - 1), (SB, 2 * SB - 1)]
        );
    }

    /// The cache survives a restart: a fresh cache over the same directory
    /// reseeds what's on disk (via `SEEK_HOLE`, Linux-only), so a read after the
    /// restart skips the network the same way an in-process hit would.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_disk_cache_survives_a_restart() {
        init_test_free_pool(16);
        let dir = tempfile::tempdir().unwrap();
        let loc = remote_loc(spawn_server(1));
        let mut before = requester_with(cache_in(&dir));
        fetch(&mut before, &loc, 0, 4096);
        settle(&mut before);
        memory_ctx().compressed_cache().clear();

        let mut after = requester_with(cache_in(&dir));
        fetch(&mut after, &loc, 0, 4096);
        settle(&mut after);

        assert_cached(&loc, 0, 4096);
    }
}
