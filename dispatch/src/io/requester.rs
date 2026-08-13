use crate::Identifier;
use crate::io::backend::IOBackend;
use crate::io::cached_http::CachedHttpEngine;
use crate::io::disk_cache::DiskCache;
use crate::io::slot_events::{self, SlotIoEvent, SlotIoOutcome};
use crate::io::{
    Completion, DataFlowRequest, FailedRead, FsReadRequest, FsRequest, HttpGetRequest, HttpRequest,
    OpenFile, RemoteReadTime, RemoteSplit,
};
use crate::memory::compressed_cache::{MAX_JOINABLE_WORKERS, WaiterRegistration};
use crate::memory::memory_ctx;
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
    /// In-flight operator-owned local-file operations, keyed by backend id,
    /// each with the bytes already completed by earlier submissions (nonzero
    /// only for a write whose prior completion came up short and was
    /// resubmitted).
    pending_io_requests: HashMap<Identifier, (DataFlowRequest<FsRequest>, usize)>,
    /// Allocates backend disk-op ids - shared by fs reads here and the engine's
    /// cache-file reads/write-backs, so a completion routes by which map holds it.
    next_id: Identifier,
    /// Remote reads, optionally served from the on-disk cache. Ring-less like the
    /// underlying `HttpEngine`: it borrows `backend` (and `next_id`) to submit.
    http: CachedHttpEngine,
    /// Reads *joined* onto another worker's in-flight read of the same extent,
    /// parked by that extent's ring slot: no IO of this worker's own. A
    /// [`SlotIoEvent`] for the slot resolves each parked read - served from the
    /// now-resident bytes, kept parked, or (after an abandonment) performed by
    /// this worker after all. This worker's bit in the slot's waiter mask is set
    /// exactly while the slot has an entry in either map.
    waiting_fs: HashMap<usize, Vec<DataFlowRequest<FsReadRequest>>>,
    /// The remote counterpart of `waiting_fs`.
    waiting_http: HashMap<usize, Vec<DataFlowRequest<HttpGetRequest>>>,
    /// Joined reads that resolved (their bytes are resident) and now await the
    /// next [`completions`](Self::completions) drain, which yields them exactly
    /// like ring completions.
    joined_ready: Vec<std::result::Result<Completion, FailedRead>>,
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
            waiting_fs: Default::default(),
            waiting_http: Default::default(),
            joined_ready: Default::default(),
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
            waiting_fs: Default::default(),
            waiting_http: Default::default(),
            joined_ready: Default::default(),
        }
    }

    /// Build a requester with a specific HTTP client config and no disk cache.
    pub fn with_http_config(http_config: Arc<rustls::ClientConfig>) -> Self {
        Self::with_config(http_config, None)
    }

    /// Submit an operator-owned filesystem operation through the worker's
    /// shared I/O backend. Request-owned storage remains alive until completion.
    ///
    /// A read of an extent another worker's read is already filling performs no
    /// IO here: it joins that read (see [`try_join_fs_read`](Self::try_join_fs_read))
    /// and completes off its slot event instead.
    pub fn request(&mut self, request: DataFlowRequest<FsRequest>) -> Result<()> {
        let request = match self.try_join_fs_read(request) {
            Some(request) => request,
            None => return Ok(()),
        };
        match &request.request {
            FsRequest::Read(read) => self.backend.submit_read(
                read.file.as_raw_fd(),
                read.block.file_offset() as u64,
                read.block.dest(),
                read.block.len(),
                self.next_id,
            )?,
            // One op writes one run of the payload, and the completion path
            // below submits the next. A run boundary looks exactly like the
            // short write the kernel can hand back anyway.
            FsRequest::Write(write) => {
                let run = write.data.run_at(0);
                self.backend.submit_write(
                    write.file.as_raw_fd(),
                    0,
                    run.as_ptr(),
                    run.len(),
                    self.next_id,
                )?
            }
        }
        self.pending_io_requests.insert(self.next_id, (request, 0));
        self.next_id += 1;
        self.backend.submit()?;

        Ok(())
    }

    /// Submit a read for a remote region, served from the on-disk cache where
    /// possible (only the missing ranges hit the network). Delegates to the
    /// [`CachedHttpEngine`], lending it the shared backend and disk-id counter.
    ///
    /// A read of an extent another worker's read is already filling touches no
    /// transport at all: it joins that read and completes off its slot event,
    /// reported as an empty [`RemoteSplit`] since no tier served it here.
    pub fn request_http(&mut self, request: DataFlowRequest<HttpRequest>) -> Result<RemoteSplit> {
        let DataFlowRequest {
            data_flow_id,
            operator_idx,
            request,
            submitted_at,
        } = request;
        match request {
            HttpRequest::Get(request) => {
                let request = match self.try_join_http_get(DataFlowRequest {
                    data_flow_id,
                    operator_idx,
                    request,
                    submitted_at,
                }) {
                    Some(request) => request,
                    None => return Ok(RemoteSplit::default()),
                };
                self.http.get(&mut self.backend, &mut self.next_id, request)
            }
            HttpRequest::Upload(request) => {
                let bytes = request.data.len() as u64;
                self.http.upload(
                    &mut self.backend,
                    DataFlowRequest {
                        data_flow_id,
                        operator_idx,
                        request,
                        submitted_at,
                    },
                )?;
                Ok(RemoteSplit {
                    http_requests: 1,
                    http_bytes: bytes,
                    ..RemoteSplit::default()
                })
            }
        }
    }

    /// Try to ride another worker's in-flight read instead of submitting
    /// `request`: a filesystem read of a non-owner extent (someone else's read
    /// is responsible for those bytes, see [`MissingExtent`]'s fetch-ownership
    /// docs) registers this worker as a waiter and parks. Returns the request
    /// back when it must be submitted after all - it is the extent's owner,
    /// joining is disabled on this thread, it is a write, or the extent was
    /// abandoned and the read now falls to us.
    ///
    /// [`MissingExtent`]: crate::memory::compressed_cache::MissingExtent
    fn try_join_fs_read(
        &mut self,
        request: DataFlowRequest<FsRequest>,
    ) -> Option<DataFlowRequest<FsRequest>> {
        let Some(worker) = joinable_worker() else {
            return Some(request);
        };
        if !matches!(&request.request, FsRequest::Read(read) if !read.block.is_owner()) {
            return Some(request);
        }
        let DataFlowRequest {
            data_flow_id,
            operator_idx,
            request: FsRequest::Read(read),
            submitted_at,
        } = request
        else {
            unreachable!("matched a non-owner read above");
        };
        let read = DataFlowRequest {
            data_flow_id,
            operator_idx,
            request: read,
            submitted_at,
        };
        let open_file = OpenFile::Local(read.request.file.clone());
        match memory_ctx().compressed_cache().register_waiter(
            &open_file,
            &read.request.block,
            worker,
        ) {
            WaiterRegistration::Registered => {
                self.waiting_fs
                    .entry(read.request.block.slot_index())
                    .or_default()
                    .push(read);
                None
            }
            WaiterRegistration::AlreadyResident => {
                self.deregister_unless_parked(read.request.block.slot_index(), worker);
                self.complete_joined_fs_read(read);
                None
            }
            WaiterRegistration::Abandoned => {
                let DataFlowRequest {
                    data_flow_id,
                    operator_idx,
                    request,
                    submitted_at,
                } = read;
                Some(DataFlowRequest {
                    data_flow_id,
                    operator_idx,
                    request: FsRequest::Read(request),
                    submitted_at,
                })
            }
        }
    }

    /// The remote counterpart of [`try_join_fs_read`](Self::try_join_fs_read).
    fn try_join_http_get(
        &mut self,
        request: DataFlowRequest<HttpGetRequest>,
    ) -> Option<DataFlowRequest<HttpGetRequest>> {
        let Some(worker) = joinable_worker() else {
            return Some(request);
        };
        if request.request.block.is_owner() {
            return Some(request);
        }
        let open_file = OpenFile::Remote(request.request.remote.clone());
        match memory_ctx().compressed_cache().register_waiter(
            &open_file,
            &request.request.block,
            worker,
        ) {
            WaiterRegistration::Registered => {
                self.waiting_http
                    .entry(request.request.block.slot_index())
                    .or_default()
                    .push(request);
                None
            }
            WaiterRegistration::AlreadyResident => {
                self.deregister_unless_parked(request.request.block.slot_index(), worker);
                self.complete_joined_http_get(request);
                None
            }
            WaiterRegistration::Abandoned => Some(request),
        }
    }

    /// Drop this worker's waiter registration on `slot_idx` unless it still has
    /// a read parked there. Used when a registration resolved without parking
    /// (the extent was already resident), so a stale bit doesn't draw spurious
    /// events.
    fn deregister_unless_parked(&self, slot_idx: usize, worker: usize) {
        if !self.waiting_fs.contains_key(&slot_idx) && !self.waiting_http.contains_key(&slot_idx) {
            memory_ctx()
                .compressed_cache()
                .deregister_waiter(slot_idx, worker);
        }
    }

    /// A joined filesystem read's bytes are resident: stage its completion for
    /// the next [`completions`](Self::completions) drain. The owner already
    /// committed the blocks, so unlike a performed read there is nothing to
    /// commit here.
    fn complete_joined_fs_read(&mut self, read: DataFlowRequest<FsReadRequest>) {
        self.joined_ready.push(Ok(Completion::FsRead(read)));
    }

    /// The remote counterpart of [`complete_joined_fs_read`](Self::complete_joined_fs_read).
    /// No transport of this worker's served the bytes, so the read time is zero.
    fn complete_joined_http_get(&mut self, read: DataFlowRequest<HttpGetRequest>) {
        self.joined_ready
            .push(Ok(Completion::HttpGet(read, RemoteReadTime::default())));
    }

    /// A slot this worker waits on made progress: resolve every read parked on
    /// it. One whose extent is now fully resident completes; after a commit the
    /// rest keep waiting (their extent is still filling), while after an
    /// abandonment each is re-routed through [`request`](Self::request) /
    /// [`request_http`](Self::request_http) - it re-joins if the abandoned
    /// extent wasn't its own (its owner is still alive), and otherwise this
    /// worker performs the read itself. A re-performed remote read's tier split
    /// goes unrecorded (its stats were reported as it was joined); abandonment
    /// is a cancelled or failed query's teardown, so that undercount is rare.
    pub fn handle_slot_event(&mut self, event: SlotIoEvent) -> Result<()> {
        for read in self.waiting_fs.remove(&event.slot_idx).unwrap_or_default() {
            if read.request.block.is_resident() {
                self.complete_joined_fs_read(read);
            } else {
                match event.outcome {
                    SlotIoOutcome::Committed => self
                        .waiting_fs
                        .entry(event.slot_idx)
                        .or_default()
                        .push(read),
                    SlotIoOutcome::Abandoned => {
                        let DataFlowRequest {
                            data_flow_id,
                            operator_idx,
                            request,
                            submitted_at,
                        } = read;
                        self.request(DataFlowRequest {
                            data_flow_id,
                            operator_idx,
                            request: FsRequest::Read(request),
                            submitted_at,
                        })?;
                    }
                }
            }
        }
        for read in self
            .waiting_http
            .remove(&event.slot_idx)
            .unwrap_or_default()
        {
            if read.request.block.is_resident() {
                self.complete_joined_http_get(read);
            } else {
                match event.outcome {
                    SlotIoOutcome::Committed => self
                        .waiting_http
                        .entry(event.slot_idx)
                        .or_default()
                        .push(read),
                    SlotIoOutcome::Abandoned => {
                        let DataFlowRequest {
                            data_flow_id,
                            operator_idx,
                            request,
                            submitted_at,
                        } = read;
                        self.request_http(DataFlowRequest {
                            data_flow_id,
                            operator_idx,
                            request: HttpRequest::Get(request),
                            submitted_at,
                        })?;
                    }
                }
            }
        }
        // The waiter-mask bit tracks "this worker has something parked on the
        // slot"; drop it once nothing is.
        if let Some(worker) = joinable_worker() {
            self.deregister_unless_parked(event.slot_idx, worker);
        }
        Ok(())
    }

    /// Returns `true` if any disk or HTTP operation has not yet completed.
    /// Joined reads are excluded on purpose: they complete off another worker's
    /// slot event (whose send wakes this worker's park slot), never off this
    /// ring, so counting them would park [`wait`](Self::wait) on a ring that may
    /// stay silent.
    pub fn has_pending(&self) -> bool {
        self.has_file_pending() || self.has_http_pending()
    }

    /// Returns `true` if any disk read is in flight - operator reads here, plus
    /// the engine's cache-file reads / write-backs (all on the shared backend).
    pub fn has_file_pending(&self) -> bool {
        !self.pending_io_requests.is_empty() || self.http.has_disk_pending()
    }

    /// Returns `true` if any HTTP operation is in flight.
    pub fn has_http_pending(&self) -> bool {
        self.http.has_network_pending()
    }

    /// Number of HTTP operations issued but not yet completed — the depth a
    /// worker uses to decide whether to submit more.
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

        // Joined reads that resolved since the last drain complete first - their
        // bytes were committed by the worker that performed the read.
        let mut out = std::mem::take(&mut self.joined_ready);

        // Backend disk CQEs: this requester's fs reads, or the engine's cache-file
        // reads / write-backs. Disjoint id spaces, so the id's owning map sorts
        // them out. A negative result is a failure surfaced to just that dataflow.
        for &(result, ud) in &raw {
            if !disk_completion(ud) {
                continue; // HTTP socket op, already routed above
            }
            if let Some((request, done)) = self.pending_io_requests.remove(&ud) {
                // A write covers one run, and may complete shorter still (the
                // kernel caps a single write at ~2 GiB, among other reasons);
                // resubmit from where it stopped, at the matching file offset,
                // until the whole payload lands. Zero progress means the file
                // accepts no more bytes, which is an error, not a retry.
                if let FsRequest::Write(write) = &request.request {
                    let done = done + result.max(0) as usize;
                    if result > 0 && done < write.data.len() {
                        let run = write.data.run_at(done);
                        self.backend.submit_write(
                            write.file.as_raw_fd(),
                            done as u64,
                            run.as_ptr(),
                            run.len(),
                            self.next_id,
                        )?;
                        self.pending_io_requests
                            .insert(self.next_id, (request, done));
                        self.next_id += 1;
                        self.backend.submit()?;
                        continue;
                    }
                }
                let failure = match &request.request {
                    FsRequest::Read(_) if result < 0 => {
                        Some(std::io::Error::from_raw_os_error(-result))
                    }
                    FsRequest::Write(write)
                        if result < 0 || done + result as usize != write.data.len() =>
                    {
                        Some(if result < 0 {
                            std::io::Error::from_raw_os_error(-result)
                        } else {
                            std::io::Error::new(
                                std::io::ErrorKind::WriteZero,
                                "file accepted no more bytes mid-write",
                            )
                        })
                    }
                    _ => None,
                };
                if let Some(error) = failure {
                    out.push(Err(FailedRead {
                        data_flow_id: request.data_flow_id,
                        operator_idx: request.operator_idx,
                        error: error.into(),
                    }));
                } else {
                    let DataFlowRequest {
                        data_flow_id,
                        operator_idx,
                        request: fs_request,
                        submitted_at,
                    } = request;
                    let completion = match fs_request {
                        FsRequest::Read(read) => {
                            read.block.commit();
                            Completion::FsRead(DataFlowRequest {
                                data_flow_id,
                                operator_idx,
                                request: read,
                                submitted_at,
                            })
                        }
                        FsRequest::Write(write) => Completion::FsWrite(DataFlowRequest {
                            data_flow_id,
                            operator_idx,
                            request: write,
                            submitted_at,
                        }),
                    };
                    out.push(Ok(completion));
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

    /// Blocks until at least one pending read (disk or HTTP) makes progress. On
    /// Linux both ride one ring, so a single `submit_and_wait` wakes on either.
    #[cfg(target_os = "linux")]
    pub fn wait(&mut self) -> Result<()> {
        self.backend.submit_and_wait(1)?;
        Ok(())
    }

    /// Non-Linux disk and HTTP completions arrive on independent channels, so park
    /// until either is ready (without consuming it; `completions` drains both),
    /// matching the Linux single-ring wake on either kind.
    #[cfg(not(target_os = "linux"))]
    pub fn wait(&mut self) -> Result<()> {
        // Flush any staged disk ops so they are in flight before we park.
        self.backend.submit()?;
        if self.backend.has_ready_completion() || self.http.has_ready_completion() {
            return Ok(());
        }
        // Park until either channel has a completion, without consuming it
        // (completions() drains both).
        let mut select = crossbeam_channel::Select::new();
        select.recv(self.backend.completion_receiver());
        select.recv(self.http.completion_receiver());
        select.ready();
        Ok(())
    }
}

/// The current worker's global index if read-joining is enabled on this thread:
/// a slot-event router is installed (so events can reach us) and the index fits
/// the per-slot waiter mask. `None` on non-worker threads and beyond-mask
/// workers, where every read simply performs its own IO.
fn joinable_worker() -> Option<usize> {
    slot_events::with_router(|_| ())?;
    let worker = crate::worker::WORKER_IDX.get();
    (worker < MAX_JOINABLE_WORKERS).then_some(worker)
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
    use crate::io::{HttpGetRequest, HttpUploadRequest, OpenFile, RemoteFile};
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

    #[test]
    fn plain_http_upload_uses_shared_connection_engine() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let received = Arc::new(Mutex::new(Vec::new()));
        let server_received = received.clone();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut wire = Vec::new();
            let mut chunk = [0u8; 4096];
            let (head_len, content_len) = loop {
                let n = stream.read(&mut chunk).unwrap();
                assert!(n > 0);
                wire.extend_from_slice(&chunk[..n]);
                if let Some(end) = wire.windows(4).position(|w| w == b"\r\n\r\n") {
                    let head_len = end + 4;
                    let head = String::from_utf8_lossy(&wire[..head_len]);
                    let content_len = head
                        .lines()
                        .find_map(|line| {
                            line.strip_prefix("Content-Length: ")
                                .and_then(|v| v.parse::<usize>().ok())
                        })
                        .unwrap();
                    break (head_len, content_len);
                }
            };
            while wire.len() < head_len + content_len {
                let n = stream.read(&mut chunk).unwrap();
                assert!(n > 0);
                wire.extend_from_slice(&chunk[..n]);
            }
            server_received
                .lock()
                .unwrap()
                .extend_from_slice(&wire[head_len..head_len + content_len]);
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n",
                )
                .unwrap();
        });

        // The payload is ring memory, in two runs, so the upload path is
        // exercised the way a real file reaches it rather than as one buffer.
        crate::memory::init_test_free_pool(4);
        let mut allocator = crate::memory::SlabAllocator::new(false);
        let mut bytes = crate::memory::FileBytes::new();
        for part in [&b"parquet upload"[..], &b" bytes"[..]] {
            let mut run = allocator.get_slab_of_size(part.len(), false);
            run.as_mut_slice().copy_from_slice(part);
            bytes.push(run);
        }
        let data = Arc::new(bytes);
        let remote = Arc::new(
            RemoteFile::open(
                Url::parse(&format!("http://{addr}/object.parquet")).unwrap(),
                None,
                data.len() as u64,
            )
            .unwrap(),
        );
        let mut requester = IORequester::default();
        let submitted_at = std::time::Instant::now();
        let mut upload = DataFlowRequest::new(
            7,
            11,
            HttpRequest::Upload(HttpUploadRequest {
                remote,
                data: data.clone(),
            }),
        );
        upload.submitted_at = Some(submitted_at);
        let split = requester.request_http(upload).unwrap();
        assert_eq!(
            split,
            RemoteSplit {
                http_requests: 1,
                http_bytes: data.len() as u64,
                ..RemoteSplit::default()
            }
        );

        let completion = loop {
            if let Some(completion) = requester.completions().unwrap().into_iter().next() {
                match completion {
                    Ok(completion) => break completion,
                    Err(failed) => panic!("upload failed: {}", failed.error),
                }
            }
            requester.wait().unwrap();
        };
        match completion {
            Completion::HttpUpload(request) => {
                assert_eq!(request.data_flow_id, 7);
                assert_eq!(request.operator_idx, 11);
                assert_eq!(request.submitted_at, Some(submitted_at));
            }
            _ => panic!("expected an HTTP upload completion"),
        }
        server.join().unwrap();
        let sent: Vec<u8> = data.runs().flatten().copied().collect();
        assert_eq!(*received.lock().unwrap(), sent);
        assert_eq!(Arc::strong_count(&data), 1);
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
    fn fetch(requester: &mut IORequester, loc: &OpenFile, offset: usize, len: usize) {
        fetch_split(requester, loc, offset, len);
    }

    /// Fetch `[offset, offset + len)` of `loc` and block until it completes,
    /// returning the per-tier read split summed over the read's blocks (disk-cache
    /// hits versus network fetches).
    fn fetch_split(
        requester: &mut IORequester,
        loc: &OpenFile,
        offset: usize,
        len: usize,
    ) -> RemoteSplit {
        let OpenFile::Remote(remote) = loc else {
            panic!("test fetches over http")
        };
        let lookups = memory_ctx().compressed_cache().get(loc, offset, len);
        let mut split = RemoteSplit::default();
        let mut submitted = 0;
        for lookup in &lookups {
            if let Some(block) = lookup.missing() {
                let req = HttpRequest::Get(HttpGetRequest {
                    remote: remote.clone(),
                    block: block.clone(),
                });
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
    fn assert_cached(loc: &OpenFile, offset: usize, len: usize) {
        let hit = memory_ctx().compressed_cache().get(loc, offset, len);
        assert_eq!(hit.len(), 1);
        assert!(hit[0].missing().is_none(), "expected a cache hit");
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
        let loc = OpenFile::Remote(remote);
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
        let loc = OpenFile::Remote(remote);
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

    /// Spawn a server that accepts connections and drops them without responding,
    /// so a range read fails terminally (the exchange hits EOF). Accepts a few in
    /// case the engine reconnects.
    fn spawn_closing_server() -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        thread::spawn(move || {
            for _ in 0..4 {
                match listener.accept() {
                    Ok((stream, _)) => drop(stream),
                    Err(_) => break,
                }
            }
        });

        port
    }

    /// Submit a remote read and drive it to completion, returning every per-read
    /// result so a test can assert success or failure (the failure-aware sibling
    /// of `fetch_split`).
    fn fetch_results(
        requester: &mut IORequester,
        loc: &OpenFile,
        offset: usize,
        len: usize,
    ) -> Vec<std::result::Result<Completion, FailedRead>> {
        let OpenFile::Remote(remote) = loc else {
            panic!("test fetches over http")
        };
        let lookups = memory_ctx().compressed_cache().get(loc, offset, len);
        let mut submitted = 0;
        for lookup in &lookups {
            if let Some(block) = lookup.missing() {
                let req = HttpRequest::Get(HttpGetRequest {
                    remote: remote.clone(),
                    block: block.clone(),
                });
                requester
                    .request_http(DataFlowRequest::new(0, 0, req))
                    .unwrap();
                submitted += 1;
            }
        }
        assert!(submitted > 0, "expected blocks to read");

        let mut results = Vec::new();
        while results.len() < submitted {
            if requester.has_pending() {
                requester.wait().unwrap();
            }
            results.extend(requester.completions().unwrap());
        }
        results
    }

    /// A terminal transport failure surfaces as a `FailedRead`, never a committed
    /// block or a hang. Exercises the async failure path: a pool thread reports an
    /// error, `take_failed` buckets it, and the drain maps it to the dataflow.
    #[test]
    fn a_failed_http_read_surfaces_a_failed_read() {
        init_test_free_pool(16);
        let loc = remote_loc(spawn_closing_server());
        let mut requester = IORequester::with_http_config(client_config());

        let results = fetch_results(&mut requester, &loc, 0, 4096);

        assert!(
            results.iter().all(|r| r.is_err()),
            "every http read should fail"
        );
    }

    /// Enable read-joining on this test thread: install a one-worker slot-event
    /// router and identify the thread as worker 0, returning the injector the
    /// events land in.
    fn enable_joining_for_this_thread() -> crossbeam_channel::Receiver<SlotIoEvent> {
        let waker = Arc::new(crate::waker::WorkerWaker::new(1));
        let (router, mut receivers) = crate::io::slot_events::SlotEventRouter::create(
            1,
            crate::waker::WakerSet::new(vec![waker], 1),
        );
        crate::io::slot_events::install_worker_slot_events(router);
        crate::worker::WORKER_IDX.set(0);
        receivers.remove(0)
    }

    /// A local file holding `len` pattern bytes, plus its open handle.
    fn pattern_file(dir: &tempfile::TempDir, len: usize) -> Arc<std::fs::File> {
        let path = dir.path().join("data");
        std::fs::write(&path, (0..len).map(pattern).collect::<Vec<u8>>()).unwrap();
        Arc::new(std::fs::File::open(&path).unwrap())
    }

    /// The one fs read a `get` of `[offset, offset+len)` produces, owning or
    /// riding the extent according to what the cache resolved.
    fn fs_read_of(
        file: &Arc<std::fs::File>,
        open_file: &OpenFile,
        offset: usize,
        len: usize,
    ) -> FsRequest {
        let block = memory_ctx().compressed_cache().get(open_file, offset, len)[0]
            .take_missing()
            .expect("an unread range yields a missing extent");
        FsRequest::Read(crate::io::FsReadRequest {
            file: file.clone(),
            block,
        })
    }

    #[test]
    fn a_joined_read_completes_off_the_owners_commit() {
        init_test_free_pool(16);
        let events = enable_joining_for_this_thread();
        let dir = tempfile::tempdir().unwrap();
        let file = pattern_file(&dir, 4096);
        let open_file = OpenFile::Local(file.clone());
        memory_ctx()
            .compressed_cache()
            .open_entry(open_file.clone());
        let mut requester = IORequester::default();
        let owner_read = fs_read_of(&file, &open_file, 0, 4096);
        let joined_read = fs_read_of(&file, &open_file, 0, 4096);

        requester
            .request(DataFlowRequest::new(1, 0, owner_read))
            .unwrap();
        requester
            .request(DataFlowRequest::new(2, 0, joined_read))
            .unwrap();
        let owner_completions = loop {
            let completions = requester.completions().unwrap();
            if !completions.is_empty() {
                break completions;
            }
            requester.wait().unwrap();
        };
        // The owner's commit sent the slot event; resolving it completes the
        // joined read with no IO of its own (nothing further is pending).
        assert!(!requester.has_pending(), "the joined read submitted no IO");
        requester
            .handle_slot_event(events.try_recv().expect("a commit event"))
            .unwrap();
        let joined_completions = requester.completions().unwrap();

        let [Ok(Completion::FsRead(owner))] = &owner_completions[..] else {
            panic!("expected the owner's read to complete first");
        };
        let [Ok(Completion::FsRead(joined))] = &joined_completions[..] else {
            panic!("expected the joined read to complete off the event");
        };
        assert_eq!(owner.data_flow_id, 1);
        assert_eq!(joined.data_flow_id, 2);
        assert_cached(&open_file, 0, 4096);
    }

    #[test]
    fn an_abandoned_extent_is_read_by_its_waiter() {
        init_test_free_pool(16);
        let events = enable_joining_for_this_thread();
        let dir = tempfile::tempdir().unwrap();
        let file = pattern_file(&dir, 4096);
        let open_file = OpenFile::Local(file.clone());
        memory_ctx()
            .compressed_cache()
            .open_entry(open_file.clone());
        let mut requester = IORequester::default();
        let FsRequest::Read(owner_read) = fs_read_of(&file, &open_file, 0, 4096) else {
            unreachable!("fs_read_of builds reads");
        };
        // Keep the joiner's lookup: after an abandonment the extent is unmapped,
        // so this retained view is how the taken-over read's bytes are consumed.
        let mut joiner_lookups = memory_ctx().compressed_cache().get(&open_file, 0, 4096);
        let joined_read = FsRequest::Read(crate::io::FsReadRequest {
            file: file.clone(),
            block: joiner_lookups[0].take_missing().unwrap(),
        });
        requester
            .request(DataFlowRequest::new(7, 3, joined_read))
            .unwrap();
        assert!(!requester.has_pending(), "the joined read submitted no IO");

        // The owner is torn down (as a cancelled query's would be) without ever
        // performing its read; the waiter must take the read over.
        drop(owner_read);
        requester
            .handle_slot_event(events.try_recv().expect("an abandonment event"))
            .unwrap();

        assert!(
            requester.has_pending(),
            "the waiter performs the read itself"
        );
        let completions = loop {
            let completions = requester.completions().unwrap();
            if !completions.is_empty() {
                break completions;
            }
            requester.wait().unwrap();
        };
        let [Ok(Completion::FsRead(taken_over))] = &completions[..] else {
            panic!("expected the taken-over read to complete");
        };
        assert_eq!(taken_over.data_flow_id, 7);
        assert_eq!(taken_over.operator_idx, 3);
        let bytes = joiner_lookups.remove(0).into_data();
        assert_eq!(bytes.len(), 4096);
        for (i, &b) in bytes.iter().enumerate() {
            assert_eq!(b, pattern(i), "byte {i}");
        }
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
    fn remote_loc(port: u16) -> OpenFile {
        let url = Url::parse(&format!("https://127.0.0.1:{port}/obj")).unwrap();
        let loc = OpenFile::Remote(Arc::new(RemoteFile::open(url, None, 1 << 20).unwrap()));
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
