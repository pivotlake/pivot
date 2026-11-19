use crate::Identifier;
use crate::io::backend::IOBackend;
use crate::io::cached_http::CachedHttpEngine;
use crate::io::disk_cache::DiskCache;
use crate::io::hardware_queues::{InFlightPermit, WorkerHardwareQueues};
use crate::io::{
    Completion, DataFlowRequest, FailedIO, FileRange, FsDirectReadRequest, FsReadRequest,
    FsRequest, FsWriteRequest, HttpDirectGetRequest, HttpGetRequest, HttpRequest,
    HttpUploadRequest, OpenFile, PendingDirectReadRequest, PendingReadRequest, PendingWriteRequest,
    ReadData, ReadRequestId, ReadResponse, RemoteSplit, SlabRead,
};
use crate::memory::cache::MissingExtent;
use crate::memory::{CacheLookup, FrozenSlab, memory_ctx};
use crate::stats::StatsCollector;
use crate::worker::WORKER_IDX;
use bytes::Bytes;
use std::collections::{HashMap, VecDeque};
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
    #[error("the cache fill this read was following failed on another worker's I/O ring")]
    PiggybackedReadFailed,
}

type Result<T, E = Error> = std::result::Result<T, E>;

/// Number of submission-queue entries in each worker's io_uring. Logical
/// registration submits through this requester immediately; the backend flushes
/// each accepted operation to the kernel.
pub const RING_SIZE: u32 = 64;

/// One in-flight operator-owned local-file operation.
struct InFlightFsOp {
    request: DataFlowRequest<FsRequest>,
    /// Bytes already completed by earlier submissions (nonzero only for a
    /// write or a direct read whose prior completion came up short and was
    /// resubmitted).
    bytes_done: usize,
    /// The hardware-queue slots the operation holds. Dropping it releases
    /// them and wakes any workers waiting on the queue, so it lives exactly
    /// as long as the operation (a resubmitted write keeps it).
    permit: InFlightPermit,
    /// The runs a vectored write on the device refers to, which the kernel
    /// reads from here until the write completes. `None` for a read and for
    /// a write of a single run.
    _write_runs: Option<WriteRuns>,
}

/// The array of runs one vectored write refers to, kept at a stable address
/// for as long as the write is in flight.
struct WriteRuns(#[allow(dead_code)] Box<[libc::iovec]>);

// SAFETY: the runs point into ring memory that the write's request keeps
// alive; the array is only ever read by the kernel.
unsafe impl Send for WriteRuns {}

/// The most runs one vectored write takes, which is the kernel's limit.
const MOST_RUNS_PER_WRITE: usize = 1024;

/// Where a logical operator request must be delivered when it becomes ready.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RequestRoute {
    pub data_flow_id: Identifier,
    pub operator_idx: Identifier,
}

/// A ready logical response paired with its destination operator.
pub(crate) struct RoutedReadResponse {
    pub route: RequestRoute,
    pub response: ReadResponse,
}

/// Where one requested range's bytes are, once its reads have landed.
enum RangeSource {
    /// The cache fragments covering the range, in file order, so it basically
    /// represents for example:
    ///
    /// -----------------\[found=5\]\[missing=10\]\[found=3\]...
    ///        offset=321
    Cache(Vec<CacheLookup>),
    /// The slab one direct read fills, here once that read has landed.
    Direct {
        physical_read_id: Identifier,
        filled: Option<FrozenSlab>,
    },
}

/// A requested range: the offset in the file where it begins, and where its
/// bytes are.
struct RangeLookups {
    offset: usize,
    source: RangeSource,
}

impl RangeLookups {
    fn into_ready(self) -> ReadData {
        ReadData {
            offset: self.offset,
            bytes: match self.source {
                RangeSource::Cache(lookups) => {
                    lookups.into_iter().map(CacheLookup::into_data).collect()
                }
                RangeSource::Direct { filled, .. } => vec![Bytes::from_owner(
                    filled.expect("every direct read of a ready read has landed"),
                )],
            },
        }
    }
}

struct PendingRead {
    route: RequestRoute,
    id: ReadRequestId,
    open_file: OpenFile,
    requested_range_lookups: Vec<RangeLookups>,
    /// Number of physical cache fills this logical request still awaits.
    remaining: usize,
}

impl PendingRead {
    fn into_response(self) -> RoutedReadResponse {
        RoutedReadResponse {
            route: self.route,
            response: ReadResponse::new(
                self.id,
                self.open_file,
                self.requested_range_lookups
                    .into_iter()
                    .map(RangeLookups::into_ready)
                    .collect(),
            ),
        }
    }
}

pub(crate) enum IORead {
    Fs(DataFlowRequest<FsRequest>),
    Http(DataFlowRequest<HttpRequest>),
}

impl IORead {
    /// The cache extent this pending read is waiting for.
    pub(crate) fn missing_extent(&self) -> &MissingExtent {
        match self {
            Self::Fs(request) => match &request.request {
                FsRequest::Read(read) => &read.block,
                FsRequest::ReadDirect(_) => {
                    unreachable!("direct reads are never registered with the cache")
                }
                FsRequest::Write(_) => unreachable!("registered reads never contain writes"),
            },
            Self::Http(request) => match &request.request {
                HttpRequest::Get(read) => &read.block,
                HttpRequest::GetDirect(_) => {
                    unreachable!("direct reads are never registered with the cache")
                }
                HttpRequest::Upload(_) => {
                    unreachable!("registered reads never contain uploads")
                }
            },
        }
    }

    pub(crate) fn data_flow_id(&self) -> Identifier {
        match self {
            Self::Fs(request) => request.data_flow_id,
            Self::Http(request) => request.data_flow_id,
        }
    }

    pub(crate) fn pending_read_id(&self) -> Identifier {
        match self {
            Self::Fs(request) => request.pending_read_id,
            Self::Http(request) => request.pending_read_id,
        }
        .expect("the requester assigns every registered read an id")
    }
}

/// Bridges dataflow operators and the I/O backend, holding state on outstanding
/// requests and returning responses together with the original requested context.
///
/// Operators enqueue logical file ranges in [`OperatorIO`](crate::io::OperatorIO).
/// This is the boundary where those ranges become physical cache extents and
/// transport operations: registration resolves a read through the compressed
/// cache, submits each missing extent's fill immediately, and the read's
/// response is assembled once its last fill lands.
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
    /// Logical reads awaiting physical fills, by slot; a `None` slot is free
    /// and listed in `free_pending_reads_slots`.
    pending_reads: Vec<Option<PendingRead>>,
    /// A vector of all the slots currently unused in pending_reads
    free_pending_reads_slots: Vec<usize>,
    /// Maps between a physical id for the read to the index of the `PendingRead` object in
    /// pending_reads
    physical_read_ids_to_pending_reads_indexes: HashMap<Identifier, usize>,
    next_physical_read_id: Identifier,
    /// Logical reads whose every fill has landed, awaiting the worker's drain.
    ready_reads: Vec<RoutedReadResponse>,
    /// In-flight operator-owned local-file operations, keyed by backend id.
    pending_io_requests: HashMap<Identifier, InFlightFsOp>,
    /// Local-file operations accepted while their hardware queue was at
    /// capacity, submitted in order as slots free.
    fs_backlog: VecDeque<DataFlowRequest<FsRequest>>,
    /// This worker's per-device hardware-queue counters, shared with every
    /// worker whose CPU submits into the same queue.
    fs_queues: WorkerHardwareQueues,
    /// Logical reads following extents currently being filled by another
    /// worker. They need no transport request of their own; the worker polls
    /// their shared extent state once per loop.
    piggybacked_reads: Vec<IORead>,
    /// Allocates backend disk-op ids - shared by fs reads here and the engine's
    /// cache-file reads/write-backs, so a completion routes by which map holds it.
    next_id: Identifier,
    /// Remote reads, optionally served from the on-disk cache. Ring-less like the
    /// underlying `HttpEngine`: it borrows `backend` (and `next_id`) to submit.
    http: CachedHttpEngine,
    /// Operations that have completed and not yet been taken (see
    /// [`collect_completions`](Self::collect_completions)). Kept here rather
    /// than returned, so that one caller can take only the completions it is
    /// after and leave the rest for the worker's next pass.
    completed: Vec<std::result::Result<Completion, FailedIO>>,
    /// This worker's budget of bytes asked for from local disks and not yet
    /// started on.
    disk_read_ahead: crate::io::ReadAheadBudget,
    /// This worker's budget of bytes asked for from object storage and not
    /// yet started on.
    network_read_ahead: crate::io::ReadAheadBudget,
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
            pending_reads: Vec::new(),
            free_pending_reads_slots: Vec::new(),
            physical_read_ids_to_pending_reads_indexes: HashMap::new(),
            next_physical_read_id: 0,
            ready_reads: Vec::new(),
            pending_io_requests: Default::default(),
            fs_backlog: VecDeque::new(),
            fs_queues: WorkerHardwareQueues::new(),
            piggybacked_reads: Vec::new(),
            next_id: 0,
            http: CachedHttpEngine::with_default_config(disk_cache)
                .expect("Unable to create http engine"),
            completed: Vec::new(),
            disk_read_ahead: crate::io::ReadAheadBudget::for_this_worker(
                crate::io::ReadMedium::Disk,
            ),
            network_read_ahead: crate::io::ReadAheadBudget::for_this_worker(
                crate::io::ReadMedium::Network,
            ),
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
            pending_reads: Vec::new(),
            free_pending_reads_slots: Vec::new(),
            physical_read_ids_to_pending_reads_indexes: HashMap::new(),
            next_physical_read_id: 0,
            ready_reads: Vec::new(),
            pending_io_requests: Default::default(),
            fs_backlog: VecDeque::new(),
            fs_queues: WorkerHardwareQueues::new(),
            piggybacked_reads: Vec::new(),
            next_id: 0,
            http: CachedHttpEngine::new(http_config, disk_cache)
                .expect("Unable to create http engine"),
            completed: Vec::new(),
            disk_read_ahead: crate::io::ReadAheadBudget::for_this_worker(
                crate::io::ReadMedium::Disk,
            ),
            network_read_ahead: crate::io::ReadAheadBudget::for_this_worker(
                crate::io::ReadMedium::Network,
            ),
        }
    }

    /// Build a requester with a specific HTTP client config and no disk cache.
    pub fn with_http_config(http_config: Arc<rustls::ClientConfig>) -> Self {
        Self::with_config(http_config, None)
    }

    /// [`request_read`](Self::request_read) with the cache lookups resolved
    /// here, as [`OperatorIO`](crate::io::OperatorIO) resolves them before
    /// borrowing the requester.
    #[cfg(test)]
    fn request_read_resolving(
        &mut self,
        data_flow_id: Identifier,
        operator_idx: Identifier,
        request: PendingReadRequest,
        stats: &mut StatsCollector,
    ) -> Result<()> {
        let cache_lookups = resolve_read_through_cache(&request);
        self.request_read(data_flow_id, operator_idx, request, cache_lookups, stats)
    }

    /// Route every local-file operation through `queue`, so a test controls
    /// the in-flight capacity and which requesters share it.
    #[cfg(test)]
    fn force_fs_queue(&mut self, queue: Arc<crate::io::hardware_queues::HardwareQueue>) {
        self.fs_queues.force(queue);
    }

    /// Take remote reads from `limiter` instead of the process's receive
    /// allowance, so a test controls the rate.
    #[cfg(all(test, target_os = "linux"))]
    fn force_receive_rate_limiter(&mut self, limiter: Arc<crate::io::ReceiveRateLimiter>) {
        self.http.force_receive_rate_limiter(limiter);
    }

    /// Keep the step deadlines of remote reads, before any doubling, between
    /// `shortest` and `longest`, so a test controls how soon a silent peer is
    /// given up on.
    #[cfg(all(test, target_os = "linux"))]
    fn force_step_deadline_bounds(
        &mut self,
        shortest: std::time::Duration,
        longest: std::time::Duration,
    ) {
        self.http.force_step_deadline_bounds(shortest, longest);
    }

    /// Ask `servers` for host addresses instead of the machine's name
    /// servers, so a test controls what a host resolves to.
    #[cfg(all(test, target_os = "linux"))]
    fn force_name_servers(&mut self, servers: Vec<std::net::SocketAddr>) {
        self.http.force_name_servers(servers);
    }

    /// Look a host up again once its answer is `refresh_after` old, and
    /// forget what it resolved to once the answer is `forget_after` old.
    #[cfg(all(test, target_os = "linux"))]
    fn force_address_ages(
        &mut self,
        refresh_after: std::time::Duration,
        forget_after: std::time::Duration,
    ) {
        self.http.force_address_ages(refresh_after, forget_after);
    }

    /// Resolve one logical read through the memory caches and immediately submit
    /// every physical cache fill it still needs.
    pub fn request_read(
        &mut self,
        data_flow_id: Identifier,
        operator_idx: Identifier,
        request: PendingReadRequest,
        cache_lookups: Vec<Vec<CacheLookup>>,
        stats: &mut StatsCollector,
    ) -> Result<()> {
        let route = RequestRoute {
            data_flow_id,
            operator_idx,
        };
        let mut fs_requests = Vec::new();
        let mut http_requests = Vec::new();
        for request in self.register_and_create_io_reads(route, request, cache_lookups) {
            if !request.missing_extent().is_fill_owner() {
                request.missing_extent().subscribe(WORKER_IDX.get());
                self.piggybacked_reads.push(request);
                continue;
            }
            match request {
                IORead::Fs(request) => fs_requests.push(request),
                IORead::Http(request) => http_requests.push(request),
            }
        }

        stats.record_issued_disk(&mut fs_requests);
        let mut fs_queue = fs_requests.into_iter();
        while let Some(request) = fs_queue.next() {
            if let Err(error) = self.submit_fs_request_to_backend(request) {
                // The failed submit withdrew its own registration; the reads
                // still queued behind it are ours to withdraw.
                self.withdraw_unsubmitted_reads(fs_queue, http_requests);
                return Err(error);
            }
        }

        stats.stamp_issued(&mut http_requests);
        let mut http_queue = http_requests.into_iter();
        while let Some(request) = http_queue.next() {
            match self.submit_http_request_to_backend(request) {
                Ok(split) => stats.record_issued_remote(split),
                Err(error) => {
                    self.withdraw_unsubmitted_reads(std::iter::empty(), http_queue);
                    return Err(error);
                }
            }
        }
        // Issued once the whole logical read is registered, so its ranges that
        // lie close together in the object share a request.
        let http_requests = self.http.issue_registered_fetches(&mut self.backend)?;
        stats.record_issued_remote(RemoteSplit {
            http_requests,
            ..RemoteSplit::default()
        });

        Ok(())
    }

    /// Undo the bookkeeping `register_read` created for a read that will never
    /// be submitted (its own submit failed, or an earlier read's in the same
    /// batch did). Writes and uploads register nothing, so for them both
    /// arguments are `None` and this is a no-op.
    ///
    /// Removing the extent from the shared cache is the critical part:
    /// registration published it as "a fill is in flight", so every other
    /// query touching the range subscribes to it and sleeps until it commits
    /// or fails. Once we know no submission will happen, nothing can ever
    /// commit or fail it, and those subscribers would sleep forever. Removing
    /// it lets the next reader start a fill of its own.
    fn withdraw_registered_read(
        &mut self,
        cache_extent: Option<(OpenFile, MissingExtent)>,
        pending_read_id: Option<Identifier>,
    ) {
        if let Some((open_file, block)) = cache_extent {
            block.remove_from_cache(&open_file);
        }
        if let Some(id) = pending_read_id {
            let _ = self.fail_pending_read(id);
        }
    }

    /// Withdraw every read left in a batch after one submit failed. A single
    /// logical read can register several physical reads; returning the error
    /// after cleaning up only the failing one would leave the rest holding
    /// cache extents no submission will ever fill.
    fn withdraw_unsubmitted_reads(
        &mut self,
        fs_requests: impl IntoIterator<Item = DataFlowRequest<FsRequest>>,
        http_requests: impl IntoIterator<Item = DataFlowRequest<HttpRequest>>,
    ) {
        for request in fs_requests {
            let cache_extent = fs_cache_extent(&request);
            self.withdraw_registered_read(cache_extent, request.pending_read_id);
        }
        for request in http_requests {
            let cache_extent = http_cache_extent(&request);
            self.withdraw_registered_read(cache_extent, request.pending_read_id);
        }
    }

    /// Select the transport for one logical write and submit it immediately.
    pub fn request_write(
        &mut self,
        data_flow_id: Identifier,
        operator_idx: Identifier,
        request: PendingWriteRequest,
        stats: &mut StatsCollector,
    ) -> Result<()> {
        let PendingWriteRequest {
            id,
            open_file,
            offset,
            data,
            method,
        } = request;
        match open_file {
            OpenFile::Local(file) => {
                let mut request = DataFlowRequest::new(
                    data_flow_id,
                    operator_idx,
                    FsRequest::Write(FsWriteRequest {
                        id,
                        file,
                        offset,
                        data,
                    }),
                );
                stats.record_issued_disk(std::slice::from_mut(&mut request));
                self.submit_fs_request_to_backend(request)
            }
            OpenFile::Remote(remote) => {
                let mut request = DataFlowRequest::new(
                    data_flow_id,
                    operator_idx,
                    HttpRequest::Upload(HttpUploadRequest {
                        remote,
                        data,
                        method,
                        etag: None,
                        response_body: Vec::new(),
                    }),
                );
                stats.stamp_issued(std::slice::from_mut(&mut request));
                let split = self.submit_http_request_to_backend(request)?;
                stats.record_issued_remote(split);
                Ok(())
            }
        }
    }

    pub(crate) fn take_ready_reads(&mut self) -> Vec<RoutedReadResponse> {
        std::mem::take(&mut self.ready_reads)
    }

    /// Discard logical read state for a dataflow that is no longer running.
    /// Physical reads already submitted are allowed to finish and warm the
    /// cache; their later completions simply find no waiter. Dropping a
    /// backlogged read drops its extent's fill owner, which removes the extent
    /// and wakes any followers, exactly like a cancelled in-flight owner.
    pub fn cancel_dataflow(&mut self, data_flow_id: Identifier) {
        let cancelled_slots: Vec<_> = self
            .pending_reads
            .iter_mut()
            .enumerate()
            .filter_map(|(slot, request)| {
                (request.as_ref()?.route.data_flow_id == data_flow_id).then_some(slot)
            })
            .collect();
        for slot in &cancelled_slots {
            if self.pending_reads[*slot].take().is_some() {
                self.free_pending_reads_slots.push(*slot);
            }
        }
        self.physical_read_ids_to_pending_reads_indexes
            .retain(|_, slot| !cancelled_slots.contains(slot));
        self.ready_reads
            .retain(|ready| ready.route.data_flow_id != data_flow_id);
        self.piggybacked_reads
            .retain(|request| request.data_flow_id() != data_flow_id);
        self.fs_backlog
            .retain(|request| request.data_flow_id != data_flow_id);
    }

    /// Resolve one logical read through the compressed cache and track it
    /// until every extent it still needs is filled; returns those fills as
    /// physical reads to submit.
    fn register_and_create_io_reads(
        &mut self,
        route: RequestRoute,
        request: PendingReadRequest,
        cache_lookups: Vec<Vec<CacheLookup>>,
    ) -> Vec<IORead> {
        let mut locations = Vec::with_capacity(request.locations.len());
        let mut physical_reads = Vec::new();

        for (location, lookups) in request.locations.iter().zip(cache_lookups) {
            for lookup in &lookups {
                if let Some(block) = lookup.missing_extent() {
                    let id = self.next_physical_read_id;
                    self.next_physical_read_id += 1;
                    physical_reads.push(match &request.open_file {
                        OpenFile::Local(file) => IORead::Fs(
                            DataFlowRequest::new(
                                route.data_flow_id,
                                route.operator_idx,
                                FsRequest::Read(FsReadRequest {
                                    file: file.clone(),
                                    block: block.clone(),
                                }),
                            )
                            .with_pending_read_id(id),
                        ),
                        OpenFile::Remote(remote) => IORead::Http(
                            DataFlowRequest::new(
                                route.data_flow_id,
                                route.operator_idx,
                                HttpRequest::Get(HttpGetRequest {
                                    remote: remote.clone(),
                                    block: block.clone(),
                                }),
                            )
                            .with_pending_read_id(id),
                        ),
                    });
                }
            }
            locations.push(RangeLookups {
                offset: location.offset,
                source: RangeSource::Cache(lookups),
            });
        }

        let physical_read_ids: Vec<_> =
            physical_reads.iter().map(IORead::pending_read_id).collect();
        self.track_pending_read(
            route,
            request.id,
            request.open_file,
            locations,
            &physical_read_ids,
        );
        physical_reads
    }

    /// Track one logical read until every physical read in `physical_read_ids`
    /// has landed; a read with none to wait for is ready at once.
    fn track_pending_read(
        &mut self,
        route: RequestRoute,
        id: ReadRequestId,
        open_file: OpenFile,
        locations: Vec<RangeLookups>,
        physical_read_ids: &[Identifier],
    ) {
        let remaining = physical_read_ids.len();
        let slot = self.insert_pending_read(PendingRead {
            route,
            id,
            open_file,
            requested_range_lookups: locations,
            remaining,
        });
        for id in physical_read_ids {
            self.physical_read_ids_to_pending_reads_indexes
                .insert(*id, slot);
        }
        if remaining == 0 {
            self.finish_pending_read(slot);
        }
    }

    /// Read ranges of a file around the compressed cache, into the slabs the
    /// caller supplied, and track the read until every range has landed.
    /// Each range is one physical read: a read of a local file, or a range
    /// GET of a remote object. Remote ranges that follow each other in the
    /// object share a request, as a cached read's do.
    pub fn request_read_direct(
        &mut self,
        data_flow_id: Identifier,
        operator_idx: Identifier,
        request: PendingDirectReadRequest,
        stats: &mut StatsCollector,
    ) -> Result<()> {
        let route = RequestRoute {
            data_flow_id,
            operator_idx,
        };
        let PendingDirectReadRequest { id, file, reads } = request;
        let mut fs_requests = Vec::new();
        let mut http_requests = Vec::new();
        let mut physical_read_ids = Vec::with_capacity(reads.len());
        let mut tracked = Vec::with_capacity(reads.len());
        for SlabRead { offset, slab, len } in reads {
            assert!(len <= slab.size(), "a direct read fits its slab");
            let physical_read_id = self.next_physical_read_id;
            self.next_physical_read_id += 1;
            physical_read_ids.push(physical_read_id);
            tracked.push(RangeLookups {
                offset,
                source: RangeSource::Direct {
                    physical_read_id,
                    filled: None,
                },
            });
            match &file {
                OpenFile::Local(file) => fs_requests.push(
                    DataFlowRequest::new(
                        data_flow_id,
                        operator_idx,
                        FsRequest::ReadDirect(FsDirectReadRequest {
                            file: file.clone(),
                            offset: offset as u64,
                            slab,
                            len,
                        }),
                    )
                    .with_pending_read_id(physical_read_id),
                ),
                OpenFile::Remote(remote) => http_requests.push(
                    DataFlowRequest::new(
                        data_flow_id,
                        operator_idx,
                        HttpRequest::GetDirect(HttpDirectGetRequest {
                            remote: remote.clone(),
                            offset: offset as u64,
                            slab,
                            len,
                        }),
                    )
                    .with_pending_read_id(physical_read_id),
                ),
            }
        }
        self.track_pending_read(route, id, file, tracked, &physical_read_ids);

        stats.record_issued_disk(&mut fs_requests);
        let mut fs_queue = fs_requests.into_iter();
        while let Some(request) = fs_queue.next() {
            if let Err(error) = self.submit_fs_request_to_backend(request) {
                self.withdraw_unsubmitted_reads(fs_queue, http_requests);
                return Err(error);
            }
        }

        stats.stamp_issued(&mut http_requests);
        let mut http_queue = http_requests.into_iter();
        while let Some(request) = http_queue.next() {
            match self.submit_http_request_to_backend(request) {
                Ok(split) => stats.record_issued_remote(split),
                Err(error) => {
                    self.withdraw_unsubmitted_reads(std::iter::empty(), http_queue);
                    return Err(error);
                }
            }
        }
        // Issued once every range is registered, so the ranges that follow
        // each other in the object share a request.
        let http_requests = self.http.issue_registered_fetches(&mut self.backend)?;
        stats.record_issued_remote(RemoteSplit {
            http_requests,
            ..RemoteSplit::default()
        });
        Ok(())
    }

    /// Hand the slab a direct read filled to the logical read waiting for
    /// it. That read may have been cancelled or failed meanwhile, and then
    /// the slab is simply dropped.
    fn land_direct_read(&mut self, physical_read_id: Identifier, filled_slab: FrozenSlab) {
        let Some(slot) = self
            .physical_read_ids_to_pending_reads_indexes
            .get(&physical_read_id)
        else {
            return;
        };
        let Some(pending) = self.pending_reads[*slot].as_mut() else {
            return;
        };
        let filled = pending
            .requested_range_lookups
            .iter_mut()
            .find_map(|range| match &mut range.source {
                RangeSource::Direct {
                    physical_read_id: id,
                    filled,
                } if *id == physical_read_id => Some(filled),
                _ => None,
            })
            .expect("a tracked direct read has a range waiting for its slab");
        *filled = Some(filled_slab);
    }

    fn insert_pending_read(&mut self, request: PendingRead) -> usize {
        if let Some(slot) = self.free_pending_reads_slots.pop() {
            self.pending_reads[slot] = Some(request);
            slot
        } else {
            self.pending_reads.push(Some(request));
            self.pending_reads.len() - 1
        }
    }

    /// One physical read of a tracked read landed; the read becomes ready
    /// once its last one does.
    fn complete_pending_read(&mut self, physical_read_id: Identifier) {
        let Some(slot) = self
            .physical_read_ids_to_pending_reads_indexes
            .remove(&physical_read_id)
        else {
            // The logical request may have been cancelled while this physical
            // read was still in flight.
            return;
        };
        let Some(request) = self.pending_reads[slot].as_mut() else {
            return;
        };
        request.remaining -= 1;
        if request.remaining == 0 {
            self.finish_pending_read(slot);
        }
    }

    /// Fail the logical request owning one physical read. Its other reads may
    /// still complete and warm the cache, but no longer have a callback target.
    fn fail_pending_read(&mut self, physical_read_id: Identifier) -> Option<RequestRoute> {
        let slot = self
            .physical_read_ids_to_pending_reads_indexes
            .remove(&physical_read_id)?;
        let request = self.pending_reads[slot].take()?;
        self.physical_read_ids_to_pending_reads_indexes
            .retain(|_, routed_slot| *routed_slot != slot);
        self.free_pending_reads_slots.push(slot);
        Some(request.route)
    }

    fn finish_pending_read(&mut self, slot: usize) {
        let request = self.pending_reads[slot]
            .take()
            .expect("finishing an empty request slot");
        debug_assert_eq!(request.remaining, 0);
        self.free_pending_reads_slots.push(slot);
        self.ready_reads.push(request.into_response());
    }

    /// Reserve the slots `request` needs on the hardware queue the file's
    /// device serves this worker with: one for each device request the
    /// operation becomes. `None` means the queue has no room for them and the
    /// operation must wait in the backlog; this worker is then registered for
    /// a ring wake on the next release of slots, so it can park with nothing
    /// of its own in flight.
    fn acquire_fs_slots(&mut self, request: &DataFlowRequest<FsRequest>) -> Option<InFlightPermit> {
        let (bytes, runs) = match &request.request {
            FsRequest::Read(read) => (read.block.len(), 1),
            FsRequest::ReadDirect(read) => (read.slab.size(), 1),
            FsRequest::Write(write) => (write.data.len(), write.data.count_runs()),
        };
        let device_idx = request.request.file().device_idx();
        let queue = self.fs_queues.queue_for_device(device_idx);
        let slots = queue.count_slots(bytes, runs);
        if let Some(permit) = queue.try_acquire(slots) {
            return Some(permit);
        }
        // Register before the deciding attempt so a release between the two
        // cannot be missed (see `register_waiter`); a successful retry means
        // no one needs to wake us after all.
        let worker_idx = WORKER_IDX.get();
        queue.register_waiter(worker_idx);
        match queue.try_acquire(slots) {
            Some(permit) => {
                queue.deregister_waiter(worker_idx);
                Some(permit)
            }
            None => None,
        }
    }

    /// Refill the backend from the backlog as far as the freed slots allow. A
    /// failure fails that operation the same way a registration-time failure
    /// does, then surfaces the error.
    fn drain_fs_backlog(&mut self) -> Result<()> {
        while let Some(request) = self.fs_backlog.pop_front() {
            let permit = match self.acquire_fs_slots(&request) {
                Some(permit) => permit,
                None => {
                    self.fs_backlog.push_front(request);
                    break;
                }
            };
            self.submit_fs_request_now(request, permit)?;
        }
        Ok(())
    }

    /// Accept an operator-owned filesystem operation: submit it through the
    /// worker's shared I/O backend, or hold it in the backlog while its
    /// hardware queue is at capacity. Request-owned storage remains alive
    /// until completion.
    fn submit_fs_request_to_backend(&mut self, request: DataFlowRequest<FsRequest>) -> Result<()> {
        match self.acquire_fs_slots(&request) {
            Some(permit) => self.submit_fs_request_now(request, permit),
            None => {
                self.fs_backlog.push_back(request);
                Ok(())
            }
        }
    }

    /// Unconditionally submit one filesystem operation to the backend,
    /// carrying the hardware-queue slot it holds. A failure withdraws the
    /// request's read registration before the error is returned, so the
    /// caller has nothing left to clean up for it.
    fn submit_fs_request_now(
        &mut self,
        request: DataFlowRequest<FsRequest>,
        permit: InFlightPermit,
    ) -> Result<()> {
        let cache_extent = fs_cache_extent(&request);
        let pending_read_id = request.pending_read_id;
        let queued = match &request.request {
            FsRequest::Read(read) => self
                .backend
                .submit_read(
                    read.file.as_raw_fd(),
                    read.block.file_offset() as u64,
                    read.block.dest(),
                    read.block.len(),
                    self.next_id,
                )
                .map(|()| None),
            FsRequest::ReadDirect(read) => self.submit_rest_of_direct_read(read, 0).map(|()| None),
            FsRequest::Write(write) => self.submit_rest_of_write(write, 0),
        };
        let write_runs = match queued {
            Ok(write_runs) => write_runs,
            Err(error) => {
                self.withdraw_registered_read(cache_extent, pending_read_id);
                return Err(error.into());
            }
        };
        self.pending_io_requests.insert(
            self.next_id,
            InFlightFsOp {
                request,
                bytes_done: 0,
                permit,
                _write_runs: write_runs,
            },
        );
        self.next_id += 1;
        if let Err(error) = self.backend.submit() {
            self.withdraw_registered_read(cache_extent, pending_read_id);
            return Err(error.into());
        }

        Ok(())
    }

    /// Put what is left of `read` after its first `bytes_done` bytes onto the
    /// backend's queue, under the next request id: the rest of its range,
    /// into the rest of its slab.
    fn submit_rest_of_direct_read(
        &mut self,
        read: &FsDirectReadRequest,
        bytes_done: usize,
    ) -> std::result::Result<(), crate::io::backend::Error> {
        assert!(
            bytes_done < read.slab.size(),
            "a read is resubmitted only while it has bytes left"
        );
        // SAFETY: `bytes_done` is inside the slab, checked just above.
        let rest = unsafe { read.slab.ptr.add(bytes_done) };
        self.backend.submit_read(
            read.file.as_raw_fd(),
            read.offset + bytes_done as u64,
            rest,
            read.slab.size() - bytes_done,
            self.next_id,
        )
    }

    /// Put what is left of `write` after its first `bytes_done` bytes onto the
    /// backend's queue as one operation, under the next request id: every
    /// remaining run at once, as far as a single write takes them. Returns the
    /// array of runs the operation refers to, when it is a vectored one, which
    /// has to be kept until it completes.
    fn submit_rest_of_write(
        &mut self,
        write: &FsWriteRequest,
        bytes_done: usize,
    ) -> std::result::Result<Option<WriteRuns>, crate::io::backend::Error> {
        let mut bytes_queued = 0;
        let runs: Box<[libc::iovec]> = write
            .data
            .runs_from(bytes_done)
            .take(MOST_RUNS_PER_WRITE)
            .take_while(|run| {
                bytes_queued += run.len();
                bytes_queued <= crate::io::backend::MAX_IO_OP_LEN || bytes_queued == run.len()
            })
            .map(|run| libc::iovec {
                iov_base: run.as_ptr() as *mut libc::c_void,
                iov_len: run.len(),
            })
            .collect();
        let offset = write.offset + bytes_done as u64;
        match &*runs {
            [] => unreachable!("a write is resubmitted only while it has bytes left"),
            [only] => {
                self.backend.submit_write(
                    write.file.as_raw_fd(),
                    offset,
                    only.iov_base as *const u8,
                    only.iov_len,
                    self.next_id,
                )?;
                Ok(None)
            }
            _ => {
                self.backend.submit_write_vectored(
                    write.file.as_raw_fd(),
                    offset,
                    &runs,
                    self.next_id,
                )?;
                Ok(Some(WriteRuns(runs)))
            }
        }
    }

    /// Submit a read for a remote region, served from the on-disk cache where
    /// possible (only the missing ranges hit the network). Delegates to the
    /// [`CachedHttpEngine`], lending it the shared backend and disk-id counter.
    /// A failure withdraws the request's read registration before the error
    /// is returned, so the caller has nothing left to clean up for it.
    fn submit_http_request_to_backend(
        &mut self,
        request: DataFlowRequest<HttpRequest>,
    ) -> Result<RemoteSplit> {
        let cache_extent = http_cache_extent(&request);
        let DataFlowRequest {
            data_flow_id,
            operator_idx,
            request,
            pending_read_id,
            submitted_at,
        } = request;
        let result = match request {
            HttpRequest::Get(request) => self.http.get(
                &mut self.backend,
                &mut self.next_id,
                DataFlowRequest {
                    data_flow_id,
                    operator_idx,
                    request,
                    pending_read_id,
                    submitted_at,
                },
            ),
            HttpRequest::GetDirect(request) => Ok(self.http.get_direct(DataFlowRequest {
                data_flow_id,
                operator_idx,
                request,
                pending_read_id,
                submitted_at,
            })),
            HttpRequest::Upload(request) => {
                let bytes = request.data.len() as u64;
                self.http
                    .upload(
                        &mut self.backend,
                        DataFlowRequest {
                            data_flow_id,
                            operator_idx,
                            request,
                            pending_read_id,
                            submitted_at,
                        },
                    )
                    .map(|()| RemoteSplit {
                        http_requests: 1,
                        http_bytes: bytes,
                        ..RemoteSplit::default()
                    })
            }
        };
        if result.is_err() {
            self.withdraw_registered_read(cache_extent, pending_read_id);
        }
        result
    }

    /// Returns `true` if any physical operation or piggybacked read is unresolved.
    pub fn has_pending(&self) -> bool {
        self.has_file_pending() || self.has_http_pending() || !self.piggybacked_reads.is_empty()
    }

    /// The handle a waker uses to interrupt this requester's blocking
    /// [`wait`](Self::wait) from another thread.
    pub fn wake_handle(&self) -> crate::io::RingWakeHandle {
        self.backend.wake_handle()
    }

    /// Returns `true` if any disk read is in flight or waiting in the backlog -
    /// operator reads here, plus the engine's cache-file reads / write-backs
    /// (all on the shared backend). A blocking wait is safe with a non-empty
    /// backlog: either this worker has operations of its own in flight, or it
    /// is registered on the full hardware queue and the release of a slot
    /// wakes its ring.
    pub fn has_file_pending(&self) -> bool {
        !self.pending_io_requests.is_empty()
            || !self.fs_backlog.is_empty()
            || self.http.has_disk_pending()
    }

    /// Returns `true` if any HTTP operation is in flight.
    pub fn has_http_pending(&self) -> bool {
        self.http.has_network_pending()
    }

    /// This worker's read-ahead budget for reads from `medium`.
    fn read_ahead_budget(&self, medium: crate::io::ReadMedium) -> &crate::io::ReadAheadBudget {
        match medium {
            crate::io::ReadMedium::Disk => &self.disk_read_ahead,
            crate::io::ReadMedium::Network => &self.network_read_ahead,
        }
    }

    /// Whether this worker has asked for less than its read-ahead budget for
    /// `medium`, so its operators may issue more reads from it.
    pub fn has_read_ahead_room(&self, medium: crate::io::ReadMedium) -> bool {
        self.read_ahead_budget(medium).has_room()
    }

    /// Whether this worker has asked for less than its read-ahead budget for
    /// disks or for the network. With both budgets full, no operator can
    /// issue another read.
    pub fn has_any_read_ahead_room(&self) -> bool {
        self.disk_read_ahead.has_room() || self.network_read_ahead.has_room()
    }

    /// Count `bytes` against this worker's read-ahead budget for `medium`
    /// until the returned charge is dropped.
    pub fn charge_read_ahead(
        &self,
        medium: crate::io::ReadMedium,
        bytes: usize,
    ) -> crate::io::ReadAheadCharge {
        self.read_ahead_budget(medium).charge(bytes)
    }

    /// Give this worker a read-ahead budget of `budget_bytes` for `medium`,
    /// so a test controls how much it may ask for.
    #[cfg(any(test, feature = "test-util"))]
    pub fn force_read_ahead_budget(&mut self, medium: crate::io::ReadMedium, budget_bytes: usize) {
        let budget = crate::io::ReadAheadBudget::of_bytes(budget_bytes);
        match medium {
            crate::io::ReadMedium::Disk => self.disk_read_ahead = budget,
            crate::io::ReadMedium::Network => self.network_read_ahead = budget,
        }
    }

    /// Number of HTTP operations issued but not yet completed — the depth a
    /// worker uses to decide whether to submit more.
    pub fn http_in_flight(&self) -> usize {
        self.http.network_in_flight()
    }

    /// Resolve reads following another fill of their extent. A successful fill
    /// only completes the logical read. A failed one is pushed onto `out` as
    /// a failed read, so the caller fails its logical read and the worker
    /// cancels the affected dataflow exactly as for a read this requester
    /// submitted itself.
    fn resolve_piggybacked(&mut self, out: &mut Vec<std::result::Result<Completion, FailedIO>>) {
        let mut index = 0;
        while index < self.piggybacked_reads.len() {
            let request = &self.piggybacked_reads[index];
            if request.missing_extent().has_failed() {
                let request = self.piggybacked_reads.swap_remove(index);
                let (data_flow_id, operator_idx) = match &request {
                    IORead::Fs(request) => (request.data_flow_id, request.operator_idx),
                    IORead::Http(request) => (request.data_flow_id, request.operator_idx),
                };
                out.push(Err(FailedIO {
                    data_flow_id,
                    operator_idx,
                    pending_read_id: Some(request.pending_read_id()),
                    error: Error::PiggybackedReadFailed,
                }));
            } else if request.missing_extent().is_committed() {
                let request = self.piggybacked_reads.swap_remove(index);
                let pending_read_id = request.pending_read_id();
                self.complete_pending_read(pending_read_id);
            } else {
                index += 1;
            }
        }
    }

    /// Drain finished reads, one per-read result each. The outer `Result` is for
    /// genuine ring-machinery failures; each inner result is `Ok` for a read
    /// whose bytes landed (block committed, request yielded as a [`Completion`]
    /// with the transport kind preserved) or `Err` for one that failed terminally
    /// (a [`FailedIO`] carrying the issuing dataflow and the error). A single
    /// failed read therefore never aborts the drain or tears down the worker —
    /// the worker cancels just the owning dataflow.
    ///
    /// The single per-core ring carries everything: this requester's fs reads,
    /// the [`CachedHttpEngine`]'s cache-file reads / write-backs, and its HTTP
    /// sockets. We drain it once and route each completion to its owner.
    pub fn completions(&mut self) -> Result<Vec<std::result::Result<Completion, FailedIO>>> {
        self.collect_completions()?;
        Ok(self.take_completions())
    }

    /// Every completion collected so far, leaving none behind.
    pub fn take_completions(&mut self) -> Vec<std::result::Result<Completion, FailedIO>> {
        std::mem::take(&mut self.completed)
    }

    /// Only the collected completions that `wanted` accepts, leaving the rest
    /// for [`take_completions`](Self::take_completions). For a caller that
    /// must handle its own completions now and no others, such as eviction
    /// waiting on writes from inside an operator's run.
    pub fn extract_completions_if(
        &mut self,
        mut wanted: impl FnMut(&std::result::Result<Completion, FailedIO>) -> bool,
    ) -> Vec<std::result::Result<Completion, FailedIO>> {
        self.completed
            .extract_if(.., |completion| wanted(completion))
            .collect()
    }

    /// Reap the backend's finished operations into the kept completions (see
    /// [`completions`](Self::completions) for what each one means), resolving
    /// the requester's own bookkeeping on the way: the next run of a partly
    /// written payload is submitted, cache fills are committed, and logical
    /// reads whose last fill landed become ready.
    pub fn collect_completions(&mut self) -> Result<()> {
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
            if let Some(InFlightFsOp {
                request,
                bytes_done: done,
                permit,
                _write_runs: _,
            }) = self.pending_io_requests.remove(&ud)
            {
                // A write may complete short (the kernel caps a single write
                // at ~2 GiB and at a number of runs, among other reasons);
                // resubmit from where it stopped, at the matching file offset,
                // until the whole payload lands. Zero progress means the file
                // accepts no more bytes, which is an error, not a retry.
                if let FsRequest::Write(write) = &request.request {
                    let done = done + result.max(0) as usize;
                    if result > 0 && done < write.data.len() {
                        let write_runs = self.submit_rest_of_write(write, done)?;
                        self.pending_io_requests.insert(
                            self.next_id,
                            InFlightFsOp {
                                request,
                                bytes_done: done,
                                permit,
                                _write_runs: write_runs,
                            },
                        );
                        self.next_id += 1;
                        self.backend.submit()?;
                        continue;
                    }
                }
                // A direct read may complete short as well, with the rest of
                // its range still in the file; read on from where it stopped.
                // Zero progress means the file ends before the range does.
                if let FsRequest::ReadDirect(read) = &request.request {
                    let done = done + result.max(0) as usize;
                    if result > 0 && done < read.len {
                        self.submit_rest_of_direct_read(read, done)?;
                        self.pending_io_requests.insert(
                            self.next_id,
                            InFlightFsOp {
                                request,
                                bytes_done: done,
                                permit,
                                _write_runs: None,
                            },
                        );
                        self.next_id += 1;
                        self.backend.submit()?;
                        continue;
                    }
                }
                let failure = match &request.request {
                    FsRequest::Read(_) | FsRequest::ReadDirect(_) if result < 0 => {
                        Some(std::io::Error::from_raw_os_error(-result))
                    }
                    // A direct read asks for bytes its caller knows are in
                    // the file, so a file that ends short of them is an error.
                    FsRequest::ReadDirect(read) if done + (result as usize) < read.len => {
                        Some(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "direct read stopped short of the requested range",
                        ))
                    }
                    // Cache fills are block-aligned, so a tail fill can
                    // legitimately return fewer bytes than its physical extent
                    // when it reaches EOF. Accept a short result only when it
                    // stopped exactly at the file's current EOF (ruling out an
                    // unexpected mid-file short read) and the returned prefix
                    // reached the extent's final block (so no whole cache block
                    // remains unwritten). Because the extent contains exactly
                    // the blocks intersecting the logical range, stopping at EOF
                    // before its final block means the logical range is out of
                    // bounds against the current file size, possibly because the
                    // file shrank after the range was formed.
                    FsRequest::Read(read) if (result as usize) < read.block.len() => {
                        match read.file.metadata() {
                            Err(error) => Some(error),
                            Ok(meta)
                                if result > 0
                                    && read.block.file_offset() as u64 + result as u64
                                        == meta.len()
                                    && read.block.prefix_reaches_last_block(result as usize) =>
                            {
                                None
                            }
                            Ok(_) => Some(std::io::Error::new(
                                std::io::ErrorKind::UnexpectedEof,
                                "file read stopped short of the requested range",
                            )),
                        }
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
                    if let FsRequest::Read(read) = &request.request {
                        read.block
                            .remove_from_cache(&OpenFile::Local(read.file.clone()));
                    }
                    let data_flow_id = request.data_flow_id;
                    let operator_idx = request.operator_idx;
                    out.push(Err(FailedIO {
                        data_flow_id,
                        operator_idx,
                        pending_read_id: request.pending_read_id,
                        error: error.into(),
                    }));
                } else {
                    let DataFlowRequest {
                        data_flow_id,
                        operator_idx,
                        request: fs_request,
                        pending_read_id,
                        submitted_at,
                    } = request;
                    let completion = match fs_request {
                        FsRequest::Read(read) => {
                            // Full reads cover the extent. An accepted short read
                            // reached EOF inside its final block, so committing
                            // the touched prefix commits every extent block.
                            read.block.commit_prefix(result as usize);
                            Completion::FsRead(DataFlowRequest {
                                data_flow_id,
                                operator_idx,
                                request: read,
                                pending_read_id,
                                submitted_at,
                            })
                        }
                        FsRequest::ReadDirect(read) => {
                            let range = FileRange::new(read.offset as usize, read.len);
                            if let Some(id) = pending_read_id {
                                self.land_direct_read(id, read.slab.freeze());
                            }
                            Completion::FsReadDirect(DataFlowRequest {
                                data_flow_id,
                                operator_idx,
                                request: range,
                                pending_read_id,
                                submitted_at,
                            })
                        }
                        FsRequest::Write(write) => Completion::FsWrite(DataFlowRequest {
                            data_flow_id,
                            operator_idx,
                            request: write,
                            pending_read_id,
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

        // The completions above released their hardware-queue slots (and this
        // worker may have been ring-woken by another worker's release); put
        // backlogged operations on the device before doing anything else with
        // this pass.
        self.drain_fs_backlog()?;

        // HTTP reads the engine finished this pass (and any write-backs they queue).
        self.http
            .drain(&mut self.backend, &mut self.next_id, &mut out)?;
        for (physical_read_id, filled_slab) in self.http.take_landed_direct_reads() {
            self.land_direct_read(physical_read_id, filled_slab);
        }

        // Reads waiting on fills the ring did not carry for this requester.
        self.resolve_piggybacked(&mut out);

        // Physical transport is complete. Resolve the requester's private
        // logical physical_read_ids_to_pending_reads_indexes before the worker sees completions or failures.
        for completion in &out {
            match completion {
                Ok(Completion::FsRead(request)) => {
                    request.request.block.wake_subscribers();
                    if let Some(id) = request.pending_read_id {
                        self.complete_pending_read(id);
                    }
                }
                Ok(Completion::HttpGet(request, _)) => {
                    request.request.block.wake_subscribers();
                    if let Some(id) = request.pending_read_id {
                        self.complete_pending_read(id);
                    }
                }
                Ok(Completion::FsReadDirect(request) | Completion::HttpGetDirect(request)) => {
                    if let Some(id) = request.pending_read_id {
                        self.complete_pending_read(id);
                    }
                }
                Ok(Completion::FsWrite(_) | Completion::HttpUpload(_)) => {}
                Err(failed) => {
                    if let Some(id) = failed.pending_read_id {
                        let _ = self.fail_pending_read(id);
                    }
                }
            }
        }

        self.completed.append(&mut out);
        Ok(())
    }

    /// Blocks until at least one pending read (disk or HTTP) makes progress. On
    /// Linux both ride one ring, so a single `submit_and_wait` wakes on either.
    #[cfg(target_os = "linux")]
    pub fn wait(&mut self) -> Result<()> {
        // Nothing registered may be left unissued across a wait: its
        // completion is what the wait would be for.
        self.http.issue_registered_fetches(&mut self.backend)?;
        // A signal, such as a profiler or debugger attaching, interrupts the
        // wait before anything completed; that is not progress or an error.
        loop {
            match self.backend.submit_and_wait(1) {
                Ok(_) => return Ok(()),
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error.into()),
            }
        }
    }

    /// Non-Linux disk and HTTP completions arrive on independent channels, so park
    /// until either is ready (without consuming it; `completions` drains both),
    /// matching the Linux single-ring wake on either kind.
    #[cfg(not(target_os = "linux"))]
    pub fn wait(&mut self) -> Result<()> {
        // Nothing registered may be left unissued across a wait: its
        // completion is what the wait would be for.
        self.http.issue_registered_fetches(&mut self.backend)?;
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

/// The cache extent a queued filesystem operation registered, paired with the
/// file it belongs to, so a failed batch can remove it again. Writes register
/// nothing.
/// Resolve each of `request`'s ranges through the compressed cache it names,
/// one lookup list per range. This allocates cache slots for the ranges that are
/// missing, and allocating can evict, so it runs before the requester is
/// borrowed: eviction may need the requester to wait on spill writes.
pub(crate) fn resolve_read_through_cache(request: &PendingReadRequest) -> Vec<Vec<CacheLookup>> {
    request
        .locations
        .iter()
        .map(|location| {
            memory_ctx().caches().compressed(request.cache).get(
                &request.open_file,
                location.offset,
                location.len,
            )
        })
        .collect()
}

fn fs_cache_extent(request: &DataFlowRequest<FsRequest>) -> Option<(OpenFile, MissingExtent)> {
    request
        .request
        .as_read()
        .map(|read| (OpenFile::Local(read.file.clone()), read.block.clone()))
}

/// The HTTP counterpart of [`fs_cache_extent`]: the extent a range GET
/// registered. Uploads register nothing.
fn http_cache_extent(request: &DataFlowRequest<HttpRequest>) -> Option<(OpenFile, MissingExtent)> {
    match &request.request {
        HttpRequest::Get(read) => Some((OpenFile::Remote(read.remote.clone()), read.block.clone())),
        HttpRequest::GetDirect(_) | HttpRequest::Upload(_) => None,
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
    use crate::io::UploadMethod;
    use crate::io::hardware_queues::HardwareQueue;
    use crate::io::{
        FileRange, HttpGetRequest, HttpUploadRequest, LocalFile, OpenFile, PendingReadRequest,
        ReadRequestId, RemoteFile, open_direct_read,
    };
    use crate::memory::cache::{CacheKey, DecompressedPageMeta, RingCache};
    use crate::memory::{init_test_free_pool, memory_ctx};
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::AtomicUsize;
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};
    use tempfile::TempDir;
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

    fn pending_local_read(open_file: OpenFile) -> PendingReadRequest {
        PendingReadRequest {
            id: ReadRequestId(0),
            open_file,
            locations: vec![FileRange::new(0, 4096)],
            cache: get_compressed_cache_id(),
        }
    }

    fn disabled_stats() -> StatsCollector {
        let (tx, _rx) = mpsc::channel();
        StatsCollector::new(tx, false)
    }

    #[test]
    fn reads_past_the_in_flight_cap_wait_their_turn_and_still_complete() {
        init_test_free_pool(4);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        let capacity = 8;
        let blocks = capacity + 4;
        let content: Vec<u8> = (0..blocks).flat_map(|b| vec![b as u8; 4096]).collect();
        std::fs::write(&path, &content).unwrap();
        let open_file =
            OpenFile::Local(LocalFile::new(std::fs::File::open(path).unwrap()).unwrap());
        let mut requester = IORequester::default();
        requester.force_fs_queue(HardwareQueue::with_capacity(capacity));

        for block in 0..blocks {
            requester
                .request_read_resolving(
                    1,
                    2,
                    PendingReadRequest {
                        id: ReadRequestId(block),
                        open_file: open_file.clone(),
                        locations: vec![FileRange::new(block * 4096, 4096)],
                        cache: get_compressed_cache_id(),
                    },
                    &mut disabled_stats(),
                )
                .unwrap();
        }
        let mut responses = Vec::new();
        for _ in 0..100 {
            if responses.len() == blocks {
                break;
            }
            requester.wait().unwrap();
            requester.completions().unwrap();
            responses.extend(requester.take_ready_reads());
        }

        assert_eq!(responses.len(), blocks);
        for routed in responses {
            let block = routed.response.id().0;
            assert_eq!(routed.response.into_bytes(), vec![block as u8; 4096]);
        }
    }

    /// A direct read of `ranges`, each into a slab of its own.
    fn pending_direct_read(
        file: &crate::io::LocalFile,
        ranges: &[FileRange],
    ) -> PendingDirectReadRequest {
        let mut allocator = crate::memory::SlabAllocator::new(false);
        PendingDirectReadRequest {
            id: ReadRequestId(0),
            file: OpenFile::Local(file.clone()),
            reads: ranges
                .iter()
                .map(|range| SlabRead {
                    offset: range.offset,
                    slab: allocator.get_slab_of_size(range.len, false),
                    len: range.len,
                })
                .collect(),
        }
    }

    #[test]
    fn a_direct_read_brings_back_exactly_the_ranges_asked_for() {
        init_test_free_pool(2);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        std::fs::write(&path, (0..5500).map(pattern).collect::<Vec<_>>()).unwrap();
        let file = LocalFile::new(std::fs::File::open(path).unwrap()).unwrap();
        let ranges = [FileRange::new(100, 3000), FileRange::new(4000, 1500)];
        let mut requester = IORequester::default();

        requester
            .request_read_direct(
                1,
                2,
                pending_direct_read(&file, &ranges),
                &mut disabled_stats(),
            )
            .unwrap();
        let mut responses = Vec::new();
        for _ in 0..100 {
            if !responses.is_empty() {
                break;
            }
            requester.wait().unwrap();
            requester.completions().unwrap();
            responses.extend(requester.take_ready_reads());
        }

        let locations = responses.pop().unwrap().response.into_locations();
        assert_eq!(locations.len(), 2);
        for (range, data) in ranges.iter().zip(locations) {
            assert_eq!(data.offset, range.offset);
            let bytes: Vec<u8> = data
                .bytes
                .iter()
                .flat_map(|run| run.iter().copied())
                .collect();
            assert_eq!(
                bytes,
                (range.offset..range.offset + range.len)
                    .map(pattern)
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn a_direct_read_that_comes_back_short_goes_on_from_where_it_stopped() {
        init_test_free_pool(1);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        std::fs::write(&path, (0..1000).map(pattern).collect::<Vec<_>>()).unwrap();
        let file = LocalFile::new(std::fs::File::open(&path).unwrap()).unwrap();
        let mut requester = IORequester::default();
        requester
            .request_read_direct(
                1,
                2,
                pending_direct_read(&file, &[FileRange::new(500, 1000)]),
                &mut disabled_stats(),
            )
            .unwrap();
        // The read has come back with the 500 bytes the file had; the rest of
        // its range is written before that result is collected.
        requester.wait().unwrap();
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(&(1000..1500).map(pattern).collect::<Vec<_>>())
            .unwrap();

        let mut responses = Vec::new();
        for _ in 0..100 {
            if !responses.is_empty() {
                break;
            }
            requester.completions().unwrap();
            responses.extend(requester.take_ready_reads());
            if responses.is_empty() {
                requester.wait().unwrap();
            }
        }

        let locations = responses.pop().unwrap().response.into_locations();
        let bytes: Vec<u8> = locations[0]
            .bytes
            .iter()
            .flat_map(|run| run.iter().copied())
            .collect();
        assert_eq!(bytes, (500..1500).map(pattern).collect::<Vec<_>>());
    }

    #[test]
    fn a_direct_read_past_the_end_of_the_file_fails() {
        init_test_free_pool(1);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        std::fs::write(&path, vec![1u8; 1000]).unwrap();
        let file = LocalFile::new(std::fs::File::open(path).unwrap()).unwrap();
        let mut requester = IORequester::default();

        requester
            .request_read_direct(
                1,
                2,
                pending_direct_read(&file, &[FileRange::new(500, 1000)]),
                &mut disabled_stats(),
            )
            .unwrap();
        let mut failures = Vec::new();
        for _ in 0..100 {
            if !failures.is_empty() {
                break;
            }
            requester.wait().unwrap();
            failures.extend(
                requester
                    .completions()
                    .unwrap()
                    .into_iter()
                    .filter_map(Result::err),
            );
        }

        assert_eq!(failures.len(), 1);
        assert!(failures[0].error.to_string().contains("stopped short"));
        assert!(requester.take_ready_reads().is_empty());
    }

    #[test]
    fn a_released_shared_queue_slot_wakes_the_waiting_worker() {
        init_test_free_pool(4);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        std::fs::write(&path, [vec![1u8; 4096], vec![2u8; 4096]].concat()).unwrap();
        let open_file =
            OpenFile::Local(LocalFile::new(std::fs::File::open(path).unwrap()).unwrap());
        let shared_queue = HardwareQueue::with_capacity(1);
        let mut first = IORequester::default();
        let mut second = IORequester::default();
        first.force_fs_queue(shared_queue.clone());
        second.force_fs_queue(shared_queue);

        first
            .request_read_resolving(
                1,
                2,
                PendingReadRequest {
                    id: ReadRequestId(0),
                    open_file: open_file.clone(),
                    locations: vec![FileRange::new(0, 4096)],
                    cache: get_compressed_cache_id(),
                },
                &mut disabled_stats(),
            )
            .unwrap();
        second
            .request_read_resolving(
                3,
                4,
                PendingReadRequest {
                    id: ReadRequestId(1),
                    open_file,
                    locations: vec![FileRange::new(4096, 4096)],
                    cache: get_compressed_cache_id(),
                },
                &mut disabled_stats(),
            )
            .unwrap();
        let second_backlogged = second.pending_io_requests.is_empty() && second.has_file_pending();
        first.wait().unwrap();
        let waker = crate::waker::worker_waker();
        waker.register_ring_waker(0, second.wake_handle());
        let last_seen = waker.ring_wake_count();
        assert!(waker.begin_ring_wait(0, last_seen));
        first.completions().unwrap(); // completes the read, releasing the slot and waking `second`
        second.wait().unwrap(); // returns on the wake, with nothing of its own in flight
        waker.end_ring_wait(0);
        second.completions().unwrap(); // the drain puts the backlogged read on the device
        let mut responses = Vec::new();
        for _ in 0..100 {
            if !responses.is_empty() {
                break;
            }
            second.wait().unwrap();
            second.completions().unwrap();
            responses.extend(second.take_ready_reads());
        }

        assert!(second_backlogged);
        let response = responses.pop().unwrap().response;
        assert_eq!(response.into_bytes(), vec![2u8; 4096]);
    }

    #[test]
    fn a_read_piggybacks_on_an_existing_fill() {
        init_test_free_pool(4);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        std::fs::write(&path, vec![7; 4096]).unwrap();
        let open_file =
            OpenFile::Local(LocalFile::new(std::fs::File::open(path).unwrap()).unwrap());
        let mut owner = IORequester::default();
        let mut follower = IORequester::default();

        owner
            .request_read_resolving(
                1,
                2,
                pending_local_read(open_file.clone()),
                &mut disabled_stats(),
            )
            .unwrap();
        follower
            .request_read_resolving(3, 4, pending_local_read(open_file), &mut disabled_stats())
            .unwrap();
        let owner_has_physical_read = owner.has_file_pending();
        let follower_has_physical_read = follower.has_file_pending();
        owner.wait().unwrap();
        let owner_completions = owner.completions().unwrap();
        let follower_completions = follower.completions().unwrap();
        let response = follower.take_ready_reads().pop().unwrap().response;

        assert!(owner_has_physical_read);
        assert!(!follower_has_physical_read);
        assert_eq!(owner_completions.len(), 1);
        assert!(follower_completions.is_empty());
        assert_eq!(response.into_bytes(), vec![7; 4096]);
    }

    #[test]
    fn a_failed_fill_fails_its_follower() {
        init_test_free_pool(4);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        std::fs::write(&path, vec![0; 4096]).unwrap();
        let write_only = std::fs::OpenOptions::new().write(true).open(path).unwrap();
        let open_file = OpenFile::Local(LocalFile::new(write_only).unwrap());
        let mut owner = IORequester::default();
        let mut follower = IORequester::default();
        let mut retry = IORequester::default();

        owner
            .request_read_resolving(
                1,
                2,
                pending_local_read(open_file.clone()),
                &mut disabled_stats(),
            )
            .unwrap();
        follower
            .request_read_resolving(
                3,
                4,
                pending_local_read(open_file.clone()),
                &mut disabled_stats(),
            )
            .unwrap();
        owner.wait().unwrap();
        let owner_failure = owner.completions().unwrap().pop().unwrap();
        let follower_failure = follower.completions().unwrap().pop().unwrap();
        let follower_failed_from_owner = matches!(
            follower_failure,
            Err(FailedIO {
                error: Error::PiggybackedReadFailed,
                ..
            })
        );
        retry
            .request_read_resolving(5, 6, pending_local_read(open_file), &mut disabled_stats())
            .unwrap();

        assert!(owner_failure.is_err());
        assert!(follower_failed_from_owner);
        assert!(retry.has_file_pending());
    }

    #[test]
    fn a_failed_submit_withdraws_its_registration() {
        init_test_free_pool(4);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        std::fs::write(&path, vec![0; 4096]).unwrap();
        let readable = std::fs::File::open(path).unwrap();
        let open_file = OpenFile::Local(LocalFile::new(readable).unwrap());
        let mut owner = IORequester::default();
        let mut retry = IORequester::default();
        owner.backend.reject_reads();

        let owner_result = owner.request_read_resolving(
            1,
            2,
            pending_local_read(open_file.clone()),
            &mut disabled_stats(),
        );
        retry
            .request_read_resolving(3, 4, pending_local_read(open_file), &mut disabled_stats())
            .unwrap();

        assert!(owner_result.is_err());
        assert!(!owner.has_pending());
        assert!(retry.has_file_pending());
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

    /// Read one whole upload request off `stream` and return its body.
    fn read_upload_body(stream: &mut std::net::TcpStream) -> Vec<u8> {
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
        wire[head_len..head_len + content_len].to_vec()
    }

    /// An upload payload held as ring memory, in two runs, so the upload path
    /// is exercised the way a real file reaches it rather than as one buffer.
    fn build_upload_payload() -> Arc<crate::memory::FileBytes> {
        crate::memory::init_test_free_pool(4);
        let mut allocator = crate::memory::SlabAllocator::new(false);
        let mut bytes = crate::memory::FileBytes::new();
        for part in [&b"parquet upload"[..], &b" bytes"[..]] {
            let mut run = allocator.get_slab_of_size(part.len(), false);
            run.as_mut_slice().copy_from_slice(part);
            bytes.push_slab(run);
        }
        Arc::new(bytes)
    }

    #[test]
    fn plain_http_upload_uses_shared_connection_engine() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let received = Arc::new(Mutex::new(Vec::new()));
        let server_received = received.clone();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let body = read_upload_body(&mut stream);
            server_received.lock().unwrap().extend_from_slice(&body);
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n",
                )
                .unwrap();
        });

        let data = build_upload_payload();
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
                method: UploadMethod::Put,
                etag: None,
                response_body: Vec::new(),
            }),
        );
        upload.submitted_at = Some(submitted_at);
        let split = requester.submit_http_request_to_backend(upload).unwrap();
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

    /// A `POST` whose body is the test payload, to the listener at `addr`.
    fn build_post(addr: std::net::SocketAddr) -> DataFlowRequest<HttpRequest> {
        let data = build_upload_payload();
        let remote = Arc::new(
            RemoteFile::open(
                Url::parse(&format!("http://{addr}/object.parquet?uploads=")).unwrap(),
                None,
                data.len() as u64,
            )
            .unwrap(),
        );
        DataFlowRequest::new(
            7,
            11,
            HttpRequest::Upload(HttpUploadRequest {
                remote,
                data,
                method: UploadMethod::Post,
                etag: None,
                response_body: Vec::new(),
            }),
        )
    }

    #[test]
    fn a_post_is_sent_as_a_post_and_its_chunked_answer_comes_back() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request_line = [0u8; 5];
            stream.read_exact(&mut request_line).unwrap();
            read_upload_body(&mut stream);
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n\
                      6\r\n<Uploa\r\n7\r\ndId>7</\r\n0\r\n\r\n",
                )
                .unwrap();
            request_line
        });
        let mut requester = IORequester::default();

        requester
            .submit_http_request_to_backend(build_post(addr))
            .unwrap();
        let completion = loop {
            if let Some(completion) = requester.completions().unwrap().into_iter().next() {
                break completion;
            }
            requester.wait().unwrap();
        };

        let Ok(Completion::HttpUpload(request)) = completion else {
            panic!("expected a completed POST");
        };
        assert_eq!(request.request.response_body, b"<UploadId>7</");
        assert_eq!(&server.join().unwrap(), b"POST ");
    }

    #[test]
    fn a_post_answered_with_a_server_error_fails_without_being_sent_again() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut overloaded, _) = listener.accept().unwrap();
            read_upload_body(&mut overloaded);
            overloaded
                .write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n")
                .unwrap();
            listener.set_nonblocking(true).unwrap();
            thread::sleep(std::time::Duration::from_millis(200));
            listener.accept().is_ok()
        });
        let mut requester = IORequester::default();

        requester
            .submit_http_request_to_backend(build_post(addr))
            .unwrap();
        let completion = loop {
            if let Some(completion) = requester.completions().unwrap().into_iter().next() {
                break completion;
            }
            requester.wait().unwrap();
        };

        assert!(completion.is_err(), "the POST fails");
        let was_sent_again = server.join().unwrap();
        assert!(!was_sent_again);
    }

    /// Only the io_uring engine re-issues a failed request.
    #[cfg(target_os = "linux")]
    #[test]
    fn upload_is_retried_after_a_server_error() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut overloaded, _) = listener.accept().unwrap();
            read_upload_body(&mut overloaded);
            overloaded
                .write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n")
                .unwrap();
            let (mut recovered, _) = listener.accept().unwrap();
            let body = read_upload_body(&mut recovered);
            recovered
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
                .unwrap();
            body
        });
        let data = build_upload_payload();
        let url = Url::parse(&format!("http://{addr}/object.parquet")).unwrap();
        let remote = Arc::new(RemoteFile::open(url, None, data.len() as u64).unwrap());
        let upload = HttpRequest::Upload(HttpUploadRequest {
            remote,
            data: data.clone(),
            method: UploadMethod::Put,
            etag: None,
            response_body: Vec::new(),
        });
        let mut requester = IORequester::default();

        requester
            .submit_http_request_to_backend(DataFlowRequest::new(7, 11, upload))
            .unwrap();
        let completion = loop {
            if let Some(completion) = requester.completions().unwrap().into_iter().next() {
                break completion;
            }
            requester.wait().unwrap();
        };

        assert!(matches!(completion, Ok(Completion::HttpUpload(_))));
        let sent: Vec<u8> = data.runs().flatten().copied().collect();
        assert_eq!(server.join().unwrap(), sent);
    }

    /// A store closes a keep-alive connection it has not heard from, and may
    /// never answer what is sent on it afterwards. The next upload goes out
    /// on a new connection instead of the closed one.
    #[cfg(target_os = "linux")]
    #[test]
    fn an_upload_avoids_a_pooled_connection_the_store_closed() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (closed_sender, closed_receiver) = std::sync::mpsc::channel();
        let server = thread::spawn(move || {
            let (mut closed, _) = listener.accept().unwrap();
            read_upload_body(&mut closed);
            closed
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
                .unwrap();
            // Closed for sending only: the socket still takes what arrives on
            // it and answers none of it.
            closed.shutdown(std::net::Shutdown::Write).unwrap();
            closed_sender.send(()).unwrap();
            let (mut fresh, _) = listener.accept().unwrap();
            let body = read_upload_body(&mut fresh);
            fresh
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
                .unwrap();
            (closed, body)
        });
        // More than the two sockets hold between them, so an upload that
        // nobody reads cannot finish sending.
        init_test_free_pool(20);
        let mut allocator = crate::memory::SlabAllocator::new(false);
        let mut bytes = crate::memory::FileBytes::new();
        for _ in 0..16 {
            bytes.push_slab(allocator.get_slab_of_size(crate::memory::BUFFER_SIZE, true));
        }
        let data = Arc::new(bytes);
        let url = Url::parse(&format!("http://{addr}/object.parquet")).unwrap();
        let remote = Arc::new(RemoteFile::open(url, None, data.len() as u64).unwrap());
        let mut requester = IORequester::default();
        let upload_and_wait = |requester: &mut IORequester| {
            let upload = HttpRequest::Upload(HttpUploadRequest {
                remote: remote.clone(),
                data: data.clone(),
                method: UploadMethod::Put,
                etag: None,
                response_body: Vec::new(),
            });
            requester
                .submit_http_request_to_backend(DataFlowRequest::new(7, 11, upload))
                .unwrap();
            loop {
                if let Some(completion) = requester.completions().unwrap().into_iter().next() {
                    break completion;
                }
                requester.wait().unwrap();
            }
        };
        assert!(upload_and_wait(&mut requester).is_ok());
        closed_receiver.recv().unwrap();
        // The close crosses the loopback after the server's call returns.
        thread::sleep(Duration::from_millis(50));

        let completion = upload_and_wait(&mut requester);

        assert!(matches!(completion, Ok(Completion::HttpUpload(_))));
        let sent: Vec<u8> = data.runs().flatten().copied().collect();
        assert_eq!(server.join().unwrap().1, sent);
    }

    #[test]
    fn upload_completion_carries_the_response_etag() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            read_upload_body(&mut stream);
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nETag: \"part-1\"\r\nContent-Length: 0\r\n\r\n")
                .unwrap();
        });
        let data = build_upload_payload();
        let url = Url::parse(&format!("http://{addr}/object.parquet")).unwrap();
        let remote = Arc::new(RemoteFile::open(url, None, data.len() as u64).unwrap());
        let upload = HttpRequest::Upload(HttpUploadRequest {
            remote,
            data,
            method: UploadMethod::Put,
            etag: None,
            response_body: Vec::new(),
        });
        let mut requester = IORequester::default();

        requester
            .submit_http_request_to_backend(DataFlowRequest::new(7, 11, upload))
            .unwrap();
        let completion = loop {
            if let Some(completion) = requester.completions().unwrap().into_iter().next() {
                break completion;
            }
            requester.wait().unwrap();
        };

        let Ok(Completion::HttpUpload(request)) = completion else {
            panic!("expected an HTTP upload completion");
        };
        assert_eq!(request.request.etag.as_deref(), Some("\"part-1\""));
        server.join().unwrap();
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

    /// Spawn a server whose first connection is accepted and then never
    /// answered, as when the peer's reply is lost on the way, and whose second
    /// connection serves one range GET.
    #[cfg(target_os = "linux")]
    fn spawn_silent_then_serving_server() -> u16 {
        let server_config = server_tls_config();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        thread::spawn(move || {
            let (_silent, _) = listener.accept().unwrap();
            let (tcp, _) = listener.accept().unwrap();
            let conn = rustls::ServerConnection::new(server_config).unwrap();
            let mut tls = rustls::StreamOwned::new(conn, tcp);
            serve_one_range(&mut tls);
        });

        port
    }

    /// Spawn a server that accepts every connection and never answers on any.
    #[cfg(target_os = "linux")]
    fn spawn_silent_server() -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        thread::spawn(move || {
            let _silent_connections: Vec<_> = listener.incoming().collect();
        });

        port
    }

    /// Shared log of the `[start, end]` ranges a recording server was asked for.
    type RangeLog = Arc<Mutex<Vec<(usize, usize)>>>;

    /// Spawn a server that records every range it's asked for (so a test can
    /// assert exactly which bytes went to the network) and serves them on as
    /// many keep-alive connections as it is given. Returns the port and the
    /// shared request log.
    fn spawn_recording_server() -> (u16, RangeLog) {
        let server_config = server_tls_config();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let log = Arc::new(Mutex::new(Vec::new()));
        let log_for_thread = log.clone();

        thread::spawn(move || {
            for tcp in listener.incoming() {
                let server_config = server_config.clone();
                let log = log_for_thread.clone();
                thread::spawn(move || {
                    let conn = rustls::ServerConnection::new(server_config).unwrap();
                    let mut tls = rustls::StreamOwned::new(conn, tcp.unwrap());
                    loop {
                        let head = read_head(&mut tls);
                        if head.is_empty() {
                            break; // client closed the connection
                        }
                        let (start, end) = parse_range(&head);
                        log.lock().unwrap().push((start, end));
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
        let lookups = memory_ctx().page_caches().compressed.get(loc, offset, len);
        let mut split = RemoteSplit::default();
        let mut submitted = 0;
        for lookup in &lookups {
            if let Some(block) = lookup.missing_extent() {
                let req = HttpRequest::Get(HttpGetRequest {
                    remote: remote.clone(),
                    block: block.clone(),
                });
                let piece = requester
                    .submit_http_request_to_backend(DataFlowRequest::new(0, 0, req))
                    .unwrap();
                split.disk_cache_requests += piece.disk_cache_requests;
                split.disk_cache_bytes += piece.disk_cache_bytes;
                split.http_requests += piece.http_requests;
                split.http_bytes += piece.http_bytes;
                submitted += 1;
            }
        }
        assert!(submitted > 0, "expected blocks to read");
        split.http_requests += requester
            .http
            .issue_registered_fetches(&mut requester.backend)
            .unwrap();

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
        let hit = memory_ctx().page_caches().compressed.get(loc, offset, len);
        assert_eq!(hit.len(), 1);
        assert!(hit[0].missing_extent().is_none(), "expected a cache hit");
        let bytes = hit.into_iter().next().unwrap().into_data();
        assert_eq!(bytes.len(), len);
        for (i, &b) in bytes.iter().enumerate() {
            assert_eq!(b, pattern(offset + i), "byte {i}");
        }
    }

    /// Read every range of `ranges` from `loc` as the parts of one logical
    /// read, and block until all of them have landed.
    fn fetch_together(requester: &mut IORequester, loc: &OpenFile, ranges: &[(usize, usize)]) {
        let OpenFile::Remote(remote) = loc else {
            panic!("test fetches over http")
        };
        let mut lookups = Vec::new();
        for &(offset, len) in ranges {
            lookups.extend(memory_ctx().page_caches().compressed.get(loc, offset, len));
        }
        for lookup in &lookups {
            let block = lookup.missing_extent().expect("an uncached range").clone();
            let request = HttpRequest::Get(HttpGetRequest {
                remote: remote.clone(),
                block,
            });
            requester
                .submit_http_request_to_backend(DataFlowRequest::new(0, 0, request))
                .unwrap();
        }
        settle(requester);
    }

    /// Two ranges of one read that touch in the object are one request.
    #[test]
    fn ranges_next_to_each_other_are_fetched_in_one_request() {
        init_test_free_pool(16);
        let (port, requested) = spawn_recording_server();
        let loc = remote_loc(port);
        let mut requester = IORequester::with_http_config(client_config());

        fetch_together(&mut requester, &loc, &[(0, 4096), (4096, 8192)]);

        assert_eq!(*requested.lock().unwrap(), vec![(0, 12287)]);
        assert_cached(&loc, 0, 4096);
        assert_cached(&loc, 4096, 8192);
    }

    /// Two ranges of one read with no more between them than may be dropped
    /// are one request, and what lies between them is not kept.
    #[test]
    fn ranges_a_little_apart_are_fetched_in_one_request_that_drops_the_gap() {
        init_test_free_pool(16);
        let (port, requested) = spawn_recording_server();
        let loc = remote_loc(port);
        let mut requester = IORequester::with_http_config(client_config());
        requester.http.force_max_dropped_gap_bytes(8192);

        fetch_together(&mut requester, &loc, &[(0, 4096), (12288, 8192)]);

        assert_eq!(*requested.lock().unwrap(), vec![(0, 20479)]);
        assert_cached(&loc, 0, 4096);
        assert_cached(&loc, 12288, 8192);
        let between = memory_ctx().page_caches().compressed.get(&loc, 4096, 8192);
        assert!(
            between
                .iter()
                .all(|lookup| lookup.missing_extent().is_some())
        );
    }

    /// Two ranges of one read that do not touch are fetched on their own.
    #[test]
    fn ranges_apart_are_fetched_separately() {
        init_test_free_pool(16);
        let (port, requested) = spawn_recording_server();
        let loc = remote_loc(port);
        let mut requester = IORequester::with_http_config(client_config());

        fetch_together(&mut requester, &loc, &[(0, 4096), (512 * 1024, 4096)]);

        let mut requested = requested.lock().unwrap().clone();
        requested.sort();
        assert_eq!(requested, vec![(0, 4095), (512 * 1024, 512 * 1024 + 4095)]);
        assert_cached(&loc, 0, 4096);
        assert_cached(&loc, 512 * 1024, 4096);
    }

    #[test]
    fn https_range_read_lands_in_cache_and_reuses_connection() {
        init_test_free_pool(16);
        let port = spawn_server(2);

        let url = Url::parse(&format!("https://127.0.0.1:{port}/obj")).unwrap();
        let remote = Arc::new(RemoteFile::open(url, None, 1 << 20).unwrap());
        let loc = OpenFile::Remote(remote);

        let mut requester = IORequester::with_http_config(client_config());

        // First read: connect + TLS handshake + range GET into the cache slot.
        fetch(&mut requester, &loc, 0, 4096);
        assert_cached(&loc, 0, 4096);

        // Second read of a different region: must reuse the pooled keep-alive
        // connection (the server only accepts one TCP connection).
        fetch(&mut requester, &loc, 4096, 2 * 4096);
        assert_cached(&loc, 4096, 2 * 4096);
    }

    /// Opening a remote object again keeps what memory caches for it: the reopen
    /// is the same object, so its bytes hit without another request.
    #[test]
    fn a_reopened_remote_object_hits_what_memory_cached() {
        init_test_free_pool(16);
        let mut requester = IORequester::with_http_config(client_config());
        let port = spawn_server(1);
        fetch(&mut requester, &remote_loc(port), 0, 4096);
        settle(&mut requester);

        let reopened = remote_loc(port);

        assert_cached(&reopened, 0, 4096);
    }

    #[test]
    fn stale_pooled_connection_is_retried_on_a_fresh_one() {
        init_test_free_pool(16);
        let port = spawn_stale_pool_server();

        let url = Url::parse(&format!("https://127.0.0.1:{port}/obj")).unwrap();
        let remote = Arc::new(RemoteFile::open(url, None, 1 << 20).unwrap());
        let loc = OpenFile::Remote(remote);

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

    /// A read whose connection goes silent is issued again on another one
    /// rather than left waiting on it.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_read_on_a_silent_connection_is_issued_again() {
        init_test_free_pool(16);
        let loc = remote_loc(spawn_silent_then_serving_server());
        let mut requester = IORequester::with_http_config(client_config());

        fetch(&mut requester, &loc, 0, 4096);

        assert_cached(&loc, 0, 4096);
    }

    /// A read against a peer that never answers fails once its deadline has
    /// stopped growing, instead of being issued again without end.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_read_on_a_peer_that_never_answers_fails() {
        init_test_free_pool(16);
        let loc = remote_loc(spawn_silent_server());
        let mut requester = IORequester::with_http_config(client_config());
        requester.force_step_deadline_bounds(Duration::from_millis(1), Duration::from_millis(1));

        let results = fetch_results(&mut requester, &loc, 0, 4096);

        assert!(results.iter().all(|result| result.is_err()));
    }

    /// What a test's name server does with the first questions it is asked.
    #[cfg(target_os = "linux")]
    enum FirstQuestions {
        Answered,
        /// This many get no answer.
        Ignored(usize),
        /// This many are answered with a server failure.
        Failed(usize),
    }

    /// Spawn a name server that answers questions with `addresses`, after
    /// treating the first ones as `first_questions` says. Returns where it
    /// listens and a channel that receives once per question asked.
    #[cfg(target_os = "linux")]
    fn spawn_name_server(
        addresses: Vec<std::net::Ipv4Addr>,
        first_questions: FirstQuestions,
    ) -> (std::net::SocketAddr, std::sync::mpsc::Receiver<()>) {
        let socket = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        let listening_at = socket.local_addr().unwrap();
        let (asked_sender, asked_receiver) = std::sync::mpsc::channel();

        thread::spawn(move || {
            let mut question = [0u8; 512];
            for question_number in 0..usize::MAX {
                let (len, asker) = socket.recv_from(&mut question).unwrap();
                // The test may have stopped listening for questions.
                let _ = asked_sender.send(());
                // The asker's id, the flags of an answer, one question and one
                // record per address, then the question as it was asked.
                let mut answer = question[..2].to_vec();
                answer.extend_from_slice(&[0x81, 0x80, 0, 1, 0, addresses.len() as u8]);
                answer.extend_from_slice(&[0, 0, 0, 0]);
                answer.extend_from_slice(&question[12..len]);
                match first_questions {
                    FirstQuestions::Ignored(count) if question_number < count => continue,
                    FirstQuestions::Failed(count) if question_number < count => {
                        // The code of a server failure, and no records.
                        answer[3] = 0x82;
                        answer[7] = 0;
                    }
                    _ => {
                        for address in &addresses {
                            // A pointer to the name in the question, the address
                            // record type, the internet class, five seconds to
                            // live, four bytes.
                            answer.extend_from_slice(&[0xC0, 12, 0, 1, 0, 1, 0, 0, 0, 5, 0, 4]);
                            answer.extend_from_slice(&address.octets());
                        }
                    }
                }
                socket.send_to(&answer, asker).unwrap();
            }
        });

        (listening_at, asked_receiver)
    }

    /// The object `/obj` of a store reached by the host name `host`.
    #[cfg(target_os = "linux")]
    fn named_remote_loc(host: &str, port: u16) -> OpenFile {
        let url = Url::parse(&format!("https://{host}:{port}/obj")).unwrap();
        OpenFile::Remote(Arc::new(RemoteFile::open(url, None, 1 << 20).unwrap()))
    }

    /// A store named by host is found by asking the name server, with no
    /// lookup when the object is opened.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_read_resolves_its_host_through_the_name_server() {
        init_test_free_pool(16);
        let (name_server, _asked) = spawn_name_server(
            vec![std::net::Ipv4Addr::LOCALHOST],
            FirstQuestions::Answered,
        );
        let loc = named_remote_loc("resolved.store.test", spawn_server(1));
        let mut requester = IORequester::with_http_config(client_config());
        requester.force_name_servers(vec![name_server]);

        fetch(&mut requester, &loc, 0, 4096);

        assert_cached(&loc, 0, 4096);
    }

    /// A question the name server never answers is asked again instead of
    /// being waited on.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_lookup_whose_answer_is_lost_is_asked_again() {
        init_test_free_pool(16);
        let (name_server, asked) = spawn_name_server(
            vec![std::net::Ipv4Addr::LOCALHOST],
            FirstQuestions::Ignored(1),
        );
        let loc = named_remote_loc("lost.store.test", spawn_server(1));
        let mut requester = IORequester::with_http_config(client_config());
        requester.force_name_servers(vec![name_server]);

        fetch(&mut requester, &loc, 0, 4096);

        assert_cached(&loc, 0, 4096);
        assert_eq!(asked.try_iter().count(), 2);
    }

    /// A name server that answers with a failure is asked again instead of
    /// failing the read.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_lookup_a_name_server_fails_is_asked_again() {
        init_test_free_pool(16);
        let (name_server, _asked) = spawn_name_server(
            vec![std::net::Ipv4Addr::LOCALHOST],
            FirstQuestions::Failed(1),
        );
        let loc = named_remote_loc("failed.store.test", spawn_server(1));
        let mut requester = IORequester::with_http_config(client_config());
        requester.force_name_servers(vec![name_server]);

        fetch(&mut requester, &loc, 0, 4096);

        assert_cached(&loc, 0, 4096);
    }

    /// A lookup the first name server stays silent on is answered by the
    /// second.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_lookup_goes_to_the_next_name_server_when_one_is_silent() {
        init_test_free_pool(16);
        let (silent, _asked_silent) =
            spawn_name_server(Vec::new(), FirstQuestions::Ignored(usize::MAX));
        let (answering, _asked) = spawn_name_server(
            vec![std::net::Ipv4Addr::LOCALHOST],
            FirstQuestions::Answered,
        );
        let loc = named_remote_loc("second.store.test", spawn_server(1));
        let mut requester = IORequester::with_http_config(client_config());
        requester.force_name_servers(vec![silent, answering]);

        fetch(&mut requester, &loc, 0, 4096);

        assert_cached(&loc, 0, 4096);
    }

    /// A host no name server answers for fails the read.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_read_of_a_host_no_name_server_answers_for_fails() {
        init_test_free_pool(16);
        let (name_server, _asked) =
            spawn_name_server(Vec::new(), FirstQuestions::Ignored(usize::MAX));
        let loc = named_remote_loc("unanswered.store.test", 443);
        let mut requester = IORequester::with_http_config(client_config());
        requester.force_name_servers(vec![name_server]);

        let results = fetch_results(&mut requester, &loc, 0, 4096);

        assert!(results.iter().all(|result| result.is_err()));
    }

    /// A name server that is not running refuses each question at once. The
    /// attempts are still spaced out, so it has time to come back.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_lookup_waits_before_asking_a_name_server_that_is_down_again() {
        init_test_free_pool(16);
        let nobody_listening = std::net::UdpSocket::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        let loc = named_remote_loc("down.store.test", 443);
        let mut requester = IORequester::with_http_config(client_config());
        requester.force_name_servers(vec![nobody_listening]);
        let started = Instant::now();

        let results = fetch_results(&mut requester, &loc, 0, 4096);

        assert!(results.iter().all(|result| result.is_err()));
        assert!(started.elapsed() >= Duration::from_millis(400));
    }

    /// A host resolves to several addresses, and a read whose connection one
    /// of them refuses goes to the others.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_read_connects_to_another_address_after_one_refuses() {
        init_test_free_pool(16);
        let nobody_listening = [
            std::net::Ipv4Addr::new(127, 0, 0, 2),
            std::net::Ipv4Addr::new(127, 0, 0, 3),
        ];
        let (name_server, _asked) = spawn_name_server(
            vec![
                nobody_listening[0],
                nobody_listening[1],
                std::net::Ipv4Addr::LOCALHOST,
            ],
            FirstQuestions::Answered,
        );
        let loc = named_remote_loc("several.store.test", spawn_server(1));
        let mut requester = IORequester::with_http_config(client_config());
        requester.force_name_servers(vec![name_server]);

        fetch(&mut requester, &loc, 0, 4096);

        assert_cached(&loc, 0, 4096);
    }

    /// A host is looked up once while its answer is recent, and again by the
    /// first new connection after the answer has aged.
    #[cfg(target_os = "linux")]
    #[test]
    fn an_aged_answer_is_looked_up_again() {
        init_test_free_pool(16);
        let (name_server, asked) = spawn_name_server(
            vec![std::net::Ipv4Addr::LOCALHOST],
            FirstQuestions::Answered,
        );
        let mut requester = IORequester::with_http_config(client_config());
        requester.force_name_servers(vec![name_server]);
        requester.force_address_ages(Duration::from_millis(50), Duration::from_secs(30));
        let first_store = named_remote_loc("aged.store.test", spawn_server(1));
        let second_store = named_remote_loc("aged.store.test", spawn_server(1));
        fetch(&mut requester, &first_store, 0, 4096);
        thread::sleep(Duration::from_millis(60));
        let asked_while_recent = asked.try_iter().count();

        fetch(&mut requester, &second_store, 0, 4096);

        assert_eq!(asked_while_recent, 1);
        asked.recv_timeout(Duration::from_secs(5)).unwrap();
    }

    /// Hold `requester`'s remote reads to a receive allowance that, as of
    /// `now`, owes one 4 KiB read and takes `payoff_time` to pay it off.
    #[cfg(target_os = "linux")]
    fn owe_one_read(requester: &mut IORequester, payoff_time: Duration, now: Instant) {
        let limiter = crate::io::ReceiveRateLimiter::new(4096 * 20, payoff_time);
        limiter.try_take(2 * 4096, now).unwrap();
        requester.force_receive_rate_limiter(Arc::new(limiter));
    }

    /// A read issued while the receive allowance is owing is held until the
    /// allowance has grown back, and then completes.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_read_past_the_receive_allowance_waits_for_it_to_refill() {
        init_test_free_pool(16);
        let loc = remote_loc(spawn_server(1));
        let mut requester = IORequester::with_http_config(client_config());
        let payoff_time = Duration::from_millis(50);
        let started_at = Instant::now();
        owe_one_read(&mut requester, payoff_time, started_at);

        fetch(&mut requester, &loc, 0, 4096);

        assert!(started_at.elapsed() >= payoff_time);
        assert_cached(&loc, 0, 4096);
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
    ) -> Vec<std::result::Result<Completion, FailedIO>> {
        let OpenFile::Remote(remote) = loc else {
            panic!("test fetches over http")
        };
        let lookups = memory_ctx().page_caches().compressed.get(loc, offset, len);
        let mut submitted = 0;
        for lookup in &lookups {
            if let Some(block) = lookup.missing_extent() {
                let req = HttpRequest::Get(HttpGetRequest {
                    remote: remote.clone(),
                    block: block.clone(),
                });
                requester
                    .submit_http_request_to_backend(DataFlowRequest::new(0, 0, req))
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

    /// Spawn a server whose `206` carries a Content-Length shorter than the
    /// requested range (a truncating proxy, or an object replaced by a smaller
    /// one), followed by that many body bytes.
    fn spawn_truncating_server() -> u16 {
        let server_config = server_tls_config();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        thread::spawn(move || {
            let (tcp, _) = listener.accept().unwrap();
            let conn = rustls::ServerConnection::new(server_config).unwrap();
            let mut tls = rustls::StreamOwned::new(conn, tcp);
            let head = read_head(&mut tls);
            let (start, end) = parse_range(&head);
            let len = (end - start).div_ceil(2);
            let body: Vec<u8> = (0..len).map(|i| pattern(start + i)).collect();
            let resp = format!(
                "HTTP/1.1 206 Partial Content\r\n\
                 Content-Length: {len}\r\n\
                 Content-Range: bytes {start}-{}/1000000\r\n\
                 Connection: keep-alive\r\n\r\n",
                start + len - 1
            );
            tls.write_all(resp.as_bytes()).unwrap();
            tls.write_all(&body).unwrap();
            tls.flush().unwrap();
        });

        port
    }

    /// Spawn a server that clamps each requested range to `object_size`, the way
    /// a real object store serves a block-aligned read overhanging the object's
    /// end.
    fn spawn_clamping_server(object_size: usize) -> u16 {
        let server_config = server_tls_config();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        thread::spawn(move || {
            let (tcp, _) = listener.accept().unwrap();
            let conn = rustls::ServerConnection::new(server_config).unwrap();
            let mut tls = rustls::StreamOwned::new(conn, tcp);
            let head = read_head(&mut tls);
            let (start, end) = parse_range(&head);
            let end = end.min(object_size - 1);
            let len = end - start + 1;
            let body: Vec<u8> = (0..len).map(|i| pattern(start + i)).collect();
            let resp = format!(
                "HTTP/1.1 206 Partial Content\r\n\
                 Content-Length: {len}\r\n\
                 Content-Range: bytes {start}-{end}/{object_size}\r\n\
                 Connection: keep-alive\r\n\r\n"
            );
            tls.write_all(resp.as_bytes()).unwrap();
            tls.write_all(&body).unwrap();
            tls.flush().unwrap();
        });

        port
    }

    /// A `206` whose body is shorter than the requested range must fail the read
    /// and leave every block uncommitted - committing would let later lookups
    /// serve whatever bytes the slot already held as file content.
    #[test]
    fn a_truncated_range_response_fails_instead_of_committing() {
        init_test_free_pool(16);
        let loc = remote_loc(spawn_truncating_server());
        let mut requester = IORequester::with_http_config(client_config());

        let results = fetch_results(&mut requester, &loc, 0, 4096);

        assert!(results.iter().all(|r| r.is_err()), "the read must fail");
        let lookups = memory_ctx().page_caches().compressed.get(&loc, 0, 4096);
        assert!(
            lookups[0].missing_extent().is_some(),
            "no block may be committed"
        );
    }

    /// A block-aligned read overhanging the object's end is served clamped by the
    /// store; the clamped body must land and commit rather than be rejected.
    #[test]
    fn a_tail_read_clamped_to_the_object_size_commits() {
        const SIZE: usize = 6000;
        init_test_free_pool(16);
        let url = Url::parse(&format!(
            "https://127.0.0.1:{}/obj",
            spawn_clamping_server(SIZE)
        ))
        .unwrap();
        let loc = OpenFile::Remote(Arc::new(RemoteFile::open(url, None, SIZE as u64).unwrap()));
        let mut requester = IORequester::with_http_config(client_config());

        fetch(&mut requester, &loc, 4096, SIZE - 4096);

        assert_cached(&loc, 4096, SIZE - 4096);
    }

    /// A remote fill whose EOF-clamped body ends before its final cache block
    /// must fail for the owner and followers rather than expose a wholly
    /// unwritten block.
    #[test]
    fn a_remote_fill_ending_before_its_last_block_fails() {
        const SIZE: usize = 2000;
        init_test_free_pool(16);
        let url = Url::parse(&format!(
            "https://127.0.0.1:{}/obj",
            spawn_clamping_server(SIZE)
        ))
        .unwrap();
        let loc = OpenFile::Remote(Arc::new(RemoteFile::open(url, None, SIZE as u64).unwrap()));
        let mut requester = IORequester::with_http_config(client_config());

        let results = fetch_results(&mut requester, &loc, 0, 2 * 4096);

        assert!(results.iter().all(|r| r.is_err()), "the read must fail");
        let lookups = memory_ctx().page_caches().compressed.get(&loc, 0, 2 * 4096);
        assert!(
            lookups[0].missing_extent().is_some(),
            "no block may be committed"
        );
    }

    /// Submit one logical read of `[offset, offset + len)` of `loc` and drive it
    /// until it either delivers its bytes or fails, returning the outcome.
    fn drive_fs_read(
        requester: &mut IORequester,
        loc: &OpenFile,
        offset: usize,
        len: usize,
    ) -> std::result::Result<Vec<u8>, FailedIO> {
        requester
            .request_read_resolving(
                0,
                0,
                PendingReadRequest {
                    id: ReadRequestId(0),
                    open_file: loc.clone(),
                    locations: vec![FileRange::new(offset, len)],
                    cache: get_compressed_cache_id(),
                },
                &mut disabled_stats(),
            )
            .unwrap();
        for _ in 0..100 {
            if requester.has_pending() {
                requester.wait().unwrap();
            }
            for completion in requester.completions().unwrap() {
                completion?;
            }
            if let Some(routed) = requester.take_ready_reads().into_iter().next() {
                return Ok(routed.response.into_bytes());
            }
        }
        panic!("the read neither completed nor failed");
    }

    /// A registered direct-I/O file holding `pattern` bytes, `size` bytes long.
    fn patterned_local_file(dir: &tempfile::TempDir, size: usize) -> OpenFile {
        let path = dir.path().join("data");
        let body: Vec<u8> = (0..size).map(pattern).collect();
        std::fs::write(&path, body).unwrap();
        OpenFile::Local(open_direct_read(&path).unwrap())
    }

    /// A block-aligned tail fill may stop at EOF inside its final block; because
    /// it reached that block, the extent must commit and serve later hits.
    #[test]
    fn a_block_aligned_tail_fill_commits_when_eof_is_in_last_block() {
        const SIZE: usize = 6000;
        init_test_free_pool(16);
        let dir = tempfile::tempdir().unwrap();
        let loc = patterned_local_file(&dir, SIZE);
        let mut requester = IORequester::default();

        let bytes = match drive_fs_read(&mut requester, &loc, 0, SIZE) {
            Ok(bytes) => bytes,
            Err(failed) => panic!("the read must succeed: {}", failed.error),
        };

        assert_eq!(bytes, (0..SIZE).map(pattern).collect::<Vec<u8>>());
        assert_cached(&loc, 0, SIZE);
    }

    /// A read whose blocks lie wholly past the file's end gets no bytes; it must
    /// fail and leave the blocks uncommitted, not serve stale slot contents.
    #[test]
    fn a_local_read_wholly_past_eof_fails_instead_of_committing() {
        init_test_free_pool(16);
        let dir = tempfile::tempdir().unwrap();
        let loc = patterned_local_file(&dir, 2000);
        let mut requester = IORequester::default();

        let result = drive_fs_read(&mut requester, &loc, 4096, 4096);

        assert!(result.is_err(), "the read must fail");
        let lookups = memory_ctx().page_caches().compressed.get(&loc, 4096, 4096);
        assert!(
            lookups[0].missing_extent().is_some(),
            "no block may be committed"
        );
    }

    /// A local fill leaving a whole cache block unwritten must fail for the
    /// owner and followers rather than complete with a garbage tail.
    #[test]
    fn a_local_fill_leaving_a_whole_block_unwritten_fails() {
        init_test_free_pool(16);
        let dir = tempfile::tempdir().unwrap();
        let loc = patterned_local_file(&dir, 2000);
        let mut requester = IORequester::default();

        let result = drive_fs_read(&mut requester, &loc, 0, 2 * 4096);

        assert!(result.is_err(), "the read must fail");
        let lookups = memory_ctx().page_caches().compressed.get(&loc, 0, 2 * 4096);
        assert!(
            lookups[0].missing_extent().is_some(),
            "no block may be committed"
        );
    }

    /// A terminal transport failure surfaces as a `FailedIO`, never a committed
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
        OpenFile::Remote(Arc::new(RemoteFile::open(url, None, 1 << 20).unwrap()))
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
        memory_ctx().page_caches().compressed.clear();

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
        memory_ctx().page_caches().compressed.clear();
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
        memory_ctx().page_caches().compressed.clear();

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
        memory_ctx().page_caches().compressed.clear();

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
        memory_ctx().page_caches().compressed.clear();

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
        memory_ctx().page_caches().compressed.clear();

        let mut after = requester_with(cache_in(&dir));
        fetch(&mut after, &loc, 0, 4096);
        settle(&mut after);

        assert_cached(&loc, 0, 4096);
    }

    const ROUTE: RequestRoute = RequestRoute {
        data_flow_id: 7,
        operator_idx: 3,
    };

    fn registered_file() -> (OpenFile, TempDir) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("data");
        std::fs::write(&path, vec![0u8; 1 << 20]).unwrap();
        let open_file =
            OpenFile::Local(LocalFile::new(std::fs::File::open(path).unwrap()).unwrap());
        (open_file, dir)
    }

    fn pending_read(
        file: OpenFile,
        locations: impl IntoIterator<Item = FileRange>,
    ) -> (ReadRequestId, PendingReadRequest) {
        let id = ReadRequestId(0);
        let request = PendingReadRequest {
            id,
            open_file: file,
            locations: locations.into_iter().collect(),
            cache: get_compressed_cache_id(),
        };
        (id, request)
    }

    fn register_local(
        requester: &mut IORequester,
        route: RequestRoute,
        request: PendingReadRequest,
    ) -> Vec<DataFlowRequest<FsRequest>> {
        let cache_lookups = resolve_read_through_cache(&request);
        requester
            .register_and_create_io_reads(route, request, cache_lookups)
            .into_iter()
            .map(|request| match request {
                IORead::Fs(request) => request,
                IORead::Http(_) => panic!("expected a filesystem read"),
            })
            .collect()
    }

    fn physical_id(request: &DataFlowRequest<FsRequest>) -> Identifier {
        request.pending_read_id.unwrap()
    }

    fn complete_reads(
        requester: &mut IORequester,
        requests: impl IntoIterator<Item = DataFlowRequest<FsRequest>>,
    ) {
        for request in requests {
            let id = physical_id(&request);
            let FsRequest::Read(read) = request.request else {
                panic!("expected a read")
            };
            read.block.commit();
            requester.complete_pending_read(id);
        }
    }

    fn warm_compressed_cache(requester: &mut IORequester, file: OpenFile, location: FileRange) {
        let (_, pending) = pending_read(file, [location]);
        let physical = register_local(requester, ROUTE, pending);

        complete_reads(requester, physical);
        requester.take_ready_reads();
    }

    fn cache_decompressed(file: OpenFile, location: FileRange) {
        let key = CacheKey::new(file, location.offset, location.len);
        memory_ctx()
            .page_caches()
            .decompressed
            .insert(key, |fill| {
                fill.get_bytes(4)[0].as_mut_slice().copy_from_slice(b"data");
                Ok::<_, std::convert::Infallible>(DecompressedPageMeta::new(
                    vec![bytes::Bytes::from_static(b"header")],
                    Arc::new(AtomicUsize::new(0)),
                ))
            })
            .unwrap();
    }

    /// The compressed cache page reads go through.
    fn get_compressed_cache_id() -> crate::memory::cache::CacheId {
        memory_ctx().page_caches().compressed.id()
    }

    #[test]
    fn uncached_request_becomes_ready_after_its_physical_read() {
        init_test_free_pool(4);
        let (file, _dir) = registered_file();
        let (id, pending) = pending_read(file, [FileRange::new(17, 100)]);
        let mut requester = IORequester::default();

        let physical = register_local(&mut requester, ROUTE, pending);
        let physical_count = physical.len();
        complete_reads(&mut requester, physical);
        let ready = requester.take_ready_reads().pop().unwrap();

        assert_eq!(physical_count, 1);
        assert_eq!(ready.route, ROUTE);
        assert_eq!(ready.response.id(), id);
        assert_eq!(ready.response.into_bytes().len(), 100);
    }

    #[test]
    fn compressed_cache_hit_needs_no_transport_io() {
        init_test_free_pool(4);
        let (file, _dir) = registered_file();
        let location = FileRange::new(0, 4096);
        let mut requester = IORequester::default();
        warm_compressed_cache(&mut requester, file.clone(), location);
        let (id, pending) = pending_read(file, [location]);

        let physical = register_local(&mut requester, ROUTE, pending);
        let ready = requester.take_ready_reads().pop().unwrap();

        assert!(physical.is_empty());
        assert_eq!(ready.response.id(), id);
    }

    /// A read is answered from the file's stored bytes alone: a decompressed
    /// page cached over its range neither spares the read nor replaces its
    /// bytes (a footer probe spanning a small file's data pages gets the file).
    #[test]
    fn a_read_ignores_decompressed_pages_cached_over_its_range() {
        init_test_free_pool(4);
        let (file, _dir) = registered_file();
        cache_decompressed(file.clone(), FileRange::new(0, 4096));
        let (_, pending) = pending_read(file, [FileRange::new(0, 8192)]);
        let mut requester = IORequester::default();

        let physical = register_local(&mut requester, ROUTE, pending);
        let needed_io = !physical.is_empty();
        complete_reads(&mut requester, physical);
        let response = requester.take_ready_reads().pop().unwrap().response;

        assert!(needed_io);
        assert_eq!(response.into_bytes(), vec![0u8; 8192]);
    }

    #[test]
    fn identical_requests_use_separate_physical_reads() {
        init_test_free_pool(4);
        let (file, _dir) = registered_file();
        let first_route = RequestRoute {
            data_flow_id: 1,
            operator_idx: 2,
        };
        let second_route = RequestRoute {
            data_flow_id: 3,
            operator_idx: 4,
        };
        let (_, first) = pending_read(file.clone(), [FileRange::new(0, 4096)]);
        let (_, second) = pending_read(file, [FileRange::new(0, 4096)]);
        let mut requester = IORequester::default();

        let mut physical = register_local(&mut requester, first_route, first);
        physical.extend(register_local(&mut requester, second_route, second));
        let physical_count = physical.len();
        complete_reads(&mut requester, physical);
        let mut routes: Vec<_> = requester
            .take_ready_reads()
            .into_iter()
            .map(|ready| ready.route)
            .collect();
        routes.sort_by_key(|route| route.data_flow_id);

        assert_eq!(physical_count, 2);
        assert_eq!(routes, vec![first_route, second_route]);
    }

    #[test]
    fn failed_physical_read_discards_its_logical_request() {
        init_test_free_pool(4);
        let (file, _dir) = registered_file();
        let (_, pending) =
            pending_read(file, [FileRange::new(0, 4096), FileRange::new(32768, 4096)]);
        let mut requester = IORequester::default();
        let physical = register_local(&mut requester, ROUTE, pending);
        let ids: Vec<_> = physical.iter().map(physical_id).collect();

        let failed_route = requester.fail_pending_read(ids[0]);
        requester.complete_pending_read(ids[1]);
        let ready = requester.take_ready_reads();

        assert_eq!(failed_route, Some(ROUTE));
        assert!(ready.is_empty());
    }

    #[test]
    fn one_logical_request_reads_multiple_locations() {
        init_test_free_pool(4);
        let (file, _dir) = registered_file();
        let (_, pending) =
            pending_read(file, [FileRange::new(0, 4096), FileRange::new(32768, 4096)]);
        let mut requester = IORequester::default();

        let physical = register_local(&mut requester, ROUTE, pending);
        let physical_count = physical.len();
        complete_reads(&mut requester, physical);
        let locations = requester
            .take_ready_reads()
            .pop()
            .unwrap()
            .response
            .into_locations();

        assert_eq!(physical_count, 2);
        assert_eq!(locations.len(), 2);
    }
}
