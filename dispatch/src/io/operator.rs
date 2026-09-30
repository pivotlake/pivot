//! The logical I/O interface exposed to operators.
//!
//! Operators describe *what* file bytes they need. They do not look up cache
//! entries, allocate cache extents, or retain destination pointers. The
//! worker's requester performs those physical tasks the moment the operator
//! asks, through the [`OperatorIO`] handle dispatch wraps around each call.

use super::OpenFile;
use crate::memory::FileBytes;
use bytes::Bytes;
use std::sync::Arc;

/// Identifies one logical read within an operator node.
///
/// Identifiers are local to an [`OperatorIO`]. Dispatch routes them together
/// with the dataflow and operator identifiers, so operators can use this small
/// value directly as a key in their own in-flight maps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReadRequestId(pub usize);

/// One byte range requested from a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileRange {
    pub offset: usize,
    pub len: usize,
}

impl FileRange {
    pub fn new(offset: usize, len: usize) -> Self {
        Self { offset, len }
    }
}

/// One resolved piece of a requested file range.
///
/// A range may contain a mixture of pages already present in the decompressed
/// cache and gaps served by the compressed cache. Pieces remain in file order.
pub enum ReadData {
    /// Raw file bytes from the compressed cache. Several buffers are possible
    /// because one logical range may cross cache extents.
    Compressed { offset: usize, bytes: Vec<Bytes> },
    /// A decompressed block whose compressed source occupied `span` bytes at
    /// `offset`. `header` is opaque metadata stored with the block by its owner.
    Decompressed {
        offset: usize,
        span: usize,
        header: Vec<Bytes>,
        data: Vec<Bytes>,
    },
}

/// The completed result of one logical read.
pub struct ReadResponse {
    id: ReadRequestId,
    open_file: OpenFile,
    /// One response list per requested [`FileRange`], in the same order.
    locations: Vec<Vec<ReadData>>,
}

impl ReadResponse {
    pub(crate) fn new(
        id: ReadRequestId,
        open_file: OpenFile,
        locations: Vec<Vec<ReadData>>,
    ) -> Self {
        Self {
            id,
            open_file,
            locations,
        }
    }

    pub fn id(&self) -> ReadRequestId {
        self.id
    }

    pub fn open_file(&self) -> &OpenFile {
        &self.open_file
    }

    pub fn into_locations(self) -> Vec<Vec<ReadData>> {
        self.locations
    }

    /// Consume a one-location response of a
    /// [`read_raw_bytes`](OperatorIO::read_raw_bytes) request and concatenate its cache
    /// fragments: a raw-bytes read never resolves to a decompressed page.
    pub fn into_bytes(self) -> Vec<u8> {
        assert_eq!(
            self.locations.len(),
            1,
            "into_bytes requires exactly one requested location"
        );
        let mut bytes = Vec::new();
        for part in self.locations.into_iter().next().unwrap() {
            match part {
                ReadData::Compressed { bytes: runs, .. } => {
                    for run in runs {
                        bytes.extend_from_slice(&run);
                    }
                }
                ReadData::Decompressed { .. } => {
                    panic!("into_bytes on a read that was not raw_bytes_only")
                }
            }
        }
        bytes
    }
}

/// A logical read waiting for the worker to register it with the caches.
pub struct PendingReadRequest {
    pub id: ReadRequestId,
    pub open_file: OpenFile,
    pub locations: Vec<FileRange>,
    /// Whether the read is answered only with the file's stored bytes (the
    /// compressed cache), never a decompressed page. Set for a read whose
    /// ranges are not made of whole Parquet pages, such as a footer's: the
    /// decompressed cache answers a range with every page starting inside it.
    pub raw_bytes_only: bool,
}

/// A logical write waiting for the worker to select its transport.
pub struct PendingWriteRequest {
    pub open_file: OpenFile,
    pub data: Arc<FileBytes>,
}

/// The logical I/O handle passed to an operator whenever it runs.
///
/// A thin borrow of the worker's requester, bound to the operator's routing
/// identity by dispatch. A read or write submits immediately; the operator
/// retains only its logical request identifiers and domain state, and never
/// sees the requester's worker-lifecycle surface.
pub struct OperatorIO<'a> {
    requester: &'a mut crate::io::IORequester,
    data_flow_id: crate::Identifier,
    operator_idx: crate::Identifier,
    stats: &'a mut crate::stats::StatsCollector,
    /// The dataflow's logical read-id counter, owned by the flow so
    /// identifiers stay unique across this handle's short lifetimes.
    next_read_id: &'a mut usize,
}

impl<'a> OperatorIO<'a> {
    pub(crate) fn new(
        requester: &'a mut crate::io::IORequester,
        data_flow_id: crate::Identifier,
        operator_idx: crate::Identifier,
        stats: &'a mut crate::stats::StatsCollector,
        next_read_id: &'a mut usize,
    ) -> Self {
        Self {
            requester,
            data_flow_id,
            operator_idx,
            stats,
            next_read_id,
        }
    }

    /// Request one or more locations from a single file.
    pub fn read(
        &mut self,
        open_file: OpenFile,
        locations: impl IntoIterator<Item = FileRange>,
    ) -> Result<ReadRequestId, crate::io::IORequesterError> {
        self.submit_read(open_file, locations, false)
    }

    /// Request one or more locations from a single file as the bytes stored
    /// there ([`PendingReadRequest::raw_bytes_only`]), for a response read
    /// with [`ReadResponse::into_bytes`].
    pub fn read_raw_bytes(
        &mut self,
        open_file: OpenFile,
        locations: impl IntoIterator<Item = FileRange>,
    ) -> Result<ReadRequestId, crate::io::IORequesterError> {
        self.submit_read(open_file, locations, true)
    }

    fn submit_read(
        &mut self,
        open_file: OpenFile,
        locations: impl IntoIterator<Item = FileRange>,
        raw_bytes_only: bool,
    ) -> Result<ReadRequestId, crate::io::IORequesterError> {
        let id = ReadRequestId(*self.next_read_id);
        *self.next_read_id += 1;
        let request = PendingReadRequest {
            id,
            open_file,
            locations: locations.into_iter().collect(),
            raw_bytes_only,
        };
        self.requester
            .request_read(self.data_flow_id, self.operator_idx, request, self.stats)?;
        Ok(id)
    }

    /// Write `data` to a local file or remote object.
    pub fn write(
        &mut self,
        open_file: OpenFile,
        data: Arc<FileBytes>,
    ) -> Result<(), crate::io::IORequesterError> {
        let request = PendingWriteRequest { open_file, data };
        self.requester
            .request_write(self.data_flow_id, self.operator_idx, request, self.stats)
    }

    /// Point this handle at another operator of the same dataflow. A traversal
    /// builds one handle and retargets it per node, which keeps the per-node
    /// cost of an IO-free pass at a single store.
    #[inline]
    pub(crate) fn set_operator_idx(&mut self, operator_idx: crate::Identifier) {
        self.operator_idx = operator_idx;
    }
}

/// Owns everything an [`OperatorIO`] borrows, so a standalone operator test
/// can hand its operator a real handle without running a worker.
#[cfg(any(test, feature = "test-util"))]
pub struct TestOperatorIO {
    requester: crate::io::IORequester,
    stats: crate::stats::StatsCollector,
    next_read_id: usize,
}

#[cfg(any(test, feature = "test-util"))]
impl Default for TestOperatorIO {
    fn default() -> Self {
        Self {
            requester: crate::io::IORequester::default(),
            stats: crate::stats::StatsCollector::disabled(),
            next_read_id: 0,
        }
    }
}

#[cfg(any(test, feature = "test-util"))]
impl TestOperatorIO {
    pub fn io(&mut self) -> OperatorIO<'_> {
        OperatorIO::new(
            &mut self.requester,
            0,
            0,
            &mut self.stats,
            &mut self.next_read_id,
        )
    }
}
