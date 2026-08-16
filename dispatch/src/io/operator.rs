//! The logical I/O interface exposed to operators.
//!
//! Operators describe *what* file bytes they need. They do not look up cache
//! entries, allocate cache extents, or retain destination pointers. The worker's
//! requester performs those physical tasks later, during its
//! `register_pending_io` stage, using its private request tracker.

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

    /// Consume a one-location raw-byte response and concatenate its cache
    /// fragments. Intended for metadata reads, whose ranges never describe a
    /// decompressed data page.
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
                    panic!("raw file read unexpectedly resolved to decompressed data")
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
}

/// A logical write waiting for the worker to select its transport.
pub struct PendingWriteRequest {
    pub open_file: OpenFile,
    pub data: Arc<FileBytes>,
}

/// Per-node I/O state passed to an operator whenever it can produce more work.
///
/// The two pending vectors deliberately live beside the operator rather than
/// inside it. This keeps scheduling concerns in dispatch and lets domain
/// operators retain only their logical request identifiers and domain state.
pub struct OperatorIO {
    pending_read_requests: Vec<PendingReadRequest>,
    pending_write_requests: Vec<PendingWriteRequest>,
    next_read_id: usize,
}

impl Default for OperatorIO {
    fn default() -> Self {
        Self {
            pending_read_requests: Vec::new(),
            pending_write_requests: Vec::new(),
            next_read_id: 0,
        }
    }
}

impl OperatorIO {
    /// Request one or more locations from a single file.
    pub fn read(
        &mut self,
        open_file: OpenFile,
        locations: impl IntoIterator<Item = FileRange>,
    ) -> ReadRequestId {
        let id = ReadRequestId(self.next_read_id);
        self.next_read_id += 1;
        self.pending_read_requests.push(PendingReadRequest {
            id,
            open_file,
            locations: locations.into_iter().collect(),
        });
        super::note_pending_io();
        id
    }

    /// Write `data` to a local file or remote object.
    pub fn write(&mut self, open_file: OpenFile, data: Arc<FileBytes>) {
        self.pending_write_requests
            .push(PendingWriteRequest { open_file, data });
        super::note_pending_io();
    }

    pub(crate) fn take_pending_read_requests(&mut self) -> Vec<PendingReadRequest> {
        std::mem::take(&mut self.pending_read_requests)
    }

    pub(crate) fn take_pending_write_requests(&mut self) -> Vec<PendingWriteRequest> {
        std::mem::take(&mut self.pending_write_requests)
    }
}
