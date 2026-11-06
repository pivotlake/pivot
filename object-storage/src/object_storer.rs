//! Object writes through the io_uring ring.
//!
//! [`store_objects`] writes each given object in full, over the worker pool,
//! through the same ring and HTTP transport every data file is uploaded with:
//! the twin of [`load_objects`](crate::load_objects), for writes. A metadata
//! file a commit writes (a manifest, a manifest list) lands the way its data
//! files did.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::sync::Arc;

use dispatch::io::{
    FsWriteRequest, HttpUploadRequest, LocalFile, OpenFile, OperatorIO, RemoteFile,
};
use dispatch::memory::{FileBytes, SlabAllocator};
use dispatch::{
    BUFFER_SIZE, DataFlowDispatcher, Sender, Unary, UnaryError, UnaryFactory, stealable,
    values_input,
};

use crate::{DataFileLocation, FileRef};

/// An object to write: its store identity, where the store sinks it, and its
/// bytes.
pub struct ObjectWrite {
    pub file: FileRef,
    pub sink: DataFileLocation,
    pub bytes: Vec<u8>,
}

/// Write every object in `writes` in full, in parallel over the pool, and
/// return once each has landed. Drives the dataflow, so it must run on the
/// coordinator, as [`load_objects`](crate::load_objects) must.
pub fn store_objects(
    dispatcher: &DataFlowDispatcher,
    writes: Vec<ObjectWrite>,
) -> Result<(), dispatch::DataFlowError> {
    if writes.is_empty() {
        return Ok(());
    }
    let workers = dispatcher.worker_count();
    let channels: Vec<_> = stealable::<ObjectWrite>(dispatcher.topology())
        .into_iter()
        .collect();
    let storers: Vec<_> = (0..workers).map(|_| ObjectStorerFactory).collect();
    values_input(dispatcher, writes)
        .chain(channels, storers)
        .collect()?;
    Ok(())
}

/// Builds each worker's [`ObjectStorer`].
struct ObjectStorerFactory;

impl UnaryFactory<ObjectWrite, FileRef> for ObjectStorerFactory {
    type Unary = ObjectStorer;

    fn build_unary(self) -> Self::Unary {
        ObjectStorer {
            in_flight: HashMap::new(),
        }
    }
}

/// Issues each write it is handed as one ring write, and reports the object
/// once the ring has landed it.
struct ObjectStorer {
    /// Writes issued and not yet landed, keyed by the address of their bytes:
    /// the request shares that `Arc`, so a completion finds its object by it.
    in_flight: HashMap<usize, FileRef>,
}

impl ObjectStorer {
    fn landed(&mut self, id: usize, sender: &mut dyn Sender<FileRef>) -> dispatch::UnaryResult<()> {
        let file = self
            .in_flight
            .remove(&id)
            .expect("completion for an object write not in flight");
        sender.send(file)?;
        Ok(())
    }
}

impl Unary<ObjectWrite, FileRef> for ObjectStorer {
    fn consume(
        &mut self,
        write: ObjectWrite,
        _sender: &mut dyn Sender<FileRef>,
        io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        let ObjectWrite { file, sink, bytes } = write;
        let open_file = match sink {
            DataFileLocation::Local(path) => {
                let handle = OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(path)
                    .map_err(operator_error)?;
                OpenFile::Local(LocalFile::new(handle).map_err(operator_error)?)
            }
            DataFileLocation::Remote { url, auth } => OpenFile::Remote(Arc::new(
                RemoteFile::open(url, auth, file.size).map_err(operator_error)?,
            )),
        };
        let data = Arc::new(ring_runs_of(&bytes));
        let id = Arc::as_ptr(&data) as usize;
        io.write(open_file, data)?;
        self.in_flight.insert(id, file);
        Ok(())
    }

    fn process_fs_write_response(
        &mut self,
        sender: &mut dyn Sender<FileRef>,
        request: FsWriteRequest,
    ) -> dispatch::UnaryResult<()> {
        self.landed(Arc::as_ptr(&request.data) as usize, sender)
    }

    fn process_http_upload_response(
        &mut self,
        sender: &mut dyn Sender<FileRef>,
        request: HttpUploadRequest,
    ) -> dispatch::UnaryResult<()> {
        self.landed(Arc::as_ptr(&request.data) as usize, sender)
    }

    /// A write lands later on the ring, so the worker is not done until every
    /// one it issued has.
    fn has_pending_work(&self) -> bool {
        !self.in_flight.is_empty()
    }

    fn finish(&mut self, _sender: &mut dyn Sender<FileRef>) -> dispatch::UnaryResult<bool> {
        Ok(self.in_flight.is_empty())
    }
}

/// `bytes` copied into ring memory, in runs of at most one buffer each, as a
/// ring write takes them.
fn ring_runs_of(bytes: &[u8]) -> FileBytes {
    let mut allocator = SlabAllocator::new(false);
    let mut runs = FileBytes::new();
    for chunk in bytes.chunks(BUFFER_SIZE) {
        let mut run = allocator.get_slab_of_size(chunk.len(), false);
        run.as_mut_slice().copy_from_slice(chunk);
        runs.push(run);
    }
    runs
}

fn operator_error(error: impl std::error::Error + Send + Sync + 'static) -> UnaryError {
    UnaryError::Operator(Box::new(error))
}
