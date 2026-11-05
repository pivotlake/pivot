//! Object reads through the io_uring ring.
//!
//! [`load_objects`] fetches every byte of each given object, and
//! [`load_file_slices`] a slice of each, over the worker pool, so the
//! read goes through the same compressed cache and disk cache as a table's
//! column chunks: an immutable metadata file (a table log, a manifest) read
//! this way is served from cache the next time round.
//!
//! [`download_objects`] fetches objects whose size is not known: a ring read
//! needs one, so a remote object is downloaded whole in one request that its
//! response sizes, and a local file is sized by the filesystem. Nothing of
//! it is cached, since what is read this way (a manifest list) is read once
//! per version and kept parsed.

use crate::item_injector::ItemInjectorFactory;
use crate::{DataFile, DataFileLocation, FileRef, ObjectPath};
use dispatch::io::{
    FileRange, OpenFile, OperatorIO, ReadRequestId, ReadResponse, RemoteFile, open_direct_read,
};
use dispatch::memory::memory_ctx;
use dispatch::{
    DataFlowDispatcher, OperatorSpec, RootUnaryOperatorFactory, Sender, Unary, UnaryFactory,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

/// A slice of an object to read: the object, and the byte range of it, which
/// must lie within it.
#[derive(Clone)]
pub struct FileSlice {
    pub file: DataFile,
    pub range: FileRange,
}

/// An object to read whole without knowing its size: its path, and where its
/// bytes live.
#[derive(Clone)]
pub struct UnsizedObject {
    pub path: ObjectPath,
    pub source: DataFileLocation,
}

/// A slice of an object, read: the object's store identity, the byte range,
/// and the bytes stored there.
pub struct LoadedObject {
    pub file: FileRef,
    pub range: FileRange,
    pub bytes: Vec<u8>,
}

/// What tells one requested slice from another: its object's path and its
/// range of it.
#[derive(PartialEq, Eq, Hash)]
struct SliceKey {
    path: ObjectPath,
    range: FileRange,
}

/// Reads one worker keeps in flight before admitting the next, so per-read
/// latency overlaps.
const MAX_READS_IN_FLIGHT: usize = 32;

/// Read every object in `files` in full, in parallel over the pool, and
/// collect them on the coordinator: one object per file, in the order of
/// `files`, whose paths must be distinct. Drives the dataflow, so it must run
/// on the **coordinator** (a
/// thread outside the dispatch workers, such as the one planning a query):
/// it blocks until every worker has finished its share, so a worker calling
/// it (from an operator, or a `run_on_worker` closure) would wait on itself.
pub fn load_objects(
    dispatcher: &DataFlowDispatcher,
    files: &[DataFile],
) -> Result<Vec<LoadedObject>, dispatch::DataFlowError> {
    let slices: Vec<FileSlice> = files
        .iter()
        .map(|file| FileSlice {
            file: file.clone(),
            range: FileRange::new(0, file.file.size as usize),
        })
        .collect();
    load_file_slices(dispatcher, &slices)
}

/// Read each slice in `slices`, in parallel over the pool, and collect them
/// on the coordinator: one read per slice, in the order of `slices`, which
/// must be distinct as path and range. The same coordinator rule as
/// [`load_objects`] applies.
pub fn load_file_slices(
    dispatcher: &DataFlowDispatcher,
    slices: &[FileSlice],
) -> Result<Vec<LoadedObject>, dispatch::DataFlowError> {
    if slices.is_empty() {
        return Ok(Vec::new());
    }
    let injector = ItemInjectorFactory::new(slices);
    let workers = dispatcher.worker_count();
    let siblings = Arc::new(AtomicUsize::new(workers));
    let factories: Vec<_> = (0..workers)
        .map(|_| {
            RootUnaryOperatorFactory::new(ObjectFetcherFactory, injector.clone(), siblings.clone())
        })
        .collect();
    let loaded: Vec<LoadedObject> = OperatorSpec::new(dispatcher.clone(), factories).collect()?;
    // Workers finish in any order: hand the reads back in the order of `slices`.
    let mut reads: HashMap<SliceKey, LoadedObject> = loaded
        .into_iter()
        .map(|read| {
            let key = SliceKey {
                path: read.file.path.clone(),
                range: read.range,
            };
            (key, read)
        })
        .collect();
    Ok(slices
        .iter()
        .map(|slice| {
            reads
                .remove(&SliceKey {
                    path: slice.file.file.path.clone(),
                    range: slice.range,
                })
                .expect("every distinct slice is read once")
        })
        .collect())
}

/// Read every object in `objects` whole, in parallel over the pool, and
/// collect them on the coordinator: one object per entry, in the order of
/// `objects`, whose paths must be distinct, each with the size it turned out
/// to have. The same coordinator rule as [`load_objects`] applies.
pub fn download_objects(
    dispatcher: &DataFlowDispatcher,
    objects: &[UnsizedObject],
) -> Result<Vec<LoadedObject>, dispatch::DataFlowError> {
    if objects.is_empty() {
        return Ok(Vec::new());
    }
    let injector = ItemInjectorFactory::new(objects);
    let workers = dispatcher.worker_count();
    let siblings = Arc::new(AtomicUsize::new(workers));
    let factories: Vec<_> = (0..workers)
        .map(|_| {
            RootUnaryOperatorFactory::new(
                ObjectDownloaderFactory,
                injector.clone(),
                siblings.clone(),
            )
        })
        .collect();
    let loaded: Vec<LoadedObject> = OperatorSpec::new(dispatcher.clone(), factories).collect()?;
    let mut downloads: HashMap<ObjectPath, LoadedObject> = loaded
        .into_iter()
        .map(|download| (download.file.path.clone(), download))
        .collect();
    Ok(objects
        .iter()
        .map(|object| {
            downloads
                .remove(&object.path)
                .expect("every distinct object is downloaded once")
        })
        .collect())
}

/// Builds each worker's [`ObjectFetcher`].
struct ObjectFetcherFactory;

impl UnaryFactory<FileSlice, LoadedObject> for ObjectFetcherFactory {
    type Unary = ObjectFetcher;

    fn build_unary(self) -> Self::Unary {
        ObjectFetcher {
            in_flight: HashMap::new(),
        }
    }
}

/// Fetches each slice it is handed as one logical read, and emits the bytes
/// once the ring has them.
struct ObjectFetcher {
    in_flight: HashMap<ReadRequestId, (FileRef, FileRange)>,
}

impl Unary<FileSlice, LoadedObject> for ObjectFetcher {
    fn consume(
        &mut self,
        slice: FileSlice,
        sender: &mut dyn Sender<LoadedObject>,
        io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        let FileSlice {
            file: DataFile { file, source },
            range,
        } = slice;
        if range.len == 0 {
            // Nothing to read, and the ring is never asked for an empty range.
            sender.send(LoadedObject {
                file,
                range,
                bytes: Vec::new(),
            })?;
            return Ok(());
        }
        let open_file = source
            .open_read(file.size)
            .map_err(|error| dispatch::UnaryError::Operator(Box::new(error)))?;
        // Register the freshly opened descriptor, so a reused fd cannot serve
        // stale extents of whatever it named before.
        memory_ctx()
            .compressed_cache()
            .open_entry(open_file.clone());
        // One range, however large: the ring tiles it into cache slots
        // itself, as it does a column chunk of any size.
        let id = io.read(open_file, [range])?;
        self.in_flight.insert(id, (file, range));
        Ok(())
    }

    fn ready_for_more_work(&mut self) -> bool {
        self.in_flight.len() < MAX_READS_IN_FLIGHT
    }

    fn process_read_response(
        &mut self,
        sender: &mut dyn Sender<LoadedObject>,
        _io: &mut OperatorIO,
        response: ReadResponse,
    ) -> dispatch::UnaryResult<()> {
        let (file, range) = self
            .in_flight
            .remove(&response.id())
            .expect("response for an unknown object read");
        sender.send(LoadedObject {
            file,
            range,
            bytes: response.into_bytes(),
        })?;
        Ok(())
    }

    fn finish(&mut self, _sender: &mut dyn Sender<LoadedObject>) -> dispatch::UnaryResult<bool> {
        Ok(self.in_flight.is_empty())
    }
}

/// Builds each worker's [`ObjectDownloader`].
struct ObjectDownloaderFactory;

impl UnaryFactory<UnsizedObject, LoadedObject> for ObjectDownloaderFactory {
    type Unary = ObjectDownloader;

    fn build_unary(self) -> Self::Unary {
        ObjectDownloader {
            in_flight: HashMap::new(),
        }
    }
}

/// Reads each object it is handed whole, sized by whatever knows the size: a
/// remote object's response, a local file's filesystem. Emits the bytes once
/// they are in.
struct ObjectDownloader {
    in_flight: HashMap<ReadRequestId, ObjectPath>,
}

impl Unary<UnsizedObject, LoadedObject> for ObjectDownloader {
    fn consume(
        &mut self,
        object: UnsizedObject,
        sender: &mut dyn Sender<LoadedObject>,
        io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        let UnsizedObject { path, source } = object;
        let id = match source {
            DataFileLocation::Remote { url, auth } => {
                let remote = RemoteFile::open_unsized(url, auth)
                    .map_err(|error| dispatch::UnaryError::Operator(Box::new(error)))?;
                io.download(Arc::new(remote))?
            }
            DataFileLocation::Local(local_path) => {
                let size = std::fs::metadata(&local_path)
                    .map_err(|error| dispatch::UnaryError::Operator(Box::new(error)))?
                    .len();
                if size == 0 {
                    // Nothing to read, and the ring is never asked for an
                    // empty range.
                    sender.send(LoadedObject {
                        file: FileRef { path, size },
                        range: FileRange::new(0, 0),
                        bytes: Vec::new(),
                    })?;
                    return Ok(());
                }
                let open_file = OpenFile::Local(
                    open_direct_read(&local_path)
                        .map_err(|error| dispatch::UnaryError::Operator(Box::new(error)))?,
                );
                // Register the freshly opened descriptor, so a reused fd
                // cannot serve stale extents of whatever it named before.
                memory_ctx()
                    .compressed_cache()
                    .open_entry(open_file.clone());
                io.read(open_file, [FileRange::new(0, size as usize)])?
            }
        };
        self.in_flight.insert(id, path);
        Ok(())
    }

    fn ready_for_more_work(&mut self) -> bool {
        self.in_flight.len() < MAX_READS_IN_FLIGHT
    }

    fn process_read_response(
        &mut self,
        sender: &mut dyn Sender<LoadedObject>,
        _io: &mut OperatorIO,
        response: ReadResponse,
    ) -> dispatch::UnaryResult<()> {
        let path = self
            .in_flight
            .remove(&response.id())
            .expect("response for an unknown object download");
        let bytes = response.into_bytes();
        sender.send(LoadedObject {
            file: FileRef {
                path,
                size: bytes.len() as u64,
            },
            range: FileRange::new(0, bytes.len()),
            bytes,
        })?;
        Ok(())
    }

    fn finish(&mut self, _sender: &mut dyn Sender<LoadedObject>) -> dispatch::UnaryResult<bool> {
        Ok(self.in_flight.is_empty())
    }
}
