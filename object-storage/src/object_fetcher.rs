//! Whole-object reads through the io_uring ring.
//!
//! [`load_objects`] fetches every byte of each given object over the worker
//! pool, so the read goes through the same compressed cache and disk cache as
//! a table's column chunks: an immutable metadata file (a table log, a manifest)
//! read this way is served from cache the next time round.

use crate::file_injector::FileInjectorFactory;
use crate::{DataFile, DataFileLocation, FileRef, ObjectPath};
use dispatch::io::{FileRange, OperatorIO, ReadRequestId, ReadResponse};
use dispatch::memory::memory_ctx;
use dispatch::{
    DataFlowDispatcher, OperatorSpec, RootUnaryOperatorFactory, Sender, Unary, UnaryFactory,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

/// An immutable object located for a whole read, without its byte length.
#[derive(Clone)]
pub struct ObjectSource {
    pub path: ObjectPath,
    pub source: DataFileLocation,
}

#[derive(Clone)]
struct ObjectRead {
    path: ObjectPath,
    source: DataFileLocation,
    size: Option<u64>,
}

/// Read immutable objects without a listing or HEAD request for their sizes.
/// Like [`load_objects`], this drives a dataflow and must run on the coordinator.
pub fn load_whole_objects(
    dispatcher: &DataFlowDispatcher,
    files: &[ObjectSource],
) -> Result<Vec<LoadedObject>, dispatch::DataFlowError> {
    let files: Vec<_> = files
        .iter()
        .map(|file| ObjectRead {
            path: file.path.clone(),
            source: file.source.clone(),
            size: None,
        })
        .collect();
    fetch_objects(dispatcher, &files)
}

/// One object read in full: its store identity and every byte of it.
pub struct LoadedObject {
    pub file: FileRef,
    pub bytes: Vec<u8>,
}

/// Whole-object reads one worker keeps in flight before admitting the next
/// object, so per-object latency overlaps.
const MAX_OBJECTS_IN_FLIGHT: usize = 32;

/// Read every object in `files` in full, in parallel over the pool, and
/// collect them on the coordinator. Completion order is whichever the workers
/// finish in. Drives the dataflow, so it must run on the **coordinator** (a
/// thread outside the dispatch workers, such as the one planning a query):
/// it blocks until every worker has finished its share, so a worker calling
/// it (from an operator, or a `run_on_worker` closure) would wait on itself.
pub fn load_objects(
    dispatcher: &DataFlowDispatcher,
    files: &[DataFile],
) -> Result<Vec<LoadedObject>, dispatch::DataFlowError> {
    let files: Vec<_> = files
        .iter()
        .map(|file| ObjectRead {
            path: file.file.path.clone(),
            source: file.source.clone(),
            size: Some(file.file.size),
        })
        .collect();
    fetch_objects(dispatcher, &files)
}

fn fetch_objects(
    dispatcher: &DataFlowDispatcher,
    files: &[ObjectRead],
) -> Result<Vec<LoadedObject>, dispatch::DataFlowError> {
    if files.is_empty() {
        return Ok(Vec::new());
    }
    let injector = FileInjectorFactory::new(files);
    let workers = dispatcher.worker_count();
    let siblings = Arc::new(AtomicUsize::new(workers));
    let factories: Vec<_> = (0..workers)
        .map(|_| {
            RootUnaryOperatorFactory::new(ObjectFetcherFactory, injector.clone(), siblings.clone())
        })
        .collect();
    OperatorSpec::new(dispatcher.clone(), factories).collect()
}

/// Builds each worker's [`ObjectFetcher`].
struct ObjectFetcherFactory;

impl UnaryFactory<ObjectRead, LoadedObject> for ObjectFetcherFactory {
    type Unary = ObjectFetcher;

    fn build_unary(self) -> Self::Unary {
        ObjectFetcher {
            in_flight: HashMap::new(),
        }
    }
}

/// Fetches each object it is handed as one logical read of the whole object,
/// and emits the bytes once the ring has them.
struct ObjectFetcher {
    in_flight: HashMap<ReadRequestId, ObjectPath>,
}

impl Unary<ObjectRead, LoadedObject> for ObjectFetcher {
    fn consume(
        &mut self,
        file: ObjectRead,
        sender: &mut dyn Sender<LoadedObject>,
        io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        let ObjectRead { path, source, size } = file;
        if size == Some(0) {
            sender.send(LoadedObject {
                file: FileRef { path, size: 0 },
                bytes: Vec::new(),
            })?;
            return Ok(());
        }
        let open_file = match size {
            Some(size) => source.open_read(size),
            None => source.open_whole_read(),
        }
        .map_err(|error| dispatch::UnaryError::Operator(Box::new(error)))?;
        memory_ctx()
            .compressed_cache()
            .open_entry(open_file.clone());
        let id = match size {
            Some(size) => io.read_raw_bytes(open_file, [FileRange::new(0, size as usize)])?,
            None => io.read_whole_file(open_file)?,
        };
        self.in_flight.insert(id, path);
        Ok(())
    }

    fn ready_for_more_work(&mut self) -> bool {
        self.in_flight.len() < MAX_OBJECTS_IN_FLIGHT
    }

    fn process_read_response(
        &mut self,
        sender: &mut dyn Sender<LoadedObject>,
        _io: &mut OperatorIO,
        response: ReadResponse,
    ) -> dispatch::UnaryResult<()> {
        let file = self
            .in_flight
            .remove(&response.id())
            .expect("response for an unknown object read");
        let bytes = response.into_bytes();
        sender.send(LoadedObject {
            file: FileRef {
                path: file,
                size: bytes.len() as u64,
            },
            bytes,
        })?;
        Ok(())
    }

    fn finish(&mut self, _sender: &mut dyn Sender<LoadedObject>) -> dispatch::UnaryResult<bool> {
        Ok(self.in_flight.is_empty())
    }
}

#[cfg(all(test, feature = "test-support"))]
mod tests {
    use super::*;
    use crate::test_support;
    use dispatch::Dispatch;
    use std::sync::OnceLock;

    #[test]
    fn whole_objects_load_from_s3_without_provided_lengths() {
        let Some(backend) = test_support::s3("whole-object-reads") else {
            eprintln!("skipping whole-object S3 test: MinIO unavailable");
            return;
        };
        static DISPATCH: OnceLock<Dispatch> = OnceLock::new();
        let dispatcher = DISPATCH
            .get_or_init(|| Dispatch::spin_up(2, 32, None))
            .dispatcher();
        let expected = [("empty", Vec::new()), ("metadata", vec![37; 51_239])];
        let sources: Vec<_> = expected
            .iter()
            .map(|(name, bytes)| {
                let path = ObjectPath::new(*name);
                backend.store.put(&path, bytes).unwrap();
                ObjectSource {
                    source: backend.store.source(&path).unwrap(),
                    path,
                }
            })
            .collect();

        let mut loaded = load_whole_objects(dispatcher, &sources).unwrap();
        loaded.sort_by(|left, right| left.file.path.as_str().cmp(right.file.path.as_str()));

        for (loaded, (name, bytes)) in loaded.into_iter().zip(expected) {
            assert_eq!(loaded.file.path.as_str(), name);
            assert_eq!(loaded.file.size, bytes.len() as u64);
            assert_eq!(loaded.bytes, bytes);
        }
    }
}
