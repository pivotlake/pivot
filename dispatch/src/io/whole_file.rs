//! Whole-file assembly with bounded ring residency. The caller ultimately owns
//! a Vec, but network and disk work holds at most 64 KiB of ring memory per read.
//! Small reads share slots. A write-back finishes before its buffer is reused.

use super::backend::IOBackend;
use super::disk_cache::{DiskCache, Object};
use super::http::{HttpEngine, WholeChunk};
use super::{
    DataFlowRequest, FailedIO, OpenFile, ReadData, ReadRequestId, ReadResponse, RemoteSplit,
};
use crate::Identifier;
use crate::memory::{Slab, SlabAllocator, memory_ctx};
use crate::request_tracker::{RequestRoute, RoutedReadResponse};
use std::collections::HashMap;
use std::os::fd::AsRawFd;
use std::sync::Arc;

const BLOCK_SIZE: usize = super::disk_cache::BLOCK_SIZE;
const READ_BUFFER_SIZE: usize = 64 * 1024;
type Result<T> = std::result::Result<T, super::IORequesterError>;

pub(crate) struct WholeRequest {
    pub id: ReadRequestId,
    pub file: OpenFile,
}

pub struct WholeReadStats {
    pub remote: RemoteSplit,
    pub disk_requests: u64,
    pub disk_bytes: u64,
}

struct WholeRead {
    request: DataFlowRequest<WholeRequest>,
    length: Option<usize>,
    bytes: Vec<u8>,
    buffer: Option<Slab>,
    filled: usize,
    /// A bounded transport chunk waiting for the ring slot's write-back.
    pending: Vec<u8>,
    network_done: bool,
    disk_pending: bool,
    object: Option<Arc<Object>>,
    /// A local source or a completed disk-cache object. False means a GET.
    disk_source: bool,
    stats: WholeReadStats,
}

struct DiskOp {
    read: Identifier,
    length: usize,
    write: bool,
}

#[derive(Default)]
pub(crate) struct WholeFiles {
    reads: HashMap<Identifier, WholeRead>,
    allocator: Option<SlabAllocator>,
    disk_ops: HashMap<Identifier, DiskOp>,
    pub ready: Vec<RoutedReadResponse>,
    pub completed: Vec<DataFlowRequest<WholeReadStats>>,
    pub failed: Vec<FailedIO>,
    #[cfg(test)]
    pub reject_cache_writes: bool,
}

impl WholeFiles {
    pub fn start(
        &mut self,
        backend: &mut IOBackend,
        http: &mut HttpEngine,
        cache: Option<&DiskCache>,
        disk_id: &mut Identifier,
        id: Identifier,
        request: DataFlowRequest<WholeRequest>,
    ) -> Result<()> {
        if let Some(bytes) = memory_ctx()
            .compressed_cache()
            .read_whole(&request.request.file)
        {
            self.ready.push(make_response(request, bytes));
            return Ok(());
        }
        let (length, object, disk_source) = match &request.request.file {
            OpenFile::Local(file) => (Some(file.metadata()?.len() as usize), None, true),
            OpenFile::Remote(remote) => match cache.and_then(|cache| cache.open_whole(remote)) {
                Some((object, length)) => (Some(length as usize), Some(object), true),
                None => (None, None, false),
            },
        };
        let mut read = WholeRead {
            request,
            length,
            object,
            disk_source,
            bytes: Vec::new(),
            buffer: None,
            filled: 0,
            pending: Vec::new(),
            network_done: false,
            disk_pending: false,
            stats: WholeReadStats {
                remote: RemoteSplit::default(),
                disk_requests: 0,
                disk_bytes: 0,
            },
        };
        if disk_source {
            if length == Some(0) {
                self.ready.push(make_response(read.request, Vec::new()));
                return Ok(());
            }
            read.buffer = Some(allocate_buffer(&mut self.allocator, length.unwrap()));
        } else if let OpenFile::Remote(remote) = &read.request.request.file {
            #[cfg(target_os = "linux")]
            http.start_whole(&mut backend.ring, id, remote.clone())?;
            #[cfg(not(target_os = "linux"))]
            http.start_whole(id, remote.clone())?;
            read.stats.remote.http_requests = 1;
        }
        self.reads.insert(id, read);
        if disk_source && let Err(error) = self.submit_disk_read(backend, disk_id, id) {
            if !self.reads[&id].disk_pending {
                self.reads.remove(&id);
            }
            return Err(error);
        }
        Ok(())
    }

    fn submit_disk_read(
        &mut self,
        backend: &mut IOBackend,
        disk_id: &mut Identifier,
        id: Identifier,
    ) -> Result<()> {
        let read = self.reads.get_mut(&id).unwrap();
        let length = (read.length.unwrap() - read.bytes.len()).min(READ_BUFFER_SIZE);
        let fd = match (&read.object, &read.request.request.file) {
            (Some(object), _) => object.fd(),
            (_, OpenFile::Local(file)) => file.as_raw_fd(),
            _ => unreachable!(),
        };
        backend.submit_read(
            fd,
            read.bytes.len() as u64,
            read.buffer.as_ref().unwrap().ptr,
            length.div_ceil(BLOCK_SIZE) * BLOCK_SIZE,
            *disk_id,
        )?;
        self.disk_ops.insert(
            *disk_id,
            DiskOp {
                read: id,
                length,
                write: false,
            },
        );
        *disk_id += 1;
        read.disk_pending = true;
        if read.object.is_some() {
            read.stats.remote.disk_cache_requests += 1;
            read.stats.remote.disk_cache_bytes += length as u64;
        } else {
            read.stats.disk_requests += 1;
            read.stats.disk_bytes += length as u64;
        }
        backend.submit()?;
        Ok(())
    }

    pub fn receive(
        &mut self,
        cache: Option<&DiskCache>,
        id: Identifier,
        chunk: WholeChunk,
    ) -> Result<()> {
        let read = self.reads.get_mut(&id).expect("chunk for a whole read");
        if let Some(length) = chunk.length {
            let length = usize::try_from(length)
                .map_err(|_| std::io::Error::other("object length exceeds address space"))?;
            read.length = Some(length);
            if let OpenFile::Remote(remote) = &read.request.request.file {
                let remote = Arc::new(remote.with_size(length as u64));
                read.object = cache.and_then(|cache| cache.open_object(&remote));
                read.request.request.file = OpenFile::Remote(remote);
            }
            if length > 0 {
                read.buffer = Some(allocate_buffer(&mut self.allocator, length));
            }
        }
        read.bytes
            .try_reserve(chunk.bytes.len())
            .map_err(std::io::Error::other)?;
        read.bytes.extend_from_slice(&chunk.bytes);
        read.stats.remote.http_bytes += chunk.bytes.len() as u64;
        assert!(
            read.pending.is_empty(),
            "transport must wait for its chunk acknowledgement"
        );
        read.pending = chunk.bytes;
        Ok(())
    }

    pub fn finish_network(&mut self, id: Identifier) -> bool {
        if let Some(read) = self.reads.get_mut(&id) {
            read.network_done = true;
            true
        } else {
            false
        }
    }

    pub fn drive(
        &mut self,
        backend: &mut IOBackend,
        http: &mut HttpEngine,
        cache: Option<&DiskCache>,
        disk_id: &mut Identifier,
        id: Identifier,
    ) -> Result<()> {
        loop {
            let Some(read) = self.reads.get_mut(&id) else {
                return Ok(());
            };
            if read.disk_pending || read.disk_source {
                return Ok(());
            }
            let capacity = read.buffer.as_ref().map_or(0, Slab::size);
            let take = read.pending.len().min(capacity - read.filled);
            if take > 0 {
                read.buffer.as_mut().unwrap().as_mut_slice()[read.filled..read.filled + take]
                    .copy_from_slice(&read.pending[..take]);
                read.pending.drain(..take);
                read.filled += take;
            }
            let flush = read.filled > 0 && read.filled == capacity
                || (read.network_done && read.pending.is_empty() && read.filled > 0);
            if flush {
                let offset = read.bytes.len() - read.pending.len() - read.filled;
                publish_cache_chunk(
                    &read.request.request.file,
                    offset,
                    &read.buffer.as_ref().unwrap().as_slice()[..read.filled],
                );
                if let Some(object) = &read.object {
                    let length = read.filled.div_ceil(BLOCK_SIZE) * BLOCK_SIZE;
                    read.buffer.as_mut().unwrap().as_mut_slice()[read.filled..length].fill(0);
                    backend.submit_write(
                        object.fd(),
                        offset as u64,
                        read.buffer.as_ref().unwrap().ptr,
                        length,
                        *disk_id,
                    )?;
                    self.disk_ops.insert(
                        *disk_id,
                        DiskOp {
                            read: id,
                            length,
                            write: true,
                        },
                    );
                    *disk_id += 1;
                    read.disk_pending = true;
                    backend.submit()?;
                    return Ok(());
                }
                read.filled = 0;
                continue;
            }
            if read.network_done && read.pending.is_empty() {
                self.finish(cache, id)?;
            } else {
                #[cfg(target_os = "linux")]
                http.resume_whole(&mut backend.ring, id)?;
                #[cfg(not(target_os = "linux"))]
                http.resume_whole(id)?;
            }
            return Ok(());
        }
    }

    pub fn complete_disk(
        &mut self,
        cache: Option<&DiskCache>,
        id: Identifier,
        result: i32,
    ) -> Option<Identifier> {
        let operation = self.disk_ops.remove(&id)?;
        #[cfg(test)]
        let result = if operation.write && self.reject_cache_writes {
            -libc::EIO
        } else {
            result
        };
        let read = self.reads.get_mut(&operation.read).unwrap();
        read.disk_pending = false;
        if result < 0
            || (operation.write && result as usize != operation.length)
            || (!operation.write && (result as usize) < operation.length)
        {
            let error = if result < 0 {
                std::io::Error::from_raw_os_error(-result)
            } else {
                std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "incomplete whole-file disk operation",
                )
            };
            if operation.write {
                // Caching is optional. Keep receiving the body, but never publish
                // a completion marker for a cache file whose write-back failed.
                tracing::warn!(%error, "whole-object cache write-back failed");
                read.object = None;
                read.filled = 0;
            } else {
                self.fail(operation.read, error.into());
            }
            return Some(operation.read);
        }
        if operation.write {
            let offset = read.bytes.len() - read.pending.len() - read.filled;
            cache
                .unwrap()
                .mark_resident(read.object.as_ref().unwrap(), offset, operation.length);
            read.filled = 0;
        } else {
            let data = &read.buffer.as_ref().unwrap().as_slice()[..operation.length];
            publish_cache_chunk(&read.request.request.file, read.bytes.len(), data);
            read.bytes.extend_from_slice(data);
        }
        Some(operation.read)
    }

    pub fn advance_disk(
        &mut self,
        backend: &mut IOBackend,
        http: &mut HttpEngine,
        cache: Option<&DiskCache>,
        disk_id: &mut Identifier,
        id: Identifier,
    ) -> Result<()> {
        let Some(read) = self.reads.get(&id) else {
            return Ok(());
        };
        if read.disk_source {
            if read.bytes.len() == read.length.unwrap() {
                self.finish(cache, id)
            } else {
                self.submit_disk_read(backend, disk_id, id)
            }
        } else {
            self.drive(backend, http, cache, disk_id, id)
        }
    }

    fn finish(&mut self, cache: Option<&DiskCache>, id: Identifier) -> Result<()> {
        let read = &self.reads[&id];
        let length = read.length.expect("finished whole read knows its length");
        if read.bytes.len() != length {
            self.fail(
                id,
                std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "whole-object body did not match Content-Length",
                )
                .into(),
            );
            return Ok(());
        }
        let read = self.reads.remove(&id).unwrap();
        if self.reads.is_empty() {
            self.allocator = None;
        }
        if matches!(read.request.request.file, OpenFile::Remote(_)) {
            memory_ctx()
                .compressed_cache()
                .mark_whole(read.request.request.file.clone(), length);
        }
        if !read.disk_source
            && let (Some(cache), Some(object)) = (cache, &read.object)
            && let Err(error) = cache.mark_whole(object, length as u64)
        {
            tracing::warn!(%error, "cannot persist whole-object cache completion");
        }
        self.completed.push(DataFlowRequest {
            data_flow_id: read.request.data_flow_id,
            operator_idx: read.request.operator_idx,
            request: read.stats,
            tracked_read_id: None,
            submitted_at: read.request.submitted_at,
        });
        self.ready.push(make_response(read.request, read.bytes));
        Ok(())
    }

    pub fn fail(&mut self, id: Identifier, error: super::IORequesterError) -> bool {
        if let Some(read) = self.reads.remove(&id) {
            if self.reads.is_empty() {
                self.allocator = None;
            }
            assert!(
                !read.disk_pending,
                "a live disk operation owns its destination"
            );
            self.failed.push(FailedIO {
                data_flow_id: read.request.data_flow_id,
                operator_idx: read.request.operator_idx,
                tracked_read_id: None,
                error,
            });
            true
        } else {
            false
        }
    }

    pub fn has_read(&self, id: Identifier) -> bool {
        self.reads.contains_key(&id)
    }
    pub fn network_in_flight(&self) -> usize {
        self.reads
            .values()
            .filter(|read| !read.disk_source && !read.network_done)
            .count()
    }
    pub fn has_disk_pending(&self) -> bool {
        !self.disk_ops.is_empty()
    }
    pub fn has_pending(&self) -> bool {
        !self.reads.is_empty()
    }
}

fn allocate_buffer(allocator: &mut Option<SlabAllocator>, length: usize) -> Slab {
    let capacity = length.min(READ_BUFFER_SIZE).div_ceil(BLOCK_SIZE) * BLOCK_SIZE;
    allocator
        .get_or_insert_with(|| SlabAllocator::new(false))
        .get_aligned_slab(capacity, BLOCK_SIZE, false)
}

fn make_response(mut request: DataFlowRequest<WholeRequest>, bytes: Vec<u8>) -> RoutedReadResponse {
    if let OpenFile::Remote(remote) = &request.request.file {
        request.request.file = OpenFile::Remote(Arc::new(remote.with_size(bytes.len() as u64)));
    }
    RoutedReadResponse {
        route: RequestRoute {
            data_flow_id: request.data_flow_id,
            operator_idx: request.operator_idx,
        },
        response: ReadResponse::new(
            request.request.id,
            request.request.file,
            vec![vec![ReadData::Compressed {
                offset: 0,
                bytes: vec![bytes::Bytes::from(bytes)],
            }]],
        ),
    }
}

/// Copy a completed, aligned chunk into only the cache fills owned by this
/// operation. Concurrent readers may already own or have filled other extents.
fn publish_cache_chunk(file: &OpenFile, offset: usize, data: &[u8]) {
    for lookup in memory_ctx()
        .compressed_cache()
        .get(file, offset, data.len())
    {
        if let Some(missing) = lookup.missing()
            && missing.fill_owner()
        {
            let start = missing.file_offset() - offset;
            let length = missing.len().min(data.len() - start);
            unsafe {
                std::ptr::copy_nonoverlapping(data[start..].as_ptr(), missing.dest(), length);
            }
            missing.commit_prefix(length);
            missing.wake_subscribers();
        }
    }
}
