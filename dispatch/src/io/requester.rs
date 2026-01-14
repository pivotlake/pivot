use crate::identified::Identifier;
use crate::io::PipelineRequest;
use crate::io::backend::IOBackend;
use crate::io::cache::CACHE;
use crate::io::disk_buffer::{DIO_ALIGNMENT, DiskBuffer};
use bytes::Bytes;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Backend(#[from] super::backend::Error),
    #[error("{0}")]
    IO(#[from] std::io::Error),
}

type Result<T, E = Error> = std::result::Result<T, E>;

pub struct IORequester {
    backend: IOBackend,
    pending_io_requests: HashMap<Identifier, (DiskBuffer, PipelineRequest, usize, usize)>,
    max_io_in_parallel: usize,
    next_id: Identifier,
}

impl IORequester {
    pub fn new(backend: IOBackend) -> Self {
        Self {
            backend,
            pending_io_requests: Default::default(),
            max_io_in_parallel: 1,
            next_id: 0,
        }
    }

    pub fn backend(&self) -> &IOBackend {
        &self.backend
    }

    pub fn request(&mut self, request: PipelineRequest) -> Result<()> {
        let location = &request.request.location;

        // Align offset DOWN to nearest alignment boundary
        let aligned_offset = location.offset & !(*DIO_ALIGNMENT - 1);
        // How far into the aligned buffer your actual data starts
        let offset_within_buffer = location.offset - aligned_offset;
        // Size needs to cover the padding + actual data, rounded UP to alignment
        let aligned_size =
            (offset_within_buffer + location.size + *DIO_ALIGNMENT - 1) & !(*DIO_ALIGNMENT - 1);

        let mut buffer = DiskBuffer::new(aligned_size)?;

        self.backend.submit_read(
            location.raw_fd,
            aligned_offset as u64,
            &mut buffer,
            aligned_size,
            self.next_id,
        )?;
        self.pending_io_requests.insert(
            self.next_id,
            (buffer, request, offset_within_buffer, aligned_size),
        );
        self.next_id += 1;
        self.backend.submit()?;

        Ok(())
    }

    pub fn has_available(&mut self) -> bool {
        !self.backend.is_submission_full()
            && self.pending_io_requests.len() < self.max_io_in_parallel
    }

    pub fn has_pending(&mut self) -> bool {
        !self.pending_io_requests.is_empty()
    }

    pub fn completions(&mut self) -> Result<impl Iterator<Item = (Bytes, PipelineRequest)>> {
        let identifiers = self.backend.completions()?;
        Ok(identifiers.into_iter().map(|(size, i)| {
            let (buffer, request, offset, aligned_size) =
                self.pending_io_requests.remove(&i).unwrap();

            // It *should* supported to receive an IO response from the kernel that wasn't the size
            // we requested, but this is not currently supported - and is a ticking time bomb :)
            if size != aligned_size {
                panic!(
                    "request {:?} does not match returned {:?}",
                    request.request.location.size + offset,
                    size
                );
            }

            let bytes =
                Bytes::from_owner(buffer).slice(offset..request.request.location.size + offset);
            CACHE.insert(request.request.location.clone(), bytes.clone());
            (bytes, request)
        }))
    }

    pub fn wait(&mut self) -> Result<()> {
        self.backend.submit_and_wait(1)?;
        Ok(())
    }
}
