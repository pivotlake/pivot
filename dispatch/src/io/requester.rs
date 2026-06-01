use crate::Identifier;
use crate::io::DataFlowRequest;
use crate::io::backend::IOBackend;
use crate::memory::{BUFFER_SIZE, ReadBuffer, WriteBuffer, memory_ctx};
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

const RING_SIZE: u32 = 32;

/// Bridges dataflow operators and the I/O backend, holding state on outstanding requests and
/// returning responses together with the original requested context.
///
/// Held one per worker
pub struct IORequester {
    backend: IOBackend,
    pending_io_requests: HashMap<Identifier, (WriteBuffer, DataFlowRequest)>,
    next_id: Identifier,
}

impl IORequester {
    pub fn new() -> Self {
        Self {
            backend: IOBackend::new(RING_SIZE).expect("Unable to create backend"),
            pending_io_requests: Default::default(),
            next_id: 0,
        }
    }

    /// Acquires a dirty write buffer, submits a read to the backend, and
    /// flushes immediately.
    pub fn request(&mut self, request: DataFlowRequest) -> Result<()> {
        let mut buffer = memory_ctx().get_write_buffer(false);

        self.backend.submit_read(
            request.request.location.raw_fd,
            request.request.location.offset as u64,
            &mut buffer,
            BUFFER_SIZE,
            self.next_id,
        )?;
        self.pending_io_requests
            .insert(self.next_id, (buffer, request));
        self.next_id += 1;
        self.backend.submit()?;

        Ok(())
    }

    /// Returns `true` if any reads have not yet completed.
    pub fn has_pending(&mut self) -> bool {
        !self.pending_io_requests.is_empty()
    }

    /// Drains completed reads from the backend, inserts each buffer into the
    /// file cache, and returns them paired with the originating request.
    pub fn completions(&mut self) -> Result<impl Iterator<Item = (ReadBuffer, DataFlowRequest)>> {
        let identifiers = self.backend.completions()?;
        Ok(identifiers.into_iter().map(|(_size, i)| {
            let (buffer, request) = self.pending_io_requests.remove(&i).unwrap();
            let read_buffer = memory_ctx()
                .file_cache()
                .insert(request.request.location.clone(), buffer);
            (read_buffer, request)
        }))
    }

    /// Blocks until at least one pending read completes.
    pub fn wait(&mut self) -> Result<()> {
        self.backend.submit_and_wait(1)?;
        Ok(())
    }
}
