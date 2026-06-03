use crate::Identifier;
use crate::io::DataFlowRequest;
use crate::io::backend::IOBackend;
use crate::memory::ReadBuffer;
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
    /// In-flight reads, each holding a pin on its destination slot so it can't
    /// be evicted out from under the read (e.g. on cancellation).
    pending_io_requests: HashMap<Identifier, (ReadBuffer, DataFlowRequest)>,
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

    /// Submits the block's read straight into its (pinned) cache slot and
    /// flushes immediately. No intermediate buffer: the slot region is the read
    /// target. We hold our own pin on the slot until the read completes, so it
    /// survives the read even if the issuing query is cancelled meanwhile.
    pub fn request(&mut self, request: DataFlowRequest) -> Result<()> {
        let pin = request.request.block.pin();
        self.backend.submit_read(
            request.request.fd,
            request.request.block.file_offset as u64,
            request.request.block.dest(),
            request.request.block.len,
            self.next_id,
        )?;
        self.pending_io_requests.insert(self.next_id, (pin, request));
        self.next_id += 1;
        self.backend.submit()?;

        Ok(())
    }

    /// Returns `true` if any reads have not yet completed.
    pub fn has_pending(&mut self) -> bool {
        !self.pending_io_requests.is_empty()
    }

    /// Drains completed reads — each block's bytes have already landed in its
    /// cache slot, so we just [`commit`](crate::memory::file_cache::MissingBlock::commit)
    /// (mark its sub-blocks valid) and yield the originating request so the
    /// issuing operator can count it down.
    pub fn completions(&mut self) -> Result<impl Iterator<Item = DataFlowRequest>> {
        let identifiers = self.backend.completions()?;
        Ok(identifiers.into_iter().map(|(_size, i)| {
            let (pin, request) = self.pending_io_requests.remove(&i).unwrap();
            request.request.block.commit();
            drop(pin); // release our read-duration pin now the bytes are committed
            request
        }))
    }

    /// Blocks until at least one pending read completes.
    pub fn wait(&mut self) -> Result<()> {
        self.backend.submit_and_wait(1)?;
        Ok(())
    }
}
