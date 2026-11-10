//! [`SlabWriter`]: an append-only byte stream held on the ring, and
//! [`SlabReader`] to read one back.
//!
//! An index being built is megabytes of bytes produced a few at a time. They
//! are gathered in a small chunk and moved to an exactly sized slab each time
//! it fills, so the stream ends up as a list of full slabs that can be handed
//! to a file write or kept until a merge reads them.

use crate::memory::{FileBytes, Slab, SlabAllocator};
use crate::text_index::Error;
use crate::text_index::format::ByteSink;

/// Bytes gathered before they are moved to a slab.
const CHUNK_LEN: usize = 256 * 1024;

pub(crate) struct SlabWriter {
    allocator: SlabAllocator,
    chunk: Vec<u8>,
    slabs: Vec<Slab>,
    len: usize,
}

impl SlabWriter {
    pub fn new() -> Self {
        Self {
            allocator: SlabAllocator::new(false),
            chunk: Vec::with_capacity(CHUNK_LEN),
            slabs: Vec::new(),
            len: 0,
        }
    }

    /// Bytes written so far: the offset the next byte lands at.
    pub fn len(&self) -> usize {
        self.len
    }

    /// The stream as the slabs it was written into, in order.
    pub fn into_slabs(mut self) -> Vec<Slab> {
        self.move_chunk_to_slab();
        self.slabs
    }

    /// The stream as the bytes of a file to write.
    pub fn into_file_bytes(self) -> FileBytes {
        let mut file = FileBytes::new();
        for slab in self.into_slabs() {
            file.push_slab(slab);
        }
        file
    }

    fn move_chunk_to_slab(&mut self) {
        if self.chunk.is_empty() {
            return;
        }
        let mut slab = self.allocator.get_slab_of_size(self.chunk.len(), false);
        slab.as_mut_slice().copy_from_slice(&self.chunk);
        self.slabs.push(slab);
        self.chunk.clear();
    }
}

impl ByteSink for SlabWriter {
    fn write_bytes(&mut self, mut bytes: &[u8]) {
        self.len += bytes.len();
        while !bytes.is_empty() {
            let take = bytes.len().min(CHUNK_LEN - self.chunk.len());
            self.chunk.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];
            if self.chunk.len() == CHUNK_LEN {
                self.move_chunk_to_slab();
            }
        }
    }
}

/// Reads a stream a [`SlabWriter`] produced, front to back.
pub(crate) struct SlabReader<'a> {
    slabs: &'a [Slab],
    offset_in_first: usize,
}

impl<'a> SlabReader<'a> {
    pub fn new(slabs: &'a [Slab]) -> Self {
        Self {
            slabs,
            offset_in_first: 0,
        }
    }

    pub fn is_at_end(&self) -> bool {
        self.slabs.is_empty()
    }

    /// The unread bytes of the slab being read. Empty only at the end.
    fn available(&self) -> &'a [u8] {
        match self.slabs.first() {
            Some(slab) => &slab.as_slice()[self.offset_in_first..],
            None => &[],
        }
    }

    fn consume(&mut self, len: usize) {
        self.offset_in_first += len;
        if self.offset_in_first == self.slabs[0].size() {
            self.slabs = &self.slabs[1..];
            self.offset_in_first = 0;
        }
    }

    pub fn read_exact(&mut self, out: &mut [u8]) -> Result<(), Error> {
        let mut filled = 0;
        while filled < out.len() {
            let available = self.available();
            if available.is_empty() {
                return Err(Error::corrupt("stream is truncated"));
            }
            let take = available.len().min(out.len() - filled);
            out[filled..filled + take].copy_from_slice(&available[..take]);
            filled += take;
            self.consume(take);
        }
        Ok(())
    }

    pub fn read_varint(&mut self) -> Result<u64, Error> {
        let mut value = 0u64;
        for index in 0..10 {
            let mut byte = [0u8; 1];
            self.read_exact(&mut byte)?;
            value |= ((byte[0] & 0x7f) as u64) << (7 * index);
            if byte[0] & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(Error::corrupt("varint is truncated"))
    }

    /// Append the next `count` little-endian `u32`s to `out`.
    pub fn read_u32s_into(&mut self, count: usize, out: &mut Vec<u32>) -> Result<(), Error> {
        let mut remaining = count;
        while remaining > 0 {
            let available = self.available();
            let whole = (available.len() / 4).min(remaining);
            if whole == 0 {
                // The next value straddles two slabs, or the stream ended.
                let mut value = [0u8; 4];
                self.read_exact(&mut value)?;
                out.push(u32::from_le_bytes(value));
                remaining -= 1;
                continue;
            }
            out.extend(
                available[..4 * whole]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|value| u32::from_le_bytes(*value)),
            );
            self.consume(4 * whole);
            remaining -= whole;
        }
        Ok(())
    }
}
