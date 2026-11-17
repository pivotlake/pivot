//! [`SlabStream`]: an append-only byte stream held on the ring, and
//! [`SlabReader`] to read one back.
//!
//! Building an index produces megabytes a few bytes at a time: the sorted
//! runs, the index file, and the sections held beside the file until their
//! place in it comes. Each is a stream, written straight into chunks carved
//! from the ring, so nothing of an index being built lives outside it.
//!
//! ```text
//!   written:  [ chunk ████████ ][ chunk ████████ ][ chunk ████░░░░ ]
//!
//!   taken:    [ chunk ████████ ][ chunk ████████ ][ ████ ]
//!                                                        [ ░░░░ ]  starts the next stream
//! ```
//!
//! A stream does not own the allocator its chunks come from: streams that
//! live and die together share one, and each write names it.

use crate::memory::{Slab, SlabAllocator};
use crate::text_index::Error;
use crate::text_index::format::ByteSink;
use std::collections::VecDeque;

/// Bytes in a chunk taken from the allocator. Eight fill a ring buffer.
const CHUNK_LEN: usize = 256 * 1024;

#[derive(Default)]
pub(crate) struct SlabStream {
    /// The stream's bytes, in order: every slab is full but the last, which
    /// is written up to `written_in_last` and holds at least one byte.
    slabs: Vec<Slab>,
    written_in_last: usize,
    /// A chunk, or the unwritten end of one, kept to start the next stream in.
    spare: Option<Slab>,
    len: usize,
}

impl SlabStream {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bytes written so far: the offset the next byte lands at.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Somewhere to append to the stream, taking new chunks from `allocator`.
    pub fn write_with<'a>(&'a mut self, allocator: &'a mut SlabAllocator) -> StreamSink<'a> {
        StreamSink {
            stream: self,
            allocator,
        }
    }

    /// The stream's bytes, in order, a chunk at a time.
    pub fn written_chunks(&self) -> impl Iterator<Item = &[u8]> + Clone + '_ {
        let last = self.slabs.len().saturating_sub(1);
        let written_in_last = self.written_in_last;
        self.slabs.iter().enumerate().map(move |(index, slab)| {
            if index == last {
                &slab.as_slice()[..written_in_last]
            } else {
                slab.as_slice()
            }
        })
    }

    /// The stream as the slabs holding it, in order, each exactly as long as
    /// what was written to it. Leaves this empty, ready for the next stream.
    pub fn take_slabs(&mut self) -> Vec<Slab> {
        if let Some(last) = self.slabs.last_mut() {
            let unwritten = last.split_off(self.written_in_last);
            if unwritten.size() > 0 {
                self.spare = Some(unwritten);
            }
        }
        self.written_in_last = 0;
        self.len = 0;
        std::mem::take(&mut self.slabs)
    }

    /// Forget the stream's bytes, keeping its first chunk for the next
    /// stream and giving the others back.
    pub fn clear(&mut self) {
        self.slabs.truncate(1);
        if let Some(first) = self.slabs.pop() {
            self.spare = Some(first);
        }
        self.written_in_last = 0;
        self.len = 0;
    }
}

/// Appends to a [`SlabStream`]. See [`SlabStream::write_with`].
pub(crate) struct StreamSink<'a> {
    stream: &'a mut SlabStream,
    allocator: &'a mut SlabAllocator,
}

impl StreamSink<'_> {
    /// Append `bytes` in one piece, never across two chunks, so a reader can
    /// look at them in place. They must fit a chunk.
    pub fn write_unsplit(&mut self, bytes: &[u8]) {
        assert!(
            bytes.len() <= CHUNK_LEN,
            "{} bytes do not fit one chunk",
            bytes.len()
        );
        if self.count_room_in_last() < bytes.len() {
            // What is left of the chunk is too short: end it here.
            let written_in_last = self.stream.written_in_last;
            if let Some(last) = self.stream.slabs.last_mut() {
                last.split_off(written_in_last);
            }
            self.start_chunk(bytes.len());
        }
        self.write_bytes(bytes);
    }

    fn count_room_in_last(&self) -> usize {
        match self.stream.slabs.last() {
            Some(last) => last.size() - self.stream.written_in_last,
            None => 0,
        }
    }

    /// Add a chunk with room for at least `min_len` bytes to the stream.
    fn start_chunk(&mut self, min_len: usize) {
        let chunk = match self.stream.spare.take() {
            Some(spare) if spare.size() >= min_len => spare,
            _ => self.allocator.get_slab_of_size(CHUNK_LEN, false),
        };
        self.stream.slabs.push(chunk);
        self.stream.written_in_last = 0;
    }
}

impl ByteSink for StreamSink<'_> {
    fn write_bytes(&mut self, mut bytes: &[u8]) {
        self.stream.len += bytes.len();
        while !bytes.is_empty() {
            let room = self.count_room_in_last();
            if room == 0 {
                self.start_chunk(1);
                continue;
            }
            let take = bytes.len().min(room);
            let written = self.stream.written_in_last;
            let last = self.stream.slabs.len() - 1;
            self.stream.slabs[last].as_mut_slice()[written..written + take]
                .copy_from_slice(&bytes[..take]);
            self.stream.written_in_last += take;
            bytes = &bytes[take..];
        }
    }
}

/// Reads a stream's slabs front to back, letting go of each once it has been
/// read, so a stream being consumed gives its memory back as it goes.
pub(crate) struct SlabReader {
    slabs: VecDeque<Slab>,
    offset_in_first: usize,
}

impl SlabReader {
    pub fn new(slabs: Vec<Slab>) -> Self {
        Self {
            slabs: slabs.into(),
            offset_in_first: 0,
        }
    }

    pub fn is_at_end(&self) -> bool {
        self.slabs.is_empty()
    }

    /// The unread bytes of the slab being read. Empty only at the end.
    pub fn peek_front(&self) -> &[u8] {
        match self.slabs.front() {
            Some(slab) => &slab.as_slice()[self.offset_in_first..],
            None => &[],
        }
    }

    /// Move past `len` bytes of [`peek_front`](Self::peek_front).
    pub fn skip(&mut self, len: usize) {
        self.offset_in_first += len;
        let is_first_read = self
            .slabs
            .front()
            .is_some_and(|first| first.size() == self.offset_in_first);
        if is_first_read {
            self.slabs.pop_front();
            self.offset_in_first = 0;
        }
    }

    pub fn read_exact(&mut self, out: &mut [u8]) -> Result<(), Error> {
        let mut filled = 0;
        while filled < out.len() {
            let available = self.peek_front();
            if available.is_empty() {
                return Err(Error::corrupt("stream is truncated"));
            }
            let take = available.len().min(out.len() - filled);
            out[filled..filled + take].copy_from_slice(&available[..take]);
            filled += take;
            self.skip(take);
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

    pub fn read_u32(&mut self) -> Result<u32, Error> {
        let mut value = [0u8; 4];
        self.read_exact(&mut value)?;
        Ok(u32::from_le_bytes(value))
    }

    /// Move the next `len` bytes to `sink`.
    pub fn copy_to(&mut self, len: usize, sink: &mut impl ByteSink) -> Result<(), Error> {
        let mut remaining = len;
        while remaining > 0 {
            let available = self.peek_front();
            if available.is_empty() {
                return Err(Error::corrupt("stream is truncated"));
            }
            let take = available.len().min(remaining);
            sink.write_bytes(&available[..take]);
            self.skip(take);
            remaining -= take;
        }
        Ok(())
    }
}
