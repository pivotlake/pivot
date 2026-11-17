//! The sparse index: which dictionary block a token would be in.
//!
//! It keeps the first token of every dictionary block and where the block is,
//! so finding a token's block is a binary search in memory and reading the
//! token is then one block:
//!
//! ```text
//!   sparse index                       dictionary blocks
//!   ┌─────────┬──────────────┐
//!   │ "apple" │ offset, len ─┼───────▶ [apple … cherry]
//!   │ "date"  │ offset, len ─┼───────▶ [date … mango]      ◀── "kiwi" sorts here
//!   │ "melon" │ offset, len ─┼───────▶ [melon … zebra]
//!   └─────────┴──────────────┘
//! ```
//!
//! The file form and the cached form ([`SparseIndex`]) are the same bytes:
//!
//! ```text
//!   ┌─────────────┬─────────────────┬──────────────────────────────┬─────────────┐
//!   │ block count │ token bytes len │ entries, 16 B × blocks       │ token bytes │
//!   │ u32         │ u32             │ (offset u64, len u32,        │             │
//!   │             │                 │  first token's end u32)      │             │
//!   └─────────────┴─────────────────┴──────────────────────────────┴─────────────┘
//! ```
//!
//! Every lookup starts here, and without it the cached dictionary blocks of
//! the file cannot be found, which is why its cache is the one the text index
//! family never gives up before the others.

use crate::io::FileRange;
use crate::memory::cache::{CacheFill, Cached};
use crate::memory::{BUFFER_SIZE, FrozenSlab, SlabAllocator};
use crate::text_index::Error;
use crate::text_index::format::{ByteSink, Footer, FragmentReader, read_u32, read_u64};
use crate::text_index::slab_stream::SlabStream;
use bytes::Bytes;

const HEADER_LEN: usize = 8;
const ENTRY_LEN: usize = 16;

/// Accumulates the first token and location of each dictionary block, in
/// block order, and writes the sparse index. What it accumulates is held on
/// the ring, in the two parts the sparse index is written as.
#[derive(Default)]
pub struct SparseIndexEncoder {
    block_count: usize,
    entries: SlabStream,
    token_bytes: SlabStream,
}

impl SparseIndexEncoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the next block, holding the record in chunks from `allocator`.
    pub fn push(&mut self, allocator: &mut SlabAllocator, first_token: &[u8], block: FileRange) {
        self.token_bytes
            .write_with(allocator)
            .write_bytes(first_token);
        let mut entries = self.entries.write_with(allocator);
        entries.write_bytes(&(block.offset as u64).to_le_bytes());
        entries.write_bytes(&(block.len as u32).to_le_bytes());
        entries.write_bytes(&(self.token_bytes.len() as u32).to_le_bytes());
        self.block_count += 1;
    }

    pub fn encoded_len(&self) -> usize {
        HEADER_LEN + self.entries.len() + self.token_bytes.len()
    }

    /// Append the sparse index to `sink`. Fails when it would not fit one ring
    /// slot, which is where a reader keeps it.
    pub fn finish(&self, sink: &mut impl ByteSink) -> Result<(), Error> {
        if self.encoded_len() > BUFFER_SIZE {
            return Err(Error::SparseIndexTooLarge {
                len: self.encoded_len(),
                max: BUFFER_SIZE,
            });
        }
        sink.write_bytes(&(self.block_count as u32).to_le_bytes());
        sink.write_bytes(&(self.token_bytes.len() as u32).to_le_bytes());
        let parts = self.entries.written_chunks();
        for chunk in parts.chain(self.token_bytes.written_chunks()) {
            sink.write_bytes(chunk);
        }
        Ok(())
    }
}

/// Where a dictionary block and the posting lists of its tokens are in the
/// index file. The lists are written right before their block, so they are
/// the bytes between the previous block and this one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockLocation {
    pub block: FileRange,
    pub postings: FileRange,
}

/// An index file's sparse index and footer, pinned in the cache for as long
/// as this is held.
pub struct SparseIndex {
    footer: Footer,
    bytes: FrozenSlab,
}

/// What the cache keeps beside a sparse index's bytes: the file's footer, so
/// a lookup that finds the sparse index cached needs nothing else from the
/// file's tail.
pub struct SparseIndexMeta {
    footer: Footer,
}

impl Cached for SparseIndex {
    type Meta = SparseIndexMeta;

    fn from_slabs(meta: &Self::Meta, slabs: Vec<FrozenSlab>) -> Self {
        let bytes = slabs
            .into_iter()
            .next()
            .expect("a sparse index is one slab");
        Self {
            footer: meta.footer,
            bytes,
        }
    }
}

impl SparseIndex {
    /// Copy the sparse index `encoded` holds (the fragments of one read, in
    /// order) into one slab carved from `fill`, and check it there. `footer`
    /// is the footer of the file it came from.
    pub fn build(
        footer: Footer,
        encoded: &[Bytes],
        fill: &mut CacheFill<'_>,
    ) -> Result<SparseIndexMeta, Error> {
        let mut reader = FragmentReader::new(encoded);
        let encoded_len = reader.remaining();
        if encoded_len > BUFFER_SIZE {
            return Err(Error::SparseIndexTooLarge {
                len: encoded_len,
                max: BUFFER_SIZE,
            });
        }
        if encoded_len < HEADER_LEN {
            return Err(Error::corrupt("sparse index is truncated"));
        }
        let mut header = [0u8; HEADER_LEN];
        reader.read_exact(&mut header)?;
        let block_count = read_u32(&header, 0) as usize;
        let token_bytes_len = read_u32(&header, 4) as usize;
        if encoded_len != HEADER_LEN + ENTRY_LEN * block_count + token_bytes_len {
            return Err(Error::corrupt("sparse index length is inconsistent"));
        }

        let mut slab = fill.get_contiguous_bytes(encoded_len);
        slab.as_mut_slice()[..HEADER_LEN].copy_from_slice(&header);
        reader.read_exact(&mut slab.as_mut_slice()[HEADER_LEN..])?;
        let encoded = slab.as_slice();
        let mut previous_end = 0;
        let mut previous_block_end = 0;
        for block in 0..block_count {
            let end = read_u32(encoded, HEADER_LEN + ENTRY_LEN * block + 12) as usize;
            if end < previous_end || end > token_bytes_len {
                return Err(Error::corrupt("sparse index tokens are inconsistent"));
            }
            previous_end = end;
            let block_offset = read_u64(encoded, HEADER_LEN + ENTRY_LEN * block) as usize;
            if block_offset < previous_block_end {
                return Err(Error::corrupt("sparse index blocks are out of order"));
            }
            previous_block_end =
                block_offset + read_u32(encoded, HEADER_LEN + ENTRY_LEN * block + 8) as usize;
        }
        Ok(SparseIndexMeta { footer })
    }

    pub fn footer(&self) -> &Footer {
        &self.footer
    }

    pub fn block_count(&self) -> usize {
        read_u32(self.bytes.as_slice(), 0) as usize
    }

    /// The dictionary block `token` is in if the index holds it: the last
    /// block whose first token is not above it. `None` when `token` sorts
    /// before every token of the index.
    pub fn find_block(&self, token: &[u8]) -> Option<BlockLocation> {
        let (mut low, mut high) = (0, self.block_count());
        while low < high {
            let middle = low + (high - low) / 2;
            if self.first_token(middle) <= token {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        let block = low.checked_sub(1)?;
        let block_range = self.block_range(block);
        let postings_start = match block {
            0 => 0,
            _ => {
                let previous = self.block_range(block - 1);
                previous.offset + previous.len
            }
        };
        Some(BlockLocation {
            block: block_range,
            postings: FileRange::new(postings_start, block_range.offset - postings_start),
        })
    }

    fn block_range(&self, block: usize) -> FileRange {
        let entry_offset = HEADER_LEN + ENTRY_LEN * block;
        let bytes = self.bytes.as_slice();
        FileRange::new(
            read_u64(bytes, entry_offset) as usize,
            read_u32(bytes, entry_offset + 8) as usize,
        )
    }

    fn first_token_end(&self, block: usize) -> usize {
        read_u32(self.bytes.as_slice(), HEADER_LEN + ENTRY_LEN * block + 12) as usize
    }

    fn first_token(&self, block: usize) -> &[u8] {
        let start = if block == 0 {
            0
        } else {
            self.first_token_end(block - 1)
        };
        let tokens = &self.bytes.as_slice()[HEADER_LEN + ENTRY_LEN * self.block_count()..];
        &tokens[start..self.first_token_end(block)]
    }
}
