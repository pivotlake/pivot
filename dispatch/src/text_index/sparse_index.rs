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
use crate::memory::cache::Cached;
use crate::memory::{BUFFER_SIZE, FrozenSlab, SlabSource};
use crate::text_index::Error;
use crate::text_index::format::{ByteSink, Footer, read_u32, read_u64};

const HEADER_LEN: usize = 8;
const ENTRY_LEN: usize = 16;

/// Accumulates the first token and location of each dictionary block, in
/// block order, and writes the sparse index.
#[derive(Default)]
pub struct SparseIndexEncoder {
    block_count: usize,
    entries: Vec<u8>,
    token_bytes: Vec<u8>,
}

impl SparseIndexEncoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, first_token: &[u8], block: FileRange) {
        self.token_bytes.extend_from_slice(first_token);
        self.entries
            .extend_from_slice(&(block.offset as u64).to_le_bytes());
        self.entries
            .extend_from_slice(&(block.len as u32).to_le_bytes());
        self.entries
            .extend_from_slice(&(self.token_bytes.len() as u32).to_le_bytes());
        self.block_count += 1;
    }

    pub fn encoded_len(&self) -> usize {
        HEADER_LEN + self.entries.len() + self.token_bytes.len()
    }

    /// Append the sparse index to `sink`. Fails when it would not fit one ring
    /// slot, which is where a reader keeps it.
    pub fn finish(self, sink: &mut impl ByteSink) -> Result<(), Error> {
        if self.encoded_len() > BUFFER_SIZE {
            return Err(Error::SparseIndexTooLarge {
                len: self.encoded_len(),
                max: BUFFER_SIZE,
            });
        }
        sink.write_bytes(&(self.block_count as u32).to_le_bytes());
        sink.write_bytes(&(self.token_bytes.len() as u32).to_le_bytes());
        sink.write_bytes(&self.entries);
        sink.write_bytes(&self.token_bytes);
        Ok(())
    }
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
    /// Check the sparse index `encoded` and copy it into one slab carved from
    /// `slabs`. `footer` is the footer of the file it came from.
    pub fn build(
        footer: Footer,
        encoded: &[u8],
        slabs: &mut impl SlabSource,
    ) -> Result<SparseIndexMeta, Error> {
        if encoded.len() > BUFFER_SIZE {
            return Err(Error::SparseIndexTooLarge {
                len: encoded.len(),
                max: BUFFER_SIZE,
            });
        }
        if encoded.len() < HEADER_LEN {
            return Err(Error::corrupt("sparse index is truncated"));
        }
        let block_count = read_u32(encoded, 0) as usize;
        let token_bytes_len = read_u32(encoded, 4) as usize;
        if encoded.len() != HEADER_LEN + ENTRY_LEN * block_count + token_bytes_len {
            return Err(Error::corrupt("sparse index length is inconsistent"));
        }
        let mut previous_end = 0;
        for block in 0..block_count {
            let end = read_u32(encoded, HEADER_LEN + ENTRY_LEN * block + 12) as usize;
            if end < previous_end || end > token_bytes_len {
                return Err(Error::corrupt("sparse index tokens are inconsistent"));
            }
            previous_end = end;
        }

        let mut slab = slabs.get_aligned_slab(encoded.len(), 8, false);
        slab.as_mut_slice().copy_from_slice(encoded);
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
    pub fn find_block(&self, token: &[u8]) -> Option<FileRange> {
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
        let entry_offset = HEADER_LEN + ENTRY_LEN * block;
        let bytes = self.bytes.as_slice();
        Some(FileRange::new(
            read_u64(bytes, entry_offset) as usize,
            read_u32(bytes, entry_offset + 8) as usize,
        ))
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
