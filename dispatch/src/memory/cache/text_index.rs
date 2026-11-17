//! The text index family: an index file's bytes as read ([`CompressedCache`])
//! and the three forms a lookup works on, each derived from part of them.
//!
//! ```text
//!   lookup "kiwi"
//!     │
//!     ▼
//!   sparse index ──▶ dictionary block ──▶ posting list ──▶ rows
//!   which block      where the token's    the rows
//!   could hold it    rows are
//! ```
//!
//! The forms are read in that order, and each is useless without the one
//! before it: a cached dictionary block can only be found through the file's
//! sparse index. So the family's shares put the whole target on the sparse
//! index, which makes the evictor step over it for as long as any other
//! member holds a slot (see [`family`](super::family)). Among the other three
//! nothing is protected and their weights decide.

use crate::io::OpenFile;
use crate::memory::cache::compressed::CompressedCache;
use crate::memory::cache::derived_cache::Cache;
use crate::text_index::{DictBlock, Postings, SparseIndex};
use std::sync::Arc;

/// The text index family's caches.
#[derive(Clone)]
pub struct TextIndexCaches {
    /// Index file bytes as read. Every read of an index file goes through
    /// this cache, so what it brings in is balanced against the forms derived
    /// from it.
    pub file_bytes: Arc<CompressedCache>,
    pub sparse_index: Arc<Cache<SparseIndex>>,
    pub dict_blocks: Arc<Cache<DictBlock>>,
    pub postings: Arc<Cache<Postings>>,
}

impl TextIndexCaches {
    /// Bytes of the index file `index_file` held in the forms a lookup uses:
    /// its sparse index, dictionary blocks and posting lists. A lookup in a
    /// file with more of them cached waits on fewer reads.
    pub fn count_cached_bytes(&self, index_file: &OpenFile) -> usize {
        self.sparse_index.count_cached_bytes(index_file)
            + self.dict_blocks.count_cached_bytes(index_file)
            + self.postings.count_cached_bytes(index_file)
    }
}
