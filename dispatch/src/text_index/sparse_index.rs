//! The sparse index: for a token, which dict index it would be in.
//!
//! One entry per dict index of the file: the smallest token that dict index
//! holds, and where the dict index is. The dict indexes hold the file's
//! tokens in ascending order, so a token belongs to the last dict index whose
//! smallest token is not above it:
//!
//! ```text
//!   smallest token   dict index at
//!   ──────────────   ─────────────
//!   "apple"          0 .. 4000
//!   "date"           4000 .. 9000      ◀── "kiwi": "date" ≤ "kiwi" < "melon"
//!   "melon"          9000 .. 12000
//! ```

use crate::io::FileRange;
use crate::memory::cache::Cached;
use crate::memory::{FrozenSlab, SlabSource};
use crate::text_index::token_ranges::{TokenRanges, TokenRangesBuilder, TokenRangesMeta};

/// An index file's sparse index, all of it in memory for as long as this is
/// held.
pub struct SparseIndex {
    smallest_tokens: TokenRanges,
}

impl SparseIndex {
    /// How many dict indexes the file has.
    pub fn count_dict_indexes(&self) -> usize {
        self.smallest_tokens.count_tokens()
    }

    /// Where the dict index is that holds `token` if the file holds it at
    /// all. `None` when `token` sorts before every token of the file.
    pub fn find_dict_index(&self, token: &[u8]) -> Option<FileRange> {
        self.smallest_tokens.find_range_at_or_below(token)
    }
}

impl Cached for SparseIndex {
    type Meta = TokenRangesMeta;

    fn from_slabs(meta: &Self::Meta, slabs: Vec<FrozenSlab>) -> Self {
        Self {
            smallest_tokens: TokenRanges::from_slabs(meta, slabs),
        }
    }
}

/// Gathers a file's dict indexes, in the order they hold the tokens, into a
/// sparse index.
#[derive(Default)]
pub struct SparseIndexBuilder {
    smallest_tokens: TokenRangesBuilder,
}

impl SparseIndexBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add the next dict index: its smallest token, which must sort after the
    /// one added before, and where it is. The memory comes from `source`.
    pub fn push(
        &mut self,
        source: &mut impl SlabSource,
        smallest_token: &[u8],
        dict_index: FileRange,
    ) {
        self.smallest_tokens
            .push(source, smallest_token, dict_index);
    }

    /// The sparse index, ready to be looked up where it was gathered.
    pub fn finish(self) -> SparseIndex {
        SparseIndex {
            smallest_tokens: self.smallest_tokens.finish(),
        }
    }

    /// What the cache whose fill the sparse index was gathered into keeps
    /// beside it.
    pub fn into_meta(self) -> TokenRangesMeta {
        self.smallest_tokens.into_meta()
    }
}
