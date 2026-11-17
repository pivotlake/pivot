//! A dict index: a run of the file's tokens in ascending order, each with
//! where its postings are.
//!
//! ```text
//!   token      postings at
//!   ─────      ───────────
//!   "date"     52000 .. 52040
//!   "fig"      52040 .. 61000
//!   "kiwi"     61000 .. 61012
//!   "mango"    61012 .. 70000
//! ```
//!
//! A file's tokens are cut into several dict indexes so that a lookup needs
//! only the one its token would be in, which the sparse index names.

use crate::io::FileRange;
use crate::memory::cache::Cached;
use crate::memory::{FrozenSlab, SlabSource};
use crate::text_index::token_ranges::{TokenRanges, TokenRangesBuilder, TokenRangesMeta};

/// One dict index, all of it in memory for as long as this is held.
pub struct DictIndex {
    tokens: TokenRanges,
}

impl DictIndex {
    pub fn count_tokens(&self) -> usize {
        self.tokens.count_tokens()
    }

    /// Where the postings of `token` are. `None` when the dict index does not
    /// hold the token.
    pub fn find_postings(&self, token: &[u8]) -> Option<FileRange> {
        self.tokens.find_range_of(token)
    }
}

impl Cached for DictIndex {
    type Meta = TokenRangesMeta;

    fn from_slabs(meta: &Self::Meta, slabs: Vec<FrozenSlab>) -> Self {
        Self {
            tokens: TokenRanges::from_slabs(meta, slabs),
        }
    }
}

/// Gathers tokens, which must arrive in ascending order, into a dict index.
#[derive(Default)]
pub struct DictIndexBuilder {
    tokens: TokenRangesBuilder,
}

impl DictIndexBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `token`, which must sort after every token added so far, and where
    /// its postings are. The memory comes from `source`.
    pub fn push(&mut self, source: &mut impl SlabSource, token: &[u8], postings: FileRange) {
        self.tokens.push(source, token, postings);
    }

    /// The dict index, ready to be looked up where it was gathered.
    pub fn finish(self) -> DictIndex {
        DictIndex {
            tokens: self.tokens.finish(),
        }
    }

    /// What the cache whose fill the dict index was gathered into keeps
    /// beside it.
    pub fn into_meta(self) -> TokenRangesMeta {
        self.tokens.into_meta()
    }
}
