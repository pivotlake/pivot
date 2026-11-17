//! A text search index over one column of one data file: for every token of
//! the column, the rows it appears in.
//!
//! An index file has three kinds of pieces, and a lookup goes through one of
//! each:
//!
//! ```text
//!   piece          one per     answers
//!   ─────          ───────     ───────
//!   sparse index   file        which dict index would hold this token
//!   dict index     token run   where this token's postings are
//!   postings       token       which rows the token appears in
//! ```
//!
//! A piece is looked up only once all of it is in memory: a lookup never
//! reads. Each kind has a cache of its own in the text index family (see
//! [`TextIndexCaches`](crate::memory::cache::TextIndexCaches)).
//!
//! In memory a piece is a [`SlabVec`](crate::memory::SlabVec), or two. It is
//! gathered by a builder whose memory comes from a
//! [`SlabSource`](crate::memory::SlabSource), so the same builder fills
//! working memory and a cache entry, and it lies in as many chunks, and as
//! many ring slots, as it needs: lookups work across them.
//!
//! [`tokenizer`] defines what a token is.

pub mod dict_index;
pub mod postings;
pub mod sparse_index;
mod token_ranges;
pub mod tokenizer;

pub use dict_index::{DictIndex, DictIndexBuilder};
pub use postings::{Postings, PostingsBuilder, PostingsMeta};
pub use sparse_index::{SparseIndex, SparseIndexBuilder};
pub use token_ranges::TokenRangesMeta;
pub use tokenizer::split_tokens;

#[cfg(test)]
mod tests;
