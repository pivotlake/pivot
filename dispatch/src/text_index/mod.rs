//! A text search index over one column of one data file: for every token of
//! the column, the rows it appears in.
//!
//! An index file has three kinds of pieces, and a lookup goes through them
//! in this order:
//!
//! ```text
//!   piece          one per     answers
//!   ─────          ───────     ───────
//!   sparse index   file        which dict index would hold this token
//!   dict index     token run   the token's rows if they are few, or else
//!                              where its postings are
//!   postings       token       which rows the token appears in
//! ```
//!
//! A piece is looked up only once all of it is in memory: a lookup never
//! reads. Each kind has a cache of its own in the text index family (see
//! [`TextIndexCaches`](crate::memory::cache::TextIndexCaches)).
//!
//! In memory a piece is a few [`SlabVec`](crate::memory::SlabVec)s. It is
//! gathered by a builder whose memory comes from a
//! [`SlabSource`](crate::memory::SlabSource), so the same builder fills
//! working memory and a cache entry, and it lies in as many chunks, and as
//! many ring slots, as it needs: lookups work across them.
//!
//! Each piece also has an encoded form, compact bytes in slabs, which
//! `encode` writes and `decode` gathers a piece back from. A piece decoded
//! into a cache takes there the bytes of its elements and no more.
//!
//! How the modules fit:
//!
//! - [`tokenizer`] defines what a token is.
//! - [`sparse_index`], [`dict_index`] and [`postings`] are the three pieces.
//! - [`index_file`] lays the pieces of one index out as a file.
//! - [`token_run`] is how the text of a data file gets to the index file:
//!   sorted a row group at a time, then merged.
//! - [`sorted_tokens`] is the run of tokens the sparse index and a dict
//!   index are each built around.
//! - [`carve_order`] tells the chunks of a piece's several vecs apart once
//!   they come back from a cache.
//! - [`encoded`] writes and reads the bytes of the encoded forms.

mod carve_order;
pub mod dict_index;
mod encoded;
pub mod index_file;
pub mod postings;
mod sorted_tokens;
pub mod sparse_index;
pub mod token_run;
pub mod tokenizer;

pub use dict_index::{
    DictIndex, DictIndexBuilder, DictIndexMeta, FewRows, MOST_ROWS_KEPT_IN_DICT, TokenRows,
};
pub use index_file::{IndexFileFooter, IndexFileWriter, TOKENS_PER_DICT_INDEX};
pub use postings::{Postings, PostingsBuilder, PostingsMeta};
pub use sparse_index::{SparseIndex, SparseIndexBuilder, SparseIndexMeta};
pub use token_run::{PlacedTokenRun, TokenRun, TokenRunBuilder};
pub use tokenizer::{LetterCase, Tokenizer, split_tokens};

/// Why encoded bytes could not be decoded.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("corrupt text index: {0}")]
    Corrupt(&'static str),
}

#[cfg(test)]
mod tests;
