//! A text search index over one column of one data file: for every token of
//! the column, the rows it appears in.
//!
//! ```text
//!   data file rows                         index file
//!   ──────────────                         ──────────
//!   0  "GET /users timeout"                sparse index    first token of each dictionary block
//!   1  "POST /users ok"         build           │
//!   2  "GET /orders timeout"   ──────▶     dictionary      GET → rows 0, 2      (kept in the entry)
//!   …                                      blocks          timeout → posting list at …
//!                                               │
//!                                          posting lists   rows of each common token
//! ```
//!
//! The index lives in its own file beside the data file. A lookup walks it
//! top down, and every level is cached in its own member of the text index
//! cache family (see [`TextIndexCaches`](crate::memory::cache::TextIndexCaches)):
//!
//! ```text
//!   has_all_tokens(message, 'GET timeout'), rows 0..128000 of the data file
//!
//!   sparse index ──▶ dictionary block(s) ──▶ posting list(s) ──▶ RowMask
//!   one per file     one per token           one per common      the rows holding
//!                                            token               every token
//! ```
//!
//! What a lookup answers is exact for the tokens the index stores: a row is
//! selected when it holds the tokens asked for, and only then. Tokens longer
//! than [`MAX_INDEXED_TOKEN_LEN`] are not stored, so a search for one cannot
//! be answered here.
//!
//! - [`tokenizer`] defines what a token is, for the index and the scalar
//!   functions alike.
//! - [`mod@format`] is the file's footer and byte-level helpers; [`postings`],
//!   [`dict_block`] and [`sparse_index`] are its three sections, each with its
//!   file form and its cached form.
//! - [`builder`] turns rows of text into an index file.
//! - [`lookup`] answers a query from an index file, reading through dispatch
//!   IO and the caches.
//! - [`operators`] are the dataflow stages that build indexes and run lookups.

pub mod builder;
mod byte_writer;
pub mod dict_block;
pub mod format;
pub mod lookup;
pub mod operators;
pub mod postings;
pub mod row_mask;
pub mod sparse_index;
pub mod tokenizer;

pub use builder::{TokenRun, TokenRunBuilder, write_index_file};
pub use dict_block::{DictBlock, DictBlockMeta};
pub use lookup::{IndexFile, LookupProgress, TextIndexLookup, TextQuery, TokenMatch};
pub use operators::{
    AssembledIndex, FileTokenRun, IndexAssembler, IndexLookups, IndexLookupsFactory, LookupRequest,
    LookupResult,
};
pub use postings::{Postings, PostingsMeta};
pub use row_mask::RowMask;
pub use sparse_index::{SparseIndex, SparseIndexMeta};
pub use tokenizer::{MAX_INDEXED_TOKEN_LEN, is_single_token, is_token_byte, split_tokens};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("corrupt text index: {0}")]
    Corrupt(String),
    #[error(
        "text index has too many distinct tokens: its sparse index needs {len} bytes \
         and at most {max} fit"
    )]
    SparseIndexTooLarge { len: usize, max: usize },
    #[error("text index rows must fit 32 bits, the file has {row_count}")]
    TooManyRows { row_count: u64 },
    #[error("text index covers {indexed_rows} rows but rows {start}..{end} were asked for")]
    RowsOutOfRange {
        indexed_rows: u64,
        start: u32,
        end: u32,
    },
    #[error("{0}")]
    Io(#[from] crate::io::IORequesterError),
}

impl Error {
    pub(crate) fn corrupt(reason: impl Into<String>) -> Self {
        Self::Corrupt(reason.into())
    }
}

#[cfg(test)]
mod tests;
