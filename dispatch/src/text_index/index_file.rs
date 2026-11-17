//! The index file: every piece of one text index, in the order a writer that
//! is handed the tokens in ascending order can produce them.
//!
//! ```text
//!   ┌──────────┬──────────┬───┬────────────┬──────────┬───┬────────────┬───┬──────────────┬────────┐
//!   │ postings │ postings │ … │ dict index │ postings │ … │ dict index │ … │ sparse index │ footer │
//!   └──────────┴──────────┴───┴────────────┴──────────┴───┴────────────┴───┴──────────────┴────────┘
//!    of the tokens of the      which holds   of the next tokens           one entry per    where the
//!    first dict index          them                                       dict index       sparse
//!                                                                                          index is
//! ```
//!
//! A dict index is closed every [`TOKENS_PER_DICT_INDEX`] tokens and written
//! right after the postings of its tokens, so a token's postings are never
//! far from the dict index that names them. The sparse index can only be
//! written once every dict index is, which puts it last, and the footer after
//! it is what a reader starts from: it has a fixed length, sits at the very
//! end of the file, and says where the sparse index is.
//!
//! A lookup then reads four things, each of them whole:
//!
//! ```text
//!   footer ─▶ sparse index ─▶ one dict index ─▶ one token's postings
//!                                               (none, for a token in few rows)
//! ```

use crate::io::FileRange;
use crate::memory::{FileBytes, SlabAllocator};
use crate::text_index::Error;
use crate::text_index::dict_index::{DictIndexBuilder, FewRows, MOST_ROWS_KEPT_IN_DICT, TokenRows};
use crate::text_index::postings::PostingsBuilder;
use crate::text_index::sparse_index::SparseIndexBuilder;

/// How many tokens a dict index holds, but for the file's last one, which
/// holds what is left. A lookup reads one dict index whole, so this is what
/// it pays to find one token.
pub const TOKENS_PER_DICT_INDEX: usize = 1024;

/// The last bytes of an index file: where its sparse index is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexFileFooter {
    pub sparse_index: FileRange,
}

impl IndexFileFooter {
    /// How many bytes the footer takes, at the end of every index file.
    pub const ENCODED_LEN: usize = 24;

    /// What the last eight bytes of an index file read: they tell it from any
    /// other file, and one written in a later layout from this one.
    const FILE_MARK: [u8; 8] = *b"pvtxtix1";

    fn encode(&self) -> [u8; Self::ENCODED_LEN] {
        let mut encoded = [0; Self::ENCODED_LEN];
        encoded[0..8].copy_from_slice(&(self.sparse_index.offset as u64).to_le_bytes());
        encoded[8..16].copy_from_slice(&(self.sparse_index.len as u64).to_le_bytes());
        encoded[16..24].copy_from_slice(&Self::FILE_MARK);
        encoded
    }

    /// The footer of a file whose last [`ENCODED_LEN`](Self::ENCODED_LEN)
    /// bytes are `encoded` and which is `file_len` bytes long. Fails when the
    /// bytes are not the end of an index file.
    pub fn decode(encoded: &[u8; Self::ENCODED_LEN], file_len: usize) -> Result<Self, Error> {
        if encoded[16..24] != Self::FILE_MARK {
            return Err(Error::Corrupt(
                "the file does not end as an index file does",
            ));
        }
        let offset = u64::from_le_bytes(encoded[0..8].try_into().expect("eight bytes"));
        let len = u64::from_le_bytes(encoded[8..16].try_into().expect("eight bytes"));
        // The sparse index is the last thing before the footer.
        let end_of_sparse_index = offset.checked_add(len);
        if end_of_sparse_index
            != file_len
                .checked_sub(Self::ENCODED_LEN)
                .map(|end| end as u64)
        {
            return Err(Error::Corrupt(
                "the sparse index does not end where the footer starts",
            ));
        }
        Ok(Self {
            sparse_index: FileRange::new(offset as usize, len as usize),
        })
    }
}

/// Writes one index file. It is handed the tokens of the indexed column in
/// ascending order, each followed by the rows it appears in, also ascending:
///
/// ```text
///   start_token("apple")  push_row(3)  push_row(17)
///   start_token("date")   push_row(0)  …
///   finish()  ─▶  the file's bytes
/// ```
///
/// What it holds at any moment is the file written so far, the dict index
/// being gathered, and the rows of the token being gathered.
#[derive(Default)]
pub struct IndexFileWriter {
    /// The file so far: postings and closed dict indexes.
    file: FileBytes,
    sparse_index: SparseIndexBuilder,

    /// The dict index being gathered, and the first of its tokens, which is
    /// what the sparse index knows it by.
    dict_index: DictIndexBuilder,
    tokens_in_dict_index: usize,
    smallest_token_of_dict_index: Vec<u8>,

    /// The token being gathered, once one was started, and how many rows it
    /// has so far. Its first few rows are held in `first_rows`. If more
    /// arrive, they all go to `postings`.
    token: Vec<u8>,
    has_token: bool,
    row_count: usize,
    first_rows: [u32; MOST_ROWS_KEPT_IN_DICT],
    postings: PostingsBuilder,
}

impl IndexFileWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start the next token, which must sort after every token so far. Its
    /// rows follow through [`push_row`](Self::push_row). The memory of what
    /// gets written comes from `allocator`.
    pub fn start_token(&mut self, allocator: &mut SlabAllocator, token: &[u8]) {
        self.write_gathered_token(allocator);
        self.token.clear();
        self.token.extend_from_slice(token);
        self.has_token = true;
    }

    /// Add a row the token being gathered appears in, which must be above
    /// its rows so far.
    pub fn push_row(&mut self, allocator: &mut SlabAllocator, row: u32) {
        assert!(self.has_token, "a row belongs to a token started before it");
        if self.row_count < MOST_ROWS_KEPT_IN_DICT {
            self.first_rows[self.row_count] = row;
        } else {
            // One row too many to keep in the dict index: from here on the
            // token's rows are postings, starting with the ones held back.
            if self.row_count == MOST_ROWS_KEPT_IN_DICT {
                for first_row in self.first_rows {
                    self.postings.push(allocator, first_row);
                }
            }
            self.postings.push(allocator, row);
        }
        self.row_count += 1;
    }

    /// Put the token being gathered into the dict index being gathered: with
    /// its rows if they are few, or else with where its postings, written to
    /// the file here, are. Closes the dict index once it is full.
    fn write_gathered_token(&mut self, allocator: &mut SlabAllocator) {
        if !self.has_token {
            // Nothing was gathered yet: this is the file's first token
            // starting, or a file with no tokens finishing.
            return;
        }
        assert!(self.row_count > 0, "a token appears in at least one row");
        let rows = if self.row_count <= MOST_ROWS_KEPT_IN_DICT {
            TokenRows::KeptInDict(FewRows::new(&self.first_rows[..self.row_count]))
        } else {
            let postings = std::mem::take(&mut self.postings).finish(allocator);
            let offset = self.file.len();
            for slab in postings.encode(allocator) {
                self.file.push_slab(slab);
            }
            TokenRows::InPostings {
                postings: FileRange::new(offset, self.file.len() - offset),
                row_count: self.row_count,
            }
        };
        if self.tokens_in_dict_index == 0 {
            self.smallest_token_of_dict_index.clear();
            self.smallest_token_of_dict_index
                .extend_from_slice(&self.token);
        }
        self.dict_index.push(allocator, &self.token, rows);
        self.tokens_in_dict_index += 1;
        self.row_count = 0;
        if self.tokens_in_dict_index == TOKENS_PER_DICT_INDEX {
            self.write_gathered_dict_index(allocator);
        }
    }

    /// Write the dict index being gathered to the file and tell the sparse
    /// index where it is.
    fn write_gathered_dict_index(&mut self, allocator: &mut SlabAllocator) {
        if self.tokens_in_dict_index == 0 {
            return;
        }
        let dict_index = std::mem::take(&mut self.dict_index).finish();
        let offset = self.file.len();
        for slab in dict_index.encode(allocator) {
            self.file.push_slab(slab);
        }
        self.sparse_index.push(
            allocator,
            &self.smallest_token_of_dict_index,
            FileRange::new(offset, self.file.len() - offset),
        );
        self.tokens_in_dict_index = 0;
    }

    /// Write what is still being gathered, then the sparse index and the
    /// footer: the whole file, ready to be written out.
    pub fn finish(mut self, allocator: &mut SlabAllocator) -> FileBytes {
        self.write_gathered_token(allocator);
        self.write_gathered_dict_index(allocator);
        let sparse_index = self.sparse_index.finish();
        let offset = self.file.len();
        for slab in sparse_index.encode(allocator) {
            self.file.push_slab(slab);
        }
        let footer = IndexFileFooter {
            sparse_index: FileRange::new(offset, self.file.len() - offset),
        };
        let mut footer_slab = allocator.get_slab_of_size(IndexFileFooter::ENCODED_LEN, false);
        footer_slab.as_mut_slice().copy_from_slice(&footer.encode());
        self.file.push_slab(footer_slab);
        self.file
    }
}
