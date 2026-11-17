//! A dict index: a run of the file's tokens in ascending order, each with the
//! rows it appears in or with where those are.
//!
//! ```text
//!   token      rows
//!   ─────      ────
//!   "date"     3, 17, 240                      kept here: the token is in few rows
//!   "fig"      in postings at 52040 .. 61000   12,000 rows
//!   "kiwi"     9001                            kept here
//!   "mango"    in postings at 61000 .. 70000   31,500 rows
//! ```
//!
//! Most tokens of a text column appear in a handful of rows. A token in at
//! most [`MOST_ROWS_KEPT_IN_DICT`] rows has them kept right here, so finding
//! them takes no read beyond the dict index. The other tokens say where
//! their postings are and how many rows those hold, which is enough to start
//! a search from its rarest token.
//!
//! A file's tokens are cut into several dict indexes so that a lookup needs
//! only the one its token would be in, which the sparse index names.
//!
//! In memory it is the tokens as [`SortedTokens`], at each token's index a
//! [`StoredTokenRows`] of 16 bytes, and the kept rows of all the tokens back
//! to back in a vec of their own.
//!
//! Encoded:
//!
//! ```text
//!   token count                                              varint
//!   bytes of all the tokens together                         varint
//!   rows kept here, of all the tokens together               varint
//!   per token:
//!     the token                                              as sorted tokens write one
//!     how many rows it appears in                            varint
//!     kept here:    each row as the rows skipped since the
//!                   row before it (since row 0, the first)   varints
//!     in postings:  where they start, less where the
//!                   postings written before them ended       signed varint
//!                   their length                             varint
//! ```

use crate::io::FileRange;
use crate::memory::cache::Cached;
use crate::memory::{FrozenSlab, Slab, SlabAllocator, SlabSource, SlabVec};
use crate::text_index::Error;
use crate::text_index::carve_order::CarveOrder;
use crate::text_index::encoded::{EncodedReader, EncodedWriter};
use crate::text_index::sorted_tokens::{SortedTokens, SortedTokensBuilder, VECS_OF_SORTED_TOKENS};

/// A token appearing in at most this many rows has them kept in its dict
/// index, and has no postings.
pub const MOST_ROWS_KEPT_IN_DICT: usize = 8;

/// The numbers, in a dict index's carve order, of its own two vecs.
const ROWS_OF_TOKENS_VEC: u8 = VECS_OF_SORTED_TOKENS;
const KEPT_ROWS_VEC: u8 = VECS_OF_SORTED_TOKENS + 1;

/// Where the rows of a token are, as a dict index tells it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenRows {
    /// The token appears in few rows, and here they are.
    KeptInDict(FewRows),
    /// The token's rows are postings, at `postings` in the index file, of
    /// `row_count` rows.
    InPostings {
        postings: FileRange,
        row_count: usize,
    },
}

/// The rows of a token that appears in at most [`MOST_ROWS_KEPT_IN_DICT`]
/// rows, ascending.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FewRows {
    rows: [u32; MOST_ROWS_KEPT_IN_DICT],
    row_count: usize,
}

impl FewRows {
    /// `rows` must be ascending, each row once, and between one and
    /// [`MOST_ROWS_KEPT_IN_DICT`] of them.
    pub fn new(rows: &[u32]) -> Self {
        assert!(
            (1..=MOST_ROWS_KEPT_IN_DICT).contains(&rows.len()),
            "{} rows are not few",
            rows.len()
        );
        assert!(
            rows.is_sorted_by(|row, next_row| row < next_row),
            "rows must be in ascending order, each once"
        );
        let mut few_rows = Self {
            rows: [0; MOST_ROWS_KEPT_IN_DICT],
            row_count: rows.len(),
        };
        few_rows.rows[..rows.len()].copy_from_slice(rows);
        few_rows
    }

    pub fn as_slice(&self) -> &[u32] {
        &self.rows[..self.row_count]
    }
}

/// What a dict index stores for one token, in 16 bytes.
#[derive(Clone, Copy)]
struct StoredTokenRows {
    /// For rows kept here: the index of the token's first row among the kept
    /// rows. For postings: where they start in the index file.
    start: u64,
    /// For postings: how many bytes they take. Unused for rows kept here.
    postings_len: u32,
    /// How many rows the token appears in. At most
    /// [`MOST_ROWS_KEPT_IN_DICT`] says they are kept here.
    row_count: u32,
}

impl StoredTokenRows {
    fn is_kept_in_dict(&self) -> bool {
        self.row_count as usize <= MOST_ROWS_KEPT_IN_DICT
    }
}

/// One dict index, all of it in memory for as long as this is held.
pub struct DictIndex {
    tokens: SortedTokens,
    /// At the index of each token, where its rows are.
    rows_of_tokens: SlabVec<StoredTokenRows>,
    /// The rows of every token that keeps them here, back to back in token
    /// order.
    kept_rows: SlabVec<u32>,
}

/// What a cache keeps beside the slabs of a dict index built into it.
pub struct DictIndexMeta {
    token_count: usize,
    token_bytes_len: usize,
    kept_rows_len: usize,
    carve_order: CarveOrder,
}

impl DictIndex {
    pub fn count_tokens(&self) -> usize {
        self.tokens.count_tokens()
    }

    /// Where the rows of `token` are. `None` when the dict index does not
    /// hold the token.
    pub fn find_rows(&self, token: &[u8]) -> Option<TokenRows> {
        let index = self.tokens.find(token)?;
        Some(self.get_rows_at(index))
    }

    /// Where the rows of the token at `index` are.
    fn get_rows_at(&self, index: usize) -> TokenRows {
        let stored = self.rows_of_tokens.get(index);
        if !stored.is_kept_in_dict() {
            return TokenRows::InPostings {
                postings: FileRange::new(stored.start as usize, stored.postings_len as usize),
                row_count: stored.row_count as usize,
            };
        }
        let row_count = stored.row_count as usize;
        let mut rows = [0; MOST_ROWS_KEPT_IN_DICT];
        for (position, row) in rows[..row_count].iter_mut().enumerate() {
            *row = self.kept_rows.get(stored.start as usize + position);
        }
        TokenRows::KeptInDict(FewRows { rows, row_count })
    }

    /// The dict index in its encoded form, in slabs from `allocator`.
    pub fn encode(&self, allocator: &mut SlabAllocator) -> Vec<Slab> {
        let mut writer = EncodedWriter::new(allocator);
        writer.write_varint(self.tokens.count_tokens() as u64);
        writer.write_varint(self.tokens.count_token_bytes() as u64);
        writer.write_varint(self.kept_rows.len() as u64);
        let mut end_of_postings_before = 0;
        self.tokens
            .encode_each(&mut writer, |writer, index| match self.get_rows_at(index) {
                TokenRows::KeptInDict(few_rows) => {
                    writer.write_varint(few_rows.row_count as u64);
                    let mut next_possible_row = 0u64;
                    for &row in few_rows.as_slice() {
                        writer.write_varint(row as u64 - next_possible_row);
                        next_possible_row = row as u64 + 1;
                    }
                }
                TokenRows::InPostings {
                    postings,
                    row_count,
                } => {
                    writer.write_varint(row_count as u64);
                    writer.write_signed_varint(
                        postings.offset as i64 - end_of_postings_before as i64,
                    );
                    writer.write_varint(postings.len as u64);
                    end_of_postings_before = postings.offset + postings.len;
                }
            });
        writer.into_slabs()
    }

    /// Gather the dict index `encoded` holds, with memory from `source`.
    /// Fails when `encoded` is not what [`encode`](Self::encode) writes.
    pub fn decode(
        encoded: &[Slab],
        source: &mut impl SlabSource,
    ) -> Result<DictIndexBuilder, Error> {
        let mut reader = EncodedReader::new(encoded);
        let token_count = reader.read_len()?;
        let token_bytes_len = reader.read_len()?;
        let kept_rows_len = reader.read_len()?;
        let mut builder = DictIndexBuilder {
            tokens: SortedTokensBuilder::with_exact_lens(token_count, token_bytes_len),
            rows_of_tokens: SlabVec::with_exact_len(token_count),
            kept_rows: SlabVec::with_exact_len(kept_rows_len),
            kept_rows_len: 0,
            carve_order: CarveOrder::default(),
        };
        let mut end_of_postings_before = 0i64;
        for _ in 0..token_count {
            builder
                .tokens
                .decode_token(&mut reader, source, &mut builder.carve_order)?;
            let row_count = u32::try_from(reader.read_varint()?)
                .ok()
                .filter(|&row_count| row_count > 0)
                .ok_or(Error::Corrupt("a token's row count is not one it can have"))?;
            let mut stored = StoredTokenRows {
                start: builder.kept_rows_len as u64,
                postings_len: 0,
                row_count,
            };
            if stored.is_kept_in_dict() {
                if row_count as usize > kept_rows_len - builder.kept_rows_len {
                    return Err(Error::Corrupt("more rows are kept than it says"));
                }
                let mut next_possible_row = 0u64;
                for _ in 0..row_count {
                    let row = next_possible_row
                        .checked_add(reader.read_varint()?)
                        .and_then(|row| u32::try_from(row).ok())
                        .ok_or(Error::Corrupt("a row does not fit 32 bits"))?;
                    builder
                        .carve_order
                        .push(KEPT_ROWS_VEC, &mut builder.kept_rows, source, row);
                    next_possible_row = row as u64 + 1;
                }
                builder.kept_rows_len += row_count as usize;
            } else {
                stored.start = end_of_postings_before
                    .checked_add(reader.read_signed_varint()?)
                    .and_then(|start| u64::try_from(start).ok())
                    .ok_or(Error::Corrupt("postings start outside any file"))?;
                stored.postings_len = u32::try_from(reader.read_varint()?)
                    .map_err(|_| Error::Corrupt("postings are longer than 4 GiB"))?;
                end_of_postings_before = stored
                    .start
                    .checked_add(stored.postings_len as u64)
                    .and_then(|end| i64::try_from(end).ok())
                    .ok_or(Error::Corrupt("postings end outside any file"))?;
            }
            builder.carve_order.push(
                ROWS_OF_TOKENS_VEC,
                &mut builder.rows_of_tokens,
                source,
                stored,
            );
        }
        if builder.tokens.count_token_bytes() != token_bytes_len {
            return Err(Error::Corrupt(
                "the tokens take another number of bytes than it says",
            ));
        }
        if builder.kept_rows_len != kept_rows_len {
            return Err(Error::Corrupt("fewer rows are kept than it says"));
        }
        reader.finish()?;
        Ok(builder)
    }
}

impl Cached for DictIndex {
    type Meta = DictIndexMeta;

    fn from_slabs(meta: &Self::Meta, slabs: Vec<FrozenSlab>) -> Self {
        let [token_ends, token_bytes, rows_of_tokens, kept_rows] =
            meta.carve_order.deal_slabs(slabs);
        Self {
            tokens: SortedTokens::from_chunks(
                token_ends,
                token_bytes,
                meta.token_count,
                meta.token_bytes_len,
            ),
            rows_of_tokens: SlabVec::from_chunks(rows_of_tokens, meta.token_count),
            kept_rows: SlabVec::from_chunks(kept_rows, meta.kept_rows_len),
        }
    }
}

/// Gathers tokens, which must arrive in ascending order, into a dict index.
#[derive(Default)]
pub struct DictIndexBuilder {
    tokens: SortedTokensBuilder,
    rows_of_tokens: SlabVec<StoredTokenRows>,
    kept_rows: SlabVec<u32>,
    kept_rows_len: usize,
    carve_order: CarveOrder,
}

impl DictIndexBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `token`, which must sort after every token added so far, and
    /// where its rows are. A token in at most [`MOST_ROWS_KEPT_IN_DICT`] rows
    /// must come with the rows themselves. The memory comes from `source`.
    pub fn push(&mut self, source: &mut impl SlabSource, token: &[u8], rows: TokenRows) {
        let stored = match rows {
            TokenRows::KeptInDict(few_rows) => StoredTokenRows {
                start: self.kept_rows_len as u64,
                postings_len: 0,
                row_count: few_rows.row_count as u32,
            },
            TokenRows::InPostings {
                postings,
                row_count,
            } => {
                assert!(
                    row_count > MOST_ROWS_KEPT_IN_DICT,
                    "a token in {row_count} rows keeps them in its dict index"
                );
                StoredTokenRows {
                    start: postings.offset as u64,
                    postings_len: u32::try_from(postings.len)
                        .expect("postings are shorter than 4 GiB"),
                    row_count: u32::try_from(row_count).expect("rows are counted in 32 bits"),
                }
            }
        };
        self.tokens.push(source, &mut self.carve_order, token);
        if let TokenRows::KeptInDict(few_rows) = rows {
            self.carve_order.extend_from_slice(
                KEPT_ROWS_VEC,
                &mut self.kept_rows,
                source,
                few_rows.as_slice(),
            );
            self.kept_rows_len += few_rows.row_count;
        }
        self.carve_order
            .push(ROWS_OF_TOKENS_VEC, &mut self.rows_of_tokens, source, stored);
    }

    /// The dict index, ready to be looked up where it was gathered.
    pub fn finish(self) -> DictIndex {
        DictIndex {
            tokens: self.tokens.finish(),
            rows_of_tokens: self.rows_of_tokens,
            kept_rows: self.kept_rows,
        }
    }

    /// What the cache whose fill the dict index was gathered into keeps
    /// beside it.
    pub fn into_meta(self) -> DictIndexMeta {
        DictIndexMeta {
            token_count: self.tokens.count_tokens(),
            token_bytes_len: self.tokens.count_token_bytes(),
            kept_rows_len: self.kept_rows_len,
            carve_order: self.carve_order,
        }
    }
}
