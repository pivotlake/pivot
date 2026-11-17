//! [`TokenLookup`]: which rows of a data file a token appears in, found in
//! the file's index.
//!
//! ```text
//!   the file's last bytes ──▶ footer ──▶ sparse index ──▶ a dict index ──▶ the token's postings
//!   one read, which usually              (cached)         (cached)         (cached, and none for
//!   holds the sparse index too                                             a token in few rows)
//! ```
//!
//! Each piece is read whole and decoded straight into its cache, and only
//! then looked up: a lookup never searches a piece that is not all in
//! memory. A piece found in its cache is not read at all, so a second lookup
//! in the same file usually reads at most a token's postings, and a file
//! whose sparse index is cached is not even asked for its footer.
//!
//! The lookup does no reading of its own. It says which range of the index
//! file it needs next, and is handed the bytes when whoever drives it has
//! them:
//!
//! ```text
//!   loop {
//!       match lookup.advance() {
//!           NeedsRead(range) => lookup.supply(the bytes of `range`)?,
//!           Found(rows)      => break rows,
//!       }
//!   }
//! ```

use crate::io::{FileRange, OpenFile};
use crate::memory::cache::CacheKey;
use crate::memory::memory_ctx;
use crate::text_index::dict_index::{FewRows, TokenRows};
use crate::text_index::index_file::IndexFileFooter;
use crate::text_index::{DictIndex, Error, Postings, SparseIndex};
use std::ops::Range;

/// How many of an index file's last bytes a lookup reads first. The footer
/// is the last of them, and a sparse index that fits before it comes with
/// the same read.
const TAIL_READ_LEN: usize = 64 * 1024;

/// The rows of a data file a token appears in, as the file's index has them.
pub enum RowsWithToken {
    /// The index does not hold the token: it appears in no row.
    NoRows,
    /// The token appears in few rows, kept in its dict index.
    Few(FewRows),
    /// The token's postings, held in memory for as long as this is.
    Postings(Postings),
}

impl RowsWithToken {
    /// How many rows of the file the token appears in.
    pub fn count_rows(&self) -> usize {
        match self {
            RowsWithToken::NoRows => 0,
            RowsWithToken::Few(few_rows) => few_rows.as_slice().len(),
            RowsWithToken::Postings(postings) => postings.count_rows(),
        }
    }

    /// Visit every row of `rows` the token appears in, ascending.
    pub fn for_each_row_in(&self, rows: Range<u32>, mut visit: impl FnMut(u32)) {
        match self {
            RowsWithToken::NoRows => {}
            RowsWithToken::Few(few_rows) => {
                for &row in few_rows.as_slice() {
                    if rows.contains(&row) {
                        visit(row);
                    }
                }
            }
            RowsWithToken::Postings(postings) => postings.for_each_row_in(rows, visit),
        }
    }
}

/// Where a lookup stands.
pub enum LookupProgress {
    /// The lookup needs this range of the index file, which
    /// [`TokenLookup::supply`] takes the bytes of.
    NeedsRead(FileRange),
    /// The token's rows.
    Found(RowsWithToken),
}

/// What a lookup asked to be read and has not been handed yet.
enum AwaitedRead {
    Nothing,
    /// The file's last bytes: the footer, and maybe the sparse index.
    Tail(FileRange),
    SparseIndex,
    DictIndex(FileRange),
    Postings(FileRange),
}

/// One lookup of one token in one index file.
pub struct TokenLookup {
    index_file: OpenFile,
    index_file_len: usize,
    token: Vec<u8>,
    tail_read_len: usize,
    awaited: AwaitedRead,
    /// Where the sparse index is, once the footer said so and it was not
    /// among the bytes the footer came with.
    sparse_index_range: Option<FileRange>,
    sparse_index: Option<SparseIndex>,
    dict_index: Option<DictIndex>,
    postings: Option<Postings>,
}

impl TokenLookup {
    /// A lookup of `token` in `index_file`, which is `index_file_len` bytes
    /// long.
    pub fn new(index_file: OpenFile, index_file_len: usize, token: &[u8]) -> Self {
        Self {
            index_file,
            index_file_len,
            token: token.to_vec(),
            tail_read_len: TAIL_READ_LEN,
            awaited: AwaitedRead::Nothing,
            sparse_index_range: None,
            sparse_index: None,
            dict_index: None,
            postings: None,
        }
    }

    /// Read `tail_read_len` of the file's last bytes first, instead of the
    /// usual number.
    pub fn with_tail_read_len(mut self, tail_read_len: usize) -> Self {
        assert!(
            tail_read_len >= IndexFileFooter::ENCODED_LEN,
            "the first read holds the footer"
        );
        self.tail_read_len = tail_read_len;
        self
    }

    /// The key a piece of the index file at `range` is cached under.
    fn key_of(&self, range: FileRange) -> CacheKey {
        CacheKey::new(self.index_file.clone(), range.offset, range.len)
    }

    /// The key the file's sparse index is cached under: the whole file, since
    /// it is found without knowing where in the file it is, and everything
    /// else in the file is found through it.
    fn key_of_sparse_index(&self) -> CacheKey {
        CacheKey::new(self.index_file.clone(), 0, self.index_file_len)
    }

    /// Go as far as what is in memory allows: to the token's rows, or to the
    /// next piece that has to be read.
    pub fn advance(&mut self) -> LookupProgress {
        let caches = memory_ctx().text_index_caches();

        if self.sparse_index.is_none() {
            self.sparse_index = caches.sparse_indexes.get(&self.key_of_sparse_index());
        }
        let Some(sparse_index) = &self.sparse_index else {
            return match self.sparse_index_range {
                Some(sparse_index_range) => {
                    self.awaited = AwaitedRead::SparseIndex;
                    LookupProgress::NeedsRead(sparse_index_range)
                }
                None => {
                    let tail_len = self.index_file_len.min(self.tail_read_len);
                    let tail = FileRange::new(self.index_file_len - tail_len, tail_len);
                    self.awaited = AwaitedRead::Tail(tail);
                    LookupProgress::NeedsRead(tail)
                }
            };
        };

        let Some(dict_index_range) = sparse_index.find_dict_index(&self.token) else {
            return LookupProgress::Found(RowsWithToken::NoRows);
        };
        if self.dict_index.is_none() {
            self.dict_index = caches.dict_indexes.get(&self.key_of(dict_index_range));
        }
        let Some(dict_index) = &self.dict_index else {
            self.awaited = AwaitedRead::DictIndex(dict_index_range);
            return LookupProgress::NeedsRead(dict_index_range);
        };

        let postings_range = match dict_index.find_rows(&self.token) {
            None => return LookupProgress::Found(RowsWithToken::NoRows),
            Some(TokenRows::KeptInDict(few_rows)) => {
                return LookupProgress::Found(RowsWithToken::Few(few_rows));
            }
            Some(TokenRows::InPostings { postings, .. }) => postings,
        };
        if self.postings.is_none() {
            self.postings = caches.postings.get(&self.key_of(postings_range));
        }
        match self.postings.take() {
            Some(postings) => LookupProgress::Found(RowsWithToken::Postings(postings)),
            None => {
                self.awaited = AwaitedRead::Postings(postings_range);
                LookupProgress::NeedsRead(postings_range)
            }
        }
    }

    /// Take the bytes of the range [`advance`](Self::advance) last asked for,
    /// in the buffers the read returned them in, and decode the piece they
    /// are into its cache. Fails when they are not that piece.
    pub fn supply(&mut self, bytes: &[impl AsRef<[u8]>]) -> Result<(), Error> {
        let caches = memory_ctx().text_index_caches();
        match std::mem::replace(&mut self.awaited, AwaitedRead::Nothing) {
            AwaitedRead::Nothing => panic!("the lookup asked for no read"),
            AwaitedRead::Tail(tail_range) => {
                let tail: Vec<u8> = bytes
                    .iter()
                    .flat_map(|buffer| buffer.as_ref())
                    .copied()
                    .collect();
                if tail.len() != tail_range.len || tail.len() < IndexFileFooter::ENCODED_LEN {
                    return Err(Error::Corrupt("the index file is shorter than its footer"));
                }
                let footer_bytes = tail[tail.len() - IndexFileFooter::ENCODED_LEN..]
                    .try_into()
                    .expect("the footer's length");
                let footer = IndexFileFooter::decode(footer_bytes, self.index_file_len)?;
                let sparse_index_range = footer.sparse_index;
                if sparse_index_range.offset < tail_range.offset {
                    // The sparse index starts before what was read: it is
                    // the next read.
                    self.sparse_index_range = Some(sparse_index_range);
                    return Ok(());
                }
                let start_in_tail = sparse_index_range.offset - tail_range.offset;
                let encoded = &tail[start_in_tail..start_in_tail + sparse_index_range.len];
                self.sparse_index = Some(
                    caches
                        .sparse_indexes
                        .insert(self.key_of_sparse_index(), |fill| {
                            Ok::<_, Error>(SparseIndex::decode(&[encoded], fill)?.into_meta())
                        })?,
                );
            }
            AwaitedRead::SparseIndex => {
                self.sparse_index = Some(
                    caches
                        .sparse_indexes
                        .insert(self.key_of_sparse_index(), |fill| {
                            Ok::<_, Error>(SparseIndex::decode(bytes, fill)?.into_meta())
                        })?,
                );
            }
            AwaitedRead::DictIndex(range) => {
                self.dict_index = Some(caches.dict_indexes.insert(self.key_of(range), |fill| {
                    Ok::<_, Error>(DictIndex::decode(bytes, fill)?.into_meta())
                })?);
            }
            AwaitedRead::Postings(range) => {
                self.postings = Some(caches.postings.insert(self.key_of(range), |fill| {
                    Ok::<_, Error>(Postings::decode(bytes, fill)?.into_meta(fill))
                })?);
            }
        }
        Ok(())
    }
}
