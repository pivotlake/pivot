//! Answering a query from an index file.
//!
//! A [`TextIndexLookup`] resolves one query against one index file. It walks
//! the index top down, and at each level takes what the caches hold and reads
//! only the rest:
//!
//! ```text
//!                    cached?            not cached
//!   sparse index     pin it             read the file's tail, build it, cache it
//!        │
//!   dict blocks      pin each           read the missing ones, decode, cache
//!        │           (one per token)
//!   posting lists    pin each           read the missing ones, cache
//!        │           (tokens whose rows are not in their dictionary entry)
//!        ▼
//!   MatchingRows     which rows hold all / any of the tokens
//! ```
//!
//! Reads go through the operator's [`OperatorIO`] into the text index family's
//! file cache, so a lookup is a small state machine: [`advance`] goes as far
//! as it can and either finishes or leaves one read outstanding, and the
//! operator hands the response back through [`process_read_response`].
//!
//! ## Fewer, larger reads
//!
//! A level's reads are issued as one request, and ranges with only a few
//! tens of kilobytes between them are fetched as one: an index's sections are
//! small next to what a round trip to storage costs.
//!
//! The posting lists of a dictionary block's tokens sit in the file right
//! before the block, so a block that has to be read brings them along when
//! they are small. The lookup keeps what it read until it is done, and finds
//! those lists there instead of reading again:
//!
//! ```text
//!   file:   … │ posting lists of block 7 │ dict block 7 │ …
//!              └────────── one read ──────────────────┘
//! ```
//!
//! A cold lookup therefore usually waits twice: for the tail, and for the
//! blocks with their lists.
//!
//! ## Exact answers
//!
//! The answer is exactly the rows that match. The index holds every row of
//! every token it stores, split the way [`tokenizer`](super::tokenizer)
//! splits text, so a row is in the answer when and only when the text of the
//! row holds the tokens. A scan can therefore read the rows a lookup names
//! and nothing else, without checking them against the text.
//!
//! [`advance`]: TextIndexLookup::advance
//! [`process_read_response`]: TextIndexLookup::process_read_response

use crate::io::{FileRange, OpenFile, OperatorIO, ReadRequestId, ReadResponse};
use crate::memory::cache::{CacheKey, TextIndexCaches};
use crate::memory::memory_ctx;
use crate::text_index::Error;
use crate::text_index::dict_block::{CopiedInlineRows, DictBlock, PostingsLocation, TokenEntry};
use crate::text_index::format::{FOOTER_LEN, Footer, FragmentReader};
use crate::text_index::postings::Postings;
use crate::text_index::row_mask::RowMask;
use crate::text_index::sparse_index::SparseIndex;
use crate::text_index::tokenizer::{MAX_INDEXED_TOKEN_LEN, split_tokens};
use bytes::Bytes;
use std::ops::Range;
use std::sync::Arc;

/// Bytes asked for from the end of an index file by the first read: the
/// footer, and with it the sparse index unless that is unusually large.
const TAIL_PROBE_LEN: usize = 64 * 1024;

/// Two wanted ranges with at most this many bytes between them are fetched as
/// one read, bytes in between included.
const MAX_GAP_READ_THROUGH: usize = 64 * 1024;

/// A dictionary block is read together with its tokens' posting lists when
/// those take at most this many bytes.
const MAX_POSTINGS_READ_WITH_BLOCK: usize = 256 * 1024;

/// How a query's tokens combine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TokenMatch {
    /// A row matches when it holds every token.
    All,
    /// A row matches when it holds at least one token.
    Any,
}

/// The tokens to look for and how they combine.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TextQuery {
    tokens: Vec<Box<[u8]>>,
    token_match: TokenMatch,
}

impl TextQuery {
    /// A query for the tokens of `text`, split the way indexed text is.
    pub fn from_text(text: &[u8], token_match: TokenMatch) -> Self {
        let mut tokens: Vec<Box<[u8]>> = split_tokens(text).map(Box::from).collect();
        tokens.sort_unstable();
        tokens.dedup();
        Self {
            tokens,
            token_match,
        }
    }

    pub fn tokens(&self) -> impl Iterator<Item = &[u8]> {
        self.tokens.iter().map(|token| token.as_ref())
    }

    pub fn token_match(&self) -> TokenMatch {
        self.token_match
    }

    /// Whether an index can answer the query: it names a token, and every
    /// token it names is one an index stores. A query naming no token holds
    /// for every row with text and for no row without, which an index cannot
    /// tell apart; one for a longer token has to read the text itself.
    pub fn is_answerable_from_index(&self) -> bool {
        !self.tokens.is_empty()
            && self
                .tokens()
                .all(|token| token.len() <= MAX_INDEXED_TOKEN_LEN)
    }
}

/// An index file to look in.
#[derive(Clone, Debug)]
pub struct IndexFile {
    pub file: OpenFile,
    /// The file's size in bytes.
    pub len: usize,
}

/// How far a lookup has got.
pub enum LookupProgress {
    /// A read is outstanding; hand its response to
    /// [`process_read_response`](TextIndexLookup::process_read_response).
    Reading(ReadRequestId),
    /// The rows of the indexed file that match the query.
    Done(MatchingRows),
}

/// The rows of one token, as the lookup found them.
enum RowSet {
    /// The few rows a dictionary entry holds itself, ascending.
    Inline(CopiedInlineRows),
    List(Postings),
}

impl RowSet {
    /// Select in `mask`, which covers exactly `rows`, the set's rows in `rows`.
    fn select_rows(&self, rows: &Range<u32>, mask: &mut RowMask) {
        match self {
            RowSet::Inline(inline_rows) => {
                for &row in inline_rows.as_slice() {
                    if rows.contains(&row) {
                        mask.select((row - rows.start) as usize);
                    }
                }
            }
            RowSet::List(postings) => postings.select_rows(rows.clone(), mask),
        }
    }
}

enum Matches {
    /// No row matches.
    None,
    /// The rows in every one of the sets, the smallest set first.
    InAll(Vec<RowSet>),
    /// The rows in at least one of the sets.
    InAny(Vec<RowSet>),
}

/// A finished lookup's answer: which rows of the indexed file match the
/// query.
///
/// The posting lists behind the answer stay pinned in the cache for as long
/// as this is held, and it answers for whichever rows it is asked about, so
/// nothing sized by the whole file is built to answer for part of it.
pub struct MatchingRows {
    indexed_rows: u64,
    matches: Matches,
}

impl MatchingRows {
    /// Rows in the indexed file.
    pub fn indexed_row_count(&self) -> u64 {
        self.indexed_rows
    }

    /// Whether no row of the indexed file matches.
    pub fn is_empty(&self) -> bool {
        matches!(self.matches, Matches::None)
    }

    /// The matching rows among `rows`: bit `i` of the mask is row
    /// `rows.start + i`.
    pub fn select_rows(&self, rows: Range<u32>) -> Result<RowMask, Error> {
        if rows.end as u64 > self.indexed_rows {
            return Err(Error::RowsOutOfRange {
                indexed_rows: self.indexed_rows,
                start: rows.start,
                end: rows.end,
            });
        }
        let row_count = rows.len();
        Ok(match &self.matches {
            Matches::None => RowMask::none(row_count),
            Matches::InAny(row_sets) => {
                let mut selected = RowMask::none(row_count);
                for row_set in row_sets {
                    row_set.select_rows(&rows, &mut selected);
                }
                selected
            }
            Matches::InAll(row_sets) => {
                let mut selected = RowMask::all(row_count);
                for row_set in row_sets {
                    if selected.selects_none() {
                        break;
                    }
                    let mut in_set = RowMask::none(row_count);
                    row_set.select_rows(&rows, &mut in_set);
                    selected.intersect_with(&in_set);
                }
                selected
            }
        })
    }
}

/// Where one token's rows come from, before its posting list is pinned.
enum RowSource {
    Inline(CopiedInlineRows),
    List(PostingsLocation),
}

impl RowSource {
    fn count_rows(&self) -> usize {
        match self {
            RowSource::Inline(rows) => rows.as_slice().len(),
            RowSource::List(list) => list.row_count as usize,
        }
    }
}

/// What the dictionary says the answer will be made of.
enum RowSources {
    /// The index holds no token a match could be made of.
    None,
    /// The rows of each token the answer is made of, the fewest first.
    Tokens(Vec<RowSource>),
}

/// Bytes a read brought in.
struct FetchedSpan {
    range: FileRange,
    bytes: Vec<Bytes>,
}

pub struct TextIndexLookup {
    index: IndexFile,
    query: Arc<TextQuery>,
    sparse_index: Option<SparseIndex>,
    dict_blocks: Vec<(FileRange, DictBlock)>,
    postings: Vec<(PostingsLocation, Postings)>,
    /// Everything read so far. Kept until the lookup is done, so that bytes a
    /// read brought along beside what it was issued for are found here.
    fetched: Vec<FetchedSpan>,
    /// The read being waited for and the ranges it was issued for.
    outstanding: Option<(ReadRequestId, Vec<FileRange>)>,
}

impl TextIndexLookup {
    /// A lookup of `query` in `index`. The query must be
    /// [answerable](TextQuery::is_answerable_from_index).
    pub fn new(index: IndexFile, query: Arc<TextQuery>) -> Self {
        assert!(
            query.is_answerable_from_index(),
            "an index cannot answer the query"
        );
        Self {
            index,
            query,
            sparse_index: None,
            dict_blocks: Vec::new(),
            postings: Vec::new(),
            fetched: Vec::new(),
            outstanding: None,
        }
    }

    /// Take the lookup as far as it goes without waiting: to its answer, or to
    /// the next read it needs.
    pub fn advance(&mut self, io: &mut OperatorIO) -> Result<LookupProgress, Error> {
        assert!(self.outstanding.is_none(), "a read is still outstanding");
        let caches = memory_ctx().text_index_caches();

        if self.sparse_index.is_none() {
            self.sparse_index = caches.sparse_index.get(&self.sparse_index_key());
        }
        if self.sparse_index.is_none() {
            let Some(footer) = self.slice_fetched(self.footer_range()?) else {
                let tail_len = self.index.len.min(TAIL_PROBE_LEN);
                let tail = FileRange::new(self.index.len - tail_len, tail_len);
                return self.read(io, caches, vec![tail]);
            };
            let mut footer_bytes = [0u8; FOOTER_LEN];
            FragmentReader::new(&footer).read_exact(&mut footer_bytes)?;
            let footer = Footer::decode(&footer_bytes, self.index.len)?;
            let sparse_range =
                FileRange::new(footer.sparse_offset as usize, footer.sparse_len as usize);
            let Some(encoded) = self.slice_fetched(sparse_range) else {
                return self.read(io, caches, vec![sparse_range]);
            };
            let sparse_index = caches
                .sparse_index
                .insert(self.sparse_index_key(), |fill| {
                    SparseIndex::build(footer, &encoded, fill)
                })?;
            self.sparse_index = Some(sparse_index);
        }

        let sparse_index = self
            .sparse_index
            .as_ref()
            .expect("the sparse index is pinned");
        let block_locations: Vec<_> = self
            .query
            .tokens()
            .filter_map(|token| sparse_index.find_block(token))
            .collect();
        let mut wanted = Vec::new();
        for location in block_locations {
            if self.dict_block(location.block).is_some() {
                continue;
            }
            let key = self.key_of(location.block);
            let dict_block = match caches.dict_blocks.get(&key) {
                Some(dict_block) => dict_block,
                None => {
                    let Some(encoded) = self.slice_fetched(location.block) else {
                        wanted.push(if location.postings.len <= MAX_POSTINGS_READ_WITH_BLOCK {
                            FileRange::new(
                                location.postings.offset,
                                location.postings.len + location.block.len,
                            )
                        } else {
                            location.block
                        });
                        continue;
                    };
                    caches
                        .dict_blocks
                        .insert(key, |fill| DictBlock::build(&encoded, fill))?
                }
            };
            self.dict_blocks.push((location.block, dict_block));
        }
        if !wanted.is_empty() {
            return self.read(io, caches, wanted);
        }

        let row_sources = match self.choose_row_sources() {
            RowSources::None => return Ok(self.finish(Matches::None)),
            RowSources::Tokens(row_sources) => row_sources,
        };
        let mut wanted = Vec::new();
        for row_source in &row_sources {
            let RowSource::List(list) = row_source else {
                continue;
            };
            if self.postings.iter().any(|(held, _)| held == list) {
                continue;
            }
            let key = self.key_of(list.range());
            let postings = match caches.postings.get(&key) {
                Some(postings) => postings,
                None => {
                    let Some(encoded) = self.slice_fetched(list.range()) else {
                        wanted.push(list.range());
                        continue;
                    };
                    caches
                        .postings
                        .insert(key, |fill| Postings::build(&encoded, fill))?
                }
            };
            self.postings.push((*list, postings));
        }
        if !wanted.is_empty() {
            return self.read(io, caches, wanted);
        }

        let row_sets = row_sources
            .into_iter()
            .map(|row_source| match row_source {
                RowSource::Inline(rows) => RowSet::Inline(rows),
                RowSource::List(list) => {
                    let pinned = self
                        .postings
                        .iter()
                        .position(|(held, _)| *held == list)
                        .expect("the token's posting list is pinned");
                    RowSet::List(self.postings.swap_remove(pinned).1)
                }
            })
            .collect();
        let matches = match self.query.token_match() {
            TokenMatch::All => Matches::InAll(row_sets),
            TokenMatch::Any => Matches::InAny(row_sets),
        };
        Ok(self.finish(matches))
    }

    /// Take in the response to the outstanding read and carry on.
    pub fn process_read_response(
        &mut self,
        response: ReadResponse,
        io: &mut OperatorIO,
    ) -> Result<LookupProgress, Error> {
        let (read, ranges) = self.outstanding.take().expect("no read is outstanding");
        assert_eq!(
            response.id(),
            read,
            "response to a read this lookup did not issue"
        );
        let fetched = response.into_locations();
        if fetched.len() != ranges.len() {
            return Err(Error::corrupt(
                "a read came back with a different number of ranges than asked for",
            ));
        }
        for (range, data) in ranges.into_iter().zip(fetched) {
            if data.bytes.iter().map(Bytes::len).sum::<usize>() != range.len {
                return Err(Error::corrupt("index file is shorter than its listed size"));
            }
            self.fetched.push(FetchedSpan {
                range,
                bytes: data.bytes,
            });
        }
        self.advance(io)
    }

    /// Issue one read for `wanted`, joining ranges that lie close together.
    fn read(
        &mut self,
        io: &mut OperatorIO,
        caches: &TextIndexCaches,
        mut wanted: Vec<FileRange>,
    ) -> Result<LookupProgress, Error> {
        wanted.sort_unstable_by_key(|range| range.offset);
        let mut ranges: Vec<FileRange> = Vec::with_capacity(wanted.len());
        for range in wanted {
            match ranges.last_mut() {
                Some(previous)
                    if range.offset <= previous.offset + previous.len + MAX_GAP_READ_THROUGH =>
                {
                    let end = (range.offset + range.len).max(previous.offset + previous.len);
                    previous.len = end - previous.offset;
                }
                _ => ranges.push(range),
            }
        }
        let read = io.read(
            &caches.file_bytes,
            self.index.file.clone(),
            ranges.iter().copied(),
        )?;
        self.outstanding = Some((read, ranges));
        Ok(LookupProgress::Reading(read))
    }

    /// The bytes of `range`, if a read so far covered all of it.
    fn slice_fetched(&self, range: FileRange) -> Option<Vec<Bytes>> {
        let span = self.fetched.iter().find(|span| {
            span.range.offset <= range.offset
                && range.offset + range.len <= span.range.offset + span.range.len
        })?;
        let mut skip = range.offset - span.range.offset;
        let mut remaining = range.len;
        let mut slices = Vec::new();
        for fragment in &span.bytes {
            if remaining == 0 {
                break;
            }
            if skip >= fragment.len() {
                skip -= fragment.len();
                continue;
            }
            let take = (fragment.len() - skip).min(remaining);
            slices.push(fragment.slice(skip..skip + take));
            skip = 0;
            remaining -= take;
        }
        Some(slices)
    }

    /// Decide, from the tokens' dictionary entries, which rows the answer is
    /// made of. Every token's dictionary block must be pinned.
    fn choose_row_sources(&self) -> RowSources {
        let token_match = self.query.token_match();
        let mut row_sources = Vec::new();
        for token in self.query.tokens() {
            match (self.find_token_entry(token), token_match) {
                (None, TokenMatch::All) => return RowSources::None,
                (None, TokenMatch::Any) => {}
                (Some(TokenEntry::Inline(rows)), _) => {
                    row_sources.push(RowSource::Inline(rows.copy_rows()));
                }
                (Some(TokenEntry::List(list)), _) => row_sources.push(RowSource::List(list)),
            }
        }
        if row_sources.is_empty() {
            // The index holds none of the tokens an "any" asks for.
            return RowSources::None;
        }
        row_sources.sort_by_key(RowSource::count_rows);
        RowSources::Tokens(row_sources)
    }

    /// Let go of everything but what the answer is made of.
    fn finish(&mut self, matches: Matches) -> LookupProgress {
        let indexed_rows = self.indexed_row_count();
        self.fetched.clear();
        self.dict_blocks.clear();
        self.postings.clear();
        LookupProgress::Done(MatchingRows {
            indexed_rows,
            matches,
        })
    }

    fn indexed_row_count(&self) -> u64 {
        self.sparse_index
            .as_ref()
            .expect("the sparse index is pinned")
            .footer()
            .row_count
    }

    fn footer_range(&self) -> Result<FileRange, Error> {
        if self.index.len < FOOTER_LEN {
            return Err(Error::corrupt("index file is shorter than a footer"));
        }
        Ok(FileRange::new(self.index.len - FOOTER_LEN, FOOTER_LEN))
    }

    fn key_of(&self, range: FileRange) -> CacheKey {
        CacheKey::new(self.index.file.clone(), range.offset, range.len)
    }

    /// The sparse index is keyed by the footer's bytes: the one part of the
    /// file whose position is known before anything has been read.
    fn sparse_index_key(&self) -> CacheKey {
        CacheKey::new(
            self.index.file.clone(),
            self.index.len.saturating_sub(FOOTER_LEN),
            FOOTER_LEN,
        )
    }

    fn dict_block(&self, block: FileRange) -> Option<&DictBlock> {
        self.dict_blocks
            .iter()
            .find(|(held, _)| *held == block)
            .map(|(_, dict_block)| dict_block)
    }

    /// What the index holds for `token`, once its dictionary block is pinned.
    /// `None` for a token the index does not hold.
    fn find_token_entry(&self, token: &[u8]) -> Option<TokenEntry<'_>> {
        let location = self.sparse_index.as_ref()?.find_block(token)?;
        self.dict_block(location.block)
            .expect("the token's dictionary block is pinned")
            .find(token)
    }
}
