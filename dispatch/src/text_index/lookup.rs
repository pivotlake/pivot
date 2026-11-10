//! Answering a query from an index file.
//!
//! A [`TextIndexLookup`] resolves one query against one index file for one
//! range of the indexed file's rows. It walks the index top down, and at each
//! level takes what the caches hold and reads only the rest:
//!
//! ```text
//!                    cached?            not cached
//!   sparse index     pin it             read the file's tail, build it, cache it
//!        │
//!   dict blocks      pin each           read the missing ones together, decode, cache
//!        │           (one per token)
//!   posting lists    pin each           read the missing ones together, cache
//!        │           (tokens whose rows are not in their dictionary entry)
//!        ▼
//!   RowMask          the rows of the range holding all / any of the tokens
//! ```
//!
//! Reads go through the operator's [`OperatorIO`] into the text index family's
//! file cache, so a lookup is a small state machine: [`advance`] goes as far
//! as it can and either finishes or leaves one read outstanding, and the
//! operator hands the response back through [`process_read_response`]. Each
//! level's reads are issued as one request, so a lookup waits at most three
//! times (four, for a sparse index larger than the tail probe) however many
//! tokens it has.
//!
//! Everything a lookup found stays pinned in it until it is dropped, so the
//! mask is computed from memory that cannot be evicted underneath it.
//!
//! [`advance`]: TextIndexLookup::advance
//! [`process_read_response`]: TextIndexLookup::process_read_response

use crate::io::{FileRange, OpenFile, OperatorIO, ReadData, ReadRequestId, ReadResponse};
use crate::memory::cache::{CacheKey, TextIndexCaches};
use crate::memory::memory_ctx;
use crate::text_index::Error;
use crate::text_index::dict_block::{DictBlock, PostingsLocation, TokenEntry};
use crate::text_index::format::{FOOTER_LEN, Footer, join_fragments};
use crate::text_index::postings::Postings;
use crate::text_index::row_mask::RowMask;
use crate::text_index::sparse_index::SparseIndex;
use crate::text_index::tokenizer::{MAX_INDEXED_TOKEN_LEN, split_tokens};
use std::ops::Range;
use std::sync::Arc;

/// Bytes asked for from the end of an index file by the first read: the
/// footer, and with it the sparse index unless that is unusually large.
const TAIL_PROBE_LEN: usize = 64 * 1024;

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

    /// Whether an index can answer the query: every token is one an index
    /// stores. A query for a longer token has to read the text itself.
    pub fn is_answerable_from_index(&self) -> bool {
        self.tokens()
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
    /// The rows of the lookup's range that match the query.
    Done(RowMask),
}

/// What the outstanding read was for.
enum Awaiting {
    /// The end of the file: the footer and, usually, the sparse index.
    Tail {
        tail: FileRange,
    },
    /// A sparse index the tail probe did not reach.
    SparseIndex {
        footer: Footer,
    },
    DictBlocks {
        blocks: Vec<FileRange>,
    },
    Postings {
        lists: Vec<PostingsLocation>,
    },
}

pub struct TextIndexLookup {
    index: IndexFile,
    query: Arc<TextQuery>,
    rows: Range<u32>,
    sparse_index: Option<SparseIndex>,
    dict_blocks: Vec<(FileRange, DictBlock)>,
    postings: Vec<(PostingsLocation, Postings)>,
    awaiting: Option<(ReadRequestId, Awaiting)>,
}

impl TextIndexLookup {
    /// A lookup of `query` in `index` for rows `rows` of the indexed file. The
    /// query must be [answerable](TextQuery::is_answerable_from_index).
    pub fn new(index: IndexFile, query: Arc<TextQuery>, rows: Range<u32>) -> Self {
        assert!(
            query.is_answerable_from_index(),
            "the query holds a token no index stores"
        );
        Self {
            index,
            query,
            rows,
            sparse_index: None,
            dict_blocks: Vec::new(),
            postings: Vec::new(),
            awaiting: None,
        }
    }

    /// Take the lookup as far as it goes without waiting: to its result, or to
    /// the next read it needs.
    pub fn advance(&mut self, io: &mut OperatorIO) -> Result<LookupProgress, Error> {
        assert!(self.awaiting.is_none(), "a read is still outstanding");
        let caches = memory_ctx().text_index_caches();

        if self.sparse_index.is_none() {
            self.sparse_index = caches.sparse_index.get(&self.sparse_index_key());
        }
        let Some(sparse_index) = &self.sparse_index else {
            let tail = self.tail_range()?;
            return self.read(io, caches, [tail], Awaiting::Tail { tail });
        };
        let indexed_rows = sparse_index.footer().row_count;
        if self.rows.end as u64 > indexed_rows {
            return Err(Error::RowsOutOfRange {
                indexed_rows,
                start: self.rows.start,
                end: self.rows.end,
            });
        }

        let mut missing_blocks = Vec::new();
        for token in self.query.tokens() {
            let Some(block) = sparse_index.find_block(token) else {
                continue;
            };
            if self.dict_block(block).is_some() || missing_blocks.contains(&block) {
                continue;
            }
            match caches.dict_blocks.get(&self.key_of(block)) {
                Some(dict_block) => self.dict_blocks.push((block, dict_block)),
                None => missing_blocks.push(block),
            }
        }
        if !missing_blocks.is_empty() {
            let awaiting = Awaiting::DictBlocks {
                blocks: missing_blocks.clone(),
            };
            return self.read(io, caches, missing_blocks, awaiting);
        }

        let mut missing_lists = Vec::new();
        for token in self.query.tokens() {
            let Some(TokenEntry::List(list)) = self.token_entry(token) else {
                continue;
            };
            if self.posting_list(list).is_some() || missing_lists.contains(&list) {
                continue;
            }
            match caches.postings.get(&self.key_of(list.range())) {
                Some(postings) => self.postings.push((list, postings)),
                None => missing_lists.push(list),
            }
        }
        if !missing_lists.is_empty() {
            let ranges: Vec<FileRange> = missing_lists.iter().map(|list| list.range()).collect();
            let awaiting = Awaiting::Postings {
                lists: missing_lists,
            };
            return self.read(io, caches, ranges, awaiting);
        }

        Ok(LookupProgress::Done(self.select_rows()))
    }

    /// Take in the response to the outstanding read and carry on.
    pub fn process_read_response(
        &mut self,
        response: ReadResponse,
        io: &mut OperatorIO,
    ) -> Result<LookupProgress, Error> {
        let (read, awaiting) = self.awaiting.take().expect("no read is outstanding");
        assert_eq!(
            response.id(),
            read,
            "response to a read this lookup did not issue"
        );
        let caches = memory_ctx().text_index_caches();
        let mut data = response.into_locations().into_iter();
        match awaiting {
            Awaiting::Tail { tail } => {
                let bytes = next_location(&mut data)?;
                let bytes = join_fragments(&bytes.bytes);
                if bytes.len() != tail.len {
                    return Err(Error::corrupt("index file is shorter than its listed size"));
                }
                let footer = Footer::decode(&bytes[bytes.len() - FOOTER_LEN..], self.index.len)?;
                let sparse_offset = footer.sparse_offset as usize;
                if sparse_offset < tail.offset {
                    let sparse = FileRange::new(sparse_offset, footer.sparse_len as usize);
                    return self.read(io, caches, [sparse], Awaiting::SparseIndex { footer });
                }
                let start = sparse_offset - tail.offset;
                self.cache_sparse_index(caches, footer, &bytes[start..bytes.len() - FOOTER_LEN])?;
            }
            Awaiting::SparseIndex { footer } => {
                let bytes = next_location(&mut data)?;
                self.cache_sparse_index(caches, footer, &join_fragments(&bytes.bytes))?;
            }
            Awaiting::DictBlocks { blocks } => {
                for block in blocks {
                    let bytes = next_location(&mut data)?;
                    let encoded = join_fragments(&bytes.bytes);
                    let dict_block = caches
                        .dict_blocks
                        .insert(self.key_of(block), |fill| DictBlock::build(&encoded, fill))?;
                    self.dict_blocks.push((block, dict_block));
                }
            }
            Awaiting::Postings { lists } => {
                for list in lists {
                    let bytes = next_location(&mut data)?;
                    let postings = caches.postings.insert(self.key_of(list.range()), |fill| {
                        Postings::build(&bytes.bytes, fill)
                    })?;
                    self.postings.push((list, postings));
                }
            }
        }
        self.advance(io)
    }

    fn read(
        &mut self,
        io: &mut OperatorIO,
        caches: &TextIndexCaches,
        ranges: impl IntoIterator<Item = FileRange>,
        awaiting: Awaiting,
    ) -> Result<LookupProgress, Error> {
        let read = io.read(&caches.file_bytes, self.index.file.clone(), ranges)?;
        self.awaiting = Some((read, awaiting));
        Ok(LookupProgress::Reading(read))
    }

    fn cache_sparse_index(
        &mut self,
        caches: &TextIndexCaches,
        footer: Footer,
        encoded: &[u8],
    ) -> Result<(), Error> {
        let sparse_index = caches
            .sparse_index
            .insert(self.sparse_index_key(), |fill| {
                SparseIndex::build(footer, encoded, fill)
            })?;
        self.sparse_index = Some(sparse_index);
        Ok(())
    }

    fn tail_range(&self) -> Result<FileRange, Error> {
        if self.index.len < FOOTER_LEN {
            return Err(Error::corrupt("index file is shorter than a footer"));
        }
        let len = self.index.len.min(TAIL_PROBE_LEN);
        Ok(FileRange::new(self.index.len - len, len))
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

    fn posting_list(&self, list: PostingsLocation) -> Option<&Postings> {
        self.postings
            .iter()
            .find(|(held, _)| *held == list)
            .map(|(_, postings)| postings)
    }

    /// What the index holds for `token`, once its dictionary block is pinned.
    /// `None` for a token the index does not hold.
    fn token_entry(&self, token: &[u8]) -> Option<TokenEntry<'_>> {
        let block = self.sparse_index.as_ref()?.find_block(token)?;
        self.dict_block(block)
            .expect("the token's dictionary block is pinned")
            .find(token)
    }

    /// The rows of the range holding `token`.
    fn select_token_rows(&self, token: &[u8], mask: &mut RowMask) {
        match self.token_entry(token) {
            None => {}
            Some(TokenEntry::Inline(rows)) => {
                for row in rows.iter().filter(|row| self.rows.contains(row)) {
                    mask.select((row - self.rows.start) as usize);
                }
            }
            Some(TokenEntry::List(list)) => self
                .posting_list(list)
                .expect("the token's posting list is pinned")
                .select_rows(self.rows.clone(), mask),
        }
    }

    fn select_rows(&self) -> RowMask {
        let row_count = self.rows.len();
        match self.query.token_match() {
            TokenMatch::Any => {
                let mut matching = RowMask::none(row_count);
                for token in self.query.tokens() {
                    self.select_token_rows(token, &mut matching);
                }
                matching
            }
            TokenMatch::All => {
                let mut matching = RowMask::all(row_count);
                for token in self.query.tokens() {
                    let mut holding_token = RowMask::none(row_count);
                    self.select_token_rows(token, &mut holding_token);
                    matching.intersect_with(&holding_token);
                }
                matching
            }
        }
    }
}

fn next_location(data: &mut impl Iterator<Item = ReadData>) -> Result<ReadData, Error> {
    data.next()
        .ok_or_else(|| Error::corrupt("a read came back with fewer ranges than asked for"))
}
