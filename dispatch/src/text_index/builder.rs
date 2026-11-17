//! Building an index file from rows of text.
//!
//! The work is split so the expensive part parallelises: the rows of a file
//! are tokenized a contiguous range at a time, on whichever workers have them,
//! and one worker then merges the results.
//!
//! ```text
//!   rows 0..128000        ─▶ TokenRunBuilder ─▶ TokenRun ─┐
//!   rows 128000..256000   ─▶ TokenRunBuilder ─▶ TokenRun ─┼─▶ IndexMerge ─▶ index file bytes
//!   rows 256000..300000   ─▶ TokenRunBuilder ─▶ TokenRun ─┘
//! ```
//!
//! A [`TokenRun`] is one range's tokens in sorted order, each with the rows of
//! the range it appears in. Sorted runs merge in one pass: the merge always
//! takes the smallest token any run is at, and gathers its rows from every run
//! holding it, which come out ascending because the runs are taken in row
//! order.
//!
//! Everything sized by the data is held on the ring. A builder gathers tokens
//! in one ring buffer's worth of fixed tables, and writes a sorted part of the
//! run whenever they fill, so a run is one part or several:
//!
//! ```text
//!   builder, one ring buffer             run
//!   ┌─────────────┬─────────────┐
//!   │ token table │ appearances │ ─full─▶ part 0 ─┐
//!   │ token bytes │ (token,row) │ ─full─▶ part 1 ─┼─ each sorted by token
//!   └─────────────┴─────────────┘ ─end──▶ part 2 ─┘
//! ```
//!
//! An [`IndexMerge`] merges a bounded piece at a time, so the worker merging
//! a large file keeps taking its turns at everything else it runs. The parts
//! give their memory back as the merge reads through them, so the index being
//! written grows while what it is written from shrinks.

use crate::io::FileRange;
use crate::memory::{BUFFER_SIZE, FileBytes, Slab, SlabAllocator, SlabBuffer};
use crate::text_index::Error;
use crate::text_index::dict_block::{
    DictBlockEncoder, MAX_INLINE_ROWS, PostingsLocation, TOKENS_PER_BLOCK, TokenRows,
};
use crate::text_index::format::{ByteSink, Footer, write_varint};
use crate::text_index::postings::encode_postings;
use crate::text_index::slab_stream::{SlabReader, SlabStream};
use crate::text_index::sparse_index::SparseIndexEncoder;
use crate::text_index::tokenizer::{MAX_INDEXED_TOKEN_LEN, TokenBuffer, split_tokens};
use std::ops::Range;

/// How much one [`IndexMerge::advance`] merges before it returns: tokens
/// taken plus rows gathered. Enough to amortise the call, little enough that
/// a turn stays in the low milliseconds.
const WORK_PER_MERGE_STEP: usize = 1 << 18;

/// The (token, row) pairs a builder gathers before it writes them out as a
/// part. A part cannot hold more distinct tokens than this either.
const MAX_PART_APPEARANCES: usize = 1 << 16;
/// Slots of a builder's token table: twice the tokens it can hold, so a
/// lookup stays a probe or two.
const TOKEN_TABLE_LEN: usize = 2 * MAX_PART_APPEARANCES;
/// The bytes of distinct tokens a builder gathers before it writes a part.
const MAX_PART_TOKEN_BYTES: usize = 256 * 1024;

// A builder's tables fill one ring buffer and no more.
const _: () = assert!(
    size_of::<u32>() * TOKEN_TABLE_LEN
        + MAX_PART_TOKEN_BYTES
        + (3 * size_of::<u32>() + size_of::<Appearance>()) * MAX_PART_APPEARANCES
        <= BUFFER_SIZE
);

/// The tokens of one contiguous range of a file's rows, each with the rows of
/// the range it appears in.
///
/// Held on the ring as parts, each a stream of records sorted by token:
/// `token len (varint) │ token │ row count (varint) │ rows (u32 each)`.
/// A token is never cut across two slabs, so it can be compared where it is.
///
/// A part ends wherever the builder filled up, which can be in the middle of
/// a row. The row's tokens are then split between two parts, and a token on
/// both sides of the split has the row in both.
pub struct TokenRun {
    rows: Range<u32>,
    parts: Vec<Vec<Slab>>,
}

impl TokenRun {
    /// The file rows this run was built from.
    pub fn rows(&self) -> Range<u32> {
        self.rows.clone()
    }
}

/// Gathers the tokens of a contiguous range of a file's rows.
#[derive(Default)]
pub struct TokenRunBuilder {
    /// The part being gathered. Its memory is taken at the first row, so a
    /// builder that is never fed holds none.
    part: Option<PartBuilder>,
    /// The parts already written of the run being built.
    written_parts: Vec<Vec<Slab>>,
}

impl TokenRunBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the tokens of `text`, the value of file row `row`. Rows must be
    /// added in ascending order; a row with no text is simply not added.
    pub fn add_row(&mut self, row: u32, text: &[u8]) {
        let part = self.part.get_or_insert_with(PartBuilder::new);
        for token in split_tokens(text) {
            if token.len() > MAX_INDEXED_TOKEN_LEN {
                continue;
            }
            if !part.has_room_for(token) {
                self.written_parts.push(part.write_part());
            }
            part.add_appearance(token, row);
        }
    }

    /// The run of everything added since the last one, leaving the builder
    /// empty for the next range. `rows` is the whole range the builder was fed
    /// from, including rows that were not added or had no tokens.
    pub fn finish_run(&mut self, rows: Range<u32>) -> TokenRun {
        if let Some(part) = &mut self.part
            && !part.is_empty()
        {
            self.written_parts.push(part.write_part());
        }
        TokenRun {
            rows,
            parts: std::mem::take(&mut self.written_parts),
        }
    }
}

#[derive(Clone, Copy)]
struct Appearance {
    token: u32,
    row: u32,
}

/// What a builder gathers one part in: fixed tables carved once from a ring
/// buffer and used again for every part.
struct PartBuilder {
    tokens: TokenInterner,
    /// One entry per token. While rows are added, the last row the token was
    /// recorded for, so a token repeated in a row is recorded once. While the
    /// part is written, where the token's rows end in `rows_by_token`.
    row_of_token: SlabBuffer<u32>,
    /// Every (token, row) the token appears in, in row order.
    appearances: SlabBuffer<Appearance>,
    appearance_count: usize,
    /// The rows of the appearances, grouped by token, while the part is
    /// written.
    rows_by_token: SlabBuffer<u32>,
    /// Where parts are written. Its chunks come from an allocator of their
    /// own: the parts leave for the merge while the tables above stay.
    stream: SlabStream,
    stream_allocator: SlabAllocator,
}

impl PartBuilder {
    fn new() -> Self {
        let mut tables = SlabAllocator::new(true);
        Self {
            tokens: TokenInterner::new(&mut tables),
            row_of_token: tables.create_slab_buffer(MAX_PART_APPEARANCES, true),
            appearances: tables.create_slab_buffer(MAX_PART_APPEARANCES, true),
            appearance_count: 0,
            rows_by_token: tables.create_slab_buffer(MAX_PART_APPEARANCES, true),
            stream: SlabStream::new(),
            stream_allocator: SlabAllocator::new(false),
        }
    }

    fn is_empty(&self) -> bool {
        self.appearance_count == 0
    }

    /// Whether one more appearance of `token` fits, were the token new.
    fn has_room_for(&self, token: &[u8]) -> bool {
        self.appearance_count < MAX_PART_APPEARANCES && self.tokens.has_room_for(token)
    }

    /// Record that `token` appears in `row`. There must be room for it.
    fn add_appearance(&mut self, token: &[u8], row: u32) {
        let known_tokens = self.tokens.len();
        let token = self.tokens.intern(token);
        let last_row = &mut self.row_of_token.as_mut_slice()[token as usize];
        if (token as usize) < known_tokens && *last_row == row {
            return;
        }
        *last_row = row;
        self.appearances.as_mut_slice()[self.appearance_count] = Appearance { token, row };
        self.appearance_count += 1;
    }

    /// Write everything gathered as one part, sorted by token, and start
    /// empty.
    fn write_part(&mut self) -> Vec<Slab> {
        let appearances = &self.appearances.as_slice()[..self.appearance_count];
        // Group the appearances by token, keeping row order within each:
        // count each token's, turn the counts into where each group starts,
        // then place every row at its group's next free place.
        let group_ends = &mut self.row_of_token.as_mut_slice()[..self.tokens.len()];
        group_ends.fill(0);
        for appearance in appearances {
            group_ends[appearance.token as usize] += 1;
        }
        let mut group_start = 0;
        for group in group_ends.iter_mut() {
            let row_count = *group;
            *group = group_start;
            group_start += row_count;
        }
        let rows_by_token = &mut self.rows_by_token.as_mut_slice()[..appearances.len()];
        for appearance in appearances {
            let next_place = &mut group_ends[appearance.token as usize];
            rows_by_token[*next_place as usize] = appearance.row;
            *next_place += 1;
        }
        // Every group's start has now moved to its end.

        let mut sink = self.stream.write_with(&mut self.stream_allocator);
        for (token, bytes) in self.tokens.sort_tokens() {
            let group_start = match token {
                0 => 0,
                _ => group_ends[token as usize - 1] as usize,
            };
            let token_rows = &rows_by_token[group_start..group_ends[token as usize] as usize];
            write_varint(&mut sink, bytes.len() as u64);
            sink.write_unsplit(bytes);
            write_varint(&mut sink, token_rows.len() as u64);
            sink.write_u32s(token_rows);
        }
        self.tokens.clear();
        self.appearance_count = 0;
        self.stream.take_slabs()
    }
}

/// Distinct tokens, numbered in order of first appearance.
struct TokenInterner {
    hasher: ahash::RandomState,
    /// Open-addressed table of token number + 1, zero for an empty slot.
    slots: SlabBuffer<u32>,
    /// Per token, where its bytes end in `token_bytes`. They start where the
    /// token before it ends.
    token_ends: SlabBuffer<u32>,
    token_bytes: Slab,
    token_count: usize,
    token_bytes_len: usize,
}

/// The tokens an interner holds, by number.
#[derive(Clone, Copy)]
struct InternedTokens<'a> {
    token_ends: &'a [u32],
    token_bytes: &'a [u8],
}

impl<'a> InternedTokens<'a> {
    fn token(&self, token: u32) -> &'a [u8] {
        let start = match token {
            0 => 0,
            _ => self.token_ends[token as usize - 1] as usize,
        };
        &self.token_bytes[start..self.token_ends[token as usize] as usize]
    }
}

impl TokenInterner {
    fn new(tables: &mut SlabAllocator) -> Self {
        Self {
            hasher: ahash::RandomState::new(),
            slots: tables.create_slab_buffer(TOKEN_TABLE_LEN, true),
            token_ends: tables.create_slab_buffer(MAX_PART_APPEARANCES, true),
            token_bytes: tables.get_slab_of_size(MAX_PART_TOKEN_BYTES, false),
            token_count: 0,
            token_bytes_len: 0,
        }
    }

    fn len(&self) -> usize {
        self.token_count
    }

    /// Whether `token` fits, were it new.
    fn has_room_for(&self, token: &[u8]) -> bool {
        self.token_count < MAX_PART_APPEARANCES
            && self.token_bytes_len + token.len() <= MAX_PART_TOKEN_BYTES
    }

    fn tokens(&self) -> InternedTokens<'_> {
        InternedTokens {
            token_ends: &self.token_ends.as_slice()[..self.token_count],
            token_bytes: &self.token_bytes.as_slice()[..self.token_bytes_len],
        }
    }

    /// The number of `token`, giving it the next one if it is new. There must
    /// be room for it.
    fn intern(&mut self, token: &[u8]) -> u32 {
        let mut slot = self.hasher.hash_one(token) as usize & (TOKEN_TABLE_LEN - 1);
        loop {
            match self.slots.as_slice()[slot] {
                0 => break,
                occupied if self.tokens().token(occupied - 1) == token => return occupied - 1,
                _ => slot = (slot + 1) & (TOKEN_TABLE_LEN - 1),
            }
        }
        let number = self.token_count as u32;
        let start = self.token_bytes_len;
        self.token_bytes.as_mut_slice()[start..start + token.len()].copy_from_slice(token);
        self.token_bytes_len += token.len();
        self.token_ends.as_mut_slice()[self.token_count] = self.token_bytes_len as u32;
        self.slots.as_mut_slice()[slot] = number + 1;
        self.token_count += 1;
        number
    }

    /// Every token with its number, in token order. The sort is done where
    /// the table was, so the interner must be cleared before it interns
    /// again.
    fn sort_tokens(&mut self) -> impl Iterator<Item = (u32, &[u8])> + '_ {
        let tokens = InternedTokens {
            token_ends: &self.token_ends.as_slice()[..self.token_count],
            token_bytes: &self.token_bytes.as_slice()[..self.token_bytes_len],
        };
        let sorted = &mut self.slots.as_mut_slice()[..self.token_count];
        for (number, slot) in sorted.iter_mut().enumerate() {
            *slot = number as u32;
        }
        sorted.sort_unstable_by_key(|&token| tokens.token(token));
        let sorted: &[u32] = sorted;
        sorted
            .iter()
            .map(move |&token| (token, tokens.token(token)))
    }

    /// Forget every token.
    fn clear(&mut self) {
        self.slots.as_mut_slice().fill(0);
        self.token_count = 0;
        self.token_bytes_len = 0;
    }
}

/// Where the merge has got to in one part: the token it is at, whose rows are
/// not yet taken.
struct PartCursor {
    /// Positioned at the bytes of the token the cursor is at.
    reader: SlabReader,
    token_len: usize,
}

impl PartCursor {
    /// A cursor at the part's first token.
    fn open(part: Vec<Slab>) -> Result<Self, Error> {
        let mut cursor = Self {
            reader: SlabReader::new(part),
            token_len: 0,
        };
        if !cursor.read_token_len()? {
            return Err(Error::corrupt("a part of a token run has no tokens"));
        }
        Ok(cursor)
    }

    /// Move onto the next record's token. `false` at the end of the part.
    fn read_token_len(&mut self) -> Result<bool, Error> {
        if self.reader.is_at_end() {
            return Ok(false);
        }
        self.token_len = self.reader.read_varint()? as usize;
        if self.reader.peek_front().len() < self.token_len {
            return Err(Error::corrupt("a token is cut across two slabs"));
        }
        Ok(true)
    }

    fn token(&self) -> &[u8] {
        &self.reader.peek_front()[..self.token_len]
    }

    /// Append the current token's rows to `rows` and move to the next token.
    /// `false` when the part has no more.
    fn take_rows_and_advance(
        &mut self,
        rows: &mut GatheredRows,
        scratch: &mut SlabAllocator,
    ) -> Result<bool, Error> {
        self.reader.skip(self.token_len);
        let row_count = self.reader.read_varint()? as usize;
        rows.append_from(&mut self.reader, row_count, scratch)?;
        self.read_token_len()
    }
}

/// One token's rows, gathered from every part holding the token: ascending,
/// four little-endian bytes each, on the ring.
#[derive(Default)]
struct GatheredRows {
    encoded: SlabStream,
    count: usize,
    last: Option<u32>,
}

impl GatheredRows {
    fn len(&self) -> usize {
        self.count
    }

    fn clear(&mut self) {
        self.encoded.clear();
        self.count = 0;
        self.last = None;
    }

    /// Append the next `row_count` rows of `reader`, which are ascending and
    /// start no lower than the last row gathered. A first row equal to it is
    /// one row whose tokens were split between two parts, and is taken once.
    fn append_from(
        &mut self,
        reader: &mut SlabReader,
        row_count: usize,
        scratch: &mut SlabAllocator,
    ) -> Result<(), Error> {
        if row_count == 0 {
            return Err(Error::corrupt("a token of a run has no rows"));
        }
        let mut sink = self.encoded.write_with(scratch);
        let first = reader.read_u32()?;
        if self.last != Some(first) {
            sink.write_bytes(&first.to_le_bytes());
            self.count += 1;
            self.last = Some(first);
        }
        if row_count > 1 {
            reader.copy_to(4 * (row_count - 2), &mut sink)?;
            let last = reader.read_u32()?;
            sink.write_bytes(&last.to_le_bytes());
            self.count += row_count - 1;
            self.last = Some(last);
        }
        Ok(())
    }

    fn iter(&self) -> impl Iterator<Item = u32> + Clone + '_ {
        self.encoded.written_chunks().flat_map(|chunk| {
            // Rows are written whole and chunks are a multiple of their size.
            debug_assert!(chunk.len().is_multiple_of(4));
            chunk
                .as_chunks::<4>()
                .0
                .iter()
                .map(|row| u32::from_le_bytes(*row))
        })
    }
}

/// The cursors still holding tokens, as a binary heap whose top is the one at
/// the smallest token; of several at the same token, the earliest rows.
struct MergeHeap {
    cursors: Vec<PartCursor>,
    /// Heap of indices into `cursors`.
    heap: Vec<usize>,
}

impl MergeHeap {
    fn new(cursors: Vec<PartCursor>) -> Self {
        let mut merge = Self {
            heap: (0..cursors.len()).collect(),
            cursors,
        };
        for position in (0..merge.heap.len() / 2).rev() {
            merge.sift_down(position);
        }
        merge
    }

    fn top(&self) -> Option<&PartCursor> {
        self.heap.first().map(|&cursor| &self.cursors[cursor])
    }

    /// Take the top cursor's rows into `rows`, then re-place the cursor by its
    /// next token or drop it if it has none.
    fn take_top_rows(
        &mut self,
        rows: &mut GatheredRows,
        scratch: &mut SlabAllocator,
    ) -> Result<(), Error> {
        let top = self.heap[0];
        if !self.cursors[top].take_rows_and_advance(rows, scratch)? {
            self.heap.swap_remove(0);
        }
        self.sift_down(0);
        Ok(())
    }

    fn comes_before(&self, left: usize, right: usize) -> bool {
        (self.cursors[left].token(), left) < (self.cursors[right].token(), right)
    }

    fn sift_down(&mut self, mut position: usize) {
        loop {
            let mut smallest = position;
            for child in [2 * position + 1, 2 * position + 2] {
                if child < self.heap.len()
                    && self.comes_before(self.heap[child], self.heap[smallest])
                {
                    smallest = child;
                }
            }
            if smallest == position {
                return;
            }
            self.heap.swap(position, smallest);
            position = smallest;
        }
    }
}

/// Whether a merge has more to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeProgress {
    /// Tokens remain; call [`IndexMerge::advance`] again.
    Merging,
    /// Every token is merged; [`IndexMerge::finish`] gives the file.
    Merged,
}

/// A file's token runs being merged into the bytes of its index.
pub struct IndexMerge {
    merge: MergeHeap,
    /// The index file so far, in chunks from an allocator of its own: they
    /// outlive the merge, to be written to storage.
    file: SlabStream,
    file_allocator: SlabAllocator,
    /// Where everything is held that lasts only until it is written to the
    /// file: the dictionary block being gathered, the sparse index, and the
    /// rows of the token being merged.
    scratch: SlabAllocator,
    block: DictBlockEncoder,
    sparse_index: SparseIndexEncoder,
    row_count: u32,
    token_count: u64,
    /// The token being merged and its rows so far.
    token: TokenBuffer,
    rows: GatheredRows,
}

impl IndexMerge {
    /// A merge of `runs`, which together must cover a file's rows from 0
    /// without gaps or overlaps.
    pub fn new(mut runs: Vec<TokenRun>) -> Result<Self, Error> {
        runs.sort_by_key(|run| run.rows.start);
        let mut row_count = 0u32;
        for run in &runs {
            assert_eq!(
                run.rows.start, row_count,
                "token runs must cover the file's rows contiguously"
            );
            row_count = run.rows.end;
        }

        // In row order: the runs by their rows, each run's parts as written.
        let mut cursors = Vec::new();
        for part in runs.into_iter().flat_map(|run| run.parts) {
            cursors.push(PartCursor::open(part)?);
        }
        Ok(Self {
            merge: MergeHeap::new(cursors),
            file: SlabStream::new(),
            file_allocator: SlabAllocator::new(false),
            scratch: SlabAllocator::new(false),
            block: DictBlockEncoder::new(),
            sparse_index: SparseIndexEncoder::new(),
            row_count,
            token_count: 0,
            token: TokenBuffer::default(),
            rows: GatheredRows::default(),
        })
    }

    /// Merge one step's worth of tokens.
    pub fn advance(&mut self) -> Result<MergeProgress, Error> {
        let mut work_done = 0;
        while work_done < WORK_PER_MERGE_STEP {
            let Some(top) = self.merge.top() else {
                return Ok(MergeProgress::Merged);
            };
            self.token.set(top.token());
            self.rows.clear();
            while self
                .merge
                .top()
                .is_some_and(|top| top.token() == self.token.as_slice())
            {
                self.merge
                    .take_top_rows(&mut self.rows, &mut self.scratch)?;
            }

            let token = self.token.as_slice();
            let row_count = self.rows.len();
            if row_count <= MAX_INLINE_ROWS {
                let mut inline_rows = [0u32; MAX_INLINE_ROWS];
                for (inline_row, row) in inline_rows.iter_mut().zip(self.rows.iter()) {
                    *inline_row = row;
                }
                let rows = TokenRows::Inline(&inline_rows[..row_count]);
                self.block.push(&mut self.scratch, token, rows);
            } else {
                let offset = self.file.len();
                let mut file = self.file.write_with(&mut self.file_allocator);
                encode_postings(self.rows.iter(), &mut file);
                let rows = TokenRows::List(PostingsLocation {
                    offset: offset as u64,
                    len: (self.file.len() - offset) as u32,
                    row_count: row_count as u32,
                });
                self.block.push(&mut self.scratch, token, rows);
            }
            self.token_count += 1;
            if self.block.token_count() == TOKENS_PER_BLOCK {
                self.write_block();
            }
            work_done += 1 + row_count;
        }
        Ok(MergeProgress::Merging)
    }

    /// The bytes of the index file, once [`advance`](Self::advance) has
    /// merged every token.
    pub fn finish(mut self) -> Result<FileBytes, Error> {
        assert!(
            self.merge.top().is_none(),
            "the merge has tokens left to take"
        );
        if self.block.token_count() > 0 {
            self.write_block();
        }
        let footer = Footer {
            sparse_offset: self.file.len() as u64,
            sparse_len: self.sparse_index.encoded_len() as u64,
            row_count: self.row_count as u64,
            token_count: self.token_count,
        };
        let mut file = self.file.write_with(&mut self.file_allocator);
        self.sparse_index.finish(&mut file)?;
        file.write_bytes(&footer.encode());

        let mut file_bytes = FileBytes::new();
        for slab in self.file.take_slabs() {
            file_bytes.push_slab(slab);
        }
        Ok(file_bytes)
    }

    /// Write the block being gathered and record it in the sparse index.
    fn write_block(&mut self) {
        let offset = self.file.len();
        let mut file = self.file.write_with(&mut self.file_allocator);
        self.block.finish(&mut file);
        self.sparse_index.push(
            &mut self.scratch,
            self.block.first_token(),
            FileRange::new(offset, self.file.len() - offset),
        );
    }
}
