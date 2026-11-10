//! Building an index file from rows of text.
//!
//! The work is split so the expensive part parallelises: the rows of a file
//! are tokenized a contiguous range at a time, on whichever workers have them,
//! and one worker then merges the results.
//!
//! ```text
//!   rows 0..128000        ─▶ TokenRunBuilder ─▶ TokenRun ─┐
//!   rows 128000..256000   ─▶ TokenRunBuilder ─▶ TokenRun ─┼─▶ write_index_file ─▶ index file bytes
//!   rows 256000..300000   ─▶ TokenRunBuilder ─▶ TokenRun ─┘
//! ```
//!
//! A [`TokenRun`] is one range's tokens in sorted order, each with the rows of
//! the range it appears in. Sorted runs merge in one pass: the merge always
//! takes the smallest token any run is at, and gathers its rows from every run
//! holding it, which come out ascending because the runs are taken in row
//! order.

use crate::io::FileRange;
use crate::memory::{FileBytes, Slab};
use crate::text_index::Error;
use crate::text_index::byte_writer::{SlabReader, SlabWriter};
use crate::text_index::dict_block::{
    DictBlockEncoder, MAX_INLINE_ROWS, PostingsLocation, TOKENS_PER_BLOCK, TokenRows,
};
use crate::text_index::format::{ByteSink, Footer, write_varint};
use crate::text_index::postings::encode_postings;
use crate::text_index::sparse_index::SparseIndexEncoder;
use crate::text_index::tokenizer::{MAX_INDEXED_TOKEN_LEN, split_tokens};
use std::ops::Range;

/// The tokens of one contiguous range of a file's rows, sorted, each with the
/// rows of the range it appears in.
///
/// Held on the ring as a stream of records, one per token:
/// `token len (varint) │ token │ row count (varint) │ rows (u32 each)`.
pub struct TokenRun {
    rows: Range<u32>,
    records: Vec<Slab>,
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
    tokens: TokenInterner,
    /// Per token, the last row it was recorded for, so a token repeated in a
    /// row is recorded once.
    last_row_of_token: Vec<Option<u32>>,
    /// Every (token, row) the token appears in, in row order.
    appearances: Vec<Appearance>,
}

#[derive(Clone, Copy)]
struct Appearance {
    token: u32,
    row: u32,
}

impl TokenRunBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the tokens of `text`, the value of file row `row`. Rows must be
    /// added in ascending order; a row with no text is simply not added.
    pub fn add_row(&mut self, row: u32, text: &[u8]) {
        debug_assert!(self.appearances.last().is_none_or(|last| last.row <= row));
        for token in split_tokens(text) {
            if token.len() > MAX_INDEXED_TOKEN_LEN {
                continue;
            }
            let token = self.tokens.intern(token);
            if token as usize == self.last_row_of_token.len() {
                self.last_row_of_token.push(None);
            }
            if self.last_row_of_token[token as usize] == Some(row) {
                continue;
            }
            self.last_row_of_token[token as usize] = Some(row);
            self.appearances.push(Appearance { token, row });
        }
    }

    /// The run of everything added. `rows` is the whole range the builder was
    /// fed from, including rows that were not added or had no tokens.
    pub fn finish(self, rows: Range<u32>) -> TokenRun {
        let token_count = self.tokens.len();
        // Group the appearances by token, keeping row order within each.
        let mut group_ends = vec![0usize; token_count + 1];
        for appearance in &self.appearances {
            group_ends[appearance.token as usize + 1] += 1;
        }
        for token in 0..token_count {
            group_ends[token + 1] += group_ends[token];
        }
        let mut next_slot = group_ends.clone();
        let mut rows_by_token = vec![0u32; self.appearances.len()];
        for appearance in &self.appearances {
            let slot = &mut next_slot[appearance.token as usize];
            rows_by_token[*slot] = appearance.row;
            *slot += 1;
        }

        let mut sorted_tokens: Vec<u32> = (0..token_count as u32).collect();
        sorted_tokens.sort_unstable_by_key(|&token| self.tokens.token(token));

        let mut writer = SlabWriter::new();
        for token in sorted_tokens {
            let bytes = self.tokens.token(token);
            let token_rows =
                &rows_by_token[group_ends[token as usize]..group_ends[token as usize + 1]];
            write_varint(&mut writer, bytes.len() as u64);
            writer.write_bytes(bytes);
            write_varint(&mut writer, token_rows.len() as u64);
            for row in token_rows {
                writer.write_bytes(&row.to_le_bytes());
            }
        }
        TokenRun {
            rows,
            records: writer.into_slabs(),
        }
    }
}

/// Distinct tokens, numbered in order of first appearance.
struct TokenInterner {
    hasher: ahash::RandomState,
    /// Open-addressed table of token number + 1, zero for an empty slot. Its
    /// length is a power of two.
    slots: Vec<u32>,
    token_ends: Vec<u32>,
    token_bytes: Vec<u8>,
}

impl Default for TokenInterner {
    fn default() -> Self {
        Self {
            hasher: ahash::RandomState::new(),
            slots: vec![0; 1024],
            token_ends: Vec::new(),
            token_bytes: Vec::new(),
        }
    }
}

impl TokenInterner {
    fn len(&self) -> usize {
        self.token_ends.len()
    }

    fn token(&self, token: u32) -> &[u8] {
        let start = match token {
            0 => 0,
            _ => self.token_ends[token as usize - 1] as usize,
        };
        &self.token_bytes[start..self.token_ends[token as usize] as usize]
    }

    /// The number of `token`, giving it the next one if it is new.
    fn intern(&mut self, token: &[u8]) -> u32 {
        if 2 * (self.len() + 1) > self.slots.len() {
            self.grow();
        }
        let mut slot = self.first_slot_of(token);
        loop {
            match self.slots[slot] {
                0 => break,
                occupied if self.token(occupied - 1) == token => return occupied - 1,
                _ => slot = (slot + 1) & (self.slots.len() - 1),
            }
        }
        let number = self.len() as u32;
        self.token_bytes.extend_from_slice(token);
        self.token_ends.push(self.token_bytes.len() as u32);
        self.slots[slot] = number + 1;
        number
    }

    fn first_slot_of(&self, token: &[u8]) -> usize {
        self.hasher.hash_one(token) as usize & (self.slots.len() - 1)
    }

    fn grow(&mut self) {
        self.slots = vec![0; 2 * self.slots.len()];
        for number in 0..self.len() as u32 {
            let mut slot = self.first_slot_of(self.token(number));
            while self.slots[slot] != 0 {
                slot = (slot + 1) & (self.slots.len() - 1);
            }
            self.slots[slot] = number + 1;
        }
    }
}

/// Where one run's merge has got to: the token it is at, and its rows not yet
/// taken.
struct RunCursor<'a> {
    reader: SlabReader<'a>,
    token: Vec<u8>,
    row_count: usize,
}

impl<'a> RunCursor<'a> {
    /// A cursor at the run's first token, or `None` for a run with no tokens.
    fn open(run: &'a TokenRun) -> Result<Option<Self>, Error> {
        let mut cursor = Self {
            reader: SlabReader::new(&run.records),
            token: Vec::new(),
            row_count: 0,
        };
        Ok(cursor.read_token()?.then_some(cursor))
    }

    /// Read the next record's token. `false` at the end of the run.
    fn read_token(&mut self) -> Result<bool, Error> {
        if self.reader.is_at_end() {
            return Ok(false);
        }
        let token_len = self.reader.read_varint()? as usize;
        self.token.resize(token_len, 0);
        self.reader.read_exact(&mut self.token)?;
        self.row_count = self.reader.read_varint()? as usize;
        Ok(true)
    }

    /// Append the current token's rows to `rows` and move to the next token.
    /// `false` when the run has no more.
    fn take_rows_and_advance(&mut self, rows: &mut Vec<u32>) -> Result<bool, Error> {
        self.reader.read_u32s_into(self.row_count, rows)?;
        self.read_token()
    }
}

/// The cursors still holding tokens, as a binary heap whose top is the one at
/// the smallest token; of several at the same token, the earliest run.
struct MergeHeap<'a> {
    cursors: Vec<RunCursor<'a>>,
    /// Heap of indices into `cursors`.
    heap: Vec<usize>,
}

impl<'a> MergeHeap<'a> {
    fn new(cursors: Vec<RunCursor<'a>>) -> Self {
        let mut merge = Self {
            heap: (0..cursors.len()).collect(),
            cursors,
        };
        for position in (0..merge.heap.len() / 2).rev() {
            merge.sift_down(position);
        }
        merge
    }

    fn top(&self) -> Option<&RunCursor<'a>> {
        self.heap.first().map(|&cursor| &self.cursors[cursor])
    }

    /// Take the top cursor's rows into `rows`, then re-place the cursor by its
    /// next token or drop it if it has none.
    fn take_top_rows(&mut self, rows: &mut Vec<u32>) -> Result<(), Error> {
        let top = self.heap[0];
        if !self.cursors[top].take_rows_and_advance(rows)? {
            self.heap.swap_remove(0);
        }
        self.sift_down(0);
        Ok(())
    }

    fn comes_before(&self, left: usize, right: usize) -> bool {
        (self.cursors[left].token.as_slice(), left) < (self.cursors[right].token.as_slice(), right)
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

/// Merge `runs`, which together must cover a file's rows from 0 without gaps
/// or overlaps, into the bytes of the file's index.
pub fn write_index_file(mut runs: Vec<TokenRun>) -> Result<FileBytes, Error> {
    runs.sort_by_key(|run| run.rows.start);
    let mut row_count = 0u32;
    for run in &runs {
        assert_eq!(
            run.rows.start, row_count,
            "token runs must cover the file's rows contiguously"
        );
        row_count = run.rows.end;
    }

    let mut cursors = Vec::with_capacity(runs.len());
    for run in &runs {
        cursors.extend(RunCursor::open(run)?);
    }
    let mut merge = MergeHeap::new(cursors);

    let mut writer = SlabWriter::new();
    let mut block = DictBlockEncoder::new();
    let mut sparse_index = SparseIndexEncoder::new();
    let mut token_count = 0u64;
    let mut token = Vec::new();
    let mut rows = Vec::new();
    while let Some(top) = merge.top() {
        token.clear();
        token.extend_from_slice(&top.token);
        rows.clear();
        while merge.top().is_some_and(|top| top.token == token) {
            merge.take_top_rows(&mut rows)?;
        }

        if rows.len() <= MAX_INLINE_ROWS {
            block.push(&token, TokenRows::Inline(&rows));
        } else {
            let offset = writer.len();
            encode_postings(&rows, &mut writer);
            block.push(
                &token,
                TokenRows::List(PostingsLocation {
                    offset: offset as u64,
                    len: (writer.len() - offset) as u32,
                    row_count: rows.len() as u32,
                }),
            );
        }
        token_count += 1;
        if block.token_count() == TOKENS_PER_BLOCK {
            write_block(&mut block, &mut sparse_index, &mut writer);
        }
    }
    if block.token_count() > 0 {
        write_block(&mut block, &mut sparse_index, &mut writer);
    }

    let footer = Footer {
        sparse_offset: writer.len() as u64,
        sparse_len: sparse_index.encoded_len() as u64,
        row_count: row_count as u64,
        token_count,
    };
    sparse_index.finish(&mut writer)?;
    writer.write_bytes(&footer.encode());
    Ok(writer.into_file_bytes())
}

/// Write the block being accumulated and record it in the sparse index.
fn write_block(
    block: &mut DictBlockEncoder,
    sparse_index: &mut SparseIndexEncoder,
    writer: &mut SlabWriter,
) {
    let offset = writer.len();
    let first_token = block.first_token().to_vec();
    block.finish(writer);
    sparse_index.push(&first_token, FileRange::new(offset, writer.len() - offset));
}
