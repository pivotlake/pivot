//! [`TokenRun`]: the tokens of a run of rows, in ascending order, each with
//! the rows it appears in.
//!
//! An index file is written from its tokens in ascending order, but the rows
//! of a data file are tokenized a row group at a time, on whichever workers
//! get to them. So each row group's text is first turned into a token run:
//!
//! ```text
//!   row 0  "GET /users timeout"
//!   row 1  "POST /users ok"         ──▶    GET: 0, 2 │ POST: 1 │ ok: 1 │ orders: 2 │
//!   row 2  "GET /orders timeout"           timeout: 0, 2 │ users: 0, 1
//! ```
//!
//! and the runs of a file are then merged, token by token, into the file's
//! index: each run already has its tokens in order, so the merge only ever
//! looks at the next token of each.
//!
//! A run numbers its rows from its own first row. Where that row is in the
//! data file is said when the run is merged ([`PlacedTokenRun`]), which is
//! what lets a run be made before its file's row groups are laid out.

use crate::memory::{FileBytes, SlabAllocator, SlabVec};
use crate::text_index::index_file::IndexFileWriter;
use crate::text_index::tokenizer::{LetterCase, Tokenizer};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// The tokens of a run of rows, ascending, each with its rows, held on the
/// ring.
pub struct TokenRun {
    /// Where each token ends in `token_bytes`, in token order.
    token_ends: SlabVec<u32>,
    /// The tokens back to back.
    token_bytes: SlabVec<u8>,
    /// Where each token's rows end in `rows`, in token order.
    row_ends: SlabVec<u32>,
    /// The rows of every token back to back, each token's ascending.
    rows: SlabVec<u32>,
}

/// A token run and where its first row is in the data file being indexed.
pub struct PlacedTokenRun {
    pub first_row: u32,
    pub run: TokenRun,
}

impl TokenRun {
    pub fn count_tokens(&self) -> usize {
        self.token_ends.len()
    }

    /// Write the index file of a data file from the token runs of its rows.
    /// `runs` are in the order of their rows in the file. The file's bytes
    /// come from `allocator`.
    pub fn write_index_file(runs: &[PlacedTokenRun], allocator: &mut SlabAllocator) -> FileBytes {
        assert!(
            runs.is_sorted_by(|run, next_run| run.first_row <= next_run.first_row),
            "runs are merged in the order of their rows"
        );
        // The next token of each run, smallest first. Runs that are at the
        // same token come out in run order, which is the order of their rows.
        let mut next_tokens = BinaryHeap::new();
        let mut cursors: Vec<RunCursor<'_>> = runs.iter().map(RunCursor::new).collect();
        for (run_index, cursor) in cursors.iter_mut().enumerate() {
            let mut token = Vec::new();
            if cursor.advance_to_next_token(&mut token) {
                next_tokens.push(Reverse((token, run_index)));
            }
        }

        let mut writer = IndexFileWriter::new();
        let mut token_being_written: Option<Vec<u8>> = None;
        let mut rows = Vec::new();
        while let Some(Reverse((mut token, run_index))) = next_tokens.pop() {
            if token_being_written.as_ref() != Some(&token) {
                writer.start_token(allocator, &token);
                token_being_written = Some(token.clone());
            }
            let cursor = &mut cursors[run_index];
            rows.clear();
            cursor.copy_rows_of_token_into(&mut rows);
            for &row in &rows {
                let row_in_file = cursor
                    .placed
                    .first_row
                    .checked_add(row)
                    .expect("a file's rows are numbered in 32 bits");
                writer.push_row(row_in_file);
            }
            if cursor.advance_to_next_token(&mut token) {
                next_tokens.push(Reverse((token, run_index)));
            }
        }
        writer.finish(allocator)
    }
}

/// A position in a token run while it is merged: the token it is at.
struct RunCursor<'a> {
    placed: &'a PlacedTokenRun,
    /// How many of the run's tokens were visited, the one it is at included.
    tokens_visited: usize,
    /// Where the token it is at ends in the run's token bytes and rows, which
    /// is where the next one starts.
    token_end: u32,
    row_end: u32,
    /// Where the token it is at starts in the run's rows.
    row_start: u32,
}

impl<'a> RunCursor<'a> {
    fn new(placed: &'a PlacedTokenRun) -> Self {
        Self {
            placed,
            tokens_visited: 0,
            token_end: 0,
            row_end: 0,
            row_start: 0,
        }
    }

    /// Move to the run's next token and put its bytes in `token`, replacing
    /// what is there. `false`, leaving `token` alone, when the run has no
    /// more tokens.
    fn advance_to_next_token(&mut self, token: &mut Vec<u8>) -> bool {
        let run = &self.placed.run;
        if self.tokens_visited == run.count_tokens() {
            return false;
        }
        let token_start = self.token_end;
        self.token_end = run.token_ends.get(self.tokens_visited);
        self.row_start = self.row_end;
        self.row_end = run.row_ends.get(self.tokens_visited);
        self.tokens_visited += 1;
        token.clear();
        run.token_bytes
            .copy_range_into(token_start as usize..self.token_end as usize, token);
        true
    }

    /// Append the rows of the token it is at, as the run numbers them, to
    /// `rows`.
    fn copy_rows_of_token_into(&self, rows: &mut Vec<u32>) {
        self.placed
            .run
            .rows
            .copy_range_into(self.row_start as usize..self.row_end as usize, rows);
    }
}

/// Turns the text of a run of rows into a [`TokenRun`], one run after
/// another.
///
/// While the rows arrive it keeps every token it has met and every row each
/// appears in, off the ring. The run it makes of them is on the ring.
pub struct TokenRunBuilder {
    tokenizer: Tokenizer,
    /// Each token met so far and the number it was given: tokens are numbered
    /// in the order they are first met.
    numbers_of_tokens: HashMap<Box<[u8]>, u32, ahash::RandomState>,
    /// By token number, the row the token was last met in. A token met again
    /// in the same row is not counted twice.
    last_row_of_tokens: Vec<u32>,
    /// A token number and a row for each row a token appears in, in the
    /// order the rows arrived.
    appearances: Vec<(u32, u32)>,
}

impl TokenRunBuilder {
    pub fn new(letter_case: LetterCase) -> Self {
        Self {
            tokenizer: Tokenizer::new(letter_case),
            numbers_of_tokens: HashMap::default(),
            last_row_of_tokens: Vec::new(),
            appearances: Vec::new(),
        }
    }

    /// Add the text of `row`, counted from the run's first row. Rows arrive
    /// in ascending order.
    pub fn push_text(&mut self, row: u32, text: &[u8]) {
        let Self {
            tokenizer,
            numbers_of_tokens,
            last_row_of_tokens,
            appearances,
        } = self;
        tokenizer.for_each_token(text, |token| {
            let Some(&number) = numbers_of_tokens.get(token) else {
                let number = last_row_of_tokens.len() as u32;
                numbers_of_tokens.insert(token.into(), number);
                last_row_of_tokens.push(row);
                appearances.push((number, row));
                return;
            };
            let last_row = &mut last_row_of_tokens[number as usize];
            if *last_row != row {
                *last_row = row;
                appearances.push((number, row));
            }
        });
    }

    /// The run of the text added since the last run was finished, in memory
    /// from `allocator`. The builder is left empty, ready for the next run.
    pub fn finish(&mut self, allocator: &mut SlabAllocator) -> TokenRun {
        let mut tokens_in_order: Vec<(&[u8], u32)> = self
            .numbers_of_tokens
            .iter()
            .map(|(token, &number)| (&**token, number))
            .collect();
        tokens_in_order.sort_unstable_by_key(|(token, _)| *token);

        // The rows are gathered by token, in the tokens' order. They arrived
        // by row, so each token's come out ascending.
        let mut row_counts = vec![0u32; self.last_row_of_tokens.len()];
        for &(number, _) in &self.appearances {
            row_counts[number as usize] += 1;
        }
        let mut next_row_slots = vec![0u32; row_counts.len()];
        let mut rows_before = 0;
        for &(_, number) in &tokens_in_order {
            next_row_slots[number as usize] = rows_before;
            rows_before += row_counts[number as usize];
        }
        let mut rows_by_token = vec![0u32; self.appearances.len()];
        for &(number, row) in &self.appearances {
            let slot = &mut next_row_slots[number as usize];
            rows_by_token[*slot as usize] = row;
            *slot += 1;
        }

        let mut run = TokenRun {
            token_ends: SlabVec::new(),
            token_bytes: SlabVec::new(),
            row_ends: SlabVec::new(),
            rows: SlabVec::new(),
        };
        let mut token_bytes_len = 0usize;
        for &(token, number) in &tokens_in_order {
            run.token_bytes.extend_from_slice(allocator, token);
            token_bytes_len += token.len();
            let token_end =
                u32::try_from(token_bytes_len).expect("a run's tokens take less than 4 GiB");
            run.token_ends.push(allocator, token_end);
            // Every row of the token is placed by now, so its next slot is
            // where its rows end.
            run.row_ends
                .push(allocator, next_row_slots[number as usize]);
        }
        run.rows.extend_from_slice(allocator, &rows_by_token);

        drop(tokens_in_order);
        self.numbers_of_tokens.clear();
        self.last_row_of_tokens.clear();
        self.appearances.clear();
        run
    }
}
