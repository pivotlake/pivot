//! Postings: the rows a token appears in.
//!
//! ```text
//!   postings of "kiwi"     SlabVec<u32>    │ 3 │ 17 │ 18 │ 240 │ 9001 │ …
//! ```
//!
//! The rows are positions in the indexed data file, ascending, each once.
//! They lie in as many chunks as they need, and a row is found by binary
//! search over all of them.

use crate::memory::cache::Cached;
use crate::memory::{FrozenSlab, SlabSource, SlabVec};

/// One token's postings, all of them in memory for as long as this is held.
pub struct Postings {
    rows: SlabVec<u32>,
}

/// What a cache keeps beside the slabs of postings built into it.
pub struct PostingsMeta {
    row_count: usize,
}

impl Postings {
    pub fn count_rows(&self) -> usize {
        self.rows.len()
    }

    /// Whether the token appears in `row`.
    pub fn contains_row(&self, row: u32) -> bool {
        let rows_below = self.rows.find_partition_point(|_, other| other < row);
        rows_below < self.rows.len() && self.rows.get(rows_below) == row
    }

    /// Visit every row the token appears in, ascending.
    pub fn for_each_row(&self, f: impl FnMut(u32)) {
        self.rows.for_each(f);
    }
}

impl Cached for Postings {
    type Meta = PostingsMeta;

    fn from_slabs(meta: &Self::Meta, slabs: Vec<FrozenSlab>) -> Self {
        Self {
            rows: SlabVec::from_chunks(slabs, meta.row_count),
        }
    }
}

/// Gathers the rows of a token, which must arrive ascending, into postings.
#[derive(Default)]
pub struct PostingsBuilder {
    rows: SlabVec<u32>,
    row_count: usize,
    last_row: u32,
}

impl PostingsBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `row`, which must be above every row added so far. The memory
    /// comes from `source`.
    pub fn push(&mut self, source: &mut impl SlabSource, row: u32) {
        assert!(
            self.row_count == 0 || self.last_row < row,
            "rows must be added in ascending order, each once"
        );
        self.rows.push(source, row);
        self.row_count += 1;
        self.last_row = row;
    }

    /// The postings, ready to be looked up where they were gathered.
    pub fn finish(self) -> Postings {
        Postings { rows: self.rows }
    }

    /// What the cache whose fill the postings were gathered into keeps beside
    /// them.
    pub fn into_meta(self) -> PostingsMeta {
        PostingsMeta {
            row_count: self.row_count,
        }
    }
}
