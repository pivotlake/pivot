//! Postings: the rows a token appears in.
//!
//! ```text
//!   postings of "kiwi"     SlabVec<u32>    │ 3 │ 17 │ 18 │ 240 │ 9001 │ …
//! ```
//!
//! The rows are positions in the indexed data file, ascending, each once.
//! They lie in as many chunks as they need, and a row is found by binary
//! search over all of them.
//!
//! Encoded, postings are a row count and then the rows in whichever of two
//! forms is smaller:
//!
//! ```text
//!   row count        varint
//!   form             one byte, absent when there are no rows
//!
//!   form 0, gaps     per row: how many rows were skipped since the row
//!                    before it (since row 0 for the first), as a varint
//!   form 1, bitmap   first row (varint), bit count (varint), then the bits,
//!                    eight to a byte, lowest first: bit i is row first + i
//! ```
//!
//! A token in few rows takes a byte or two a row as gaps. A token in most
//! rows takes about a bit a row as a bitmap.

use crate::memory::cache::Cached;
use crate::memory::{FrozenSlab, Slab, SlabAllocator, SlabSource, SlabVec};
use crate::text_index::Error;
use crate::text_index::encoded::{EncodedReader, EncodedWriter};

const GAPS_FORM: u8 = 0;
const BITMAP_FORM: u8 = 1;

/// One token's postings, all of them in memory for as long as this is held.
pub struct Postings {
    rows: SlabVec<u32>,
}

/// What a cache keeps beside the slabs of postings built into it.
pub struct PostingsMeta {
    row_count: usize,
}

impl Postings {
    /// The postings in their encoded form, in slabs from `allocator`.
    pub fn encode(&self, allocator: &mut SlabAllocator) -> Vec<Slab> {
        let mut writer = EncodedWriter::new(allocator);
        let row_count = self.rows.len();
        writer.write_varint(row_count as u64);
        if row_count == 0 {
            return writer.into_slabs();
        }

        let first_row = self.rows.get(0);
        let bit_count = (self.rows.get(row_count - 1) - first_row) as u64 + 1;
        let bitmap_len = EncodedWriter::count_varint_bytes(first_row as u64)
            + EncodedWriter::count_varint_bytes(bit_count)
            + bit_count.div_ceil(8) as usize;
        let (mut gaps_len, mut next_possible_row) = (0, 0u64);
        self.rows.for_each(|row| {
            gaps_len += EncodedWriter::count_varint_bytes(row as u64 - next_possible_row);
            next_possible_row = row as u64 + 1;
        });

        if gaps_len <= bitmap_len {
            writer.write_byte(GAPS_FORM);
            let mut next_possible_row = 0u64;
            self.rows.for_each(|row| {
                writer.write_varint(row as u64 - next_possible_row);
                next_possible_row = row as u64 + 1;
            });
        } else {
            writer.write_byte(BITMAP_FORM);
            writer.write_varint(first_row as u64);
            writer.write_varint(bit_count);
            let (mut byte, mut byte_index) = (0u8, 0);
            self.rows.for_each(|row| {
                let bit = (row - first_row) as usize;
                while byte_index < bit / 8 {
                    writer.write_byte(byte);
                    byte = 0;
                    byte_index += 1;
                }
                byte |= 1 << (bit % 8);
            });
            writer.write_byte(byte);
        }
        writer.into_slabs()
    }

    /// Gather the postings `encoded` holds, with memory from `source`. Fails
    /// when `encoded` is not what [`encode`](Self::encode) writes.
    pub fn decode(
        encoded: &[Slab],
        source: &mut impl SlabSource,
    ) -> Result<PostingsBuilder, Error> {
        let mut reader = EncodedReader::new(encoded);
        let mut builder = PostingsBuilder::new();
        let row_count = reader.read_varint()?;
        if row_count == 0 {
            reader.finish()?;
            return Ok(builder);
        }

        match reader.read_byte()? {
            GAPS_FORM => {
                let mut next_possible_row = 0u64;
                for _ in 0..row_count {
                    let row = next_possible_row
                        .checked_add(reader.read_varint()?)
                        .and_then(|row| u32::try_from(row).ok())
                        .ok_or(Error::Corrupt("a row does not fit 32 bits"))?;
                    builder.push(source, row);
                    next_possible_row = row as u64 + 1;
                }
            }
            BITMAP_FORM => {
                let first_row = reader.read_varint()?;
                let bit_count = reader.read_varint()?;
                if first_row
                    .checked_add(bit_count)
                    .is_none_or(|end| end > u32::MAX as u64 + 1)
                {
                    return Err(Error::Corrupt("a row does not fit 32 bits"));
                }
                for first_bit_of_byte in (0..bit_count).step_by(8) {
                    let byte = reader.read_byte()?;
                    for bit in first_bit_of_byte..bit_count.min(first_bit_of_byte + 8) {
                        if byte & (1 << (bit % 8)) != 0 {
                            builder.push(source, (first_row + bit) as u32);
                        }
                    }
                }
                if builder.row_count as u64 != row_count {
                    return Err(Error::Corrupt(
                        "a bitmap holds another number of rows than it says",
                    ));
                }
            }
            _ => return Err(Error::Corrupt("postings are in an unknown form")),
        }
        reader.finish()?;
        Ok(builder)
    }

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
