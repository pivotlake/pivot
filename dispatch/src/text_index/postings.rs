//! Postings: the rows a token appears in.
//!
//! The rows of the indexed data file are cut into segments of 65,536 rows,
//! and postings keep the token's rows segment by segment. Inside a segment a
//! row is named by its position there, which fits 16 bits, and a segment
//! keeps its rows in whichever of two ways is smaller:
//!
//! ```text
//!   rows of the file     0 ……… 65,535 │ 65,536 ……… 131,071 │ 131,072 ……… 196,607 │ …
//!   segment              0            │ 1                  │ 2                   │
//!
//!   the token is in      3 rows       │ none               │ 40,000 rows
//!   kept as              3 positions  │ nothing            │ 65,536 bits
//!                        6 bytes      │                    │ 8 KB
//! ```
//!
//! - **listed**: the positions of its rows, ascending. Two bytes a row.
//! - **as bits**: one bit for every row of the segment, set where the token
//!   appears. 8 KB whatever the number of rows.
//!
//! Listing more than [`MOST_ROWS_LISTED`] rows would take more than the bits
//! do, so that is where a segment changes from one to the other. A rare
//! token costs two bytes a row, a token in every row a bit a row, and a
//! token that is common in one part of the file and rare in another gets
//! each where it fits.
//!
//! In memory postings are two vecs:
//!
//! ```text
//!   segments             SlabVec<Segment>   one entry per segment the token is in:
//!                                           its number, its row count, where its
//!                                           positions or bits start
//!   positions_and_bits   SlabVec<u16>       every segment's positions, or its bits
//!                                           as 4,096 words of 16, back to back
//! ```
//!
//! A row is found by a binary search for its segment and then, in a listed
//! segment, another for its position. A range of rows is answered a segment
//! at a time, skipping straight to the first segment the range reaches.
//!
//! The encoded form is the same thing written out, so decoding is a copy
//! with checks:
//!
//! ```text
//!   segment count                                    varint
//!   per segment:  its number, its row count less 1   two bytes each
//!   per segment:  its positions or its bits          two bytes each
//! ```

use crate::memory::cache::Cached;
use crate::memory::{FrozenSlab, Slab, SlabAllocator, SlabSource, SlabVec};
use crate::text_index::Error;
use crate::text_index::carve_order::CarveOrder;
use crate::text_index::encoded::{EncodedReader, EncodedWriter};
use std::ops::Range;

/// How many rows of the file one segment covers.
const ROWS_PER_SEGMENT: u64 = 1 << 16;
/// The most segments a file has: its rows are numbered in 32 bits.
const MOST_SEGMENTS: usize = 1 << 16;
/// A segment with more of the token's rows than this keeps them as bits.
/// This many listed rows take 8 KB, which is what a segment's bits take.
pub const MOST_ROWS_LISTED: usize = 4096;
/// A segment's bits, in words of 16: `ROWS_PER_SEGMENT / 16`.
const WORDS_OF_SEGMENT_BITS: usize = 4096;

/// The numbers of the two vecs in the postings' carve order.
const SEGMENTS_VEC: u8 = 0;
const POSITIONS_AND_BITS_VEC: u8 = 1;

/// One segment the token appears in.
#[derive(Clone, Copy)]
struct Segment {
    /// Which segment of the file it is: it covers the rows from
    /// `number * 65,536`.
    number: u16,
    /// How many of its rows the token appears in, less one, so that a
    /// segment the token fills fits 16 bits too.
    row_count_less_one: u16,
    /// Where its positions or its bits start in `positions_and_bits`.
    data_start: u32,
}

impl Segment {
    /// The first row of the file the segment covers.
    fn first_row(&self) -> u64 {
        self.number as u64 * ROWS_PER_SEGMENT
    }

    fn count_rows(&self) -> usize {
        self.row_count_less_one as usize + 1
    }

    /// Whether the segment lists its rows' positions, rather than keeping a
    /// bit for every row.
    fn is_listed(&self) -> bool {
        self.count_rows() <= MOST_ROWS_LISTED
    }

    /// Where the segment's positions or bits are in `positions_and_bits`.
    fn data(&self) -> Range<usize> {
        let len = if self.is_listed() {
            self.count_rows()
        } else {
            WORDS_OF_SEGMENT_BITS
        };
        self.data_start as usize..self.data_start as usize + len
    }
}

/// A stretch of consecutive rows, as postings hand a range of rows out: the
/// range comes as runs, in order, that together cover it exactly.
enum RowRun {
    /// `len` rows from `first_row` on, none of which the token appears in.
    Without { first_row: u32, len: u32 },
    /// At most 64 rows from `first_row` on. Bit `i` of `rows_with_token`
    /// says whether the token appears in row `first_row + i`.
    Mixed {
        first_row: u32,
        len: u32,
        rows_with_token: u64,
    },
}

/// A caller's selection over a range of rows: one bit a row, 64 rows to a
/// word, the range's first row in the lowest bit of the first word.
struct SelectedRows<'a> {
    first_row: u32,
    bits: &'a mut [u64],
}

impl<'a> SelectedRows<'a> {
    fn new(rows: &Range<u32>, bits: &'a mut [u64]) -> Self {
        assert!(
            bits.len() * 64 >= rows.len(),
            "{} words do not hold a bit for each of {} rows",
            bits.len(),
            rows.len()
        );
        Self {
            first_row: rows.start,
            bits,
        }
    }

    /// Select the rows `rows` marks among the 64 from `first_row` on.
    fn select(&mut self, first_row: u32, rows: u64) {
        let offset = (first_row - self.first_row) as usize;
        let (word, shift) = (offset / 64, offset % 64);
        self.bits[word] |= rows << shift;
        // The 64 rows lie across two words unless they start at a word's
        // first bit.
        if shift > 0 && rows >> (64 - shift) != 0 {
            self.bits[word + 1] |= rows >> (64 - shift);
        }
    }

    /// Deselect the rows `rows` marks among the 64 from `first_row` on.
    fn deselect(&mut self, first_row: u32, rows: u64) {
        let offset = (first_row - self.first_row) as usize;
        let (word, shift) = (offset / 64, offset % 64);
        self.bits[word] &= !(rows << shift);
        if shift > 0 && rows >> (64 - shift) != 0 {
            self.bits[word + 1] &= !(rows >> (64 - shift));
        }
    }

    /// Deselect `len` rows from `first_row` on.
    fn deselect_run(&mut self, first_row: u32, len: u32) {
        let mut deselected = 0;
        while deselected < len {
            let rows = (len - deselected).min(64);
            self.deselect(first_row + deselected, u64::MAX >> (64 - rows));
            deselected += rows;
        }
    }
}

/// One token's postings, all of them in memory for as long as this is held.
pub struct Postings {
    /// The segments the token appears in, by ascending number.
    segments: SlabVec<Segment>,
    /// Every segment's positions or bits, in segment order.
    positions_and_bits: SlabVec<u16>,
    row_count: usize,
}

/// What a cache keeps beside the slabs of postings built into it.
pub struct PostingsMeta {
    segment_count: usize,
    positions_and_bits_len: usize,
    row_count: usize,
    carve_order: CarveOrder,
}

impl Postings {
    /// How many rows the token appears in.
    pub fn count_rows(&self) -> usize {
        self.row_count
    }

    /// Whether the token appears in `row`.
    pub fn contains_row(&self, row: u32) -> bool {
        let number = (row >> 16) as u16;
        let position = row as u16;
        let segment_count = self.segments.len();
        let segment_index = self
            .segments
            .find_partition_point(0..segment_count, |_, segment| segment.number < number);
        if segment_index == segment_count {
            return false;
        }
        let segment = self.segments.get(segment_index);
        if segment.number != number {
            return false;
        }
        let data = segment.data();
        if !segment.is_listed() {
            let word = self
                .positions_and_bits
                .get(data.start + position as usize / 16);
            return word & (1 << (position % 16)) != 0;
        }
        let index = self
            .positions_and_bits
            .find_partition_point(data.clone(), |_, listed| listed < position);
        index < data.end && self.positions_and_bits.get(index) == position
    }

    /// Visit every row the token appears in, ascending.
    pub fn for_each_row(&self, mut visit: impl FnMut(u32)) {
        self.segments.for_each(|segment| {
            let first_row = segment.first_row() as u32;
            if segment.is_listed() {
                for index in segment.data() {
                    visit(first_row + self.positions_and_bits.get(index) as u32);
                }
                return;
            }
            for (word_index, index) in segment.data().enumerate() {
                let mut word = self.positions_and_bits.get(index);
                while word != 0 {
                    visit(first_row + word_index as u32 * 16 + word.trailing_zeros());
                    word &= word - 1;
                }
            }
        });
    }

    /// Select, in `selected`, every row of `rows` the token appears in. Rows
    /// already selected stay selected, so calling this for several tokens
    /// selects the rows any of them appears in.
    ///
    /// `selected` holds one bit for each row of `rows`, 64 rows to a word:
    /// row `rows.start + i` is bit `i % 64` of word `i / 64`.
    pub fn select_rows(&self, rows: Range<u32>, selected: &mut [u64]) {
        let mut selected = SelectedRows::new(&rows, selected);
        self.for_each_run_in(rows, |run| {
            if let RowRun::Mixed {
                first_row,
                rows_with_token,
                ..
            } = run
            {
                selected.select(first_row, rows_with_token);
            }
        });
    }

    /// Keep selected, in `selected`, only the rows of `rows` the token
    /// appears in. No row becomes selected, so calling this for several
    /// tokens leaves the rows all of them appear in.
    ///
    /// `selected` is laid out as for [`select_rows`](Self::select_rows).
    pub fn keep_selected_rows(&self, rows: Range<u32>, selected: &mut [u64]) {
        let mut selected = SelectedRows::new(&rows, selected);
        self.for_each_run_in(rows, |run| match run {
            RowRun::Without { first_row, len } => selected.deselect_run(first_row, len),
            RowRun::Mixed {
                first_row,
                len,
                rows_with_token,
            } => {
                let rows_of_run = u64::MAX >> (64 - len);
                selected.deselect(first_row, !rows_with_token & rows_of_run);
            }
        });
    }

    /// Hand `rows` out as runs, in order: stretches the token is absent
    /// from, each as long as it gets, and pieces of at most 64 rows wherever
    /// the token appears.
    fn for_each_run_in(&self, rows: Range<u32>, mut visit: impl FnMut(RowRun)) {
        let end = rows.end as u64;
        let mut row = rows.start as u64;
        let first_number = (rows.start >> 16) as u16;
        let segment_count = self.segments.len();
        let mut segment_index = self
            .segments
            .find_partition_point(0..segment_count, |_, segment| segment.number < first_number);
        while row < end {
            // The first segment not yet walked holds `row` or starts after
            // it.
            let segment = (segment_index < segment_count).then(|| self.segments.get(segment_index));
            let next_row_in_a_segment = segment.map_or(end, |segment| segment.first_row().min(end));
            if next_row_in_a_segment > row {
                visit(RowRun::Without {
                    first_row: row as u32,
                    len: (next_row_in_a_segment - row) as u32,
                });
                row = next_row_in_a_segment;
                continue;
            }
            let segment = segment.expect("a segment holds the row");
            let walked_end = (segment.first_row() + ROWS_PER_SEGMENT).min(end);
            if segment.is_listed() {
                self.walk_listed_segment(segment, row..walked_end, &mut visit);
            } else {
                self.walk_segment_bits(segment, row..walked_end, &mut visit);
            }
            row = walked_end;
            segment_index += 1;
        }
    }

    /// Hand out `rows`, which lie in the listed segment `segment`, as runs.
    fn walk_listed_segment(
        &self,
        segment: Segment,
        rows: Range<u64>,
        visit: &mut impl FnMut(RowRun),
    ) {
        let data = segment.data();
        let first_position = (rows.start - segment.first_row()) as u16;
        let mut index = self
            .positions_and_bits
            .find_partition_point(data.clone(), |_, listed| listed < first_position);
        let mut row = rows.start;
        while index < data.end {
            let row_with_token = segment.first_row() + self.positions_and_bits.get(index) as u64;
            if row_with_token >= rows.end {
                break;
            }
            if row_with_token > row {
                visit(RowRun::Without {
                    first_row: row as u32,
                    len: (row_with_token - row) as u32,
                });
            }
            // A piece of up to 64 rows starting at this row, taking in every
            // listed row that falls inside it.
            let piece_end = (row_with_token + 64).min(rows.end);
            let mut rows_with_token = 0u64;
            while index < data.end {
                let listed_row = segment.first_row() + self.positions_and_bits.get(index) as u64;
                if listed_row >= piece_end {
                    break;
                }
                rows_with_token |= 1 << (listed_row - row_with_token);
                index += 1;
            }
            visit(RowRun::Mixed {
                first_row: row_with_token as u32,
                len: (piece_end - row_with_token) as u32,
                rows_with_token,
            });
            row = piece_end;
        }
        if row < rows.end {
            visit(RowRun::Without {
                first_row: row as u32,
                len: (rows.end - row) as u32,
            });
        }
    }

    /// Hand out `rows`, which lie in `segment`, a segment kept as bits, as
    /// runs of 64 rows.
    fn walk_segment_bits(
        &self,
        segment: Segment,
        rows: Range<u64>,
        visit: &mut impl FnMut(RowRun),
    ) {
        let data = segment.data();
        let mut row = rows.start;
        while row < rows.end {
            let len = (rows.end - row).min(64);
            // The piece's bits lie in up to five of the segment's words of
            // 16: gather those, then drop the bits before the piece and
            // after it.
            let position = (row - segment.first_row()) as usize;
            let last_position = position + len as usize - 1;
            let mut gathered = 0u128;
            for (place, word_index) in (position / 16..=last_position / 16).enumerate() {
                let word = self.positions_and_bits.get(data.start + word_index);
                gathered |= (word as u128) << (16 * place);
            }
            let rows_of_piece = u64::MAX >> (64 - len);
            visit(RowRun::Mixed {
                first_row: row as u32,
                len: len as u32,
                rows_with_token: (gathered >> (position % 16)) as u64 & rows_of_piece,
            });
            row += len;
        }
    }

    /// The postings in their encoded form, in slabs from `allocator`.
    pub fn encode(&self, allocator: &mut SlabAllocator) -> Vec<Slab> {
        let mut writer = EncodedWriter::new(allocator);
        writer.write_varint(self.segments.len() as u64);
        self.segments.for_each(|segment| {
            writer.write_u16(segment.number);
            writer.write_u16(segment.row_count_less_one);
        });
        self.positions_and_bits
            .for_each(|position_or_bits| writer.write_u16(position_or_bits));
        writer.into_slabs()
    }

    /// Gather the postings `encoded` holds, with memory from `source`. Fails
    /// when `encoded` is not what [`encode`](Self::encode) writes.
    pub fn decode(
        encoded: &[Slab],
        source: &mut impl SlabSource,
    ) -> Result<PostingsBuilder, Error> {
        let mut reader = EncodedReader::new(encoded);
        let segment_count = reader.read_len()?;
        if segment_count > MOST_SEGMENTS {
            return Err(Error::Corrupt("there are more segments than a file has"));
        }
        let mut builder = PostingsBuilder {
            segments: SlabVec::with_exact_len(segment_count),
            ..PostingsBuilder::default()
        };
        let mut number_before = None;
        for _ in 0..segment_count {
            let segment = Segment {
                number: reader.read_u16()?,
                row_count_less_one: reader.read_u16()?,
                data_start: builder.positions_and_bits_len as u32,
            };
            if number_before.is_some_and(|number_before| number_before >= segment.number) {
                return Err(Error::Corrupt("the segments are not in ascending order"));
            }
            number_before = Some(segment.number);
            builder
                .carve_order
                .push(SEGMENTS_VEC, &mut builder.segments, source, segment);
            builder.segment_count += 1;
            builder.positions_and_bits_len += segment.data().len();
            builder.row_count += segment.count_rows();
        }

        builder.positions_and_bits = SlabVec::with_exact_len(builder.positions_and_bits_len);
        let mut positions_or_bits = Vec::new();
        for segment_index in 0..segment_count {
            let segment = builder.segments.get(segment_index);
            positions_or_bits.clear();
            reader.read_u16s_into(segment.data().len(), &mut positions_or_bits)?;
            if segment.is_listed() {
                if !positions_or_bits.is_sorted_by(|position, next| position < next) {
                    return Err(Error::Corrupt(
                        "a segment's rows are not in ascending order",
                    ));
                }
            } else {
                let rows_set: u32 = positions_or_bits.iter().map(|word| word.count_ones()).sum();
                if rows_set as usize != segment.count_rows() {
                    return Err(Error::Corrupt(
                        "a segment's bits hold another number of rows than it says",
                    ));
                }
            }
            builder.carve_order.extend_from_slice(
                POSITIONS_AND_BITS_VEC,
                &mut builder.positions_and_bits,
                source,
                &positions_or_bits,
            );
        }
        reader.finish()?;
        Ok(builder)
    }
}

impl Cached for Postings {
    type Meta = PostingsMeta;

    fn from_slabs(meta: &Self::Meta, slabs: Vec<FrozenSlab>) -> Self {
        let [segments, positions_and_bits] = meta.carve_order.deal_slabs(slabs);
        Self {
            segments: SlabVec::from_chunks(segments, meta.segment_count),
            positions_and_bits: SlabVec::from_chunks(
                positions_and_bits,
                meta.positions_and_bits_len,
            ),
            row_count: meta.row_count,
        }
    }
}

/// Gathers the rows of a token, which must arrive ascending, into postings.
///
/// The rows of the segment being gathered are held aside until the segment
/// is over, since only then is it known whether it lists them or keeps them
/// as bits. So finishing takes a source too, for the last segment.
#[derive(Default)]
pub struct PostingsBuilder {
    segments: SlabVec<Segment>,
    positions_and_bits: SlabVec<u16>,
    segment_count: usize,
    positions_and_bits_len: usize,
    row_count: usize,
    carve_order: CarveOrder,
    /// The row added last.
    row_before: Option<u32>,
    /// The positions, in their segment, of the rows added since the last
    /// segment was written.
    gathered_positions: Vec<u16>,
}

impl PostingsBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `row`, which must be above every row added so far. The memory
    /// comes from `source`.
    pub fn push(&mut self, source: &mut impl SlabSource, row: u32) {
        if let Some(row_before) = self.row_before {
            assert!(
                row_before < row,
                "rows must be added in ascending order, each once"
            );
            if row_before >> 16 != row >> 16 {
                self.write_gathered_segment(source);
            }
        }
        self.gathered_positions.push(row as u16);
        self.row_before = Some(row);
    }

    /// Write the segment whose rows were gathered, listed or as bits.
    fn write_gathered_segment(&mut self, source: &mut impl SlabSource) {
        if self.gathered_positions.is_empty() {
            return;
        }
        let row_before = self.row_before.expect("a row was gathered");
        let segment = Segment {
            number: (row_before >> 16) as u16,
            row_count_less_one: (self.gathered_positions.len() - 1) as u16,
            data_start: self.positions_and_bits_len as u32,
        };
        self.carve_order
            .push(SEGMENTS_VEC, &mut self.segments, source, segment);
        if segment.is_listed() {
            self.carve_order.extend_from_slice(
                POSITIONS_AND_BITS_VEC,
                &mut self.positions_and_bits,
                source,
                &self.gathered_positions,
            );
        } else {
            let mut bits = [0u16; WORDS_OF_SEGMENT_BITS];
            for &position in &self.gathered_positions {
                bits[position as usize / 16] |= 1 << (position % 16);
            }
            self.carve_order.extend_from_slice(
                POSITIONS_AND_BITS_VEC,
                &mut self.positions_and_bits,
                source,
                &bits,
            );
        }
        self.segment_count += 1;
        self.positions_and_bits_len += segment.data().len();
        self.row_count += segment.count_rows();
        self.gathered_positions.clear();
    }

    /// The postings, ready to be looked up where they were gathered. The
    /// last segment's memory comes from `source`.
    pub fn finish(mut self, source: &mut impl SlabSource) -> Postings {
        self.write_gathered_segment(source);
        Postings {
            segments: self.segments,
            positions_and_bits: self.positions_and_bits,
            row_count: self.row_count,
        }
    }

    /// What the cache whose fill the postings were gathered into keeps
    /// beside them. The last segment's memory comes from `source`.
    pub fn into_meta(mut self, source: &mut impl SlabSource) -> PostingsMeta {
        self.write_gathered_segment(source);
        PostingsMeta {
            segment_count: self.segment_count,
            positions_and_bits_len: self.positions_and_bits_len,
            row_count: self.row_count,
            carve_order: self.carve_order,
        }
    }
}
