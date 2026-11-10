//! Posting lists: the rows a token appears in.
//!
//! A list is the token's row positions in the indexed file, ascending, stored
//! in the portable Roaring bitmap layout (the variant without run containers):
//!
//! ```text
//!   ┌────────┬───────┬──────────────────┬──────────────┬───────────┬───────────┬ ─ ─
//!   │ cookie │ count │ (key, rows - 1)  │ offset       │ container │ container │
//!   │ u32    │ u32   │ u16, u16 × count │ u32 × count  │ 0         │ 1         │
//!   └────────┴───────┴──────────────────┴──────────────┴───────────┴───────────┴ ─ ─
//!   └──────────────────── header ──────────────────────┘
//! ```
//!
//! Rows sharing their upper 16 bits (the key) share a container. A container
//! of up to 4096 rows is the sorted array of their lower 16 bits; a fuller one
//! is a 65536-bit bitmap. So a rare token costs two bytes a row and a common
//! one never more than a bit a row.
//!
//! Nothing in the layout needs decoding, so the cached form ([`Postings`]) is
//! the same bytes: the header in one slab and each container in its own, none
//! of them straddling a ring slot.

use crate::memory::cache::Cached;
use crate::memory::{FrozenSlab, SlabSource};
use crate::text_index::Error;
use crate::text_index::format::{ByteSink, FragmentReader, read_u16, read_u32, read_u64};
use crate::text_index::row_mask::RowMask;
use bytes::Bytes;
use std::ops::Range;

const COOKIE: u32 = 12346;
const LIST_PREFIX_LEN: usize = 8;
/// Header bytes per container: its key and row count, then its offset.
const HEADER_BYTES_PER_CONTAINER: usize = 8;
/// A container with more rows than this is a bitmap.
const MAX_ARRAY_ROWS: usize = 4096;
const BITMAP_LEN: usize = 8192;
const ROWS_PER_CONTAINER: u64 = 1 << 16;

fn payload_len(row_count: usize) -> usize {
    if row_count > MAX_ARRAY_ROWS {
        BITMAP_LEN
    } else {
        2 * row_count
    }
}

/// Append the posting list of `rows`, which must be ascending without
/// repeats, to `sink`.
pub fn encode_postings(rows: &[u32], sink: &mut impl ByteSink) {
    debug_assert!(rows.windows(2).all(|pair| pair[0] < pair[1]));
    let containers: Vec<&[u32]> = rows
        .chunk_by(|left, right| left >> 16 == right >> 16)
        .collect();

    sink.write_bytes(&COOKIE.to_le_bytes());
    sink.write_bytes(&(containers.len() as u32).to_le_bytes());
    for container in &containers {
        sink.write_bytes(&((container[0] >> 16) as u16).to_le_bytes());
        sink.write_bytes(&((container.len() - 1) as u16).to_le_bytes());
    }
    let mut offset = LIST_PREFIX_LEN + HEADER_BYTES_PER_CONTAINER * containers.len();
    for container in &containers {
        sink.write_bytes(&(offset as u32).to_le_bytes());
        offset += payload_len(container.len());
    }
    let mut payload = Vec::with_capacity(BITMAP_LEN);
    for container in &containers {
        payload.clear();
        if container.len() > MAX_ARRAY_ROWS {
            payload.resize(BITMAP_LEN, 0);
            for row in *container {
                let low = (row & 0xffff) as usize;
                payload[low / 8] |= 1 << (low % 8);
            }
        } else {
            for row in *container {
                payload.extend_from_slice(&(*row as u16).to_le_bytes());
            }
        }
        sink.write_bytes(&payload);
    }
}

/// A posting list, pinned in the cache for as long as this is held.
pub struct Postings {
    header: FrozenSlab,
    /// One slab per container, in key order.
    containers: Vec<FrozenSlab>,
}

/// What the cache keeps beside a posting list's bytes: nothing, the bytes
/// describe themselves.
pub struct PostingsMeta;

impl Cached for Postings {
    type Meta = PostingsMeta;

    fn from_slabs(_meta: &Self::Meta, slabs: Vec<FrozenSlab>) -> Self {
        let mut slabs = slabs.into_iter();
        let header = slabs.next().expect("a posting list has a header slab");
        Self {
            header,
            containers: slabs.collect(),
        }
    }
}

impl Postings {
    /// Copy the posting list `encoded` holds (the fragments of one read, in
    /// order) into slabs carved from `slabs`: the header first, then one slab
    /// per container.
    pub fn build(encoded: &[Bytes], slabs: &mut impl SlabSource) -> Result<PostingsMeta, Error> {
        let mut reader = FragmentReader::new(encoded);
        let mut prefix = [0u8; LIST_PREFIX_LEN];
        reader.read_exact(&mut prefix)?;
        if read_u32(&prefix, 0) != COOKIE {
            return Err(Error::corrupt("posting list cookie is missing"));
        }
        let container_count = read_u32(&prefix, 4) as usize;
        if container_count == 0 || container_count > ROWS_PER_CONTAINER as usize {
            return Err(Error::corrupt("posting list container count is invalid"));
        }

        let header_len = LIST_PREFIX_LEN + HEADER_BYTES_PER_CONTAINER * container_count;
        let mut header = slabs.get_aligned_slab(header_len, 1, false);
        header.as_mut_slice()[..LIST_PREFIX_LEN].copy_from_slice(&prefix);
        reader.read_exact(&mut header.as_mut_slice()[LIST_PREFIX_LEN..])?;
        let header = header.as_slice();

        let mut expected_offset = header_len;
        let mut previous_key = None;
        for index in 0..container_count {
            let key = container_key(header, index);
            if previous_key.is_some_and(|previous| previous >= key) {
                return Err(Error::corrupt("posting list keys are not ascending"));
            }
            previous_key = Some(key);
            let offset_at = LIST_PREFIX_LEN + 4 * container_count + 4 * index;
            if read_u32(header, offset_at) as usize != expected_offset {
                return Err(Error::corrupt("posting list offsets are inconsistent"));
            }
            let len = payload_len(container_row_count(header, index));
            let mut container = slabs.get_aligned_slab(len, 1, false);
            reader.read_exact(container.as_mut_slice())?;
            expected_offset += len;
        }
        if reader.remaining() != 0 {
            return Err(Error::corrupt("posting list has trailing bytes"));
        }
        Ok(PostingsMeta)
    }

    /// Rows in the list.
    pub fn row_count(&self) -> usize {
        (0..self.containers.len())
            .map(|index| container_row_count(self.header.as_slice(), index))
            .sum()
    }

    /// Select in `mask` every row of the list that falls in `rows`. The mask
    /// covers exactly `rows`: its bit `i` is row `rows.start + i`.
    pub fn select_rows(&self, rows: Range<u32>, mask: &mut RowMask) {
        assert_eq!(mask.len(), rows.len(), "the mask must cover the row range");
        if rows.is_empty() {
            return;
        }
        let header = self.header.as_slice();
        let first_key = (rows.start >> 16) as u16;
        let last_key = ((rows.end - 1) >> 16) as u16;
        let mut index = partition_point(self.containers.len(), |index| {
            container_key(header, index) < first_key
        });
        while index < self.containers.len() && container_key(header, index) <= last_key {
            let first_row = (container_key(header, index) as u64) << 16;
            let payload = self.containers[index].as_slice();
            if container_row_count(header, index) > MAX_ARRAY_ROWS {
                select_bitmap_rows(payload, first_row, &rows, mask);
            } else {
                select_array_rows(payload, first_row, &rows, mask);
            }
            index += 1;
        }
    }
}

fn container_key(header: &[u8], index: usize) -> u16 {
    read_u16(header, LIST_PREFIX_LEN + 4 * index)
}

fn container_row_count(header: &[u8], index: usize) -> usize {
    read_u16(header, LIST_PREFIX_LEN + 4 * index + 2) as usize + 1
}

/// The first index in `0..len` for which `is_before` is false, given that it
/// is true for a prefix of them.
fn partition_point(len: usize, is_before: impl Fn(usize) -> bool) -> usize {
    let (mut low, mut high) = (0, len);
    while low < high {
        let middle = low + (high - low) / 2;
        if is_before(middle) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low
}

fn select_array_rows(payload: &[u8], first_row: u64, rows: &Range<u32>, mask: &mut RowMask) {
    for low in payload.as_chunks::<2>().0 {
        let row = first_row + u16::from_le_bytes(*low) as u64;
        if row >= rows.end as u64 {
            return;
        }
        if row >= rows.start as u64 {
            mask.select((row - rows.start as u64) as usize);
        }
    }
}

fn select_bitmap_rows(payload: &[u8], first_row: u64, rows: &Range<u32>, mask: &mut RowMask) {
    let (start, end) = (rows.start as u64, rows.end as u64);
    let first_word = (start.saturating_sub(first_row) / 64) as usize;
    let end_word = (end.saturating_sub(first_row).div_ceil(64) as usize).min(BITMAP_LEN / 8);
    for word_index in first_word..end_word {
        let word = read_u64(payload, word_index * 8);
        if word == 0 {
            continue;
        }
        let word_first_row = first_row + word_index as u64 * 64;
        let lies_inside = word_first_row >= start && word_first_row + 64 <= end;
        if lies_inside && (word_first_row - start).is_multiple_of(64) {
            mask.select_word(((word_first_row - start) / 64) as usize, word);
            continue;
        }
        let mut remaining = word;
        while remaining != 0 {
            let row = word_first_row + remaining.trailing_zeros() as u64;
            remaining &= remaining - 1;
            if row >= start && row < end {
                mask.select((row - start) as usize);
            }
        }
    }
}
