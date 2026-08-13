//! Decode checkpoints: where a leaf decoder stood at a given row, in enough
//! detail to put a fresh decoder back there without decoding the rows before it.
//!
//! A data page's start is always a clean entry point, but a split of a row group
//! rarely begins on one. Reaching its first row by *skipping* from the page
//! start costs about what decoding those rows would for a run-length encoded
//! page, which is the whole saving gone. A checkpoint instead records the byte
//! the next value starts at, so resuming is a seek.
//!
//! Positions are recorded as a byte count into the page's payload rather than as
//! a [`ReaderPosition`]. A page's bytes arrive scattered across whichever cache
//! slots happened to hold them, and that split differs from one scan to the
//! next, so a buffer index recorded by one scan means nothing to another. The
//! byte count is a property of the page itself, and [`position_at`] resolves it
//! against whatever layout the page arrives in.

use crate::parquet::reading::decoding::leaf_decoders::rle::Run;
use bytes::Bytes;
use dispatch::memory::ReaderPosition;

/// Where a leaf decoder stands within its chunk's pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageCheckpoint {
    /// The data page the decoder is in.
    pub page_idx: usize,
    /// Bytes into that page's payload where the next value begins. Counted from
    /// the payload's start, so it already covers a nullable page's
    /// definition-level prefix.
    pub byte_offset: usize,
    /// The partially consumed run, when the position falls inside one. Only the
    /// RLE/bit-packed hybrid carries state across a value boundary; a plain
    /// page's byte offset says everything.
    pub run: Option<Run>,
}

/// A [`PageCheckpoint`] tagged with the row it addresses: what gets recorded for
/// a chunk so a later reader can enter it at that row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeafCheckpoint {
    /// The row this checkpoint addresses, counted from the row group's first.
    pub row: usize,
    pub at: PageCheckpoint,
}

/// Total bytes before `position` in a page's scattered payload.
pub fn logical_offset(data: &[Bytes], position: ReaderPosition) -> usize {
    let prior: usize = data[..position.buffer_index].iter().map(Bytes::len).sum();
    prior + position.offset
}

/// Resolves a byte count into the buffer and offset holding it, for whatever
/// layout this scan's copy of the page arrived in.
pub fn position_at(data: &[Bytes], byte_offset: usize) -> ReaderPosition {
    let mut remaining = byte_offset;
    for (buffer_index, buffer) in data.iter().enumerate() {
        if remaining < buffer.len() {
            return ReaderPosition {
                buffer_index,
                offset: remaining,
            };
        }
        remaining -= buffer.len();
    }
    // A checkpoint at the very end of the payload lands one past the last byte,
    // which belongs to the last buffer rather than to a buffer that isn't there.
    ReaderPosition {
        buffer_index: data.len().saturating_sub(1),
        offset: data.last().map(Bytes::len).unwrap_or(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buffers(sizes: &[usize]) -> Vec<Bytes> {
        sizes.iter().map(|&n| Bytes::from(vec![0u8; n])).collect()
    }

    #[test]
    fn logical_offset_counts_the_buffers_before_the_position() {
        let data = buffers(&[10, 20, 5]);

        let at = |buffer_index, offset| {
            logical_offset(
                &data,
                ReaderPosition {
                    buffer_index,
                    offset,
                },
            )
        };

        assert_eq!(at(0, 0), 0);
        assert_eq!(at(0, 7), 7);
        assert_eq!(at(1, 0), 10);
        assert_eq!(at(2, 3), 33);
    }

    /// A byte count resolves back to the buffer holding it, whatever the split.
    #[test]
    fn position_at_inverts_logical_offset() {
        let data = buffers(&[10, 20, 5]);

        for offset in 0..35 {
            assert_eq!(logical_offset(&data, position_at(&data, offset)), offset);
        }
    }

    /// The same byte count resolves correctly against a different split of the
    /// same payload, which is the point of storing a count rather than a
    /// buffer index.
    #[test]
    fn position_at_is_independent_of_how_the_payload_is_split() {
        let split = buffers(&[10, 20, 5]);
        let whole = buffers(&[35]);

        for offset in 0..35 {
            assert_eq!(
                logical_offset(&split, position_at(&split, offset)),
                logical_offset(&whole, position_at(&whole, offset))
            );
        }
    }

    #[test]
    fn position_at_clamps_to_the_end_of_the_payload() {
        let data = buffers(&[10, 5]);

        assert_eq!(logical_offset(&data, position_at(&data, 15)), 15);
        assert_eq!(logical_offset(&data, position_at(&data, 99)), 15);
    }
}
