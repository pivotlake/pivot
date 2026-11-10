//! The index file's footer and the byte-level helpers every section shares.
//!
//! ```text
//!   ┌──────────────┬────────────┬──────────────┬────────────┬ ─ ─ ┬──────────────┬────────┐
//!   │ postings of  │ dict block │ postings of  │ dict block │     │ sparse index │ footer │
//!   │ block 0      │ 0          │ block 1      │ 1          │     │              │ 48 B   │
//!   └──────────────┴────────────┴──────────────┴────────────┴ ─ ─ ┴──────────────┴────────┘
//!     ▲                 ▲                                              ▲    │
//!     │                 └── located by the sparse index ───────────────┘    │
//!     └── located by the dict block's entries        located by the footer ◀┘
//! ```
//!
//! The footer is the only fixed-position part: it sits at the end and says
//! where the sparse index is. A reader that asks for the file's tail usually
//! gets the footer and the sparse index in one read.
//!
//! Every integer in the file is little-endian.

use crate::text_index::Error;
use bytes::Bytes;
use std::borrow::Cow;

pub const FOOTER_LEN: usize = 48;
const MAGIC: [u8; 8] = *b"PIVOTTIX";
const VERSION: u32 = 1;
/// The tokenizer of [`tokenizer`](super::tokenizer): alphanumeric runs,
/// compared byte for byte.
const TOKENIZER_ALPHANUMERIC: u32 = 1;

/// What the end of an index file says about the rest of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Footer {
    pub sparse_offset: u64,
    pub sparse_len: u64,
    /// Rows of the indexed file, matching or not: the index answers for row
    /// positions `0..row_count`.
    pub row_count: u64,
    /// Distinct tokens in the index.
    pub token_count: u64,
}

impl Footer {
    pub fn encode(&self) -> [u8; FOOTER_LEN] {
        let mut bytes = [0u8; FOOTER_LEN];
        bytes[0..8].copy_from_slice(&self.sparse_offset.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.sparse_len.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.row_count.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.token_count.to_le_bytes());
        bytes[32..36].copy_from_slice(&TOKENIZER_ALPHANUMERIC.to_le_bytes());
        bytes[36..40].copy_from_slice(&VERSION.to_le_bytes());
        bytes[40..48].copy_from_slice(&MAGIC);
        bytes
    }

    /// Decode the footer of a `file_len` byte index file from its last
    /// [`FOOTER_LEN`] bytes.
    pub fn decode(bytes: &[u8], file_len: usize) -> Result<Self, Error> {
        if bytes.len() != FOOTER_LEN {
            return Err(Error::corrupt("footer is truncated"));
        }
        if bytes[40..48] != MAGIC {
            return Err(Error::corrupt("footer magic is missing"));
        }
        let version = read_u32(bytes, 36);
        if version != VERSION {
            return Err(Error::corrupt(format!("unknown format version {version}")));
        }
        let tokenizer = read_u32(bytes, 32);
        if tokenizer != TOKENIZER_ALPHANUMERIC {
            return Err(Error::corrupt(format!("unknown tokenizer {tokenizer}")));
        }
        let footer = Self {
            sparse_offset: read_u64(bytes, 0),
            sparse_len: read_u64(bytes, 8),
            row_count: read_u64(bytes, 16),
            token_count: read_u64(bytes, 24),
        };
        let Some(footer_offset) = file_len.checked_sub(FOOTER_LEN) else {
            return Err(Error::corrupt("file is shorter than a footer"));
        };
        let sparse_end = footer.sparse_offset.checked_add(footer.sparse_len);
        if sparse_end != Some(footer_offset as u64) {
            return Err(Error::corrupt("sparse index does not end at the footer"));
        }
        Ok(footer)
    }
}

/// Somewhere encoded bytes are appended to.
pub trait ByteSink {
    fn write_bytes(&mut self, bytes: &[u8]);
}

impl ByteSink for Vec<u8> {
    fn write_bytes(&mut self, bytes: &[u8]) {
        self.extend_from_slice(bytes);
    }
}

pub(crate) fn read_u16(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap())
}

pub(crate) fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

pub(crate) fn read_u64(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

/// Append `value` as a LEB128 varint.
pub(crate) fn write_varint(sink: &mut impl ByteSink, mut value: u64) {
    let mut encoded = [0u8; 10];
    let mut len = 0;
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            encoded[len] = byte;
            len += 1;
            break;
        }
        encoded[len] = byte | 0x80;
        len += 1;
    }
    sink.write_bytes(&encoded[..len]);
}

/// Read a LEB128 varint from the front of `bytes`, advancing it.
pub(crate) fn read_varint(bytes: &mut &[u8]) -> Result<u64, Error> {
    let mut value = 0u64;
    for (index, &byte) in bytes.iter().enumerate().take(10) {
        value |= ((byte & 0x7f) as u64) << (7 * index);
        if byte & 0x80 == 0 {
            *bytes = &bytes[index + 1..];
            return Ok(value);
        }
    }
    Err(Error::corrupt("varint is truncated"))
}

/// Take the next `len` bytes from the front of `bytes`, advancing it.
pub(crate) fn take_bytes<'a>(bytes: &mut &'a [u8], len: usize) -> Result<&'a [u8], Error> {
    if bytes.len() < len {
        return Err(Error::corrupt("section is truncated"));
    }
    let (taken, rest) = bytes.split_at(len);
    *bytes = rest;
    Ok(taken)
}

/// The fragments of one read as a single slice: borrowed when the read came
/// back in one piece, copied together otherwise.
pub(crate) fn join_fragments(fragments: &[Bytes]) -> Cow<'_, [u8]> {
    match fragments {
        [single] => Cow::Borrowed(single.as_ref()),
        many => Cow::Owned(
            many.iter()
                .flat_map(|fragment| fragment.iter().copied())
                .collect(),
        ),
    }
}

/// Reads the fragments of one read front to back, across their boundaries,
/// for sections too large to be worth copying into one piece first.
pub(crate) struct FragmentReader<'a> {
    fragments: &'a [Bytes],
    offset_in_first: usize,
}

impl<'a> FragmentReader<'a> {
    pub fn new(fragments: &'a [Bytes]) -> Self {
        Self {
            fragments,
            offset_in_first: 0,
        }
    }

    pub fn remaining(&self) -> usize {
        self.fragments.iter().map(Bytes::len).sum::<usize>() - self.offset_in_first
    }

    /// Fill `out` with the next bytes.
    pub fn read_exact(&mut self, out: &mut [u8]) -> Result<(), Error> {
        let mut filled = 0;
        while filled < out.len() {
            let Some(first) = self.fragments.first() else {
                return Err(Error::corrupt("section is truncated"));
            };
            let available = &first[self.offset_in_first..];
            let take = available.len().min(out.len() - filled);
            out[filled..filled + take].copy_from_slice(&available[..take]);
            filled += take;
            self.offset_in_first += take;
            if self.offset_in_first == first.len() {
                self.fragments = &self.fragments[1..];
                self.offset_in_first = 0;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_footer_round_trips() {
        let footer = Footer {
            sparse_offset: 1000,
            sparse_len: 24,
            row_count: 77,
            token_count: 5,
        };

        let decoded = Footer::decode(&footer.encode(), 1000 + 24 + FOOTER_LEN).unwrap();

        assert_eq!(decoded, footer);
    }

    #[test]
    fn a_footer_whose_sparse_index_does_not_meet_it_is_rejected() {
        let footer = Footer {
            sparse_offset: 1000,
            sparse_len: 24,
            row_count: 77,
            token_count: 5,
        };

        let decoded = Footer::decode(&footer.encode(), 5000);

        assert!(decoded.is_err());
    }

    #[test]
    fn varints_round_trip_across_their_whole_range() {
        let values = [0u64, 1, 127, 128, 300, u32::MAX as u64, u64::MAX];
        let mut encoded = Vec::new();
        for value in values {
            write_varint(&mut encoded, value);
        }

        let mut rest = encoded.as_slice();
        let decoded: Vec<u64> = values
            .iter()
            .map(|_| read_varint(&mut rest).unwrap())
            .collect();

        assert_eq!(decoded, values);
        assert!(rest.is_empty());
    }

    #[test]
    fn a_fragment_reader_reads_across_fragment_boundaries() {
        let fragments = [Bytes::from_static(b"abc"), Bytes::from_static(b"defgh")];
        let mut reader = FragmentReader::new(&fragments);
        let mut first = [0u8; 2];
        let mut second = [0u8; 5];

        reader.read_exact(&mut first).unwrap();
        reader.read_exact(&mut second).unwrap();

        assert_eq!(&first, b"ab");
        assert_eq!(&second, b"cdefg");
        assert_eq!(reader.remaining(), 1);
    }
}
