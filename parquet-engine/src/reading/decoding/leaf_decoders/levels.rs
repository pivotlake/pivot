//! Decodes Parquet definition levels from the RLE/bit-packed hybrid encoding.
//!
//! In Parquet, nullable columns prefix each data page with definition levels:
//! a row holds a value when its level equals the column's `max_def_level`, and
//! is null otherwise (an ancestor or the leaf is absent). The levels are
//! `bit_width`-bit values: one level for a directly-nullable column, but more
//! under nesting (e.g. a variant's shredded `typed_value` leaves), where a row
//! can be null at several depths.
//!
//! The result is the page's [`Presence`], one bit per row. For a column with
//! one level the stream is the presence bitmap itself, stored as runs the way
//! an RLE boolean page stores its values, so it is read by that page's decoder:
//! a repeated run becomes a bit fill and a bit-packed run a byte copy. Nested
//! levels are extracted one at a time and compared to the maximum.

use bytes::Bytes;
use dispatch::arrays::{BitSink, append_bits, set_run};
use dispatch::memory::{MultiBufferReader, ReaderPosition};

use super::boolean::BooleanRleDecoder;

/// Which rows of a page hold a value, one bit per row, `1` for present.
pub struct Presence {
    bytes: Vec<u8>,
    len: usize,
}

impl Presence {
    fn with_capacity(rows: usize) -> Self {
        Self {
            bytes: vec![0; rows.div_ceil(8)],
            len: 0,
        }
    }

    /// The bitmap's bytes, least significant bit first.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn is_present(&self, row: usize) -> bool {
        (self.bytes[row / 8] >> (row % 8)) & 1 == 1
    }

    /// How many rows from `row` on share its presence, at most `max`: found
    /// a 64-bit word at a time, by counting the trailing bits that match.
    pub fn run_from(&self, row: usize, max: usize) -> usize {
        let present = self.is_present(row);
        let end = (row + max).min(self.len);
        let mut at = row;
        while at < end {
            let byte = at / 8;
            let bit = at % 8;
            let take = (self.bytes.len() - byte).min(8);
            let mut word = [0u8; 8];
            word[..take].copy_from_slice(&self.bytes[byte..byte + take]);
            // Bits equal to the run's value become zero, so the run's length
            // in this word is the count of trailing zeros, bounded by the bits
            // the word really holds.
            let mut word = u64::from_le_bytes(word) >> bit;
            if present {
                word = !word;
            }
            let available = take * 8 - bit;
            let run = (word.trailing_zeros() as usize).min(available);
            at += run;
            if run < available {
                break;
            }
        }
        at.min(end) - row
    }

    fn count_present(&self) -> usize {
        self.bytes
            .iter()
            .map(|byte| byte.count_ones() as usize)
            .sum()
    }
}

impl BitSink for Presence {
    #[inline]
    fn append_bits(&mut self, src: &[u8], src_bit: usize, count: usize) {
        append_bits(&mut self.bytes, self.len, src, src_bit, count);
        self.len += count;
    }

    #[inline]
    fn append_run(&mut self, value: bool, count: usize) {
        if value {
            set_run(&mut self.bytes, self.len, count);
        }
        self.len += count;
    }
}

/// Decodes the definition levels at `position` in a page: the 4-byte length
/// prefix and the RLE/bit-packed hybrid stream after it, `num_values` levels
/// long. Leaves `position` at the values that follow the levels, which the
/// length places whatever padding the stream ends with.
///
/// Returns `None` when every row is present, so the caller takes the fast,
/// non-null decode path without a presence bitmap.
pub fn decode_def_levels(
    data: &[Bytes],
    position: &mut ReaderPosition,
    num_values: usize,
    max_def_level: i16,
) -> Option<Presence> {
    let levels_start = *position;
    let byte_len = MultiBufferReader::new(data, position).read_u32_le() as usize;
    let mut presence = Presence::with_capacity(num_values);
    if max_def_level == 1 {
        BooleanRleDecoder::read_from(data.to_vec(), levels_start)
            .read_bits(&mut presence, num_values);
    } else {
        let mut reader = MultiBufferReader::new(data, position);
        decode_nested_levels(
            &mut reader,
            num_values,
            byte_len,
            max_def_level,
            &mut presence,
        );
    }
    *position = levels_start;
    MultiBufferReader::new(data, position).skip(4 + byte_len);
    (presence.count_present() < num_values).then_some(presence)
}

/// Decodes `num_values` levels of more than one bit each, present where a
/// level equals `max_def_level`, into `presence`.
fn decode_nested_levels(
    reader: &mut MultiBufferReader,
    num_values: usize,
    byte_len: usize,
    max_def_level: i16,
    presence: &mut Presence,
) {
    let bit_width = bit_width_for(max_def_level);
    let max = max_def_level as u32;
    let mut bytes_read = 0usize;
    let mut len = 0usize;

    while len < num_values && bytes_read < byte_len {
        let header = read_varint_tracked(reader, &mut bytes_read);

        if header & 1 == 0 {
            // RLE run: `count` copies of one `bit_width`-bit level.
            let count = ((header >> 1) as usize).min(num_values - len);
            let value = read_le(reader, bit_width.div_ceil(8) as usize, &mut bytes_read);
            presence.append_run(value == max, count);
            len += count;
        } else {
            // Bit-packed run: (header >> 1) groups of 8 `bit_width`-bit levels,
            // little-endian within the run.
            let (mut acc, mut acc_bits) = (0u64, 0u32);
            for _ in 0..(header >> 1) as usize * 8 {
                while acc_bits < bit_width {
                    acc |= (reader.read_u8() as u64) << acc_bits;
                    bytes_read += 1;
                    acc_bits += 8;
                }
                let value = (acc & ((1u64 << bit_width) - 1)) as u32;
                acc >>= bit_width;
                acc_bits -= bit_width;
                if len < num_values {
                    presence.append_run(value == max, 1);
                    len += 1;
                }
            }
        }
    }
}

/// Bits needed to encode the levels `0..=max_def_level`.
fn bit_width_for(max_def_level: i16) -> u32 {
    (max_def_level as u32 + 1)
        .next_power_of_two()
        .trailing_zeros()
}

/// Read a little-endian unsigned value of `nbytes` (an RLE run's repeated level).
fn read_le(reader: &mut MultiBufferReader, nbytes: usize, bytes_read: &mut usize) -> u32 {
    let mut value = 0u32;
    for i in 0..nbytes {
        value |= (reader.read_u8() as u32) << (i * 8);
        *bytes_read += 1;
    }
    value
}

/// Reads an unsigned LEB128 (varint) from `reader`, incrementing
/// `bytes_read` by the number of bytes consumed.
fn read_varint_tracked(reader: &mut MultiBufferReader, bytes_read: &mut usize) -> u32 {
    let mut result = 0u32;
    let mut shift = 0;
    loop {
        let byte = reader.read_u8();
        *bytes_read += 1;
        result |= ((byte & 0x7F) as u32) << shift;
        if byte & 0x80 == 0 {
            return result;
        }
        shift += 7;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn varint(out: &mut Vec<u8>, mut v: u32) {
        loop {
            let byte = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(byte);
                return;
            }
            out.push(byte | 0x80);
        }
    }

    /// Encode `levels` as RLE runs (one value byte per run; bit_width <= 8).
    fn rle(levels: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < levels.len() {
            let run = levels[i..].iter().take_while(|&&x| x == levels[i]).count();
            varint(&mut out, (run as u32) << 1);
            out.push(levels[i]);
            i += run;
        }
        out
    }

    /// Encode `levels` as one bit-packed run of `bit_width`-bit values, LSB-first
    /// (Parquet's order), padding the final group of 8 with zeroes.
    fn bit_packed(levels: &[u8], bit_width: u32) -> Vec<u8> {
        let groups = levels.len().div_ceil(8);
        let mut padded = levels.to_vec();
        padded.resize(groups * 8, 0);
        let mut out = Vec::new();
        varint(&mut out, ((groups as u32) << 1) | 1);
        let (mut acc, mut bits) = (0u64, 0u32);
        for &l in &padded {
            acc |= (l as u64) << bits;
            bits += bit_width;
            while bits >= 8 {
                out.push((acc & 0xff) as u8);
                acc >>= 8;
                bits -= 8;
            }
        }
        out
    }

    /// Decodes a levels section made of `stream`, and checks the cursor lands
    /// on the bytes after it.
    fn decode(stream: &[u8], num_values: usize, max_def_level: i16) -> Option<Vec<bool>> {
        let mut page = (stream.len() as u32).to_le_bytes().to_vec();
        page.extend_from_slice(stream);
        page.extend_from_slice(b"values");
        let data = vec![Bytes::from(page)];
        let mut position = ReaderPosition::default();

        let presence = decode_def_levels(&data, &mut position, num_values, max_def_level);

        assert_eq!(position.offset, 4 + stream.len());
        presence.map(|presence| {
            (0..num_values)
                .map(|row| presence.is_present(row))
                .collect()
        })
    }

    #[test]
    fn all_present_is_none() {
        let bytes = rle(&[1, 1, 1]);

        assert_eq!(decode(&bytes, 3, 1), None);
    }

    #[test]
    fn rle_nulls_yield_a_mask() {
        let bytes = rle(&[1, 1, 0, 0, 1]);

        assert_eq!(
            decode(&bytes, 5, 1),
            Some(vec![true, true, false, false, true])
        );
    }

    #[test]
    fn two_level_present_only_at_max() {
        // max_def 2: present where def == 2; def == 1 is null at an ancestor.
        let bytes = rle(&[2, 1, 2, 0]);

        assert_eq!(decode(&bytes, 4, 2), Some(vec![true, false, true, false]));
    }

    #[test]
    fn two_level_all_present_is_none() {
        let bytes = rle(&[2, 2, 2]);

        assert_eq!(decode(&bytes, 3, 2), None);
    }

    #[test]
    fn three_level_bit_width_two() {
        // max_def 3 -> bit_width 2; present only where def == 3.
        let bytes = rle(&[3, 1, 3, 2, 0]);

        assert_eq!(
            decode(&bytes, 5, 3),
            Some(vec![true, false, true, false, false])
        );
    }

    #[test]
    fn bit_width_three_levels() {
        // max_def 7 -> bit_width 3.
        let bytes = rle(&[7, 0, 7, 4]);

        assert_eq!(decode(&bytes, 4, 7), Some(vec![true, false, true, false]));
    }

    #[test]
    fn bit_packed_one_level() {
        let bytes = bit_packed(&[1, 0, 1, 0, 0, 1], 1);

        assert_eq!(
            decode(&bytes, 6, 1),
            Some(vec![true, false, true, false, false, true])
        );
    }

    #[test]
    fn bit_packed_two_level() {
        let bytes = bit_packed(&[2, 0, 2, 1], 2);

        assert_eq!(decode(&bytes, 4, 2), Some(vec![true, false, true, false]));
    }

    #[test]
    fn decodes_exactly_num_values_when_run_is_longer() {
        let bytes = rle(&[0, 0, 0, 0, 0]); // a run of 5

        assert_eq!(decode(&bytes, 3, 1), Some(vec![false, false, false]));
    }

    #[test]
    fn a_null_after_present_rows_keeps_them_present() {
        let bytes = rle(&[1, 1, 1, 0]);

        assert_eq!(decode(&bytes, 4, 1), Some(vec![true, true, true, false]));
    }

    #[test]
    fn runs_are_found_across_word_boundaries() {
        // 70 present rows, 3 nulls, 20 present rows: runs that start mid-byte
        // and span more than one 64-bit word.
        let mut levels = vec![1u8; 70];
        levels.extend([0, 0, 0]);
        levels.extend([1; 20]);
        let mut page = (rle(&levels).len() as u32).to_le_bytes().to_vec();
        page.extend(rle(&levels));
        let data = vec![Bytes::from(page)];
        let mut position = ReaderPosition::default();

        let presence = decode_def_levels(&data, &mut position, 93, 1).unwrap();

        assert_eq!(presence.run_from(0, 1000), 70);
        assert_eq!(presence.run_from(5, 1000), 65);
        assert_eq!(presence.run_from(5, 10), 10);
        assert_eq!(presence.run_from(70, 1000), 3);
        assert_eq!(presence.run_from(72, 1000), 1);
        assert_eq!(presence.run_from(73, 1000), 20);
        assert_eq!(presence.run_from(92, 1000), 1);
    }
}
