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
//! an RLE boolean page stores its values, and it is decoded a word at a time:
//! a repeated run shifts ones or zeros into the word, a bit-packed run shifts
//! its bytes in, and each full word is written whole. Nested levels are
//! extracted one at a time and compared to the maximum.

use bytes::Bytes;
use dispatch::arrays::{BitSink, append_bits, count_set, set_run};
use dispatch::memory::{MultiBufferReader, ReaderPosition};

use super::boolean::decode_varint;

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

    #[cfg(test)]
    pub fn is_present(&self, row: usize) -> bool {
        (self.bytes[row / 8] >> (row % 8)) & 1 == 1
    }

    /// How many of the `count` rows from `row` on hold a value.
    pub fn count_present_in(&self, row: usize, count: usize) -> usize {
        count_set(&self.bytes, row, count)
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
    let present = if max_def_level == 1 {
        decode_one_bit_levels(data, position, num_values, &mut presence)
    } else {
        let mut reader = MultiBufferReader::new(data, position);
        decode_nested_levels(
            &mut reader,
            num_values,
            byte_len,
            max_def_level,
            &mut presence,
        );
        presence.count_present()
    };
    *position = levels_start;
    MultiBufferReader::new(data, position).skip(4 + byte_len);
    (present < num_values).then_some(presence)
}

/// Decodes `num_values` one-bit levels, the stream a column with one level
/// carries, into `presence`, and tells how many rows are present. The stream
/// is the presence bitmap stored as runs, so the bits go through a word that
/// is written whole each time it fills: a repeated run is a shift of ones or
/// zeros, a bit-packed run a shift of its bytes.
fn decode_one_bit_levels(
    data: &[Bytes],
    position: &mut ReaderPosition,
    num_values: usize,
    presence: &mut Presence,
) -> usize {
    let mut bits = WordBits::over(&mut presence.bytes);
    let mut left = num_values;
    while left > 0 {
        let (header, value) = read_run_header(data, position);
        if header & 1 == 0 {
            let count = (header >> 1).min(left);
            bits.push_run(value, count);
            left -= count;
        } else {
            // The run's groups are whole bytes at the cursor, possibly spread
            // over buffers; the stream's last group may pad past the rows.
            let mut bytes_left = header >> 1;
            while bytes_left > 0 && left > 0 {
                let buffer = &data[position.buffer_index];
                if position.offset == buffer.len() {
                    position.buffer_index += 1;
                    position.offset = 0;
                    continue;
                }
                let take = bytes_left.min(buffer.len() - position.offset);
                let count = (take * 8).min(left);
                bits.push_bytes(&buffer[position.offset..position.offset + take], count);
                left -= count;
                bytes_left -= take;
                position.offset += take;
            }
        }
    }
    presence.len = num_values;
    bits.finish()
}

/// Reads the run header at `position`: the header word and, for a repeated
/// run, its one-byte value. A header is at most a five-byte varint and that
/// byte; one the buffer holds whole is decoded in place, one at a buffer's
/// end through the reader, which crosses into the next buffer.
fn read_run_header(data: &[Bytes], position: &mut ReaderPosition) -> (usize, bool) {
    let buffer = &data[position.buffer_index];
    let bytes = &buffer[position.offset..];
    if bytes.len() >= 6 {
        let (header, used) = decode_varint(bytes);
        if header & 1 == 0 {
            position.offset += used + 1;
            (header, bytes[used] & 1 == 1)
        } else {
            position.offset += used;
            (header, false)
        }
    } else {
        let mut reader = MultiBufferReader::new(data, position);
        let header = reader.read_varint() as usize;
        let value = header & 1 == 0 && reader.read_u8() & 1 == 1;
        (header, value)
    }
}

/// Bits appended to a byte buffer through a word: the word is written whole
/// each time it fills, and the count of set bits is kept as it goes.
struct WordBits<'a> {
    bytes: &'a mut [u8],
    /// Bits written to `bytes`, a multiple of 64 until `finish`.
    written: usize,
    word: u64,
    /// Bits in `word`, below 64.
    filled: u32,
    present: usize,
}

impl<'a> WordBits<'a> {
    fn over(bytes: &'a mut [u8]) -> Self {
        Self {
            bytes,
            written: 0,
            word: 0,
            filled: 0,
            present: 0,
        }
    }

    /// Appends the low `count` bits of `bits`, `count` at most 64 and the
    /// higher bits zero.
    #[inline(always)]
    fn push(&mut self, bits: u64, count: u32) {
        let room = 64 - self.filled;
        if count < room {
            self.word |= bits << self.filled;
            self.filled += count;
            return;
        }
        self.word |= bits << self.filled;
        self.flush();
        // The bits that did not fit start the next word.
        self.word = if room == 64 { 0 } else { bits >> room };
        self.filled = count - room;
    }

    fn flush(&mut self) {
        let at = self.written / 8;
        self.bytes[at..at + 8].copy_from_slice(&self.word.to_le_bytes());
        self.present += self.word.count_ones() as usize;
        self.written += 64;
        self.word = 0;
        self.filled = 0;
    }

    /// Appends `count` bits of `value`.
    fn push_run(&mut self, value: bool, mut count: usize) {
        let pattern = if value { u64::MAX } else { 0 };
        while count > 0 {
            if self.filled == 0 && count >= 64 {
                // Whole words straight to the buffer, which is zero already.
                let words = count / 64;
                let at = self.written / 8;
                if value {
                    self.bytes[at..at + words * 8].fill(0xff);
                    self.present += words * 64;
                }
                self.written += words * 64;
                count -= words * 64;
                continue;
            }
            let take = count.min(64) as u32;
            let bits = if take == 64 {
                pattern
            } else {
                pattern & ((1u64 << take) - 1)
            };
            self.push(bits, take);
            count -= take as usize;
        }
    }

    /// Appends the first `count` bits of `chunk`, which holds them packed
    /// least significant bit first.
    fn push_bytes(&mut self, chunk: &[u8], mut count: usize) {
        let mut at = 0;
        while count >= 64 {
            let word = u64::from_le_bytes(chunk[at..at + 8].try_into().unwrap());
            self.push(word, 64);
            at += 8;
            count -= 64;
        }
        if count > 0 {
            let mut tail = [0u8; 8];
            let used = count.div_ceil(8);
            tail[..used].copy_from_slice(&chunk[at..at + used]);
            let word = u64::from_le_bytes(tail) & ((1u64 << count) - 1);
            self.push(word, count as u32);
        }
    }

    /// Writes the bits still in the word and returns the count of set bits.
    fn finish(mut self) -> usize {
        if self.filled > 0 {
            let at = self.written / 8;
            let used = (self.filled as usize).div_ceil(8);
            self.bytes[at..at + used].copy_from_slice(&self.word.to_le_bytes()[..used]);
            self.present += self.word.count_ones() as usize;
        }
        self.present
    }
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
    fn bit_packed_levels_cross_word_boundaries() {
        let levels: Vec<u8> = (0..150).map(|i| (i % 5 != 2) as u8).collect();
        let bytes = bit_packed(&levels, 1);

        let want: Vec<bool> = levels.iter().map(|&l| l == 1).collect();
        assert_eq!(decode(&bytes, 150, 1), Some(want));
    }

    #[test]
    fn runs_and_groups_mix_across_words() {
        // Ten present, a packed group with nulls, a run of 70 nulls, three
        // present: runs that start and end inside words and a group between.
        let mut stream = rle(&[1; 10]);
        stream.extend(bit_packed(&[1, 0, 1, 1, 0, 0, 1, 0], 1));
        stream.extend(rle(&[0; 70]));
        stream.extend(rle(&[1; 3]));

        let mut want = vec![true; 10];
        want.extend([true, false, true, true, false, false, true, false]);
        want.extend([false; 70]);
        want.extend([true; 3]);
        assert_eq!(decode(&stream, 91, 1), Some(want));
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
    fn counts_the_present_rows_of_any_range() {
        // 70 present rows, 3 nulls, 20 present rows.
        let mut levels = vec![1u8; 70];
        levels.extend([0, 0, 0]);
        levels.extend([1; 20]);
        let mut page = (rle(&levels).len() as u32).to_le_bytes().to_vec();
        page.extend(rle(&levels));
        let data = vec![Bytes::from(page)];
        let mut position = ReaderPosition::default();

        let presence = decode_def_levels(&data, &mut position, 93, 1).unwrap();

        assert_eq!(presence.count_present_in(0, 93), 90);
        assert_eq!(presence.count_present_in(68, 7), 4);
        assert_eq!(presence.count_present_in(70, 3), 0);
        assert_eq!(presence.count_present_in(5, 10), 10);
    }
}
