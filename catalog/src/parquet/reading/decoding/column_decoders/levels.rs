//! Decodes Parquet definition levels from the RLE/bit-packed hybrid encoding.
//!
//! In Parquet, nullable columns prefix each data page with definition levels:
//! a row holds a value when its level equals the column's `max_def_level`, and
//! is null otherwise (an ancestor or the leaf is absent). The levels are
//! `bit_width`-bit values: one level for a directly-nullable column, but more
//! under nesting (e.g. a variant's shredded `typed_value` leaves), where a row
//! can be null at several depths.

use dispatch::memory::MultiBufferReader;

/// Decode the RLE/bit-packed hybrid definition levels of a page.
///
/// Reads `byte_len` bytes from `reader` (the data after the 4-byte length
/// prefix); `max_def_level` fixes the level `bit_width` and the present
/// threshold. Returns `None` when every row is present (no nulls), so the
/// caller takes the fast, non-null decode path without materialising a mask;
/// otherwise `Some(present)` where `present[i]` means row `i` holds a value. The
/// mask is allocated lazily on the first null, so a page without nulls allocates
/// nothing.
pub fn decode_def_levels(
    reader: &mut MultiBufferReader,
    num_values: usize,
    byte_len: usize,
    max_def_level: i16,
) -> Option<Vec<bool>> {
    let bit_width = bit_width_for(max_def_level);
    let max = max_def_level as u32;
    let mut bytes_read = 0usize;
    let mut present: Option<Vec<bool>> = None;
    let mut len = 0usize;

    while len < num_values && bytes_read < byte_len {
        let header = read_varint_tracked(reader, &mut bytes_read);

        if header & 1 == 0 {
            // RLE run: `count` copies of one `bit_width`-bit level.
            let count = ((header >> 1) as usize).min(num_values - len);
            let value = read_le(reader, bit_width.div_ceil(8) as usize, &mut bytes_read);
            append_presence_run(&mut present, &mut len, num_values, value == max, count);
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
                    append_presence_run(&mut present, &mut len, num_values, value == max, 1);
                }
            }
        }
    }

    // Ensure we consumed exactly byte_len bytes from the section.
    while bytes_read < byte_len {
        reader.read_u8();
        bytes_read += 1;
    }

    present
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

/// Append a run of `n` rows that are all present (`true`) or all null
/// (`false`) to the lazily-built present mask, advancing `len`. The mask stays
/// `None` while every row seen is present; the first null materialises it and
/// backfills the present rows so far.
fn append_presence_run(
    present: &mut Option<Vec<bool>>,
    len: &mut usize,
    num_values: usize,
    value: bool,
    n: usize,
) {
    if !value && present.is_none() {
        let mut mask = Vec::with_capacity(num_values);
        mask.extend(std::iter::repeat_n(true, *len));
        *present = Some(mask);
    }
    if let Some(mask) = present {
        mask.extend(std::iter::repeat_n(value, n));
    }
    *len += n;
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
    use bytes::Bytes;
    use dispatch::memory::ReaderPosition;

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

    fn decode(bytes: &[u8], num_values: usize, max_def_level: i16) -> Option<Vec<bool>> {
        let data = vec![Bytes::copy_from_slice(bytes)];
        let mut pos = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(&data, &mut pos);
        decode_def_levels(&mut reader, num_values, bytes.len(), max_def_level)
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
    fn first_null_backfills_earlier_present_rows() {
        let bytes = rle(&[1, 1, 1, 0]);

        assert_eq!(decode(&bytes, 4, 1), Some(vec![true, true, true, false]));
    }
}
