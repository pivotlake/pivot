//! Decodes Parquet definition levels from the RLE/bit-packed hybrid encoding.
//!
//! In Parquet, nullable columns prefix each data page with definition levels
//! that indicate which rows are present (`def_level == max_def_level`) and
//! which are null. This module decodes those levels for the `bit_width == 1`
//! case (single level of nullability) into a `Vec<bool>`.

use dispatch::memory::MultiBufferReader;

/// Decode RLE/bit-packed hybrid encoded definition levels (bit_width=1).
///
/// Reads `byte_len` bytes from `reader` (the data after the 4-byte length prefix).
/// Returns a `Vec<bool>` where `true` = value present, `false` = null.
pub fn decode_def_levels(
    reader: &mut MultiBufferReader,
    num_values: usize,
    byte_len: usize,
) -> Vec<bool> {
    let mut bytes_read = 0usize;
    let mut result = Vec::with_capacity(num_values);

    while result.len() < num_values && bytes_read < byte_len {
        let header = read_varint_tracked(reader, &mut bytes_read);

        if header & 1 == 0 {
            // RLE run
            let count = (header >> 1) as usize;
            let value = reader.read_u8() != 0;
            bytes_read += 1;
            let n = count.min(num_values - result.len());
            result.extend(std::iter::repeat_n(value, n));
        } else {
            // Bit-packed run: (header >> 1) groups of 8 values
            let num_groups = (header >> 1) as usize;
            for _ in 0..num_groups {
                let byte = reader.read_u8();
                bytes_read += 1;
                for bit in 0..8u8 {
                    if result.len() >= num_values {
                        break;
                    }
                    result.push((byte >> bit) & 1 != 0);
                }
            }
        }
    }

    // Ensure we consumed exactly byte_len bytes from the section
    while bytes_read < byte_len {
        reader.read_u8();
        bytes_read += 1;
    }

    result
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
