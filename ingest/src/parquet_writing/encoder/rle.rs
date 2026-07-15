//! Parquet RLE/bit-packed hybrid encoder for dictionary indices and definition
//! levels — the write-side mirror of the reader's `RleDecoder`.
//!
//! A faithful port of the Parquet reference encoder (arrow's `RleEncoder`):
//! values are buffered eight at a time, and each group of eight is emitted as a
//! bit-packed run unless the value has been repeating for ≥ 8, in which case the
//! run becomes a (much smaller) RLE run. Two simplifications vs. arrow, both
//! valid: each bit-packed run here is a single group of eight (we don't merge
//! adjacent groups, costing one extra indicator byte per eight values), and the
//! whole stream is built straight into a `Vec<u8>` since every run is
//! byte-aligned (a group of eight values is exactly `bit_width` bytes).

/// Parquet bit-packs values in groups of this many; an RLE run also becomes
/// worthwhile once a value has repeated this many times.
const GROUP: usize = 8;

/// Encode dictionary `indices` as a Parquet RLE/bit-packed hybrid stream at
/// `bit_width` bits per index (`bit_width >= 1`).
#[cfg(test)]
pub(super) fn encode_indices(indices: &[u32], bit_width: u8) -> Vec<u8> {
    encode(indices.iter().copied(), bit_width)
}

/// Encode a stream whose values are generated on demand. Definition levels use
/// this path so a nullable leaf does not allocate an intermediate integer per
/// row before RLE compression.
pub(super) fn encode(values: impl IntoIterator<Item = u32>, bit_width: u8) -> Vec<u8> {
    let mut encoder = RleEncoder::new(bit_width);
    for value in values {
        encoder.put(value);
    }
    encoder.finish()
}

struct RleEncoder {
    bit_width: u8,
    out: Vec<u8>,
    /// Values held back for the current (possibly bit-packed) group.
    buffered: [u32; GROUP],
    num_buffered: usize,
    /// The last value seen and how many times it has repeated in a row. A repeat
    /// of a full [`GROUP`] switches the run to RLE.
    current_value: u32,
    repeat_count: usize,
}

impl RleEncoder {
    fn new(bit_width: u8) -> Self {
        Self {
            bit_width,
            out: Vec::new(),
            buffered: [0; GROUP],
            num_buffered: 0,
            current_value: 0,
            repeat_count: 0,
        }
    }

    fn put(&mut self, value: u32) {
        if value == self.current_value {
            self.repeat_count += 1;
            if self.repeat_count > GROUP {
                // Mid-RLE-run: counted, not buffered (the group it started in
                // was already flushed as RLE-pending).
                return;
            }
        } else {
            if self.repeat_count >= GROUP {
                self.flush_rle_run();
            }
            self.repeat_count = 1;
            self.current_value = value;
        }
        self.buffered[self.num_buffered] = value;
        self.num_buffered += 1;
        if self.num_buffered == GROUP {
            self.flush_group();
        }
    }

    /// A full group: bit-pack it, unless the value has repeated a whole [`GROUP`],
    /// in which case it belongs to an RLE run and is dropped here (the run's count
    /// already covers it, flushed at the next distinct value or at finish).
    fn flush_group(&mut self) {
        if self.repeat_count < GROUP {
            put_bit_packed_group(&mut self.out, &self.buffered, self.bit_width);
            self.repeat_count = 0;
        }
        self.num_buffered = 0;
    }

    fn flush_rle_run(&mut self) {
        put_rle_run(
            &mut self.out,
            self.repeat_count,
            self.current_value,
            self.bit_width,
        );
        self.num_buffered = 0;
        self.repeat_count = 0;
    }

    fn finish(mut self) -> Vec<u8> {
        if self.repeat_count == 0 && self.num_buffered == 0 {
            return self.out;
        }
        // A pure repeat (an ongoing RLE run, or a short all-equal tail) flushes as
        // RLE; anything else bit-packs the partial group, zero-padded to eight
        // (the padding lands past the page's value count, so it is never read).
        let all_repeat = self.num_buffered == 0 || self.repeat_count == self.num_buffered;
        if self.repeat_count > 0 && all_repeat {
            self.flush_rle_run();
        } else {
            self.buffered[self.num_buffered..].fill(0);
            put_bit_packed_group(&mut self.out, &self.buffered, self.bit_width);
        }
        self.out
    }
}

/// Append a bit-packed run of one group: the indicator (`1 << 1 | 1`) then the
/// group's values, `bit_width` bits each, LSB-first — exactly `bit_width` bytes.
fn put_bit_packed_group(out: &mut Vec<u8>, group: &[u32; GROUP], bit_width: u8) {
    put_vlq(out, (1 << 1) | 1);
    let mut byte = 0u8;
    let mut filled = 0u8;
    for &value in group {
        for bit in 0..bit_width {
            byte |= (((value >> bit) & 1) as u8) << filled;
            filled += 1;
            if filled == 8 {
                out.push(byte);
                byte = 0;
                filled = 0;
            }
        }
    }
    debug_assert_eq!(filled, 0, "a group of 8 is a whole number of bytes");
}

/// Append an RLE run: the indicator (`count << 1`) then the repeated value in
/// `ceil(bit_width / 8)` little-endian bytes.
fn put_rle_run(out: &mut Vec<u8>, count: usize, value: u32, bit_width: u8) {
    put_vlq(out, (count as u64) << 1);
    let value_bytes = (bit_width as usize).div_ceil(8);
    out.extend_from_slice(&value.to_le_bytes()[..value_bytes]);
}

/// Append a ULEB128 varint.
fn put_vlq(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let low = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(low);
            return;
        }
        out.push(low | 0x80);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_vlq(stream: &[u8], pos: &mut usize) -> u64 {
        let mut v = 0u64;
        let mut shift = 0;
        loop {
            let b = stream[*pos];
            *pos += 1;
            v |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return v;
            }
            shift += 7;
        }
    }

    /// A reference decoder for the hybrid stream — deliberately simple, to check
    /// the encoder round-trips.
    fn decode(stream: &[u8], bit_width: u8, count: usize) -> Vec<u32> {
        let mut values = Vec::new();
        let mut pos = 0;
        while values.len() < count {
            let header = read_vlq(stream, &mut pos);
            if header & 1 == 1 {
                // Bit-packed run of `groups * 8` values.
                let groups = (header >> 1) as usize;
                let mut bit = 0;
                for _ in 0..groups * 8 {
                    let mut value = 0u32;
                    for b in 0..bit_width {
                        let byte = stream[pos + bit / 8];
                        value |= u32::from((byte >> (bit % 8)) & 1) << b;
                        bit += 1;
                    }
                    values.push(value);
                }
                pos += bit / 8;
            } else {
                // RLE run of `count` copies of one value.
                let run = (header >> 1) as usize;
                let value_bytes = (bit_width as usize).div_ceil(8);
                let mut buf = [0u8; 4];
                buf[..value_bytes].copy_from_slice(&stream[pos..pos + value_bytes]);
                pos += value_bytes;
                let value = u32::from_le_bytes(buf);
                values.extend(std::iter::repeat_n(value, run));
            }
        }
        values.truncate(count);
        values
    }

    fn round_trip(indices: &[u32], bit_width: u8) {
        let stream = encode_indices(indices, bit_width);
        assert_eq!(decode(&stream, bit_width, indices.len()), indices);
    }

    #[test]
    fn all_equal_is_one_rle_run() {
        let indices = vec![3u32; 100];
        let stream = encode_indices(&indices, 3);
        // RLE indicator (100 << 1 = 200) as a 2-byte varint, then the value.
        assert_eq!(stream, vec![200, 1, 3]);
        round_trip(&indices, 3);
    }

    #[test]
    fn distinct_values_bit_pack() {
        round_trip(&[0, 1, 2, 3, 4, 5, 6, 7], 3);
    }

    #[test]
    fn mixed_runs_and_literals_round_trip() {
        let mut indices = vec![1, 2, 3, 4, 5]; // short literal stretch
        indices.extend(std::iter::repeat_n(6, 20)); // long run -> RLE
        indices.extend([7, 8, 9]); // trailing literals (partial group)
        round_trip(&indices, 4);
    }

    #[test]
    fn non_multiple_of_eight_round_trips() {
        round_trip(&[5, 1, 5, 2, 5], 3);
        round_trip(&[0], 1);
    }
}
