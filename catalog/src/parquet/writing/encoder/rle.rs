//! RLE/bit-packed hybrid encoder for dictionary indices — the write-side mirror
//! of the reader's `RleDecoder`. Produces the run stream a dictionary data page
//! carries after its one leading bit-width byte (which the data-page builder
//! prepends).
//!
//! A faithful port of the Parquet reference encoder (arrow's `RleEncoder`):
//! values are buffered eight at a time, and each group of eight is bit-packed
//! unless the value has been repeating for ≥ 8, in which case the run becomes a
//! (much smaller) RLE run.
//!
//! Adjacent bit-packed groups share one run header, up to
//! [`MAX_GROUPS_PER_RUN`]. Giving each group its own header instead costs a byte
//! per eight values, which is a whole extra bit per value: on a dictionary
//! column of four-bit indices that is a quarter of the column. It costs more on
//! the read side, where a header per eight values is a run to parse per eight
//! values rather than one per several hundred.
//!
//! The stream is built straight into a `Vec<u8>` since every run is byte-aligned
//! (a group of eight values is exactly `bit_width` bytes).

/// Parquet bit-packs values in groups of this many; an RLE run also becomes
/// worthwhile once a value has repeated this many times.
const GROUP: usize = 8;

/// Groups a single bit-packed run may cover. Held to 63 so the run's header
/// stays one byte, which is what other writers emit and every reader expects.
const MAX_GROUPS_PER_RUN: usize = 63;

/// Encode dictionary `indices` as a Parquet RLE/bit-packed hybrid stream at
/// `bit_width` bits per index (`bit_width >= 1`).
pub(super) fn encode_indices(indices: &[u32], bit_width: u8) -> Vec<u8> {
    let mut encoder = RleEncoder::new(bit_width);
    for &index in indices {
        encoder.put(index);
    }
    encoder.finish()
}

/// Encode definition `levels` as a Parquet RLE/bit-packed hybrid stream — the
/// same encoding as the indices, over the levels `0..=max_def_level`. A leaf
/// present on every row collapses to a single RLE run, so the levels cost a few
/// bytes when they carry no information.
pub(super) fn encode_levels(levels: &[i16], max_def_level: i16) -> Vec<u8> {
    let mut encoder = RleEncoder::new(bit_width(max_def_level as usize));
    for &level in levels {
        encoder.put(level as u32);
    }
    encoder.finish()
}

/// Bits needed to encode the values `0..=max` — at least one, so a stream whose
/// only value is zero still carries a real bit width. Mirrors the reader's
/// `bit_width_for`; the two must agree or a page's levels decode as garbage.
pub(super) fn bit_width(max: usize) -> u8 {
    if max == 0 {
        1
    } else {
        (usize::BITS - max.leading_zeros()) as u8
    }
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
    /// Bit-packed groups waiting to be written as one run, and how many.
    packed: Vec<u8>,
    packed_groups: usize,
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
            packed: Vec::new(),
            packed_groups: 0,
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

    /// A full group: bit-pack it into the pending run, unless the value has
    /// repeated a whole [`GROUP`], in which case it belongs to an RLE run and is
    /// dropped here (the run's count already covers it, flushed at the next
    /// distinct value or at finish).
    fn flush_group(&mut self) {
        if self.repeat_count < GROUP {
            pack_group(&mut self.packed, &self.buffered, self.bit_width);
            self.packed_groups += 1;
            if self.packed_groups == MAX_GROUPS_PER_RUN {
                self.flush_packed_run();
            }
            self.repeat_count = 0;
        }
        self.num_buffered = 0;
    }

    /// Write the pending bit-packed groups as one run. Every run that follows
    /// them has to call this first, since they come earlier in the stream.
    fn flush_packed_run(&mut self) {
        if self.packed_groups == 0 {
            return;
        }
        put_vlq(&mut self.out, ((self.packed_groups as u64) << 1) | 1);
        self.out.extend_from_slice(&self.packed);
        self.packed.clear();
        self.packed_groups = 0;
    }

    fn flush_rle_run(&mut self) {
        self.flush_packed_run();
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
            // Whole groups may still be waiting under no header yet.
            self.flush_packed_run();
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
            pack_group(&mut self.packed, &self.buffered, self.bit_width);
            self.packed_groups += 1;
        }
        self.flush_packed_run();
        self.out
    }
}

/// Append one group's values to a run's body, `bit_width` bits each, LSB-first —
/// exactly `bit_width` bytes, which is what lets groups be appended back to back
/// under a single header.
fn pack_group(out: &mut Vec<u8>, group: &[u32; GROUP], bit_width: u8) {
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

    /// Adjacent groups share one header, which is the difference between a byte
    /// per eight values and a byte per several hundred.
    #[test]
    fn adjacent_groups_share_one_run_header() {
        let indices: Vec<u32> = (0..80).map(|i| i % 16).collect();

        let stream = encode_indices(&indices, 4);

        // Ten groups of eight four-bit values: one header byte, then the values.
        assert_eq!(stream.len(), 1 + 80 * 4 / 8);
        assert_eq!(stream[0], (10 << 1) | 1);
        round_trip(&indices, 4);
    }

    /// A run cannot grow past what a one-byte header describes, so a long
    /// literal stretch becomes several runs rather than one oversized one.
    #[test]
    fn a_long_literal_stretch_splits_at_the_run_limit() {
        let indices: Vec<u32> = (0..MAX_GROUPS_PER_RUN as u32 * GROUP as u32 + 16)
            .map(|i| i % 16)
            .collect();

        let stream = encode_indices(&indices, 4);

        assert_eq!(stream[0], ((MAX_GROUPS_PER_RUN as u8) << 1) | 1);
        round_trip(&indices, 4);
    }

    /// A stream whose last value completes a group leaves nothing buffered, so
    /// the packed groups are all there is left to write. Dropping them at finish
    /// costs the whole page rather than a value or two.
    #[test]
    fn a_stream_ending_on_a_group_boundary_still_emits_its_groups() {
        let indices: Vec<u32> = (0..16).map(|i| i % 4).collect();

        let stream = encode_indices(&indices, 2);

        assert_eq!(decode(&stream, 2, indices.len()), indices);
    }
}
