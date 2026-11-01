use arrow_buffer::Buffer;

use crate::memory::{SlabAllocator, SlabBuffer};

/// Something bits are appended to, least significant bit first. The parquet
/// decoders produce bits through this: a boolean column's values, a column's
/// validity, or a page's presence held outside slab memory.
pub trait BitSink {
    /// Appends `count` bits of `src`, starting at bit `src_bit` of it.
    fn append_bits(&mut self, src: &[u8], src_bit: usize, count: usize);

    /// Appends `count` bits of `value`.
    fn append_run(&mut self, value: bool, count: usize);
}

/// A bitmap built on zeroed slab memory (`1` = present, or `true`), the
/// slab-array analog of arrow's `BooleanBufferBuilder`: a boolean column's
/// values, or the null bitmap that goes with any column, on the same
/// pre-faulted, accounted memory as the values rather than on the global
/// allocator. Append the rows in order, then hand the bitmap to Arrow with
/// [`into_buffer`](Self::into_buffer).
pub struct BitmapBuilder {
    bits: SlabBuffer<u8>,
    len: usize,
}

impl BitmapBuilder {
    /// A bitmap sized for `capacity` rows, all initially zero.
    pub fn with_capacity(allocator: &mut SlabAllocator, capacity: usize) -> Self {
        // Zeroed slab: zero runs are never written, and appends only set bits.
        Self {
            bits: allocator.create_slab_buffer(capacity.div_ceil(8), true),
            len: 0,
        }
    }

    /// Rows appended so far.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Appends `count` rows, all present (valid) or all null.
    pub fn append_n(&mut self, count: usize, present: bool) {
        self.append_run(present, count);
    }

    /// Appends `count` bits of `src`, starting at bit `src_bit` of it.
    #[inline]
    pub fn append_bits(&mut self, src: &[u8], src_bit: usize, count: usize) {
        let len = self.len;
        append_bits(self.bytes_mut(), len, src, src_bit, count);
        self.len += count;
    }

    /// Appends `count` bits of `value`.
    #[inline]
    pub fn append_run(&mut self, value: bool, count: usize) {
        if value {
            let len = self.len;
            set_run(self.bytes_mut(), len, count);
        }
        self.len += count;
    }

    /// The last `count_set(presence, first_bit, rows)` bits were appended
    /// densely for `rows` rows, of which `presence` marks the present ones
    /// (row 0 at bit `first_bit`): moves each to its row and clears the null
    /// rows, growing the bitmap to `rows` past where the dense bits began.
    /// Rows are walked from the last, 64 at a time: a word of output bits is
    /// gathered from the dense bits (a row's bit is the word's first dense
    /// bit plus the count of present rows before it) and written whole, so
    /// no source bit is overwritten before it is read.
    pub fn expand_present(&mut self, rows: usize, presence: &[u8], first_bit: usize) {
        let present = count_set(presence, first_bit, rows);
        let start = self.len - present;
        let bytes = self.bytes_mut();
        let mut src = present;
        let mut row = rows;
        while row > 0 {
            let take = row.min(64);
            let out = row - take;
            let word = bits_word(presence, first_bit + out, take);
            let in_word = word.count_ones() as usize;
            let base = src - in_word;
            let dense = if in_word == 0 {
                0
            } else {
                bits_word(bytes, start + base, in_word)
            };
            let mut gathered = 0u64;
            for i in 0..take {
                let before = (word & ((1u64 << i) - 1)).count_ones();
                let bit = (dense >> before) & ((word >> i) & 1);
                gathered |= bit << i;
            }
            write_bits_word(bytes, start + out, take, gathered);
            src = base;
            row = out;
        }
        self.len = start + rows;
    }

    /// Hand the bitmap to Arrow as a zero-copy slab-backed [`Buffer`].
    pub fn into_buffer(self) -> Buffer {
        self.bits.into_buffer(self.len.div_ceil(8))
    }

    /// The slab's bytes, capacity and all.
    #[inline(always)]
    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: the slab is owned by this builder and holds
        // `elements_left(0)` bytes from its start.
        unsafe {
            std::slice::from_raw_parts_mut(self.bits.ptr_at_index(0), self.bits.elements_left(0))
        }
    }
}

impl BitSink for BitmapBuilder {
    #[inline]
    fn append_bits(&mut self, src: &[u8], src_bit: usize, count: usize) {
        BitmapBuilder::append_bits(self, src, src_bit, count);
    }

    #[inline]
    fn append_run(&mut self, value: bool, count: usize) {
        BitmapBuilder::append_run(self, value, count);
    }
}

/// Writes `count` bits of `src`, from bit `src_bit`, into `dst` from bit
/// `dst_bit`, where `dst`'s bits are zero. Both sides are least significant
/// bit first. A copy of whole bytes when both are byte-aligned, a shifted
/// byte copy otherwise, with the partial bytes at either end set through
/// masks.
#[inline]
pub fn append_bits(dst: &mut [u8], dst_bit: usize, src: &[u8], src_bit: usize, count: usize) {
    let mut dst_bit = dst_bit;
    let mut src_bit = src_bit;
    let mut left = count;
    // The bits up to the destination's next byte boundary go in with one
    // mask, so the bulk below writes whole destination bytes.
    let head = (8 - dst_bit % 8) % 8;
    if head > 0 && left > 0 {
        let take = head.min(left);
        dst[dst_bit / 8] |= bits_from(src, src_bit, take) << (dst_bit % 8);
        dst_bit += take;
        src_bit += take;
        left -= take;
    }
    let whole_bytes = left / 8;
    let shift = src_bit % 8;
    let src_byte = src_bit / 8;
    let dst_byte = dst_bit / 8;
    if shift == 0 {
        dst[dst_byte..dst_byte + whole_bytes]
            .copy_from_slice(&src[src_byte..src_byte + whole_bytes]);
    } else {
        // Each destination byte is the tail of one source byte and the head
        // of the next.
        for i in 0..whole_bytes {
            dst[dst_byte + i] =
                (src[src_byte + i] >> shift) | (src[src_byte + i + 1] << (8 - shift));
        }
    }
    dst_bit += whole_bytes * 8;
    src_bit += whole_bytes * 8;
    let tail = left % 8;
    if tail > 0 {
        dst[dst_bit / 8] |= bits_from(src, src_bit, tail);
    }
}

/// Sets `count` bits of `dst` from `start` on, which are zero: a mask on the
/// first and last bytes touched, whole bytes between.
#[inline]
pub fn set_run(dst: &mut [u8], start: usize, count: usize) {
    if count == 0 {
        return;
    }
    let end = start + count;
    let first = start / 8;
    let last = (end - 1) / 8;
    let head = 0xffu8 << (start % 8);
    let tail = low_mask(end - last * 8);
    if first == last {
        dst[first] |= head & tail;
        return;
    }
    dst[first] |= head;
    dst[first + 1..last].fill(0xff);
    dst[last] |= tail;
}

/// Sets bit `bit` of `dst` to `value`, where it is zero.
#[inline(always)]
pub fn set_bit(dst: &mut [u8], bit: usize, value: bool) {
    dst[bit / 8] |= (value as u8) << (bit % 8);
}

/// Sets bit `bit` of `dst` to `value`, whatever it was.
#[inline(always)]
pub fn write_bit(dst: &mut [u8], bit: usize, value: bool) {
    let mask = 1u8 << (bit % 8);
    let byte = &mut dst[bit / 8];
    *byte = (*byte & !mask) | ((value as u8) << (bit % 8));
}

/// Bit `bit` of `bytes`.
#[inline(always)]
pub fn bit_at(bytes: &[u8], bit: usize) -> bool {
    (bytes[bit / 8] >> (bit % 8)) & 1 == 1
}

/// How many of the `count` bits of `bytes` from bit `first` on are set.
pub fn count_set(bytes: &[u8], first: usize, count: usize) -> usize {
    if count == 0 {
        return 0;
    }
    let end = first + count;
    let first_byte = first / 8;
    let last_byte = (end - 1) / 8;
    let head = 0xffu8 << (first % 8);
    let tail = low_mask(end - last_byte * 8);
    if first_byte == last_byte {
        return (bytes[first_byte] & head & tail).count_ones() as usize;
    }
    (bytes[first_byte] & head).count_ones() as usize
        + bytes[first_byte + 1..last_byte]
            .iter()
            .map(|byte| byte.count_ones() as usize)
            .sum::<usize>()
        + (bytes[last_byte] & tail).count_ones() as usize
}

/// The `count_set(presence, first_bit, slots.len())` values at the front of
/// `slots` were written densely for `slots.len()` rows, of which `presence`
/// marks the present ones (row 0 at bit `first_bit`): moves each to its row
/// and writes `default` under the null rows. Rows are walked from the last,
/// so a value always moves to a slot at or past its own and is never
/// overwritten before it is read.
///
/// The walk takes the presence 64 rows at a time and chooses the value or
/// the default without a branch, since which rows are null is not
/// predictable. The source index is carried from row to row: computing it
/// per row from a popcount measured slower than the one-cycle dependency.
pub fn expand_present<T: Copy>(slots: &mut [T], presence: &[u8], first_bit: usize, default: T) {
    let rows = slots.len();
    let present = count_set(presence, first_bit, rows);
    if present == 0 {
        slots.fill(default);
        return;
    }
    let mut src = present;
    let mut row = rows;
    while row > 0 {
        let take = row.min(64);
        let out = row - take;
        let word = bits_word(presence, first_bit + out, take);
        for i in (0..take).rev() {
            let is_present = (word >> i) & 1 == 1;
            src -= is_present as usize;
            // A null row reads slot 0, which holds a value since `present`
            // is at least one, and discards it.
            let read_at = if is_present { src } else { 0 };
            // SAFETY: `read_at < present <= rows` and `out + i < rows`.
            unsafe {
                let value = *slots.get_unchecked(read_at);
                *slots.get_unchecked_mut(out + i) = if is_present { value } else { default };
            }
        }
        row = out;
    }
}

/// Writes the low `count` bits of `word`, `count` at most 64, over the bits
/// of `bytes` from bit `first` on.
#[inline(always)]
fn write_bits_word(bytes: &mut [u8], first: usize, count: usize, word: u64) {
    let first_byte = first / 8;
    let last_byte = (first + count - 1) / 8;
    let touched = last_byte + 1 - first_byte;
    let mut buffer = [0u8; 16];
    buffer[..touched].copy_from_slice(&bytes[first_byte..=last_byte]);
    let shift = first % 8;
    let mask = if count == 64 {
        u128::MAX >> 64
    } else {
        (1u128 << count) - 1
    } << shift;
    let current = u128::from_le_bytes(buffer);
    let updated = (current & !mask) | (((word as u128) << shift) & mask);
    bytes[first_byte..=last_byte].copy_from_slice(&updated.to_le_bytes()[..touched]);
}

/// The `count` bits of `bytes` from bit `first` on, `count` at most 64, in
/// the low bits of a word.
#[inline(always)]
fn bits_word(bytes: &[u8], first: usize, count: usize) -> u64 {
    let first_byte = first / 8;
    let last_byte = (first + count - 1) / 8;
    let mut buffer = [0u8; 16];
    buffer[..last_byte + 1 - first_byte].copy_from_slice(&bytes[first_byte..=last_byte]);
    let word = (u128::from_le_bytes(buffer) >> (first % 8)) as u64;
    if count == 64 {
        word
    } else {
        word & ((1u64 << count) - 1)
    }
}

/// A byte whose low `n` bits are set, `n` at most 8.
#[inline(always)]
fn low_mask(n: usize) -> u8 {
    ((1u16 << n) - 1) as u8
}

/// The `n` bits of `bytes` from bit `bit` on, `n` at most 8, in the low bits
/// of a byte.
#[inline(always)]
fn bits_from(bytes: &[u8], bit: usize, n: usize) -> u8 {
    let shift = bit % 8;
    let mut word = bytes[bit / 8] as u16;
    if shift + n > 8 {
        word |= (bytes[bit / 8 + 1] as u16) << 8;
    }
    (word >> shift) as u8 & low_mask(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use arrow_buffer::BooleanBuffer;

    fn bits_of(builder: BitmapBuilder) -> Vec<bool> {
        let len = builder.len();
        let bits = BooleanBuffer::new(builder.into_buffer(), 0, len);
        (0..len).map(|i| bits.value(i)).collect()
    }

    /// Append `runs` of (count, present) and read the resulting bitmap back.
    fn validity(runs: &[(usize, bool)]) -> Vec<bool> {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let len: usize = runs.iter().map(|&(n, _)| n).sum();
        let mut builder = BitmapBuilder::with_capacity(&mut allocator, len);
        for &(n, present) in runs {
            builder.append_n(n, present);
        }
        bits_of(builder)
    }

    fn pack(values: &[bool]) -> Vec<u8> {
        let mut bytes = vec![0u8; values.len().div_ceil(8)];
        for (i, &value) in values.iter().enumerate() {
            bytes[i / 8] |= (value as u8) << (i % 8);
        }
        bytes
    }

    #[test]
    fn present_runs_cross_byte_boundaries() {
        let got = validity(&[(3, true), (8, false), (5, true)]);

        let mut want = vec![false; 16];
        for i in [0, 1, 2, 11, 12, 13, 14, 15] {
            want[i] = true;
        }
        assert_eq!(got, want);
    }

    #[test]
    fn whole_byte_present_runs() {
        let got = validity(&[(16, true), (8, false)]);

        assert_eq!(got, [vec![true; 16], vec![false; 8]].concat());
    }

    #[test]
    fn single_bit_runs_alternate() {
        let runs: Vec<(usize, bool)> = (0..11).map(|i| (1, i % 2 == 0)).collect();

        let got = validity(&runs);

        assert_eq!(got, (0..11).map(|i| i % 2 == 0).collect::<Vec<_>>());
    }

    #[test]
    fn counts_set_bits_over_any_range() {
        let bytes = pack(&(0..40).map(|i| i % 3 == 0).collect::<Vec<_>>());

        assert_eq!(count_set(&bytes, 0, 40), 14);
        assert_eq!(count_set(&bytes, 3, 4), 2);
        assert_eq!(count_set(&bytes, 5, 20), 7);
        assert_eq!(count_set(&bytes, 7, 0), 0);
        assert_eq!(count_set(&bytes, 39, 1), 1);
    }

    #[test]
    fn expands_dense_values_over_their_rows() {
        let presence = pack(&[
            true, true, false, true, false, false, true, true, true, false,
        ]);
        let mut slots = [10, 20, 30, 40, 50, 60, 0, 0, 0, 0];

        expand_present(&mut slots, &presence, 0, -1);

        assert_eq!(slots, [10, 20, -1, 30, -1, -1, 40, 50, 60, -1]);
    }

    #[test]
    fn expands_dense_bits_over_their_rows() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let presence = pack(&[
            false, true, true, false, true, true, true, false, false, true,
        ]);
        let mut builder = BitmapBuilder::with_capacity(&mut allocator, 13);
        builder.append_run(true, 3);
        // The six present rows' values, dense: 1 0 1 1 0 1.
        builder.append_bits(&[0b101101], 0, 6);

        builder.expand_present(10, &presence, 0);

        let mut want = vec![true, true, true];
        want.extend([
            false, true, false, false, true, true, false, false, false, true,
        ]);
        assert_eq!(bits_of(builder), want);
    }

    #[test]
    fn appends_bits_at_every_alignment() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let values: Vec<bool> = (0..100).map(|i| i % 3 == 0 || i % 7 == 0).collect();
        let packed = pack(&values);
        let mut builder = BitmapBuilder::with_capacity(&mut allocator, 100);

        // Pieces whose lengths leave the destination and the source at
        // different offsets within a byte.
        let mut at = 0;
        for piece in [3, 8, 13, 1, 24, 5, 46] {
            builder.append_bits(&packed[at / 8..], at % 8, piece);
            at += piece;
        }

        assert_eq!(bits_of(builder), values);
    }
}
