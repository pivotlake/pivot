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
