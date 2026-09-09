use arrow_buffer::Buffer;

use crate::memory::{SlabAllocator, SlabBuffer};

/// A validity bitmap (`1` = present/valid) built run-wise on slab memory, the
/// slab-array analog of arrow's `BooleanBufferBuilder`. A nullable decoded
/// column's null buffer lives on the same pre-faulted, accounted slab memory as
/// its values, rather than on the global allocator. Append the rows in order,
/// then hand the bitmap to Arrow with [`into_buffer`](Self::into_buffer).
pub struct ValidityBuilder {
    bits: SlabBuffer<u8>,
    len: usize,
}

impl ValidityBuilder {
    /// A bitmap sized for `capacity` rows, all initially null.
    pub fn with_capacity(allocator: &mut SlabAllocator, capacity: usize) -> Self {
        // Zeroed slab: absent rows stay null without being written.
        Self {
            bits: allocator.create_slab_buffer(capacity.div_ceil(8), true),
            len: 0,
        }
    }

    /// Append `count` rows, all present (valid) or all null. Null runs are
    /// already zero in the slab, so only present runs are written: whole bytes
    /// where the run spans them, edge bits otherwise.
    pub fn append_n(&mut self, count: usize, present: bool) {
        if present {
            let (mut i, end) = (self.len, self.len + count);
            while i < end && i % 8 != 0 {
                self.set(i);
                i += 1;
            }
            while i + 8 <= end {
                unsafe { *self.bits.ptr_at_index(i / 8) = 0xFF };
                i += 8;
            }
            while i < end {
                self.set(i);
                i += 1;
            }
        }
        self.len += count;
    }

    #[inline]
    fn set(&mut self, i: usize) {
        unsafe { *self.bits.ptr_at_index(i / 8) |= 1 << (i % 8) };
    }

    /// Hand the bitmap to Arrow as a zero-copy slab-backed [`Buffer`].
    pub fn into_buffer(self) -> Buffer {
        self.bits.into_buffer(self.len.div_ceil(8))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use arrow_buffer::BooleanBuffer;

    /// Append `runs` of (count, present) and read the resulting bitmap back.
    fn validity(runs: &[(usize, bool)]) -> Vec<bool> {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let len: usize = runs.iter().map(|&(n, _)| n).sum();
        let mut builder = ValidityBuilder::with_capacity(&mut allocator, len);
        for &(n, present) in runs {
            builder.append_n(n, present);
        }
        let bits = BooleanBuffer::new(builder.into_buffer(), 0, len);
        (0..len).map(|i| bits.value(i)).collect()
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
}
