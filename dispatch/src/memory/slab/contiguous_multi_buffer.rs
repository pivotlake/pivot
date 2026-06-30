use crate::memory::BUFFER_SIZE;
use crate::memory::WriteBuffer;
use crate::memory::memory_ctx;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/// Typed buffer backed by consecutive ring slots, providing truly contiguous memory
/// that spans multiple 2 MB buffers.
///
/// Unlike [`MultiSlabBuffer`](super::MultiSlabBuffer) which may use non-adjacent ring slots,
/// this buffer searches the ring for consecutive AVAILABLE slots (`used == 0`), yielding a
/// single contiguous memory region. Because the memory is contiguous, indexing is simple
/// pointer arithmetic — no slab lookup or bit-shift needed.
///
/// On drop, the backing `WriteBuffer`s are released back to the free pool.
pub struct ContiguousMultiBuffer<T> {
    /// Pointer to the start of the contiguous region (first slot's buffer pointer).
    ptr: *mut u8,
    /// The consecutive write buffers held for the lifetime of this buffer.
    _buffers: Vec<WriteBuffer>,
    _phantom: PhantomData<T>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContiguousAllocationError;

unsafe impl<T> Send for ContiguousMultiBuffer<T> {}
unsafe impl<T> Sync for ContiguousMultiBuffer<T> {}

impl<T> ContiguousMultiBuffer<T> {
    /// Searches the ring for enough consecutive available slots to hold `num_elements`
    /// elements of type `T`, acquires them, and zeroes the memory.
    ///
    /// For each candidate run, attempts a CAS on each slot from `used == 0` to WRITING.
    /// If any slot is contended, the already-acquired slots are released and the search
    /// continues past the failed slot.
    ///
    /// Returns `Err(())` if no contiguous run of the required size exists in the ring.
    pub fn new(num_elements: usize) -> Result<Self, ContiguousAllocationError> {
        let total_bytes = num_elements * size_of::<T>();
        let slot_count = total_bytes.div_ceil(BUFFER_SIZE);

        let ring = memory_ctx().ring();
        let ring_len = ring.len();

        let mut start = 0;
        while start + slot_count <= ring_len {
            let mut buffers: Vec<WriteBuffer> = Vec::with_capacity(slot_count);

            for i in 0..slot_count {
                match ring.try_write(start + i) {
                    Some(wb) => buffers.push(wb),
                    None => {
                        // Drop releases the already-acquired WriteBuffers back to
                        // the free pool. Skip past the failed slot.
                        start = start + i + 1;
                        break;
                    }
                }
            }

            if buffers.len() == slot_count {
                let ptr = buffers[0].ptr;
                for buf in &mut buffers {
                    if !buf.zeroed {
                        unsafe {
                            std::ptr::write_bytes(buf.ptr, 0, BUFFER_SIZE);
                        }
                    }
                }
                return Ok(Self {
                    ptr,
                    _buffers: buffers,
                    _phantom: PhantomData,
                });
            }
        }

        Err(ContiguousAllocationError)
    }

    /// Returns the raw pointer to the start of the contiguous region.
    #[inline(always)]
    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Returns a raw pointer to the element at `index`.
    ///
    /// Since the memory is truly contiguous, this is simple pointer arithmetic —
    /// no slab lookup needed.
    #[inline(always)]
    pub(crate) fn ptr_at_index(&self, index: usize) -> *mut T {
        let byte_offset = index * size_of::<T>();
        unsafe { self.ptr.add(byte_offset) as *mut T }
    }
}

impl<T> Index<usize> for ContiguousMultiBuffer<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &*self.ptr_at_index(index) }
    }
}

impl<T> IndexMut<usize> for ContiguousMultiBuffer<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { &mut *self.ptr_at_index(index) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;

    #[test]
    fn write_and_read() {
        init_test_free_pool(0);
        let mut buf: ContiguousMultiBuffer<u32> = ContiguousMultiBuffer::new(64).unwrap();

        buf[0] = 10;
        buf[63] = 20;

        assert_eq!(buf[0], 10);
        assert_eq!(buf[63], 20);
    }

    #[test]
    fn zeroed_on_creation() {
        init_test_free_pool(0);
        let buf: ContiguousMultiBuffer<u64> = ContiguousMultiBuffer::new(256).unwrap();

        for i in 0..256 {
            assert_eq!(buf[i], 0);
        }
    }

    #[test]
    fn spans_multiple_buffers_contiguously() {
        init_test_free_pool(0);
        let elements = BUFFER_SIZE / size_of::<u64>() + 100;
        let mut buf: ContiguousMultiBuffer<u64> = ContiguousMultiBuffer::new(elements).unwrap();

        buf[0] = 1;
        buf[elements - 1] = 2;

        assert_eq!(buf[0], 1);
        assert_eq!(buf[elements - 1], 2);

        // Verify the memory is truly contiguous — pointer arithmetic across the
        // buffer boundary works without any slab lookup.
        let first_ptr = buf.ptr_at_index(0);
        let last_ptr = buf.ptr_at_index(elements - 1);
        let expected_distance = (elements - 1) * size_of::<u64>();
        assert_eq!(
            unsafe { last_ptr.byte_offset_from(first_ptr) } as usize,
            expected_distance,
        );
    }
}
