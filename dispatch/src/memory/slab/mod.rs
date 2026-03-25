//! Slab-based sub-allocation on top of 2MB [`WriteBuffer`]s.
//!
//! A [`Slab`] is a contiguous byte region within a `WriteBuffer`. It holds an `Arc<WriteBuffer>`,
//! so the underlying 2MB buffer is only returned to the free pool once *every* slab carved from
//! it has been dropped. This means a single long-lived slab will pin the entire `WriteBuffer` —
//! avoid mixing short-lived and long-lived slabs on the same buffer.
//!
//! Two typed wrappers provide indexable access over slabs:
//!
//! - [`SlabBuffer<T>`] — backed by a single slab (must fit within one 2MB buffer). Cheaper to
//!   index since there is no buffer lookup. Use this when the data is guaranteed to be < 2MB.
//!
//! - [`MultiSlabBuffer<T>`] — backed by one or more slabs, supporting allocations that span
//!   multiple `WriteBuffer`s. Indexing requires a buffer lookup via bit-shift and mask. Use this
//!   for allocations whose size may exceed 2MB (e.g., hash tables).
//!
//! Both support `Index<usize>` / `IndexMut<usize>` for typed element access.
//!
//! [`SlabAllocator`] manages the allocation by bumping through the current `WriteBuffer` and
//! requesting new ones from the free pool as needed. It provides [`SlabAllocator::create_slab_buffer`]
//! and [`SlabAllocator::create_multi_slab_buffer`] as the primary entry points.

use crate::memory::WriteBuffer;
use std::sync::Arc;

mod allocator;
pub use allocator::SlabAllocator;

mod slab_buffer;

pub use slab_buffer::SlabBuffer;

mod multi_slab_buffer;
pub use multi_slab_buffer::MultiSlabBuffer;

mod slab_vec;
pub use slab_vec::{SlabVec, SlabVecIterator};


/// A contiguous byte region within a `WriteBuffer`.
///
/// Keeps the parent buffer alive via `Arc<WriteBuffer>`. Multiple slabs can share the same
/// buffer — the buffer is only returned to the free pool when the last slab is dropped.
pub struct Slab {
    /// Raw pointer to the start of this slab's region within the parent buffer.
    pub(crate) ptr: *mut u8,
    /// Size of this slab in bytes.
    pub(crate) size: usize,
    /// Shared ownership of the parent buffer. Dropping the last `Arc` returns the buffer
    /// to the free pool.
    _buffer: Arc<WriteBuffer>,
}

unsafe impl Send for Slab {}

unsafe impl Sync for Slab {}

impl Slab {
    /// Zeroes the slab's entire region.
    pub fn zero_out(&mut self) {
        unsafe {
            std::ptr::write_bytes(self.ptr, 0, self.size);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::memory::{SlabAllocator, init_test_free_pool};

    #[test]
    fn slab_zero_out() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(false);
        let mut slab = alloc.get_slab_of_size(256, false);
        unsafe {
            std::ptr::write_bytes(slab.ptr, 0xFF, 256);
        }

        slab.zero_out();

        let slice = unsafe { std::slice::from_raw_parts(slab.ptr, 256) };
        assert!(slice.iter().all(|&b| b == 0));
    }
}
