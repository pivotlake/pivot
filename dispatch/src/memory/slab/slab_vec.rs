//! [`SlabVec`] — a growable, slab-backed, append-only buffer.

use super::{SlabAllocator, SlabBuffer};

/// A growable, engine-backed (slab-pool) append-only buffer, as a list of
/// fixed-size chunks. Appends never reallocate; a cached base pointer makes the
/// scatter writes and aggregate reads sequential (no per-element index math).
pub struct SlabVec<T: Copy> {
    chunks: Vec<SlabBuffer<T>>,
    cur_base: *mut T,
    last_len: usize,
}

// SAFETY: for string keys the stored value embeds an `ArenaKey` (raw pointer into
// the `Arc`'d arena, which outlives the merge); `cur_base` points into a slab the
// `chunks` keep alive. (Strings never scatter today, but the bound is generic.)
unsafe impl<T: Copy> Send for SlabVec<T> {}

impl<T: Copy> Default for SlabVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy> SlabVec<T> {
    /// Elements per chunk: a ~64 KB target of `T`. Derived from `size_of::<T>()`
    /// rather than a fixed element count so a wide row (e.g. a group-by with many
    /// aggregates) can't overflow a single slab — which `create_slab_buffer`
    /// asserts against — and kept far below a full 2 MB slab so chunks bump-pack
    /// into the shared ring buffers: with thousands of radix partitions a
    /// full-slab chunk apiece would pin gigabytes for near-empty partitions, while
    /// small chunks make memory track the data scattered, not the partition count.
    const CHUNK_CAP: usize = {
        let cap = (64 * 1024) / size_of::<T>();
        if cap == 0 { 1 } else { cap }
    };

    pub fn new() -> Self {
        Self {
            chunks: Vec::new(),
            cur_base: std::ptr::null_mut(),
            last_len: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, allocator: &mut SlabAllocator, val: T) {
        if self.chunks.is_empty() || self.last_len == Self::CHUNK_CAP {
            let chunk = allocator.create_slab_buffer(Self::CHUNK_CAP, false);
            self.cur_base = chunk.ptr_at_index(0);
            self.chunks.push(chunk);
            self.last_len = 0;
        }
        unsafe { *self.cur_base.add(self.last_len) = val };
        self.last_len += 1;
    }

    /// Visit every element in insertion order — sequentially off each chunk's base.
    #[inline(always)]
    pub fn for_each(&self, mut f: impl FnMut(T)) {
        let n = self.chunks.len();
        for (ci, chunk) in self.chunks.iter().enumerate() {
            let len = if ci + 1 == n {
                self.last_len
            } else {
                Self::CHUNK_CAP
            };
            let base = chunk.ptr_at_index(0) as *const T;
            let slice = unsafe { std::slice::from_raw_parts(base, len) };
            for &val in slice {
                f(val);
            }
        }
    }
}
