//! [`SlabVec`] — a growable, slab-backed, append-only buffer.

use super::{SlabAllocator, SlabBuffer};

/// A growable, append-only buffer backed by the slab pool: a vec of fixed-size
/// chunks rather than one contiguous allocation, so appends never reallocate. A
/// cached base pointer into the current chunk keeps [`push`](Self::push) and
/// [`for_each`](Self::for_each) sequential (no per-element index math).
pub struct SlabVec<T: Copy> {
    chunks: Vec<SlabBuffer<T>>,
    cur_base: *mut T,
    last_len: usize,
}

// SAFETY: the cached `cur_base` raw pointer is the only thing blocking the auto
// `Send`. It points into a slab owned by `chunks` — address-stable pool memory the
// `SlabBuffer`s keep alive — so it stays valid when the `SlabVec` moves to another
// thread. `Send` is asserted for every `Copy` `T`, including `T`s that embed raw
// pointers; the caller is then responsible for those pointers being valid to use
// from the receiving thread.
unsafe impl<T: Copy> Send for SlabVec<T> {}

impl<T: Copy> Default for SlabVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy> SlabVec<T> {
    /// Elements per chunk: a ~64 KB target of `T`, derived from `size_of::<T>()`
    /// rather than a fixed element count so a wide `T` can't overflow a single slab
    /// (which `create_slab_buffer` asserts against). Kept well below a full 2 MB
    /// slab so chunks bump-pack into the shared ring buffers — when many small
    /// `SlabVec`s share the pool, a full-slab chunk apiece would pin a whole 2 MB
    /// buffer each, whereas small chunks let memory track total data rather than
    /// the number of vecs.
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
