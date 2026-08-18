//! [`SlabVec`] - a growable, slab-backed, append-only buffer.

use super::{SlabAllocator, SlabBuffer};

/// A growable, append-only buffer backed by the slab pool: a vec of fixed-size
/// chunks rather than one contiguous allocation, so appends never reallocate. A
/// cached base pointer into the current chunk keeps [`push`](Self::push) and
/// [`for_each`](Self::for_each) sequential (no per-element index math).
///
/// A `SlabVec` follows its element's thread-safety: it is `Send` only for
/// `T: Send`. A `Copy` type that must not cross threads (a raw pointer, say)
/// keeps blocking the impl:
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<dispatch::memory::SlabVec<*const u8>>();
/// ```
pub struct SlabVec<T: Copy> {
    chunks: Vec<SlabBuffer<T>>,
    cur_base: *mut T,
    last_len: usize,
}

// SAFETY: the cached `cur_base` raw pointer is the only thing blocking the auto
// `Send`. It points into a slab owned by `chunks` - address-stable pool memory the
// `SlabBuffer`s keep alive - so it stays valid when the `SlabVec` moves to another
// thread. `T: Send` is still required: the vec hands out its elements on the
// receiving thread, so a `T` that is unsound to move across threads (e.g. one
// borrowing a `Cell`) must keep blocking the impl.
unsafe impl<T: Copy + Send> Send for SlabVec<T> {}

impl<T: Copy> Default for SlabVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy> SlabVec<T> {
    /// Elements per chunk: a ~64 KB target of `T`, derived from `size_of::<T>()`
    /// rather than a fixed element count so a wide `T` can't overflow a single slab
    /// (which `create_slab_buffer` asserts against). Kept well below a full 2 MB
    /// slab so chunks bump-pack into the shared ring buffers - when many small
    /// `SlabVec`s share the pool, a full-slab chunk apiece would pin a whole 2 MB
    /// buffer each, whereas small chunks let memory track total data rather than
    /// the number of vecs.
    const CHUNK_CAP: usize = {
        let cap = (64 * 1024) / size_of::<T>();
        if cap == 0 { 1 } else { cap }
    };

    /// Capacity of the *first* chunk, a fraction of [`CHUNK_CAP`](Self::CHUNK_CAP).
    /// A `SlabVec` that only ever holds a handful of elements takes one small chunk
    /// rather than a full `CHUNK_CAP` one. This matters for the radix scatter: a
    /// switch allocates `RADIX_PARTITIONS` (thousands of) per-partition buffers, and
    /// at moderate cardinality most hold only a few rows. Chunks bump-pack into the
    /// shared 2MB pool buffers, so a full first chunk apiece would tie up ~8x more of
    /// the pool - buffers the compressed cache could otherwise use - for partitions that
    /// barely fill them. Buffers that do grow large take full `CHUNK_CAP` chunks from
    /// the second one on, so the per-chunk overhead for big partitions stays
    /// negligible.
    const INITIAL_CAP: usize = {
        let cap = Self::CHUNK_CAP / 8;
        if cap == 0 { 1 } else { cap }
    };

    /// Capacity of chunk `ci`: the first is [`INITIAL_CAP`](Self::INITIAL_CAP),
    /// the rest are [`CHUNK_CAP`](Self::CHUNK_CAP). The push overflow check, the
    /// new-chunk allocation, and the iteration length all derive their per-chunk
    /// size from this one function, so the small-first-chunk rule lives in one place.
    #[inline(always)]
    fn chunk_cap(ci: usize) -> usize {
        if ci == 0 {
            Self::INITIAL_CAP
        } else {
            Self::CHUNK_CAP
        }
    }

    /// Number of elements pushed. Every chunk before the last is full, so the
    /// count is the full chunks' capacities plus the last chunk's fill.
    pub fn len(&self) -> usize {
        if self.chunks.is_empty() {
            return 0;
        }
        (0..self.chunks.len() - 1)
            .map(Self::chunk_cap)
            .sum::<usize>()
            + self.last_len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn new() -> Self {
        Self {
            chunks: Vec::new(),
            cur_base: std::ptr::null_mut(),
            last_len: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, allocator: &mut SlabAllocator, val: T) {
        // Full when the chunk being filled (index `chunks.len() - 1`) is at its
        // capacity; an empty `SlabVec` allocates its first chunk straight away
        // (the `is_empty` short-circuit also guards the `len() - 1` below).
        if self.chunks.is_empty() || self.last_len == Self::chunk_cap(self.chunks.len() - 1) {
            let cap = Self::chunk_cap(self.chunks.len());
            let chunk = allocator.create_slab_buffer(cap, false);
            self.cur_base = chunk.ptr_at_index(0);
            self.chunks.push(chunk);
            self.last_len = 0;
        }
        unsafe { *self.cur_base.add(self.last_len) = val };
        self.last_len += 1;
    }

    /// Visit each chunk's elements as one contiguous slice, in insertion order.
    /// The last chunk holds `last_len` elements; every earlier one is full at its
    /// [`chunk_cap`](Self::chunk_cap). Centralises the unsafe base/length logic so
    /// the two public iterators below can't drift apart.
    #[inline(always)]
    fn for_each_chunk(&self, mut f: impl FnMut(&[T])) {
        let n = self.chunks.len();
        for (ci, chunk) in self.chunks.iter().enumerate() {
            let len = if ci + 1 == n {
                self.last_len
            } else {
                Self::chunk_cap(ci)
            };
            let base = chunk.ptr_at_index(0) as *const T;
            let slice = unsafe { std::slice::from_raw_parts(base, len) };
            f(slice);
        }
    }

    /// Visit every element in insertion order - sequentially off each chunk's base.
    #[inline(always)]
    pub fn for_each(&self, mut f: impl FnMut(T)) {
        self.for_each_chunk(|slice| {
            for &val in slice {
                f(val);
            }
        });
    }

    /// Like [`for_each`](Self::for_each), but also hands `f` a reference to the
    /// element `AHEAD` slots ahead (or `None` within `AHEAD` of the end of a
    /// chunk). This gives the caller a window to warm a cache line the upcoming
    /// element will chase (e.g. an out-of-line string blob an element only holds
    /// a handle to) before reaching it. A single closure (rather than separate
    /// prefetch and visit callbacks) lets the caller prefetch through a resource
    /// it also mutates while visiting (e.g. an aggregation target). Lookahead is
    /// per-chunk: the last `AHEAD` elements of each chunk get no peek.
    #[inline(always)]
    pub fn for_each_prefetched<const AHEAD: usize>(&self, mut f: impl FnMut(T, Option<&T>)) {
        self.for_each_chunk(|slice| {
            let len = slice.len();
            for i in 0..len {
                let ahead = (i + AHEAD < len).then(|| &slice[i + AHEAD]);
                f(slice[i], ahead);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;

    /// The `T: Send` side of the bound the struct doc's `compile_fail` example
    /// guards: an ordinary element type must keep the vec sendable.
    #[test]
    fn a_slab_vec_of_send_elements_is_send() {
        fn assert_send<T: Send>() {}

        assert_send::<SlabVec<u64>>();
    }

    /// A `SlabVec<u64>` holding `0..n`, plus the allocator that owns the slab
    /// memory its chunks point into (kept alive for the vec's lifetime).
    fn push_range(n: u64) -> (SlabVec<u64>, SlabAllocator) {
        init_test_free_pool(8);
        let mut alloc = SlabAllocator::new(false);
        let mut v = SlabVec::new();
        for i in 0..n {
            v.push(&mut alloc, i);
        }
        (v, alloc)
    }

    #[test]
    fn for_each_visits_every_element_in_insertion_order() {
        let (v, _alloc) = push_range(10_000);

        let mut seen = Vec::new();
        v.for_each(|x| seen.push(x));

        assert_eq!(seen, (0..10_000).collect::<Vec<_>>());
    }

    #[test]
    fn empty_vec_visits_nothing() {
        let (v, _alloc) = push_range(0);

        let mut count = 0;
        v.for_each(|_| count += 1);

        assert_eq!(count, 0);
    }

    #[test]
    fn single_element_lives_in_the_small_first_chunk() {
        let (v, _alloc) = push_range(1);

        let mut seen = Vec::new();
        v.for_each(|x| seen.push(x));

        assert_eq!(seen, vec![0]);
    }

    #[test]
    fn growth_past_the_first_chunk_keeps_every_element() {
        // 10_000 > INITIAL_CAP + CHUNK_CAP for u64, so this spans several chunks; a
        // wrong per-chunk length would drop or repeat elements at a boundary.
        let (v, _alloc) = push_range(10_000);

        let mut seen = Vec::new();
        v.for_each(|x| seen.push(x));

        assert_eq!(seen.len(), 10_000);
        assert_eq!(seen.first(), Some(&0));
        assert_eq!(seen.last(), Some(&9_999));
    }

    #[test]
    fn prefetched_visits_the_same_elements_as_for_each() {
        let (v, _alloc) = push_range(10_000);

        let mut seen = Vec::new();
        v.for_each_prefetched::<8>(|val, _| seen.push(val));

        assert_eq!(seen, (0..10_000).collect::<Vec<_>>());
    }

    #[test]
    fn prefetched_lookahead_points_at_the_element_ahead() {
        const AHEAD: usize = 4;
        let (v, _alloc) = push_range(10_000);

        // Values are contiguous within a chunk, so a present lookahead is exactly
        // `val + AHEAD`; it is `None` only within AHEAD of each chunk's end.
        v.for_each_prefetched::<AHEAD>(|val, ahead| {
            if let Some(&a) = ahead {
                assert_eq!(a, val + AHEAD as u64);
            }
        });
    }
}
