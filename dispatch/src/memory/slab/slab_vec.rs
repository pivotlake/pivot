//! [`SlabVec`] — a growable, slab-backed, append-only buffer.

use super::{FrozenSlab, Slab, SlabBuffer, SlabSource};
use std::cmp::Ordering;
use std::ops::Range;

/// A growable, append-only buffer backed by the slab pool: a vec of fixed-size
/// chunks rather than one contiguous allocation, so appends never reallocate. A
/// cached base pointer into the current chunk keeps [`push`](Self::push) and
/// [`for_each`](Self::for_each) sequential (no per-element index math).
pub struct SlabVec<T: Copy> {
    chunks: Vec<SlabBuffer<T>>,
    cur_base: *mut T,
    last_len: usize,
    /// Elements the chunk being filled can hold. Zero while there is no chunk.
    last_cap: usize,
    /// For a vec created to hold an exact number of elements (see
    /// [`with_exact_len`](Self::with_exact_len)): how many of them the chunks
    /// not yet carved will hold.
    left_to_carve: Option<usize>,
}

// SAFETY: the cached `cur_base` raw pointer is the only thing blocking the auto
// `Send`. It points into a slab owned by `chunks` — address-stable pool memory the
// `SlabBuffer`s keep alive — so it stays valid when the `SlabVec` moves to another
// thread. `Send` is asserted for every `Copy` `T`, including `T`s that embed raw
// pointers; the caller is then responsible for those pointers being valid to use
// from the receiving thread.
unsafe impl<T: Copy> Send for SlabVec<T> {}

// SAFETY: a shared reference only reads elements already pushed: appending
// takes `&mut self`. So sharing the vec between threads shares `T`s, which is
// sound when `T` itself may be shared.
unsafe impl<T: Copy + Sync> Sync for SlabVec<T> {}

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

    /// Capacity of the *first* chunk, a fraction of [`CHUNK_CAP`](Self::CHUNK_CAP).
    /// A `SlabVec` that only ever holds a handful of elements takes one small chunk
    /// rather than a full `CHUNK_CAP` one. This matters for the radix scatter: a
    /// switch allocates `RADIX_PARTITIONS` (thousands of) per-partition buffers, and
    /// at moderate cardinality most hold only a few rows. Chunks bump-pack into the
    /// shared 2MB pool buffers, so a full first chunk apiece would tie up ~8x more of
    /// the pool — buffers the compressed cache could otherwise use — for partitions that
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
            last_cap: 0,
            left_to_carve: None,
        }
    }

    /// A vec that will hold exactly `len` elements. Its last chunk is carved
    /// only as large as the elements it will hold, so a vec whose length is
    /// known before it is filled takes no more memory than its elements.
    /// Pushing more than `len` elements panics.
    pub fn with_exact_len(len: usize) -> Self {
        Self {
            left_to_carve: Some(len),
            ..Self::new()
        }
    }

    /// The vec whose chunks are `chunks`, in order, holding `len` elements:
    /// how a vec that was built into the slabs of a cache entry is put back
    /// together over them. Panics unless the chunks are those a vec of `len`
    /// elements has.
    ///
    /// The vec is for reading. The slabs are shared with whoever else holds
    /// the entry, so pushing to it panics.
    pub(crate) fn from_chunks(chunks: Vec<FrozenSlab>, len: usize) -> Self {
        let chunk_count = Self::count_chunks_for_len(len);
        assert_eq!(
            chunks.len(),
            chunk_count,
            "a vec of {len} elements has another number of chunks"
        );
        let elements_before_last: usize = (0..chunk_count.saturating_sub(1))
            .map(Self::chunk_cap)
            .sum();
        let last_len = len - elements_before_last;
        let chunks: Vec<SlabBuffer<T>> = chunks
            .into_iter()
            .enumerate()
            .map(|(chunk_index, chunk)| {
                // Every chunk but the last is full. The last one was carved
                // for at least the elements it holds.
                let is_last = chunk_index + 1 == chunk_count;
                let full_size = Self::chunk_cap(chunk_index) * size_of::<T>();
                assert!(
                    chunk.size == full_size || (is_last && chunk.size >= last_len * size_of::<T>()),
                    "chunk {chunk_index} is too small for its elements"
                );
                assert!(
                    (chunk.ptr as usize).is_multiple_of(align_of::<T>()),
                    "chunk {chunk_index} is not aligned for its elements"
                );
                SlabBuffer::new(Slab::new(chunk.ptr as *mut u8, chunk.size, chunk.backing))
            })
            .collect();
        Self {
            cur_base: chunks
                .last()
                .map_or(std::ptr::null_mut(), |chunk| chunk.ptr_at_index(0)),
            last_len,
            // Full, with nothing left to carve: a push finds no room.
            last_cap: last_len,
            left_to_carve: Some(0),
            chunks,
        }
    }

    /// How many chunks a vec of `len` elements lies in.
    fn count_chunks_for_len(len: usize) -> usize {
        if len <= Self::INITIAL_CAP {
            return usize::from(len > 0);
        }
        1 + (len - Self::INITIAL_CAP).div_ceil(Self::CHUNK_CAP)
    }

    /// The vec's chunks as slabs, in order, the last one cut to the elements
    /// pushed: the elements' bytes and nothing after them.
    pub fn into_slabs(self) -> Vec<Slab> {
        let last_size = self.last_len * size_of::<T>();
        let mut slabs: Vec<Slab> = self.chunks.into_iter().map(SlabBuffer::into_slab).collect();
        if let Some(last) = slabs.last_mut() {
            last.size = last_size;
        }
        slabs
    }

    /// How many chunks have been carved so far.
    pub fn count_chunks(&self) -> usize {
        self.chunks.len()
    }

    #[inline(always)]
    pub fn push(&mut self, source: &mut impl SlabSource, val: T) {
        // Full when the chunk being filled is at its capacity. An empty
        // `SlabVec` has no chunk and a capacity of zero, so it carves its
        // first chunk straight away.
        if self.last_len == self.last_cap {
            self.start_chunk(source);
        }
        unsafe { *self.cur_base.add(self.last_len) = val };
        self.last_len += 1;
    }

    /// Append every element of `values`, a chunk's worth at a time.
    pub fn extend_from_slice(&mut self, source: &mut impl SlabSource, values: &[T]) {
        let mut rest = values;
        while !rest.is_empty() {
            if self.last_len == self.last_cap {
                self.start_chunk(source);
            }
            let (fitting, after) = rest.split_at(rest.len().min(self.last_cap - self.last_len));
            // SAFETY: the chunk being filled has room for `fitting` past its
            // `last_len` elements, and `values` cannot overlap a chunk this
            // vec holds exclusively.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    fitting.as_ptr(),
                    self.cur_base.add(self.last_len),
                    fitting.len(),
                );
            }
            self.last_len += fitting.len();
            rest = after;
        }
    }

    /// Carve the next chunk from `source` and make it the one being filled.
    #[cold]
    fn start_chunk(&mut self, source: &mut impl SlabSource) {
        let full_cap = Self::chunk_cap(self.chunks.len());
        let cap = match &mut self.left_to_carve {
            None => full_cap,
            Some(left) => {
                assert!(*left > 0, "pushed past the length the vec was created for");
                let cap = full_cap.min(*left);
                *left -= cap;
                cap
            }
        };
        let chunk =
            SlabBuffer::new(source.get_contiguous_slab(cap * size_of::<T>(), align_of::<T>()));
        self.cur_base = chunk.ptr_at_index(0);
        self.chunks.push(chunk);
        self.last_len = 0;
        self.last_cap = cap;
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

    /// Visit every element in insertion order — sequentially off each chunk's base.
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

    /// The chunk holding element `index` and the element's position in it.
    #[inline(always)]
    fn locate(index: usize) -> (usize, usize) {
        if index < Self::INITIAL_CAP {
            return (0, index);
        }
        let past_first_chunk = index - Self::INITIAL_CAP;
        (
            1 + past_first_chunk / Self::CHUNK_CAP,
            past_first_chunk % Self::CHUNK_CAP,
        )
    }

    /// The elements pushed into chunk `chunk_index`.
    #[inline(always)]
    fn chunk_elements(&self, chunk_index: usize) -> &[T] {
        let len = if chunk_index + 1 == self.chunks.len() {
            self.last_len
        } else {
            Self::chunk_cap(chunk_index)
        };
        let base = self.chunks[chunk_index].ptr_at_index(0) as *const T;
        // SAFETY: the chunk's first `len` elements were pushed, and it was
        // carved aligned for `T`.
        unsafe { std::slice::from_raw_parts(base, len) }
    }

    /// Element `index`, whichever chunk it is in. Panics when `index` is past
    /// the end.
    #[inline(always)]
    pub fn get(&self, index: usize) -> T {
        let (chunk_index, index_in_chunk) = Self::locate(index);
        self.chunk_elements(chunk_index)[index_in_chunk]
    }

    /// How the elements in `range` compare, as a sequence, to `other`. The
    /// range may lie across chunks: it is compared a chunk's part at a time.
    /// Panics when `range` reaches past the end.
    pub fn compare_range(&self, range: Range<usize>, other: &[T]) -> Ordering
    where
        T: Ord,
    {
        let mut position = range.start;
        let mut other_rest = other;
        while position < range.end {
            let (chunk_index, index_in_chunk) = Self::locate(position);
            let chunk_rest = &self.chunk_elements(chunk_index)[index_in_chunk..];
            assert!(!chunk_rest.is_empty(), "element {position} is past the end");
            let part = &chunk_rest[..chunk_rest.len().min(range.end - position)];
            let compared_len = part.len().min(other_rest.len());
            match part[..compared_len].cmp(&other_rest[..compared_len]) {
                // `other` ran out first: it is a prefix of the range.
                Ordering::Equal if compared_len < part.len() => return Ordering::Greater,
                Ordering::Equal => {}
                unequal => return unequal,
            }
            position += part.len();
            other_rest = &other_rest[compared_len..];
        }
        if other_rest.is_empty() {
            Ordering::Equal
        } else {
            Ordering::Less
        }
    }

    /// The index of the first element in `range` that `is_before` rejects, or
    /// the range's end when it rejects none, by binary search across the
    /// chunks. `is_before` is given each element it is asked about together
    /// with its index, and must accept every element of the range before some
    /// point and none from there on.
    pub fn find_partition_point(
        &self,
        range: Range<usize>,
        mut is_before: impl FnMut(usize, T) -> bool,
    ) -> usize {
        let (mut low, mut high) = (range.start, range.end);
        while low < high {
            let middle = low + (high - low) / 2;
            if is_before(middle, self.get(middle)) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        low
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{SlabAllocator, init_test_free_pool};

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

    #[test]
    fn any_element_of_any_chunk_is_read_by_its_index() {
        let (v, _alloc) = push_range(10_000);

        let read: Vec<u64> = (0..10_000).map(|index| v.get(index)).collect();

        assert_eq!(read, (0..10_000).collect::<Vec<_>>());
    }

    #[test]
    #[should_panic]
    fn reading_past_the_end_panics() {
        let (v, _alloc) = push_range(10);

        v.get(10);
    }

    #[test]
    fn a_partition_point_is_found_in_whichever_chunk_it_lies() {
        let (v, _alloc) = push_range(10_000);

        let points: Vec<usize> = [0, 1, 1_024, 5_000, 9_999, 10_000]
            .into_iter()
            .map(|bound| v.find_partition_point(0..10_000, |_, value| value < bound))
            .collect();

        assert_eq!(points, [0, 1, 1_024, 5_000, 9_999, 10_000]);
    }

    #[test]
    fn a_partition_point_is_looked_for_only_inside_the_range_given() {
        let (v, _alloc) = push_range(10_000);

        let before_the_range = v.find_partition_point(2_000..3_000, |_, value| value < 5);
        let inside_the_range = v.find_partition_point(2_000..3_000, |_, value| value < 2_500);
        let after_the_range = v.find_partition_point(2_000..3_000, |_, value| value < 9_000);

        assert_eq!(
            [before_the_range, inside_the_range, after_the_range],
            [2_000, 2_500, 3_000]
        );
    }

    #[test]
    fn a_slice_appended_across_chunks_keeps_every_element() {
        init_test_free_pool(8);
        let mut alloc = SlabAllocator::new(false);
        let values: Vec<u64> = (0..10_000).collect();
        let mut v = SlabVec::new();

        v.extend_from_slice(&mut alloc, &values);

        let read: Vec<u64> = (0..10_000).map(|index| v.get(index)).collect();
        assert_eq!(read, values);
    }

    #[test]
    fn a_vec_of_exact_length_takes_only_the_bytes_of_its_elements() {
        init_test_free_pool(8);
        let mut alloc = SlabAllocator::new(false);
        let mut v = SlabVec::with_exact_len(3);

        v.extend_from_slice(&mut alloc, &[1u64, 2, 3]);

        let first_element = v.chunks[0].ptr_at_index(0) as usize;
        let next = alloc.get_slab_of_size(8, false);
        assert_eq!(next.ptr as usize - first_element, 3 * size_of::<u64>());
    }

    #[test]
    fn a_vec_of_exact_length_longer_than_a_chunk_keeps_every_element() {
        init_test_free_pool(8);
        let mut alloc = SlabAllocator::new(false);
        let values: Vec<u64> = (0..10_000).collect();
        let mut v = SlabVec::with_exact_len(10_000);

        v.extend_from_slice(&mut alloc, &values);

        let read: Vec<u64> = (0..10_000).map(|index| v.get(index)).collect();
        assert_eq!(read, values);
        assert_eq!(v.len(), 10_000);
    }

    #[test]
    #[should_panic(expected = "pushed past the length")]
    fn pushing_past_an_exact_length_panics() {
        init_test_free_pool(8);
        let mut alloc = SlabAllocator::new(false);
        let mut v = SlabVec::with_exact_len(1);
        v.push(&mut alloc, 1u64);

        v.push(&mut alloc, 2u64);
    }

    #[test]
    fn a_range_lying_across_chunks_compares_as_one_sequence() {
        let (v, _alloc) = push_range(10_000);
        let across_a_chunk_boundary: Vec<u64> = (1_000..1_100).collect();

        let same = v.compare_range(1_000..1_100, &across_a_chunk_boundary);
        let one_longer = v.compare_range(1_000..1_101, &across_a_chunk_boundary);
        let one_shorter = v.compare_range(1_000..1_099, &across_a_chunk_boundary);
        let shifted_up = v.compare_range(1_001..1_101, &across_a_chunk_boundary);

        assert_eq!(same, Ordering::Equal);
        assert_eq!(one_longer, Ordering::Greater);
        assert_eq!(one_shorter, Ordering::Less);
        assert_eq!(shifted_up, Ordering::Greater);
    }
}
