//! [`SlabTopK`]: a fixed-capacity "largest-`k`-by-key" collection on slab memory.

use super::{MultiSlabBuffer, SlabAllocator, SlabBuffer};
use crate::memory::BUFFER_SIZE;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/// Initial slot count, so a generous `cap` (a large user `LIMIT`) does not
/// reserve the whole buffer before enough items arrive to need it. The buffer
/// doubles from here up to `cap` only as items are actually retained.
const INITIAL_CAP: usize = 4096;

/// One retained item: its ordering `key` and the `value` kept alongside it.
#[derive(Clone, Copy)]
pub(crate) struct Ranked<K, V> {
    key: K,
    value: V,
}

/// A slab-backed buffer the heap can allocate and probe. Lets [`SlabTopK`] be
/// generic over its backing store (like `BaseHashTable`),
/// so the buffer is picked *once* at construction and the hot sift is
/// monomorphised with no per-row dispatch over the storage kind.
///
/// The sift does not index the buffer directly. It grabs a [`cursor`](Self::cursor)
/// once per `offer`, then does every read and write through that. The cursor is a
/// plain [`Index`]/[`IndexMut`] handle, so the accessor itself is nothing special;
/// what matters is that it is hoisted *out of the buffer struct*:
///
/// ```text
///   self.buf[i]                          let cur = self.buf.cursor(); ... cur[i]
///   -----------------------------------  -------------------------------------
///   Each access re-reads the buffer's    The base pointer is read ONCE into a
///   base pointer out of `self` (through  Copy cursor and lives in a register
///   `&mut self` the compiler can't       for the whole sift; each access is a
///   prove it stays put), then adds `i`.  bare `base + i` with no reload, which
///                                        is what the `Vec` behind a
///                                        `BinaryHeap` gets for free.
/// ```
///
/// A sift is O(log cap) accesses and `offer` runs once per group (millions of
/// calls on a high-cardinality grouped top-k), so that per-access reload is a
/// measured cost. For a contiguous [`SlabBuffer`] the cursor is a bare base
/// pointer; for the rare multi-slab buffer it is the buffer itself, whose
/// per-access slab lookup is inherent (but that path is cold).
pub(crate) trait HeapBuffer<T: Copy>: Sized {
    /// The hoisted accessor (see the trait docs). Index-addressable so the sift
    /// reads and writes it exactly like a slice.
    type Cursor<'a>: Index<usize, Output = T> + IndexMut<usize>
    where
        Self: 'a;
    fn with_slots(allocator: &mut SlabAllocator, slots: usize) -> Self;
    fn cursor(&mut self) -> Self::Cursor<'_>;
}

/// Cursor over a contiguous [`SlabBuffer`]: just its base pointer, copied into a
/// register so indexing is a single `base + i` with no reload.
#[derive(Clone, Copy)]
pub(crate) struct ContiguousCursor<T>(*mut T);

impl<T: Copy> Index<usize> for ContiguousCursor<T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, i: usize) -> &T {
        unsafe { &*self.0.add(i) }
    }
}

impl<T: Copy> IndexMut<usize> for ContiguousCursor<T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut T {
        unsafe { &mut *self.0.add(i) }
    }
}

impl<T: Copy> HeapBuffer<T> for SlabBuffer<T> {
    type Cursor<'a>
        = ContiguousCursor<T>
    where
        T: 'a;
    fn with_slots(allocator: &mut SlabAllocator, slots: usize) -> Self {
        allocator.create_slab_buffer(slots, false)
    }
    fn cursor(&mut self) -> ContiguousCursor<T> {
        ContiguousCursor(self.ptr_at_index(0))
    }
}

/// Cursor over a multi-slab buffer: the buffer itself, indexed per access. Its
/// slab lookup is inherent to spanning slabs, and this backing is only reached by
/// an enormous `cap`, so nothing is hoisted here.
pub(crate) struct MultiCursor<'a, T>(&'a mut MultiSlabBuffer<T>);

impl<T: Copy> Index<usize> for MultiCursor<'_, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, i: usize) -> &T {
        &self.0[i]
    }
}

impl<T: Copy> IndexMut<usize> for MultiCursor<'_, T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.0[i]
    }
}

impl<T: Copy> HeapBuffer<T> for MultiSlabBuffer<T> {
    type Cursor<'a>
        = MultiCursor<'a, T>
    where
        T: 'a;
    fn with_slots(allocator: &mut SlabAllocator, slots: usize) -> Self {
        allocator.create_multi_slab_buffer(slots, false)
    }
    fn cursor(&mut self) -> MultiCursor<'_, T> {
        MultiCursor(self)
    }
}

/// Slots of `Ranked<K, V>` that fit one 2 MB slab. A `cap` up to this can use the
/// contiguous [`SlabBuffer`] backing ([`SlabTopK::single`]); a larger one needs
/// the multi-slab backing ([`SlabTopK::multi`]).
pub(crate) const fn slots_per_slab<K, V>() -> usize {
    BUFFER_SIZE / size_of::<Ranked<K, V>>()
}

/// A fixed-capacity collection that retains the `cap` items with the largest
/// keys seen, backed by the slab pool rather than the global allocator.
///
/// It is a min-heap keyed on `key`: the smallest retained key sits at the root,
/// so once the heap is full an offered item only displaces the root when its key
/// is strictly larger, leaving the `cap` largest. [`offer`](Self::offer) is
/// O(log cap); [`values`](Self::values) reads the retained items back in
/// arbitrary (heap) order. The backing buffer grows lazily from [`INITIAL_CAP`]
/// up to `cap`, so a generous `cap` reserves no slab until that many rows survive.
///
/// Generic over the backing buffer `A`: [`single`](Self::single) backs it with a
/// contiguous [`SlabBuffer`] (the near-universal case), [`multi`](Self::multi)
/// with a [`MultiSlabBuffer`] for a `cap` that spills past one slab. `A` is fixed
/// at construction, so the hot sift carries no per-row branch over the storage
/// kind (see [`HeapBuffer`]).
///
/// This is the building block for an `ORDER BY <key> DESC LIMIT cap` without a
/// `BinaryHeap` on the global allocator. For an *ascending* limit (keep the
/// smallest), order the key in reverse (e.g. negate a numeric key) so
/// largest-kept maps to smallest-wanted.
pub(crate) struct SlabTopK<K: Ord + Copy, V: Copy, A: HeapBuffer<Ranked<K, V>>> {
    buf: A,
    /// Currently-allocated slot count; grows toward `cap` as items are retained.
    buffer_cap: usize,
    len: usize,
    /// The maximum number of items retained (the query's `LIMIT`).
    cap: usize,
    _phantom: PhantomData<(K, V)>,
}

/// A [`SlabTopK`] on the contiguous single-slab backing (the common case).
pub(crate) type SingleTopK<K, V> = SlabTopK<K, V, SlabBuffer<Ranked<K, V>>>;
/// A [`SlabTopK`] on the multi-slab backing, for a `cap` past one slab.
pub(crate) type MultiTopK<K, V> = SlabTopK<K, V, MultiSlabBuffer<Ranked<K, V>>>;

impl<K: Ord + Copy, V: Copy> SlabTopK<K, V, SlabBuffer<Ranked<K, V>>> {
    /// A heap backed by a single contiguous slab. The caller must ensure `cap`
    /// fits one slab (see [`slots_per_slab`]); this is the common case.
    pub fn single(allocator: &mut SlabAllocator, cap: usize) -> Self {
        Self::with_initial(allocator, cap)
    }
}

impl<K: Ord + Copy, V: Copy> SlabTopK<K, V, MultiSlabBuffer<Ranked<K, V>>> {
    /// A heap backed by multiple slabs, for a `cap` larger than one slab holds.
    pub fn multi(allocator: &mut SlabAllocator, cap: usize) -> Self {
        Self::with_initial(allocator, cap)
    }
}

impl<K: Ord + Copy, V: Copy, A: HeapBuffer<Ranked<K, V>>> SlabTopK<K, V, A> {
    fn with_initial(allocator: &mut SlabAllocator, cap: usize) -> Self {
        // Reserve `min(cap, INITIAL_CAP)` up front (at least 1 so the buffer is
        // non-empty even for a `cap == 0` heap, which `offer` never writes into);
        // it doubles toward `cap` on demand. Indices `>= len` are never read.
        let buffer_cap = INITIAL_CAP.clamp(1, cap.max(1));
        Self {
            buf: A::with_slots(allocator, buffer_cap),
            buffer_cap,
            len: 0,
            cap,
            _phantom: PhantomData,
        }
    }

    /// Returns whether [`offer`](Self::offer) could retain this key.
    #[inline]
    pub fn would_retain(&mut self, key: K) -> bool {
        self.len < self.cap || (self.cap > 0 && key > self.buf.cursor()[0].key)
    }

    /// Offer an item, keeping it only if it is among the `cap` largest by `key`.
    /// `allocator` backs the buffer's lazy growth during the fill phase; it is
    /// untouched once the heap is full.
    #[inline]
    pub fn offer(&mut self, allocator: &mut SlabAllocator, key: K, value: V) {
        if self.len < self.cap {
            if self.len == self.buffer_cap {
                self.grow(allocator);
            }
            let mut cursor = self.buf.cursor();
            cursor[self.len] = Ranked { key, value };
            sift_up(&mut cursor, self.len);
            self.len += 1;
        } else if self.cap > 0 {
            let mut cursor = self.buf.cursor();
            if key > cursor[0].key {
                cursor[0] = Ranked { key, value };
                sift_down(&mut cursor, self.len, 0);
            }
        }
    }

    /// Double the backing buffer (capped at `cap`) and copy the live entries over.
    /// Only reached during the fill phase, at most `log2(cap / INITIAL_CAP)` times.
    #[cold]
    fn grow(&mut self, allocator: &mut SlabAllocator) {
        let new_cap = (self.buffer_cap * 2).min(self.cap);
        let mut new_buf = A::with_slots(allocator, new_cap);
        {
            let old = self.buf.cursor();
            let mut new = new_buf.cursor();
            for i in 0..self.len {
                new[i] = old[i];
            }
        }
        self.buf = new_buf;
        self.buffer_cap = new_cap;
    }

    /// The smallest retained key once the heap is full: the k-th largest key
    /// offered so far. `None` while fewer than `cap` items are retained.
    #[inline]
    pub fn kth_key(&mut self) -> Option<K> {
        (self.cap > 0 && self.len == self.cap).then(|| self.buf.cursor()[0].key)
    }

    /// The retained values, in arbitrary (heap) order.
    pub fn values(&mut self) -> impl Iterator<Item = V> + '_ {
        let len = self.len;
        let cursor = self.buf.cursor();
        (0..len).map(move |i| cursor[i].value)
    }
}

/// Restore the min-heap order after writing a new item at leaf `i`. The cursor is
/// a slice-like handle of `Ranked<K, V>` (see [`HeapBuffer::cursor`]).
#[inline]
fn sift_up<K: Ord + Copy, V: Copy, C>(cursor: &mut C, mut i: usize)
where
    C: Index<usize, Output = Ranked<K, V>> + IndexMut<usize>,
{
    while i > 0 {
        let parent = (i - 1) / 2;
        if cursor[i].key >= cursor[parent].key {
            break;
        }
        swap(cursor, i, parent);
        i = parent;
    }
}

/// Restore the min-heap order after overwriting the root at `i`.
#[inline]
fn sift_down<K: Ord + Copy, V: Copy, C>(cursor: &mut C, len: usize, mut i: usize)
where
    C: Index<usize, Output = Ranked<K, V>> + IndexMut<usize>,
{
    loop {
        let (left, right) = (2 * i + 1, 2 * i + 2);
        let mut smallest = i;
        if left < len && cursor[left].key < cursor[smallest].key {
            smallest = left;
        }
        if right < len && cursor[right].key < cursor[smallest].key {
            smallest = right;
        }
        if smallest == i {
            break;
        }
        swap(cursor, i, smallest);
        i = smallest;
    }
}

#[inline]
fn swap<K: Ord + Copy, V: Copy, C>(cursor: &mut C, a: usize, b: usize)
where
    C: Index<usize, Output = Ranked<K, V>> + IndexMut<usize>,
{
    // Read both before writing: `cursor[a] = cursor[b]` would borrow the cursor
    // mutably and immutably at once.
    let (va, vb) = (cursor[a], cursor[b]);
    cursor[a] = vb;
    cursor[b] = va;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;

    /// Offering a shuffled stream retains exactly the `cap` largest keys,
    /// exercising the sift-up fill and the full-heap displacement.
    #[test]
    fn keeps_largest_cap() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(false);
        let mut heap = SlabTopK::<i64, i64, _>::single(&mut allocator, 3);

        for k in [5, 1, 9, 3, 7, 2, 8] {
            heap.offer(&mut allocator, k, k * 10);
        }

        let mut kept: Vec<i64> = heap.values().collect();
        kept.sort();
        assert_eq!(kept, vec![70, 80, 90]);
    }

    /// Fewer items than `cap`: every item is retained and no unwritten slot is read.
    #[test]
    fn cap_exceeds_input() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(false);
        let mut heap = SlabTopK::<i64, i64, _>::single(&mut allocator, 10);

        for k in [4, 2, 6] {
            heap.offer(&mut allocator, k, k);
        }

        let mut kept: Vec<i64> = heap.values().collect();
        kept.sort();
        assert_eq!(kept, vec![2, 4, 6]);
    }

    /// Ties on the key keep `cap` items and never panic.
    #[test]
    fn ties_keep_cap_items() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(false);
        let mut heap = SlabTopK::<i64, i64, _>::single(&mut allocator, 3);

        for v in 0..10 {
            heap.offer(&mut allocator, 7, v);
        }

        assert_eq!(heap.values().count(), 3);
        assert!(heap.values().all(|v| (0..10).contains(&v)));
    }

    /// A `cap` past [`INITIAL_CAP`] on the single-slab backing grows the buffer in
    /// steps while retaining the largest `cap`, so the lazy-growth copy preserves
    /// the heap.
    #[test]
    fn single_grows_within_slab() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(false);
        let cap = INITIAL_CAP * 2 + 7;

        let mut heap = SlabTopK::<i64, i64, _>::single(&mut allocator, cap);
        for k in 0..(2 * cap) as i64 {
            heap.offer(&mut allocator, k, k);
        }

        assert_eq!(heap.values().count(), cap);
        assert!(heap.values().all(|v| v >= cap as i64));
    }

    /// The multi-slab backing handles a `cap` past one 2 MB slab, growing across
    /// slabs and still retaining the largest `cap`. (`Ranked<i64,i64>` is 16 bytes,
    /// so one slab holds 131072; `cap` here exceeds that.)
    #[test]
    fn multi_spills_across_slabs() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(false);
        let cap = slots_per_slab::<i64, i64>() + 50;

        let mut heap = SlabTopK::<i64, i64, _>::multi(&mut allocator, cap);
        for k in 0..(2 * cap) as i64 {
            heap.offer(&mut allocator, k, k);
        }

        assert_eq!(heap.values().count(), cap);
        assert!(heap.values().all(|v| v >= cap as i64));
    }

    /// A zero-capacity heap retains nothing and never reads a slot.
    #[test]
    fn zero_cap_keeps_nothing() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(false);
        let mut heap = SlabTopK::<i64, i64, _>::single(&mut allocator, 0);

        heap.offer(&mut allocator, 1, 1);
        heap.offer(&mut allocator, 2, 2);

        assert_eq!(heap.values().count(), 0);
    }
}
