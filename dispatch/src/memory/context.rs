use crate::env::get_env_var_with_default;
use crate::memory::file_cache::{FileCache, SUB_BLOCKS_PER_BUFFER};
use crate::memory::free_pool::{FreePool, PoolFactory};
use crate::memory::{BUFFER_SIZE, Ring, WriteBuffer};
use crate::worker::{NUM_WORKERS, WORKER_IDX};
use std::cell::{Cell, RefCell};
use std::sync::atomic::Ordering;
use std::sync::{Arc, LazyLock};

static PANIC_ON_EVICT: LazyLock<bool> =
    LazyLock::new(|| get_env_var_with_default("PANIC_ON_EVICT", true));

thread_local! {
    static MEMORY_CONTEXT_OWNER: RefCell<Option<Box<MemoryContext>>> = const { RefCell::new(None) };
    static MEMORY_CTX_PTR: Cell<*const MemoryContext> = const { Cell::new(std::ptr::null()) };
}

pub fn init_memory_context(ctx: MemoryContext) {
    MEMORY_CONTEXT_OWNER.with_borrow_mut(|slot| {
        let b = Box::new(ctx);
        MEMORY_CTX_PTR.set(&*b as *const MemoryContext);
        *slot = Some(b);
    });
}

pub fn memory_ctx() -> &'static MemoryContext {
    unsafe { &*MEMORY_CTX_PTR.get() }
}

/// True when a [`MemoryContext`] is installed on the current thread — i.e. the
/// caller is running on a dispatch worker (or a test that called
/// `init_test_free_pool`). Worker-only APIs that reach into the per-thread
/// memory context use this to fail with a clear message instead of letting
/// [`memory_ctx`] dereference the null context pointer.
pub fn has_memory_context() -> bool {
    MEMORY_CTX_PTR.with(|p| !p.get().is_null())
}

pub struct MemoryContextFactory {
    ring: Arc<Ring>,
    file_cache: Arc<FileCache>,
    dirty_pool_factory: PoolFactory,
    zeroed_pool_factory: PoolFactory,
}

impl MemoryContextFactory {
    pub fn create_many(count: usize, buffers: usize) -> Vec<Self> {
        assert!(
            MEMORY_CTX_PTR.get().is_null(),
            "another memory context is already active!"
        );

        let ring = Arc::new(Ring::new(buffers).unwrap());
        let file_cache = Arc::new(FileCache::new(buffers));
        let mut zeroed_pool_factories = PoolFactory::create_many(count);
        let mut dirty_pool_factories = PoolFactory::create_many(count);

        (0..count)
            .map(|_| Self {
                ring: ring.clone(),
                file_cache: file_cache.clone(),
                dirty_pool_factory: dirty_pool_factories.pop().unwrap(),
                zeroed_pool_factory: zeroed_pool_factories.pop().unwrap(),
            })
            .collect()
    }

    pub fn create_memory_ctx(self) -> MemoryContext {
        let allocator = RefCell::new(SubBlockAllocator::new(self.ring.clone()));
        MemoryContext {
            ring: self.ring,
            file_cache: self.file_cache,
            dirty_pool: self.dirty_pool_factory.create_pool(),
            zeroed_pool: self.zeroed_pool_factory.create_pool(),
            allocator,
        }
    }
}

pub struct MemoryContext {
    ring: Arc<Ring>,
    file_cache: Arc<FileCache>,
    dirty_pool: FreePool,
    zeroed_pool: FreePool,
    /// This worker's private sub-block allocator for packing small cached reads
    /// into ring buffers. Holds its own [`Ring`] handle so releasing the pin never
    /// depends on the thread-local context pointer (sound during teardown / test
    /// re-init). Defined below [`MemoryContext`]'s `impl`.
    allocator: RefCell<SubBlockAllocator>,
}

impl MemoryContext {
    pub fn prefault_buffers(&self) {
        // Pre-fault buffers (strided by NUM_WORKERS) so each worker faults different pages.
        // We forget the WriteBuffer to avoid the Drop impl pushing to the dirty pool,
        // then manually release the slot and push to the zeroed pool.
        for i in (WORKER_IDX.get()..self.ring.len()).step_by(NUM_WORKERS.get()) {
            let mut write = memory_ctx().ring().try_write(i).unwrap();
            for j in (0..BUFFER_SIZE).step_by(4096) {
                write.as_mut()[j] = 1u8;
            }
            write.zero_out();
        }
    }

    pub fn file_cache(&self) -> &FileCache {
        self.file_cache.as_ref()
    }

    pub fn ring(&self) -> &Ring {
        self.ring.as_ref()
    }

    /// Return a buffer index to the pool.
    ///
    /// Routes the index to the home worker's zeroed or dirty pool. If the current thread
    /// *is* the home worker, the push goes directly into the local deque. Otherwise it is
    /// placed into the home worker's injector so the owner retrieves it on its next pop.
    pub fn push_free_idx(&self, idx: usize, zeroed: bool) {
        if zeroed {
            self.zeroed_pool.push(idx)
        } else {
            self.dirty_pool.push(idx)
        }
    }

    /// Pop a free buffer index for the current worker thread.
    ///
    /// When `prefer_zeroed` is true, tries the zeroed pool first then dirty.
    /// When false, tries dirty first then zeroed.
    pub fn pop_free_idx(&self, prefer_zeroed: bool) -> Option<usize> {
        let (first, second) = if prefer_zeroed {
            (&self.zeroed_pool, &self.dirty_pool)
        } else {
            (&self.dirty_pool, &self.zeroed_pool)
        };
        first.pop(true).or_else(|| second.pop(true))
    }

    /// Pop a dirty buffer from this worker's local deque only (no stealing).
    ///
    /// Used by background buffer-clean passes that should not pull buffers off
    /// peer workers — see [`crate::worker::Worker`]'s `clear_dirty_buffer_or_park`.
    pub fn pop_dirty_buffer(&self) -> Option<WriteBuffer> {
        self.dirty_pool
            .pop(false)
            .and_then(|i| memory_ctx().ring().try_write(i))
    }

    /// Drain this worker's dirty buffers, zeroing each and returning it to the
    /// zeroed pool. Equivalent to the engine's idle-time dirty-buffer cleanup,
    /// but run eagerly to completion rather than opportunistically.
    ///
    /// Must run on the worker thread that owns this context (its `pop`/`push`
    /// touch the per-worker free pool). Returns the number of buffers zeroed.
    /// Intended for benchmarks: re-zeroing dirtied buffers between iterations is
    /// allocation/setup work, so doing it eagerly (outside the timed region)
    /// keeps the next query's hash-table allocation from re-zeroing inline.
    pub fn zero_dirty_buffers(&self) -> usize {
        let mut zeroed = 0;
        while let Some(buf) = self.pop_dirty_buffer() {
            buf.zero_out();
            zeroed += 1;
        }
        zeroed
    }

    /// Acquire a [`WriteBuffer`] from the free pool, falling back to eviction.
    ///
    /// When `prefer_zeroed` is true, tries the zeroed pool first — use this when the caller
    /// needs zeroed memory so we can skip a memset. When false, tries the
    /// dirty pool first — use this when the caller will overwrite the buffer entirely
    /// (e.g. decompression, I/O reads) to preserve zeroed buffers for those who need them.
    /// Either way, the other pool is used as a fallback if the preferred one is empty.
    ///
    /// If the popped index's ring slot is contended, retries with a fresh index rather
    /// than evicting.
    pub fn get_write_buffer(&self, prefer_zeroed: bool) -> WriteBuffer {
        loop {
            if let Some(idx) = self.pop_free_idx(prefer_zeroed) {
                if let Some(r) = memory_ctx().ring().try_write(idx) {
                    return r;
                } else {
                    // Let's try again to get a free idx, we don't want to start evicting yet
                    continue;
                }
            }

            if *PANIC_ON_EVICT {
                panic!("Evicting");
            }
            // Nothing free! Let's evict from page cache
            return self.file_cache.evict();
        }
    }

    /// Allocate a contiguous run of `sub_block_count` sub-blocks in this worker's
    /// current buffer for a cache miss, acquiring a fresh buffer (possibly
    /// evicting) when the current one lacks room. A run never exceeds one buffer,
    /// since callers split reads at 2 MB region boundaries.
    ///
    /// The caller records the returned allocation in the cache index and reads its
    /// bytes into the buffer. The buffer stays pinned by this worker until it
    /// fills up, so the run cannot be evicted out from under an in-flight read
    /// (which also takes its own pin).
    pub(crate) fn allocate_sub_blocks(&self, sub_block_count: usize) -> SubBlockAllocation {
        debug_assert!(sub_block_count <= SUB_BLOCKS_PER_BUFFER);
        let mut allocator = self.allocator.borrow_mut();

        let has_room = allocator.buffer_index.is_some()
            && allocator.next_sub_block + sub_block_count <= SUB_BLOCKS_PER_BUFFER;
        if !has_room {
            allocator.retire();
            let mut write_buffer = self.get_write_buffer(false);
            let buffer_index = write_buffer.slot_idx;
            self.file_cache.prepare_fill_buffer(&mut write_buffer);
            // Take the buffer from exclusive (WRITING) to one reader: this worker's
            // pin. Forget the `WriteBuffer` so its `Drop` does not return the slot
            // to the free pool — `SubBlockAllocator::retire` releases it instead.
            self.ring.set_slot_zeroed(buffer_index, false);
            self.ring.set_slot_used(buffer_index, 1, Ordering::Release);
            std::mem::forget(write_buffer);
            allocator.buffer_index = Some(buffer_index);
            allocator.next_sub_block = 0;
        }

        let buffer_index = allocator.buffer_index.expect("just ensured a buffer");
        let buffer_first_sub_block = allocator.next_sub_block;
        allocator.next_sub_block += sub_block_count;
        SubBlockAllocation {
            buffer_index,
            buffer_first_sub_block,
            // Read fresh: the pinned buffer's generation can be bumped by `clear`.
            generation: self.file_cache.buffer_generation(buffer_index),
        }
    }
}

/// Where [`MemoryContext::allocate_sub_blocks`] put a run: the ring buffer holding
/// it, the sub-block offset within that buffer, and the buffer's generation at
/// allocation time (the tag the run is cached under).
pub(crate) struct SubBlockAllocation {
    pub buffer_index: usize,
    pub buffer_first_sub_block: usize,
    pub generation: u64,
}

/// A worker's private bump allocator for cached runs: it owns the current ring
/// buffer (one slot it holds a reader pin on, so the CLOCK hand cannot reclaim it
/// while it is being filled) and the bump offset into it. [`retire`](Self::retire)
/// drops the pin, turning the buffer into an ordinary evictable cache buffer.
struct SubBlockAllocator {
    ring: Arc<Ring>,
    buffer_index: Option<usize>,
    next_sub_block: usize,
}

impl SubBlockAllocator {
    fn new(ring: Arc<Ring>) -> Self {
        SubBlockAllocator {
            ring,
            buffer_index: None,
            next_sub_block: 0,
        }
    }

    /// Retire the current buffer by dropping this worker's reader pin, leaving it
    /// as an evictable cache buffer.
    fn retire(&mut self) {
        if let Some(buffer_index) = self.buffer_index.take() {
            self.ring.slots[buffer_index]
                .used
                .fetch_sub(1, Ordering::Release);
        }
    }
}

impl Drop for SubBlockAllocator {
    fn drop(&mut self) {
        self.retire();
    }
}

/// Install a fresh single-worker [`MemoryContext`] on the calling test
/// thread, with the first `dirty_count` ring slots pre-loaded into the dirty
/// pool so `memory_ctx().get_write_buffer(false)` can hand them out.
///
/// Each test thread gets its own context (and its own [`Ring`]), so tests
/// don't share state and don't need a serializing lock.
#[cfg(any(test, feature = "test-util"))]
pub fn init_test_free_pool(dirty_count: usize) {
    WORKER_IDX.set(0);
    NUM_WORKERS.set(1);
    crate::worker::install_test_worker_waker();
    let factory = MemoryContextFactory::create_many(1, 128).pop().unwrap();
    init_memory_context(factory.create_memory_ctx());
    for i in 0..dirty_count {
        memory_ctx().push_free_idx(i, false);
    }
}

#[cfg(test)]
mod tests {
    //! Tests for [`MemoryContext`] — the zeroed/dirty pair, fallback rules,
    //! and the buffer accessors. Single-pool routing lives in
    //! [`crate::memory::free_pool`]'s tests; here we only exercise behaviour
    //! that comes from owning *both* pools (plus the ring and file cache).
    //!
    //! Each test installs a fresh [`MemoryContext`] on its own thread via
    //! [`init_test_free_pool`], so there's no shared state and no lock.

    use super::*;

    /// Install a context with no buffers seeded.
    fn fresh_ctx() {
        init_test_free_pool(0);
    }

    #[test]
    fn pop_returns_none_when_both_pools_empty() {
        // Setup
        fresh_ctx();

        // Execute
        let zeroed = memory_ctx().pop_free_idx(true);
        let dirty = memory_ctx().pop_free_idx(false);

        // Assert
        assert_eq!(zeroed, None);
        assert_eq!(dirty, None);
    }

    #[test]
    fn prefer_zeroed_picks_zeroed_over_dirty() {
        // Setup: a dirty idx and a zeroed idx, both routable to worker 0.
        fresh_ctx();
        memory_ctx().push_free_idx(0, false);
        memory_ctx().push_free_idx(1, true);

        // Execute
        let popped = memory_ctx().pop_free_idx(true);

        // Assert
        assert_eq!(popped, Some(1));
    }

    #[test]
    fn prefer_dirty_picks_dirty_over_zeroed() {
        // Setup
        fresh_ctx();
        memory_ctx().push_free_idx(0, true);
        memory_ctx().push_free_idx(1, false);

        // Execute
        let popped = memory_ctx().pop_free_idx(false);

        // Assert
        assert_eq!(popped, Some(1));
    }

    #[test]
    fn prefer_zeroed_falls_back_to_dirty_when_zeroed_is_empty() {
        // Setup
        fresh_ctx();
        memory_ctx().push_free_idx(0, false);

        // Execute
        let popped = memory_ctx().pop_free_idx(true);

        // Assert
        assert_eq!(popped, Some(0));
    }

    #[test]
    fn prefer_dirty_falls_back_to_zeroed_when_dirty_is_empty() {
        // Setup
        fresh_ctx();
        memory_ctx().push_free_idx(0, true);

        // Execute
        let popped = memory_ctx().pop_free_idx(false);

        // Assert
        assert_eq!(popped, Some(0));
    }

    #[test]
    fn pop_drains_both_pools_before_returning_none() {
        // Setup: one in each pool.
        fresh_ctx();
        memory_ctx().push_free_idx(0, false);
        memory_ctx().push_free_idx(1, true);

        // Execute
        let mut seen = vec![
            memory_ctx().pop_free_idx(false).unwrap(),
            memory_ctx().pop_free_idx(false).unwrap(),
        ];
        seen.sort();
        let after = memory_ctx().pop_free_idx(false);

        // Assert
        assert_eq!(seen, vec![0, 1]);
        assert_eq!(after, None);
    }

    #[test]
    fn get_write_buffer_hands_out_a_seeded_slot() {
        // Setup
        init_test_free_pool(4);

        // Execute
        let wb = memory_ctx().get_write_buffer(false);

        // Assert
        assert!(wb.slot_idx < 4);
    }

    #[test]
    fn pop_dirty_buffer_returns_a_buffer_when_dirty_pool_has_one() {
        // Setup
        init_test_free_pool(1);

        // Execute
        let popped = memory_ctx().pop_dirty_buffer();

        // Assert
        assert!(popped.is_some());
    }

    #[test]
    fn pop_dirty_buffer_returns_none_when_dirty_pool_is_empty() {
        // Setup: seed only the zeroed pool.
        fresh_ctx();
        memory_ctx().push_free_idx(0, true);

        // Execute
        let popped = memory_ctx().pop_dirty_buffer();

        // Assert
        assert!(popped.is_none());
    }

    #[test]
    fn dropping_a_write_buffer_returns_the_slot_to_the_dirty_pool() {
        // Setup
        init_test_free_pool(1);
        let wb = memory_ctx().get_write_buffer(false);
        let slot_idx = wb.slot_idx;

        // Execute
        drop(wb);
        let recovered = memory_ctx().pop_dirty_buffer();

        // Assert
        assert_eq!(recovered.map(|b| b.slot_idx), Some(slot_idx));
    }

    #[test]
    fn zero_out_returns_the_slot_to_the_zeroed_pool() {
        // Setup
        init_test_free_pool(1);
        let wb = memory_ctx().get_write_buffer(false);
        let slot_idx = wb.slot_idx;

        // Execute
        wb.zero_out();
        let recovered_via_zeroed = memory_ctx().pop_free_idx(true);

        // Assert
        assert_eq!(recovered_via_zeroed, Some(slot_idx));
    }
}
