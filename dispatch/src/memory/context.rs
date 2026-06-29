use crate::env::get_env_var_with_default;
use crate::memory::clock::Clock;
use crate::memory::decompressed_cache::DecompressedCache;
use crate::memory::file_memory_cache::{FileMemoryCache, FillCursor};
use crate::memory::free_pool::{FreePool, PoolFactory};
use crate::memory::{BUFFER_SIZE, Ring, WriteBuffer};
use crate::worker::{NUM_WORKERS, WORKER_IDX};
use std::cell::{Cell, RefCell, UnsafeCell};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};

static PANIC_ON_EVICT: LazyLock<bool> =
    LazyLock::new(|| get_env_var_with_default("PANIC_ON_EVICT", true));

/// Target number of pre-zeroed buffers to keep on hand. Idle workers top the
/// zeroed pool up to this so allocators that need zeroed memory (group-by hash
/// tables) rarely have to zero inline. Must stay below the cache's reserve
/// (`buffers - max_slots`) so there's room to hold them. Default: an eighth of
/// the ring. Env `PIVOT_ZEROED_BUFFER_TARGET` overrides.
fn zeroed_buffer_target(buffers: usize) -> usize {
    get_env_var_with_default("PIVOT_ZEROED_BUFFER_TARGET", buffers / 8)
}

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

/// True when a [`MemoryContext`] is installed on the current thread - i.e. the
/// caller is running on a dispatch worker (or a test that called
/// `init_test_free_pool`). Worker-only APIs that reach into the per-thread
/// memory context use this to fail with a clear message instead of letting
/// [`memory_ctx`] dereference the null context pointer.
pub fn has_memory_context() -> bool {
    MEMORY_CTX_PTR.with(|p| !p.get().is_null())
}

pub struct MemoryContextFactory {
    ring: Arc<Ring>,
    file_memory_cache: Arc<FileMemoryCache>,
    decompressed_cache: Arc<DecompressedCache>,
    clock: Arc<Clock>,
    dirty_pool_factory: PoolFactory,
    zeroed_pool_factory: PoolFactory,
    zeroed_count: Arc<AtomicUsize>,
    zeroed_target: usize,
}

impl MemoryContextFactory {
    pub fn create_many(count: usize, buffers: usize) -> Vec<Self> {
        assert!(
            MEMORY_CTX_PTR.get().is_null(),
            "another memory context is already active!"
        );

        let ring = Arc::new(Ring::new(buffers).unwrap());
        let file_memory_cache = Arc::new(FileMemoryCache::new(buffers));
        let decompressed_cache = Arc::new(DecompressedCache::new());
        let clock = Arc::new(Clock::new(buffers));
        let zeroed_count = Arc::new(AtomicUsize::new(0));
        let zeroed_target = zeroed_buffer_target(buffers);
        let mut zeroed_pool_factories = PoolFactory::create_many(count);
        let mut dirty_pool_factories = PoolFactory::create_many(count);

        (0..count)
            .map(|_| Self {
                ring: ring.clone(),
                file_memory_cache: file_memory_cache.clone(),
                decompressed_cache: decompressed_cache.clone(),
                clock: clock.clone(),
                dirty_pool_factory: dirty_pool_factories.pop().unwrap(),
                zeroed_pool_factory: zeroed_pool_factories.pop().unwrap(),
                zeroed_count: zeroed_count.clone(),
                zeroed_target,
            })
            .collect()
    }

    pub fn create_memory_ctx(self) -> MemoryContext {
        MemoryContext {
            ring: self.ring,
            file_memory_cache: self.file_memory_cache,
            decompressed_cache: self.decompressed_cache,
            clock: self.clock,
            dirty_pool: self.dirty_pool_factory.create_pool(),
            zeroed_pool: self.zeroed_pool_factory.create_pool(),
            zeroed_count: self.zeroed_count,
            zeroed_target: self.zeroed_target,
            fill_cursor: UnsafeCell::new(FillCursor::empty()),
        }
    }
}

pub struct MemoryContext {
    ring: Arc<Ring>,
    file_memory_cache: Arc<FileMemoryCache>,
    decompressed_cache: Arc<DecompressedCache>,
    /// The CLOCK eviction policy over all ring slots, shared by both caches.
    clock: Arc<Clock>,
    dirty_pool: FreePool,
    zeroed_pool: FreePool,
    /// Approximate count of buffers currently in the zeroed pool (shared across
    /// workers). Bumped on push-as-zeroed, dropped on pop-from-zeroed. Drives the
    /// idle zeroing target; a small race-induced drift is harmless (it only nudges
    /// how eagerly idle workers zero).
    zeroed_count: Arc<AtomicUsize>,
    /// Target zeroed-pool size idle workers maintain (see [`zeroed_buffer_target`]).
    zeroed_target: usize,
    /// This worker's bump cursor for packing missed reads into a shared fill buffer
    /// (see [`FileMemoryCache`]). Per-thread, so interior-mutable without a lock - the
    /// `&'static MemoryContext` is really thread-local, so there is never a second
    /// accessor. Mirrors the `UnsafeCell` discipline the file cache uses for its
    /// per-slot metadata.
    fill_cursor: UnsafeCell<FillCursor>,
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

    pub fn file_memory_cache(&self) -> &FileMemoryCache {
        self.file_memory_cache.as_ref()
    }

    pub fn decompressed_cache(&self) -> &DecompressedCache {
        self.decompressed_cache.as_ref()
    }

    pub fn clock(&self) -> &Clock {
        self.clock.as_ref()
    }

    pub fn ring(&self) -> &Ring {
        self.ring.as_ref()
    }

    /// This worker's fill cursor (the bump allocator the file cache packs missed
    /// reads into). Sound because the context is per-thread, so the returned `&mut`
    /// never aliases another accessor on the same thread.
    #[allow(clippy::mut_from_ref)] // interior mutability; per-thread, single accessor
    pub(crate) fn fill_cursor(&self) -> &mut FillCursor {
        unsafe { &mut *self.fill_cursor.get() }
    }

    /// Return a buffer index to the pool.
    ///
    /// Routes the index to the home worker's zeroed or dirty pool. If the current thread
    /// *is* the home worker, the push goes directly into the local deque. Otherwise it is
    /// placed into the home worker's injector so the owner retrieves it on its next pop.
    pub fn push_free_idx(&self, idx: usize, zeroed: bool) {
        if zeroed {
            self.zeroed_count.fetch_add(1, Ordering::Relaxed);
            self.zeroed_pool.push(idx)
        } else {
            self.dirty_pool.push(idx)
        }
    }

    /// Pop a free buffer index for the current worker thread.
    ///
    /// When `prefer_zeroed` is true, tries the zeroed pool first then dirty.
    /// When false, tries dirty first then zeroed. Popping from the zeroed pool
    /// decrements [`zeroed_count`](Self::needs_zeroed_buffers).
    pub fn pop_free_idx(&self, prefer_zeroed: bool) -> Option<usize> {
        if prefer_zeroed {
            self.pop_zeroed().or_else(|| self.dirty_pool.pop(true))
        } else {
            self.dirty_pool.pop(true).or_else(|| self.pop_zeroed())
        }
    }

    /// Pop from the zeroed pool, keeping the shared zeroed count in step.
    fn pop_zeroed(&self) -> Option<usize> {
        let idx = self.zeroed_pool.pop(true)?;
        self.zeroed_count.fetch_sub(1, Ordering::Relaxed);
        Some(idx)
    }

    /// Whether the zeroed pool is below its target, so an otherwise-idle worker
    /// should zero a dirty buffer to top it up.
    pub fn needs_zeroed_buffers(&self) -> bool {
        self.zeroed_count.load(Ordering::Relaxed) < self.zeroed_target
    }

    /// Pop a dirty buffer from this worker's local deque only (no stealing).
    ///
    /// Used by background buffer-clean passes that should not pull buffers off
    /// peer workers - see [`crate::worker::Worker`]'s `clear_dirty_buffer_or_park`.
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
    /// When `prefer_zeroed` is true, tries the zeroed pool first - use this when the caller
    /// needs zeroed memory so we can skip a memset. When false, tries the
    /// dirty pool first - use this when the caller will overwrite the buffer entirely
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

            // Pool empty: evict via the shared clock, which surrenders
            // decompressed pages preferentially (they age out faster). Panic only
            // when there is nothing cheap to reclaim - an empty decompressed cache
            // means we would be evicting the compressed file cache, the
            // memory-pressure signal `PANIC_ON_EVICT` guards.
            if *PANIC_ON_EVICT && self.decompressed_cache.is_empty() {
                panic!("Evicting");
            }
            return self.file_memory_cache.evict();
        }
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
    //! Tests for [`MemoryContext`] - the zeroed/dirty pair, fallback rules,
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
