use crate::env::get_env_var_with_default;
use crate::memory::clock::{Clock, Owner};
use crate::memory::compressed_cache::{CompressedCache, FillCursor};
use crate::memory::decompressed_cache::DecompressedCache;
use crate::memory::free_pool::{FreePool, PoolFactory};
use crate::memory::{BUFFER_SIZE, Ring, WriteBuffer};
use crate::worker::{NODE_LOCAL_IDX, WORKERS_PER_NODE};
use std::cell::{Cell, RefCell, UnsafeCell};
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

/// Raw pointer to the calling thread's installed [`MemoryContext`], or null if
/// none is installed. Used to tag each [`WriteBuffer`] with its owning domain so
/// it can be freed against the right node even when dropped on another thread.
pub(crate) fn current_ctx_ptr() -> *const MemoryContext {
    MEMORY_CTX_PTR.get()
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
    compressed_cache: Arc<CompressedCache>,
    decompressed_cache: Arc<DecompressedCache>,
    clock: Arc<Clock>,
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
        let compressed_cache = Arc::new(CompressedCache::new(buffers));
        let decompressed_cache = Arc::new(DecompressedCache::new());
        let clock = Arc::new(Clock::new(buffers));
        let mut zeroed_pool_factories = PoolFactory::create_many(count);
        let mut dirty_pool_factories = PoolFactory::create_many(count);

        (0..count)
            .map(|_| Self {
                ring: ring.clone(),
                compressed_cache: compressed_cache.clone(),
                decompressed_cache: decompressed_cache.clone(),
                clock: clock.clone(),
                dirty_pool_factory: dirty_pool_factories.pop().unwrap(),
                zeroed_pool_factory: zeroed_pool_factories.pop().unwrap(),
            })
            .collect()
    }

    pub fn create_memory_ctx(self) -> MemoryContext {
        MemoryContext {
            ring: self.ring,
            compressed_cache: self.compressed_cache,
            decompressed_cache: self.decompressed_cache,
            clock: self.clock,
            dirty_pool: self.dirty_pool_factory.create_pool(),
            zeroed_pool: self.zeroed_pool_factory.create_pool(),
            fill_cursor: UnsafeCell::new(FillCursor::empty()),
        }
    }
}

pub struct MemoryContext {
    ring: Arc<Ring>,
    compressed_cache: Arc<CompressedCache>,
    decompressed_cache: Arc<DecompressedCache>,
    /// The CLOCK eviction policy over all ring slots, shared by both caches.
    clock: Arc<Clock>,
    dirty_pool: FreePool,
    zeroed_pool: FreePool,
    /// This worker's bump cursor for packing missed reads into a shared fill buffer
    /// (see [`CompressedCache`]). Per-thread, so interior-mutable without a lock - the
    /// `&'static MemoryContext` is really thread-local, so there is never a second
    /// accessor. Mirrors the `UnsafeCell` discipline the compressed cache uses for its
    /// per-slot metadata.
    fill_cursor: UnsafeCell<FillCursor>,
}

impl MemoryContext {
    pub fn prefault_buffers(&self) {
        // Pre-fault this node's ring, strided by the node-local worker index so each
        // of the node's workers faults different pages of its own domain's ring (and,
        // pinned to its node first, faults them node-local). We forget the WriteBuffer
        // to avoid the Drop impl pushing to the dirty pool, then manually release the
        // slot and push to the zeroed pool.
        for i in (NODE_LOCAL_IDX.get()..self.ring.len()).step_by(WORKERS_PER_NODE.get()) {
            let mut write = memory_ctx().ring().try_write(i).unwrap();
            for j in (0..BUFFER_SIZE).step_by(4096) {
                write.as_mut()[j] = 1u8;
            }
            write.zero_out();
        }
    }

    pub fn compressed_cache(&self) -> &CompressedCache {
        self.compressed_cache.as_ref()
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

    /// This worker's fill cursor (the bump allocator the compressed cache packs missed
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
            self.zeroed_pool.push(idx)
        } else {
            self.dirty_pool.push(idx)
        }
    }

    /// Return a free index through the pool's cross-thread injector only, never
    /// the per-worker local deque. Used when releasing a buffer from a thread
    /// that does not own this context (a shared arena dropped off-node): the
    /// local deque is single-producer and only its owning worker may push to it.
    pub(crate) fn push_free_idx_via_injector(&self, idx: usize, zeroed: bool) {
        if zeroed {
            self.zeroed_pool.push_via_injector(idx)
        } else {
            self.dirty_pool.push_via_injector(idx)
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
                // A slot listed in the free pool must genuinely be free, i.e. owned
                // by no cache. If a cache still owns it, it leaked into the pool
                // while live and is about to be handed out under two owners.
                debug_assert_eq!(
                    self.clock.owner(idx),
                    Owner::Free,
                    "pooled slot {idx} still owned by a cache",
                );
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
            // means we would be evicting the compressed cache, the
            // memory-pressure signal `PANIC_ON_EVICT` guards.
            if *PANIC_ON_EVICT && self.decompressed_cache.is_empty() {
                panic!("Evicting");
            }
            return self.evict();
        }
    }

    /// Evict a ring slot via the shared CLOCK and return it writable. The sweep
    /// picks victims across both caches, routing each to its owner's reclaim:
    /// compressed runs to [`CompressedCache::reclaim`], decompressed
    /// pages to [`DecompressedCache::reclaim`]. Decompressed pages age out
    /// faster (lower clock tier), so they are surrendered first.
    pub(crate) fn evict(&self) -> WriteBuffer {
        // Livelock guard. When a query's working set exceeds the ring, every
        // worker spins here finding nothing evictable (near-100% CPU, ~0 forward
        // progress, and the loop has no cancellation point). A healthy CLOCK finds
        // a victim within a few sweeps (counters reach zero in at most their tier
        // max), so far more than that is abnormal: warn and sleep to let peer
        // workers / ingest release slots; if it still finds nothing, the cache is
        // genuinely exhausted, so panic to abort the offending query.
        let ring_len = self.ring.len() as u64;
        let warn_at = 4 * ring_len;
        let mut iterations: u64 = 0;
        let mut panic_at: Option<u64> = None;
        loop {
            iterations += 1;
            if panic_at.is_none() && iterations == warn_at {
                tracing::warn!(iterations, "evict: no memory left to evict, sleeping");
                std::thread::sleep(std::time::Duration::from_secs(1));
                panic_at = Some(iterations + ring_len);
            } else if panic_at.is_some_and(|limit| iterations >= limit) {
                panic!(
                    "evict: still no evictable memory after sleeping ({iterations} \
                     iterations) — aborting query (cache exhausted by an oversized \
                     working set)"
                );
            }

            let (slot_idx, victim) = self.clock.advance();
            let reclaimed = match victim {
                Some(Owner::Compressed) => self.compressed_cache.reclaim(slot_idx),
                Some(Owner::Decompressed) => self.decompressed_cache.reclaim(slot_idx),
                Some(Owner::Free) | None => None,
            };
            if let Some(write_buffer) = reclaimed {
                return write_buffer;
            }
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
    crate::worker::WORKER_IDX.set(0);
    crate::worker::NUM_WORKERS.set(1);
    NODE_LOCAL_IDX.set(0);
    WORKERS_PER_NODE.set(1);
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
    //! that comes from owning *both* pools (plus the ring and compressed cache).
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
