use crate::env::get_env_var_with_default;
use crate::memory::clock::{Clock, Owner};
use crate::memory::compressed_cache::CompressedCache;
use crate::memory::decompressed_cache::DecompressedCache;
use crate::memory::fill_cursor::FillCursor;
use crate::memory::free_pool::{FreePool, PoolFactory};
use crate::memory::{BUFFER_SIZE, Ring, RingLayout, WriteBuffer};
use crate::worker::WORKER_IDX;
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
    layout: RingLayout,
    node: usize,
}

impl MemoryContextFactory {
    /// Build one factory per worker (global worker order) over a single shared
    /// ring, both caches, and the clock, partitioned per `layout`: each worker
    /// prefaults, acquires, and evicts only its own node's slot region, so its
    /// memory placement stays node-local, while all cached data remains
    /// readable (and releasable) from every worker.
    pub fn create_for_layout(layout: RingLayout) -> Vec<Self> {
        assert!(
            MEMORY_CTX_PTR.get().is_null(),
            "another memory context is already active!"
        );

        let buffers = layout.total_slots();
        let topology = layout.topology();
        let ring = Arc::new(Ring::new(buffers).unwrap());
        let compressed_cache = Arc::new(CompressedCache::new(buffers));
        let decompressed_cache = Arc::new(DecompressedCache::new(buffers));
        let clock = Arc::new(Clock::with_regions(
            topology.node_count,
            buffers / topology.node_count,
        ));
        let zeroed_pool_factories = PoolFactory::create_for_layout(layout);
        let dirty_pool_factories = PoolFactory::create_for_layout(layout);

        zeroed_pool_factories
            .into_iter()
            .zip(dirty_pool_factories)
            .enumerate()
            .map(|(worker, (zeroed_pool_factory, dirty_pool_factory))| Self {
                ring: ring.clone(),
                compressed_cache: compressed_cache.clone(),
                decompressed_cache: decompressed_cache.clone(),
                clock: clock.clone(),
                dirty_pool_factory,
                zeroed_pool_factory,
                layout,
                node: topology.node_of_worker(worker),
            })
            .collect()
    }

    /// Single-node convenience over [`create_for_layout`](Self::create_for_layout):
    /// `count` workers sharing a `buffers`-slot ring with no NUMA split.
    pub fn create_many(count: usize, buffers: usize) -> Vec<Self> {
        Self::create_for_layout(RingLayout::single_node(count, buffers))
    }

    pub fn create_memory_ctx(self) -> MemoryContext {
        MemoryContext {
            ring: self.ring,
            compressed_cache: self.compressed_cache,
            decompressed_cache: self.decompressed_cache,
            clock: self.clock,
            dirty_pool: self.dirty_pool_factory.create_pool(),
            zeroed_pool: self.zeroed_pool_factory.create_pool(),
            compressed_fill_cursor: UnsafeCell::new(FillCursor::empty()),
            decompressed_fill_cursor: UnsafeCell::new(FillCursor::empty()),
            layout: self.layout,
            node: self.node,
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
    /// This worker's bump cursors for packing entries into shared fill slots, one
    /// per cache (see [`CompressedCache`] and [`DecompressedCache`]). Per-thread, so
    /// interior-mutable without a lock - the `&'static MemoryContext` is really
    /// thread-local, so there is never a second accessor. Mirrors the `UnsafeCell`
    /// discipline the caches use for their per-slot metadata.
    compressed_fill_cursor: UnsafeCell<FillCursor>,
    decompressed_fill_cursor: UnsafeCell<FillCursor>,
    /// How ring slots map to nodes and home workers (shared by all contexts).
    layout: RingLayout,
    /// This worker's NUMA node, i.e. the ring region it allocates and evicts from.
    node: usize,
}

impl MemoryContext {
    pub fn prefault_buffers(&self) {
        // Fault in the slots this worker is home to, so under first-touch their
        // pages land on this worker's (already pinned) core's NUMA node. Homes
        // partition each node's region across that node's workers, so every
        // slot is faulted exactly once, by a worker on its own node.
        // We forget the WriteBuffer to avoid the Drop impl pushing to the dirty pool,
        // then manually release the slot and push to the zeroed pool.
        let worker = WORKER_IDX.get();
        for i in self.layout.node_slots(self.node) {
            if self.layout.home_worker(i) != worker {
                continue;
            }
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

    /// This worker's compressed-cache fill cursor (the bump allocator missed reads
    /// are packed into). Sound because the context is per-thread, so the returned
    /// `&mut` never aliases another accessor on the same thread.
    #[allow(clippy::mut_from_ref)] // interior mutability; per-thread, single accessor
    pub(crate) fn compressed_fill_cursor(&self) -> &mut FillCursor {
        unsafe { &mut *self.compressed_fill_cursor.get() }
    }

    /// This worker's decompressed-cache fill cursor (the bump allocator pages are
    /// packed into). Same soundness argument as
    /// [`compressed_fill_cursor`](Self::compressed_fill_cursor).
    #[allow(clippy::mut_from_ref)] // interior mutability; per-thread, single accessor
    pub(crate) fn decompressed_fill_cursor(&self) -> &mut FillCursor {
        unsafe { &mut *self.decompressed_fill_cursor.get() }
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
    /// peer workers - see [`crate::worker::Worker`]'s `clear_dirty_buffer_or_park`.
    ///
    /// A popped slot can be transiently locked (e.g. by an evictor probing it
    /// before backing off). The contender never re-pools an index it did not
    /// pop, so the index is pushed back - dropping it would leak the slot from
    /// every pool - and `None` is returned; the next pass retries.
    pub fn pop_dirty_buffer(&self) -> Option<WriteBuffer> {
        let idx = self.dirty_pool.pop(false)?;
        let buffer = memory_ctx().ring().try_write(idx);
        if buffer.is_none() {
            self.dirty_pool.push(idx);
        }
        buffer
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
        // Indices popped for slots that turned out transiently locked (e.g. an
        // evictor probing a pooled slot before backing off). Set aside here and
        // pushed back once a buffer is in hand: the contender never re-pools an
        // index it did not pop, so dropping one would leak the slot from every
        // pool. Held out of the pools during the loop so the retry pops a
        // *different* index instead of spinning on the locked one.
        let mut contended = Vec::new();
        let buffer = loop {
            if let Some(idx) = self.pop_free_idx(prefer_zeroed) {
                // A slot listed in the free pool must genuinely be free, i.e. owned
                // by no cache. If a cache still owns it, it leaked into the pool
                // while live and is about to be handed out under two owners.
                debug_assert_eq!(
                    self.clock.owner(idx),
                    None,
                    "pooled slot {idx} still owned by a cache",
                );
                match memory_ctx().ring().try_write(idx) {
                    Some(buffer) => break buffer,
                    // Try again to get a free idx, we don't want to start evicting yet
                    None => contended.push(idx),
                }
                continue;
            }

            // Pool empty: evict via the shared clock, which takes from whichever
            // tier is over its target share (decompressed, in the common case of
            // a large decompressed working set). Panic only when there is nothing
            // cheap to reclaim - an empty decompressed cache means we would be
            // evicting the compressed cache, the memory-pressure signal
            // `PANIC_ON_EVICT` guards.
            if *PANIC_ON_EVICT && self.decompressed_cache.is_empty() {
                panic!("Evicting");
            }
            break self.evict();
        };
        for idx in contended {
            self.push_free_idx(idx, memory_ctx().ring().slot_zeroed(idx));
        }
        buffer
    }

    /// Evict a ring slot via the shared CLOCK and return it writable. Each
    /// iteration re-picks which tier to take from - compressed while its share
    /// of cached slots exceeds the target percentage, decompressed otherwise
    /// (see [`Clock::preferred_victim_tier`]) - advances that tier's hand, and
    /// routes the victim to its cache's reclaim.
    pub(crate) fn evict(&self) -> WriteBuffer {
        // Livelock guard. When a query's working set exceeds the ring, every
        // worker spins here finding nothing evictable (near-100% CPU, ~0 forward
        // progress, and the loop has no cancellation point). A healthy hand finds
        // a victim within the lives ceiling's worth of revolutions (a counter
        // reaches zero in at most that many), so more than both hands' worth
        // combined is abnormal: warn and sleep to let peer workers / ingest
        // release slots; if it still finds nothing, the cache is genuinely
        // exhausted, so panic to abort the offending query. The sweep covers
        // only this worker's node region, so every bound is region-sized.
        let ring_len = self.layout.node_slots(self.node).len() as u64;
        let max_lives = self.clock.max_lives() as u64;
        let warn_at = (2 * max_lives + 2) * ring_len;
        let mut iterations: u64 = 0;
        let mut panic_at: Option<u64> = None;
        // A hand that has swept the full lives ceiling's worth of revolutions
        // without reclaiming anything (every candidate pinned) is stuck; fall
        // through to the other tier regardless of share rather than spinning
        // into the guard while evictable slots exist there.
        let mut ticks_without_success: u64 = 0;
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

            let mut tier = self.clock.preferred_victim_tier();
            if ticks_without_success > (max_lives + 1) * ring_len {
                tier = match tier {
                    Owner::Compressed => Owner::Decompressed,
                    Owner::Decompressed => Owner::Compressed,
                };
                // Re-arm after one fruitless revolution of the other hand too, so
                // the fallback alternates between the tiers instead of latching
                // onto one that may be empty while the preferred tier's pins have
                // long been released.
                if ticks_without_success > (max_lives + 2) * ring_len {
                    ticks_without_success = 0;
                }
            }

            let reclaimed = self
                .clock
                .advance(tier, self.node)
                .and_then(|slot| match tier {
                    Owner::Compressed => self.compressed_cache.reclaim(slot),
                    Owner::Decompressed => self.decompressed_cache.reclaim(slot),
                });
            match reclaimed {
                Some(write_buffer) => return write_buffer,
                None => ticks_without_success += 1,
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
    WORKER_IDX.set(0);
    crate::worker::NUM_WORKERS.set(1);
    crate::waker::install_test_worker_waker();
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
    fn pop_dirty_buffer_keeps_a_contended_slot_pooled() {
        // Setup: slot 0's index sits in the dirty pool while an evictor probe
        // transiently holds the slot itself.
        init_test_free_pool(1);
        let probe = memory_ctx().ring().try_write(0).unwrap();

        let while_contended = memory_ctx().pop_dirty_buffer();
        probe.release_in_place();
        let after_release = memory_ctx().pop_dirty_buffer();

        // Assert: the contended pop yields nothing, but the slot is still
        // poolable once the probe backs off.
        assert!(while_contended.is_none());
        assert_eq!(after_release.map(|b| b.slot_idx), Some(0));
    }

    #[test]
    fn get_write_buffer_repools_a_contended_slot() {
        // Setup: two dirty slots; an evictor probe transiently holds slot 1.
        init_test_free_pool(2);
        let probe = memory_ctx().ring().try_write(1).unwrap();

        let handed_out = memory_ctx().get_write_buffer(false);
        probe.release_in_place();
        let repooled = memory_ctx().pop_free_idx(false);

        // Assert: the free slot is handed out and the contended one is back in
        // a pool rather than leaked.
        assert_eq!(handed_out.slot_idx, 0);
        assert_eq!(repooled, Some(1));
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
