use crate::Identifier;
use crate::WriteRequestId;
use crate::io::{Completion, IORequester};
use crate::memory::clock::{Clock, Owner, Sweep};
use crate::memory::compressed_cache::CompressedCache;
use crate::memory::decompressed_cache::DecompressedCache;
use crate::memory::evict::DataflowEvict;
use crate::memory::fill_cursor::FillCursor;
use crate::memory::free_pool::{FreePool, PoolFactory};
use crate::memory::status::{MemoryBlockStatus, read_block_status};
use crate::memory::{BUFFER_SIZE, Ring, RingLayout, WriteBuffer};
use crate::worker::WORKER_IDX;
use std::cell::{Cell, OnceCell, RefCell, UnsafeCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Share of a node's ring slots that eviction should always be able to take
/// back without waiting on a write: slots that are free, or cached and so
/// reclaimable by the CLOCK sweep. Once fewer than this remain, the rest being
/// held by operators, the worker asks its operators to spill.
const RECLAIMABLE_SLOTS_TARGET_PERCENT: usize = 20;

/// [`RECLAIMABLE_SLOTS_TARGET_PERCENT`], or what `PIVOT_SPILL_RESERVE_PERCENT`
/// sets in its place.
fn get_reclaimable_slots_target_percent() -> usize {
    static TARGET_PERCENT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    crate::env::get_tunable(
        &TARGET_PERCENT,
        "PIVOT_SPILL_RESERVE_PERCENT",
        RECLAIMABLE_SLOTS_TARGET_PERCENT,
        |percent| percent <= 100,
    )
}

/// The share of a node's slots below which spilling starts, when
/// `PIVOT_SPILL_START_RESERVE_PERCENT` sets one apart from the target. Left
/// unset it is the target, so a worker spills as soon as the reserve is at
/// all short. Set lower, the reserve is left to run down to it and is then
/// refilled towards the target in one go, which gives each worker a batch of
/// buffers to spill together.
fn get_reclaimable_slots_start_percent() -> usize {
    static START_PERCENT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let target = get_reclaimable_slots_target_percent();
    crate::env::get_tunable(
        &START_PERCENT,
        "PIVOT_SPILL_START_RESERVE_PERCENT",
        target,
        |percent| percent <= target,
    )
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
    compressed_cache: Arc<CompressedCache>,
    decompressed_cache: Arc<DecompressedCache>,
    clock: Arc<Clock>,
    dirty_pool_factory: PoolFactory,
    zeroed_pool_factory: PoolFactory,

    buffers_spilled: Arc<[AtomicUsize]>,
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
        let buffers_spilled: Arc<[AtomicUsize]> = (0..topology.node_count)
            .map(|_| AtomicUsize::new(0))
            .collect();

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
                buffers_spilled: buffers_spilled.clone(),
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
            evictors: RefCell::new(HashMap::new()),
            buffers_spilled: self.buffers_spilled,
            own_buffers_spilled: Cell::new(0),
            io_requester: OnceCell::new(),
            spill_writes_in_flight: RefCell::new(HashMap::new()),
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
    /// Each dataflow's operators that keep ring slots outside the caches and can
    /// give them back, keyed by dataflow id. Consulted only after both caches
    /// have swept their whole region without finding a slot nobody holds. A
    /// `RefCell` because the context is reached through a shared reference; it
    /// is per worker thread, so the borrow never contends.
    evictors: RefCell<HashMap<Identifier, DataflowEvict>>,
    /// Per node, the slots its operators have handed over for spilling and
    /// still hold, whether the write is in flight or has landed. Shared by the
    /// node's workers, so a slot one worker is already spilling is not asked
    /// for again by another.
    buffers_spilled: Arc<[AtomicUsize]>,
    /// The slots this worker itself has handed over for spilling and still
    /// holds. Only this worker's eviction can take those back, so they are
    /// what tells one worker's need to spill apart from its siblings'.
    own_buffers_spilled: Cell<usize>,
    /// The worker's IO requester, attached once the worker has built it, so
    /// that eviction can wait for spill writes to land when nothing else can
    /// be reclaimed. Shared with the worker, which drives it.
    io_requester: OnceCell<Rc<RefCell<IORequester>>>,
    /// Every spill write in flight on this worker, pointing at the in-flight
    /// count of each spilled buffer it carries rows of. A count that reaches
    /// zero tells the buffer's evictor the whole buffer is on disk. Keyed by
    /// the issuing dataflow as well as the write id, because write ids are
    /// drawn per dataflow and two dataflows spilling on one worker repeat them.
    spill_writes_in_flight: RefCell<HashMap<(Identifier, WriteRequestId), Vec<Arc<AtomicUsize>>>>,
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

    /// Read what every block of the ring currently holds, in ring order.
    ///
    /// The whole ring, from whichever worker asks: a worker may only *acquire*
    /// and evict blocks in its own node's region, but reading crosses regions
    /// freely, and reading is all this does.
    pub fn read_all_blocks(&self) -> Vec<MemoryBlockStatus> {
        (0..self.ring.len())
            .map(|slot| read_block_status(&self.ring, &self.clock, &self.layout, slot))
            .collect()
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
            self.push_free_idx(idx, false);
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
        // Held out of the pools during the loop so the retry pops a *different*
        // index instead of spinning on the locked one at the LIFO top.
        let mut contended = ContendedIndices::new(self);
        loop {
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
                    Some(buffer) => return buffer,
                    None => {
                        // Try again with a fresh idx, we don't want to start
                        // evicting yet.
                        contended.set_aside(idx);
                        continue;
                    }
                }
            }

            // Pool empty: one last attempt at the set-aside indices (their
            // contenders hold slots only transiently), then hand the rest back
            // so peers can pop them while this worker is busy evicting.
            if let Some(buffer) = contended.pop_writable() {
                return buffer;
            }
            contended.release();

            // Evict via the shared clock, which takes from whichever tier is
            // over its target share (decompressed, in the common case of a
            // large decompressed working set).
            return self.evict();
        }
    }

    /// Evict a ring slot via the shared CLOCK and return it writable. Each
    /// iteration re-picks which tier to take from - compressed while its share
    /// of cached slots exceeds the target percentage, decompressed otherwise
    /// (see [`Clock::preferred_victim_tier`]) - advances that tier's hand, and
    /// routes the victim to its cache's reclaim.
    ///
    /// A tier that owns no slots in this region is not swept at all, and when
    /// neither tier owns any the registered [`Evict`] operators are asked at
    /// once. When both caches have swept their region and every slot they hold
    /// is pinned, the operators are asked before the caches are swept again.
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
        // A full drain is the most revolutions a hand needs to reach any
        // victim in its tier: one per life a slot can hold, plus the revolution
        // that finds it at zero. A hand that swept that much without reclaiming
        // anything has nothing to give, so the other tier gets a full drain of
        // its own, regardless of share, and the two keep alternating in whole
        // drains. Anything shorter for the other hand cannot reach a slot that
        // still holds lives.
        //
        // A revolution that aged nothing cannot produce a victim by going on:
        // the tier has no lives left to drain in this region, so its zero-life
        // slots (all held by readers or writers, since the revolution ended
        // without a victim) or its absence from the region will look the same
        // next time round. Such a revolution hands over at once instead of
        // repeating itself for the rest of the drain.
        let drain = (max_lives + 1) * ring_len;
        let mut ticks_without_success: u64 = 0;
        let mut aged_this_revolution: u64 = 0;
        let mut sweeping_other_tier = false;
        // How many times a tier's sweep ended with nothing to give. Once both
        // tiers have, the caches hold nothing unpinned right now, and the
        // operators keeping slots outside the caches are the only remaining
        // source.
        let mut tiers_exhausted: u32 = 0;
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

            // The owner counts are shared with peer workers, so a slot may be
            // bound between reading them and sweeping. Skipping an empty tier is
            // still safe: at worst such a slot waits one revolution to be found.
            let compressed_cached = self.clock.owned_in_region(Owner::Compressed, self.node);
            let decompressed_cached = self.clock.owned_in_region(Owner::Decompressed, self.node);
            if compressed_cached == 0 && decompressed_cached == 0 {
                // Nothing is cached in this region, so the caches have nothing to
                // give back and only the operators can.
                if let Some(buffer) = self.evict_from_dataflow_operators() {
                    return buffer;
                }
                if self.has_spill_writes_in_flight() {
                    self.wait_for_a_spill_write_to_land();
                    iterations = 0;
                }
                continue;
            }

            let mut tier = self.clock.preferred_victim_tier(self.node);
            if sweeping_other_tier {
                tier = match tier {
                    Owner::Compressed => Owner::Decompressed,
                    Owner::Decompressed => Owner::Compressed,
                };
            }
            // An empty tier has nothing to sweep; take from the other one.
            tier = match tier {
                Owner::Compressed if compressed_cached == 0 => Owner::Decompressed,
                Owner::Decompressed if decompressed_cached == 0 => Owner::Compressed,
                tier => tier,
            };

            let step = self.clock.sweep(tier, self.node);
            let reclaimed = match step {
                Sweep::Candidate(slot) => match tier {
                    Owner::Compressed => self.compressed_cache.reclaim(slot),
                    Owner::Decompressed => self.decompressed_cache.reclaim(slot),
                },
                Sweep::Skipped | Sweep::Aged => None,
            };
            if let Some(write_buffer) = reclaimed {
                return write_buffer;
            }
            ticks_without_success += 1;
            if step == Sweep::Aged {
                aged_this_revolution += 1;
            }
            if ticks_without_success.is_multiple_of(ring_len) {
                let drained = ticks_without_success.is_multiple_of(drain);
                let nothing_left_to_drain = aged_this_revolution == 0;
                if drained || nothing_left_to_drain {
                    sweeping_other_tier = !sweeping_other_tier;
                    tiers_exhausted += 1;
                    if tiers_exhausted >= 2 {
                        if let Some(buffer) = self.evict_from_dataflow_operators() {
                            return buffer;
                        }
                        if self.has_spill_writes_in_flight() {
                            self.wait_for_a_spill_write_to_land();
                            iterations = 0;
                        }
                    }
                }
                aged_this_revolution = 0;
            }
        }
    }

    /// How many slots this worker's operators should spill now: this worker's
    /// share of the node's shortfall of reclaimable slots below the target (see
    /// [`RECLAIMABLE_SLOTS_TARGET_PERCENT`]). The shortfall is split evenly
    /// across the node's workers, rounded up, because every worker on the node
    /// sees the same shortfall and would otherwise each ask for all of it.
    ///
    /// Reclaimable means free, owned by a cache, or spilled. Cached slots count
    /// because the sweep can take them back at will, and spilled slots count
    /// because eviction can take them once their write lands, so the slots
    /// that are none of these are the ones operators still hold.
    ///
    /// A spilled slot can be taken back only by the worker that spilled it, so
    /// this worker's own spilled slots are added on top of the node's. A worker
    /// that holds many sees no shortfall and one that holds few sees it first,
    /// which sends each new spill to the worker with the fewest.
    ///
    /// Nothing is asked for until the reclaimable slots are below where
    /// spilling starts (see [`get_reclaimable_slots_start_percent`]), which may
    /// be lower than the target the shortfall is then measured against. The counts
    /// are read at slightly different instants while peers allocate and
    /// release, so the result is approximate by a few slots, which a watermark
    /// tolerates.
    pub fn slots_to_spill(&self) -> usize {
        let free = self.dirty_pool.free_slots() + self.zeroed_pool.free_slots();
        let cached = self.clock.owned_in_region(Owner::Compressed, self.node)
            + self.clock.owned_in_region(Owner::Decompressed, self.node);
        let spilled = self.buffers_spilled[self.node].load(Ordering::Relaxed);
        let slots = self.layout.node_slots(self.node).len();
        let start = slots * get_reclaimable_slots_start_percent() / 100;
        let target = slots * get_reclaimable_slots_target_percent() / 100;
        let reclaimable = free + cached + spilled + self.own_buffers_spilled.get();
        if reclaimable >= start {
            return 0;
        }
        (target - reclaimable).div_ceil(self.layout.topology().workers_per_node)
    }

    /// Count `count` more of this node's slots as spilled, so that
    /// [`slots_to_spill`](Self::slots_to_spill) stops asking for them.
    pub fn record_buffers_spilled(&self, count: usize) {
        self.buffers_spilled[self.node].fetch_add(count, Ordering::Relaxed);
        self.own_buffers_spilled
            .set(self.own_buffers_spilled.get() + count);
    }

    /// Note that `count` spilled slots have been released, so they are no
    /// longer counted as spilled: once free, the pools count them instead.
    pub fn record_spilled_buffers_released(&self, count: usize) {
        let before = self.buffers_spilled[self.node].fetch_sub(count, Ordering::Relaxed);
        assert!(
            before >= count,
            "released more spilled buffers than were spilled"
        );
        self.own_buffers_spilled
            .set(self.own_buffers_spilled.get() - count);
    }

    /// Make `evictors` reachable from this worker's eviction for as long as the
    /// dataflow runs; its `DataFlow` unregisters them when it drops.
    pub fn register_dataflow_evictors(&self, dataflow_id: Identifier, evictors: DataflowEvict) {
        let replaced = self.evictors.borrow_mut().insert(dataflow_id, evictors);
        assert!(
            replaced.is_none(),
            "dataflow {dataflow_id} registered its evictors twice"
        );
    }

    /// Forget the evictor of one operator of `dataflow_id`, which has finished
    /// and released whatever it kept.
    pub fn remove_operator_evictor(&self, dataflow_id: Identifier, operator_idx: usize) {
        self.evictors
            .borrow_mut()
            .get_mut(&dataflow_id)
            .expect("an operator's evictor is removed from a dataflow that registered evictors")
            .remove_operator(operator_idx);
    }

    /// Forget the evictors registered under `dataflow_id`.
    pub fn unregister_dataflow_evictors(&self, dataflow_id: Identifier) {
        let removed = self.evictors.borrow_mut().remove(&dataflow_id);
        assert!(
            removed.is_some(),
            "dataflow {dataflow_id} never registered evictors"
        );
        // A dataflow that failed takes its queued IO with it, so a spill write
        // of its that was never submitted never lands. Left here, it would
        // keep eviction waiting for a write that is not coming.
        self.spill_writes_in_flight
            .borrow_mut()
            .retain(|(dataflow, _), _| *dataflow != dataflow_id);
    }

    /// Share the worker's IO requester with this context, once, so that
    /// eviction can wait on it (see [`evict`](Self::evict)).
    pub fn attach_io_requester(&self, requester: Rc<RefCell<IORequester>>) {
        self.io_requester
            .set(requester)
            .unwrap_or_else(|_| panic!("the worker attached its IO requester twice"));
    }

    /// Note a spill write issued by `dataflow` that carries rows of the
    /// buffers whose in-flight counts are given, so that when it lands each
    /// of them is counted down.
    pub fn record_spill_write_issued(
        &self,
        dataflow: Identifier,
        id: WriteRequestId,
        buffers_writes_in_flight: Vec<Arc<AtomicUsize>>,
    ) {
        let replaced = self
            .spill_writes_in_flight
            .borrow_mut()
            .insert((dataflow, id), buffers_writes_in_flight);
        assert!(
            replaced.is_none(),
            "spill write {id:?} of dataflow {dataflow} issued twice"
        );
    }

    pub fn has_spill_writes_in_flight(&self) -> bool {
        !self.spill_writes_in_flight.borrow().is_empty()
    }

    /// Take the spill writes that `requester` has collected as complete and
    /// count each off its buffer. The worker runs this on every pass over its
    /// completions, and eviction runs it after waiting for one.
    pub fn complete_landed_spill_writes(&self, requester: &mut IORequester) {
        let mut spill_writes_in_flight = self.spill_writes_in_flight.borrow_mut();
        let landed = requester.extract_completions_if(|completion| {
            matches!(completion, Ok(Completion::FsWrite(write)) if write.request.id.is_spill())
        });
        for completion in landed {
            let Ok(Completion::FsWrite(write)) = completion else {
                unreachable!("extracted only spill writes");
            };
            // A write of a dataflow that has since failed is no longer
            // recorded (see `unregister_dataflow_evictors`); its buffer went
            // with the dataflow, so there is nothing left to count it off.
            if let Some(buffers_writes_in_flight) =
                spill_writes_in_flight.remove(&(write.data_flow_id, write.request.id))
            {
                for writes_in_flight in buffers_writes_in_flight {
                    writes_in_flight.fetch_sub(1, Ordering::Release);
                }
            }
        }
    }

    /// Block until one of this worker's spill writes may have landed, and
    /// count the ones that did off their buffers. Every slot this worker could
    /// reclaim is held by a spilled buffer whose write has not landed yet, so
    /// the only way forward is to let the writes finish. Runs inside an
    /// operator's run, so it leaves every other completion for the worker's
    /// next pass.
    ///
    /// A write held back because its device's hardware queue is full is not on
    /// the ring, and the worker is told through its waker when a slot in that
    /// queue frees. The waker reaches only a wait that was published to it, so
    /// the wait is published the way the worker loop publishes its own: a
    /// worker whose writes are all held back would otherwise never wake.
    fn wait_for_a_spill_write_to_land(&self) {
        let requester = self
            .io_requester
            .get()
            .expect("spill writes are in flight only on a worker with an attached IO requester");
        let mut requester = requester
            .try_borrow_mut()
            .expect("eviction does not run from inside the IO requester's own calls");
        let waker = crate::waker::worker_waker();
        let node_local_idx = self
            .layout
            .topology()
            .local_index_of_worker(WORKER_IDX.get());
        // Read before the hardware queues are tried, so a slot that frees
        // after that makes the wait return at once instead of being missed.
        let last_seen = waker.ring_wake_count();
        // Take what the ring already holds, which also puts held-back writes
        // on the device as far as its queue allows.
        if self.count_off_landed_spill_writes(&mut requester) {
            return;
        }
        if waker.begin_ring_wait(node_local_idx, last_seen) {
            let waited = requester.wait();
            waker.end_ring_wait(node_local_idx);
            waited.expect("waiting on the IO ring while evicting");
        }
        self.count_off_landed_spill_writes(&mut requester);
    }

    /// Collect the ring's completions and count the spill writes among them
    /// off their buffers. Returns whether any spill write had landed.
    fn count_off_landed_spill_writes(&self, requester: &mut IORequester) -> bool {
        requester
            .collect_completions()
            .expect("collecting IO completions while evicting");
        let in_flight_before = self.spill_writes_in_flight.borrow().len();
        self.complete_landed_spill_writes(requester);
        self.spill_writes_in_flight.borrow().len() < in_flight_before
    }

    /// Ask each registered dataflow's operators for a slot they can give up and
    /// return the first one offered, or `None` when none has a slot to spare
    /// right now. The borrow is held across the calls, which is fine because an
    /// operator's `evict` only releases memory it already holds and never
    /// reaches back into this registry.
    fn evict_from_dataflow_operators(&self) -> Option<WriteBuffer> {
        self.evictors
            .borrow_mut()
            .values_mut()
            .find_map(DataflowEvict::evict)
    }
}

/// Free-pool indices set aside because their ring slot was transiently locked
/// (e.g. an evictor probing the slot before backing off). The contender never
/// re-pools an index it did not pop, so the popper must return every set-aside
/// index or the slot leaks from every pool. Dropping the holder returns them,
/// keeping the invariant across early returns and unwinds (eviction panics).
struct ContendedIndices<'a> {
    ctx: &'a MemoryContext,
    indices: Vec<usize>,
}

impl<'a> ContendedIndices<'a> {
    fn new(ctx: &'a MemoryContext) -> Self {
        Self {
            ctx,
            indices: Vec::new(),
        }
    }

    fn set_aside(&mut self, idx: usize) {
        self.indices.push(idx);
    }

    /// One more `try_write` attempt at each set-aside index, in case its
    /// contender already backed off. A single pass per index, so a long-held
    /// slot cannot spin the caller.
    fn pop_writable(&mut self) -> Option<WriteBuffer> {
        for i in 0..self.indices.len() {
            if let Some(buffer) = self.ctx.ring().try_write(self.indices[i]) {
                self.indices.swap_remove(i);
                return Some(buffer);
            }
        }
        None
    }

    /// Return every remaining index to its pool now rather than at drop, so
    /// peers can pop the slots while the holder is still alive.
    fn release(&mut self) {
        for idx in self.indices.drain(..) {
            self.ctx
                .push_free_idx(idx, self.ctx.ring().slot_zeroed(idx));
        }
    }
}

impl Drop for ContendedIndices<'_> {
    fn drop(&mut self) {
        self.release();
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
    use crate::memory::evict::{Evict, HandOverEvictor};
    use std::rc::Rc;

    #[test]
    fn evict_turns_to_dataflow_operators_once_the_caches_have_nothing() {
        init_test_free_pool(1);
        let held = memory_ctx().get_write_buffer(false);
        let held_slot = held.slot_idx;
        let log = Rc::new(RefCell::new(Vec::new()));
        let operator: Box<dyn Evict> = Box::new(HandOverEvictor {
            label: "group",
            log: log.clone(),
            buffer: RefCell::new(Some(held)),
        });
        memory_ctx().register_dataflow_evictors(7, DataflowEvict::new(vec![Some(operator)], &[]));

        let evicted = memory_ctx().evict();

        assert_eq!(evicted.slot_idx, held_slot);
        assert_eq!(*log.borrow(), ["group"]);
        memory_ctx().unregister_dataflow_evictors(7);
    }

    #[test]
    fn cached_slots_count_as_reclaimable_for_spilling() {
        // 128 slots, so the target is 25 reclaimable; 10 free plus 15 cached.
        init_test_free_pool(10);
        for idx in 10..25 {
            memory_ctx().clock().bind(idx, Owner::Compressed);
        }

        let shortfall = memory_ctx().slots_to_spill();

        assert_eq!(shortfall, 0);
    }

    #[test]
    fn slots_to_spill_is_the_shortfall_below_the_reclaimable_target_less_what_is_spilled() {
        // 128 slots, so the target is 25 reclaimable; seed 10 free, nothing cached.
        init_test_free_pool(10);

        let shortfall = memory_ctx().slots_to_spill();
        memory_ctx().record_buffers_spilled(shortfall);
        let while_spilled = memory_ctx().slots_to_spill();
        memory_ctx().record_spilled_buffers_released(shortfall);
        let after_release = memory_ctx().slots_to_spill();

        assert_eq!(shortfall, 15);
        assert_eq!(while_spilled, 0);
        assert_eq!(after_release, 15);
    }

    #[test]
    fn a_workers_own_spilled_slots_count_again_toward_its_target() {
        // 128 slots, so the target is 25 reclaimable; 10 free, nothing cached.
        init_test_free_pool(10);
        memory_ctx().record_buffers_spilled(8);

        let shortfall = memory_ctx().slots_to_spill();

        assert_eq!(shortfall, 0);
        memory_ctx().record_spilled_buffers_released(8);
    }

    #[test]
    fn slots_to_spill_is_this_workers_share_of_the_nodes_shortfall_rounded_up() {
        // Four workers on one node of 128 slots; 10 free leaves a shortfall of 15.
        WORKER_IDX.set(0);
        crate::worker::NUM_WORKERS.set(4);
        crate::waker::install_test_worker_waker();
        let mut factories = MemoryContextFactory::create_many(4, 128);
        // Building a context waits until every worker on the node has built one.
        let siblings: Vec<_> = factories
            .drain(1..)
            .enumerate()
            .map(|(i, factory)| {
                std::thread::spawn(move || {
                    WORKER_IDX.set(i + 1);
                    drop(factory.create_memory_ctx());
                })
            })
            .collect();
        init_memory_context(factories.pop().unwrap().create_memory_ctx());
        for sibling in siblings {
            sibling.join().unwrap();
        }
        for i in 0..10 {
            memory_ctx().push_free_idx(i, false);
        }

        let share = memory_ctx().slots_to_spill();

        assert_eq!(share, 4);
    }

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
        // Pop the zeroed pool directly: the contended slot's ring flag is still
        // `zeroed`, so the push-back routes it there, while a slot the loop
        // never popped would have stayed in the dirty pool. Popping either pool
        // here would pass even if the contended branch never ran.
        let repooled = memory_ctx().zeroed_pool.pop(false);

        // Assert: the free slot is handed out and the contended one is back in
        // a pool rather than leaked.
        assert_eq!(handed_out.slot_idx, 0);
        assert_eq!(repooled, Some(1));
    }

    #[test]
    fn get_write_buffer_repools_contended_slots_when_it_panics() {
        // Setup: the only pooled slot is transiently locked by an evictor
        // probe, so get_write_buffer runs out of pool and panics on evict.
        init_test_free_pool(1);
        let probe = memory_ctx().ring().try_write(0).unwrap();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            memory_ctx().get_write_buffer(false)
        }));
        probe.release_in_place();

        // Assert: the panic unwound without leaking the set-aside slot.
        assert!(result.is_err());
        assert_eq!(memory_ctx().pop_free_idx(false), Some(0));
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
