//! A CLOCK eviction policy shared by the compressed
//! [`CompressedCache`](super::compressed_cache::CompressedCache) and the
//! [`DecompressedCache`](super::decompressed_cache::DecompressedCache), indexed
//! by ring slot.
//!
//! ## Two hands per region, one target share
//!
//! Each tier has its own hand sweeping the slot array. A hand only acts on
//! slots of its own tier - anything else (free, or the other tier) is stepped
//! over without touching its counter - so a tier's blocks age only while that
//! tier is being evicted from. Which hand moves is the evictor's choice
//! ([`evict`](super::context::MemoryContext) compares the compressed tier's
//! share of its region's cached slots against a target percentage), making
//! the compressed/decompressed balance an explicit knob instead of an
//! emergent property of per-tier lifetimes. The share is judged per region
//! because the hands are: a tier over target ring-wide but held entirely in
//! another node's region has nothing this region's hand could take.
//!
//! Slot state (`owner`, `refs`) is global: any worker's hit may `touch` any
//! slot. The sweep, however, is per NUMA node: each node has its own pair of
//! hands covering only that node's slot region, so eviction only ever frees
//! slots the evicting worker may acquire (node-local memory placement, see
//! [`RingLayout`](super::RingLayout)).
//!
//! Within a tier, CLOCK rules apply: each slot has a small "lives" counter,
//! the hand decrements it on each pass, and a slot at zero is the victim. A
//! use ADDS the slot's per-touch bump (default 1, capped at [`MAX_LIVES`]) and
//! never removes lives, so use frequency separates a durable working set from
//! one-shot speculative inserts and eviction churn lands on the latter.
//!
//! ## Cross-tier reinforcement
//!
//! The same file bytes often live in both tiers at once - a compressed run and
//! the decompressed pages made from it. A read only goes to disk when *both*
//! copies are gone, so when either cache evicts a block,
//! [`reinforce`](Clock::reinforce) raises the surviving copy's per-touch bump
//! to [`REINFORCE_BUMP`]. Under pressure heavy enough that hand passes outpace
//! once-per-scan touches, ordinary blocks sink and recycle while a still-read
//! last copy climbs - the tiers settle into complements instead of dying
//! together (which is what forces disk IO).
//!
//! Everything is lock-free; a little racing between the atomics is fine because
//! eviction only needs to be approximate.

use crate::env::get_env_var_with_default;
use std::sync::LazyLock;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;

/// Hard ceiling on any slot's accumulated lives (env `PIVOT_MAX_LIVES`,
/// default 16).
static MAX_LIVES: LazyLock<u8> = LazyLock::new(|| get_env_var_with_default("PIVOT_MAX_LIVES", 16));

/// The per-touch bump of a slot whose same-bytes twin in the other tier has
/// died (env `PIVOT_REINFORCE_BUMP`, default 6). Ordinary slots gain 1 life
/// per touch; a last-copy slot gains this much, so under eviction pressure
/// that drains everyone, a block that is still being read AND has no other
/// copy climbs while equally-hot backed blocks sink - the hand then feeds on
/// blocks whose eviction costs a re-decompress, never a disk read.
static REINFORCE_BUMP: LazyLock<u8> =
    LazyLock::new(|| get_env_var_with_default("PIVOT_REINFORCE_BUMP", 6));

/// Target share of cached slots the compressed tier should occupy, in percent
/// (env `PIVOT_COMPRESSED_CACHE_PCT`, default 30). The evictor evicts from the
/// compressed tier only while its share exceeds this.
static COMPRESSED_SHARE_PCT: LazyLock<usize> =
    LazyLock::new(|| get_env_var_with_default("PIVOT_COMPRESSED_CACHE_PCT", 30));

/// The cache owning a ring slot; a free slot has no owner (`None`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Owner {
    Compressed = 1,
    Decompressed = 2,
}

const FREE: u8 = 0;

/// What one step of a tier's hand did (see [`Clock::sweep`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sweep {
    /// The slot is free or belongs to the other tier: stepped over.
    Skipped,
    /// The slot belongs to the tier and spent one of its lives.
    Aged,
    /// The slot belongs to the tier and had no life left: the victim.
    Candidate(usize),
}

impl Owner {
    fn decode(byte: u8) -> Option<Owner> {
        match byte {
            1 => Some(Owner::Compressed),
            2 => Some(Owner::Decompressed),
            _ => None,
        }
    }
}

struct Slot {
    owner: AtomicU8,
    refs: AtomicU8,
    /// Lives gained per touch: 1, or [`REINFORCE_BUMP`] once the slot's
    /// other-tier twin has died.
    bump: AtomicU8,
}

/// One CLOCK over every ring slot, swept by two per-tier hands per node
/// region.
pub struct Clock {
    slots: Box<[Slot]>,
    /// One compressed-tier sweep hand per node region; region `r` covers
    /// `r * slots_per_region..(r + 1) * slots_per_region`.
    compressed_hands: Box<[AtomicUsize]>,
    /// The decompressed tier's counterpart of `compressed_hands`.
    decompressed_hands: Box<[AtomicUsize]>,
    slots_per_region: usize,
    /// Slots currently bound [`Owner::Compressed`] / [`Owner::Decompressed`]
    /// in each region, maintained by [`bind`](Self::bind) and
    /// [`release`](Self::release), the two functions every ownership transition
    /// goes through. Per region because the hands are: a region's evictor can
    /// only take from the tiers as they stand in its own region, so that is the
    /// balance it must judge.
    compressed_counts: Box<[AtomicUsize]>,
    decompressed_counts: Box<[AtomicUsize]>,
}

impl Clock {
    /// A single-region clock over `len` slots (single-node machines, tests).
    pub fn new(len: usize) -> Self {
        Self::with_regions(1, len)
    }

    /// A clock over `region_count * slots_per_region` slots with one sweep
    /// hand per tier per region.
    pub fn with_regions(region_count: usize, slots_per_region: usize) -> Self {
        Self {
            slots: (0..region_count * slots_per_region)
                .map(|_| Slot {
                    owner: AtomicU8::new(FREE),
                    refs: AtomicU8::new(0),
                    bump: AtomicU8::new(1),
                })
                .collect(),
            compressed_hands: (0..region_count).map(|_| AtomicUsize::new(0)).collect(),
            decompressed_hands: (0..region_count).map(|_| AtomicUsize::new(0)).collect(),
            slots_per_region,
            compressed_counts: (0..region_count).map(|_| AtomicUsize::new(0)).collect(),
            decompressed_counts: (0..region_count).map(|_| AtomicUsize::new(0)).collect(),
        }
    }

    fn counts_of(&self, owner: Owner) -> &[AtomicUsize] {
        match owner {
            Owner::Compressed => &self.compressed_counts,
            Owner::Decompressed => &self.decompressed_counts,
        }
    }

    fn region_of(&self, slot: usize) -> usize {
        slot / self.slots_per_region
    }

    /// Give a free slot to `owner` with a single life and an ordinary bump: a
    /// fresh block is speculative until re-use accumulates more.
    pub fn bind(&self, slot: usize, owner: Owner) {
        self.slots[slot].refs.store(1, Relaxed);
        self.slots[slot].bump.store(1, Relaxed);
        self.slots[slot].owner.store(owner as u8, Relaxed);
        self.counts_of(owner)[self.region_of(slot)].fetch_add(1, Relaxed);
    }

    /// Mark a slot used again: gain its bump's worth of lives, up to
    /// [`MAX_LIVES`]. Every use restores what hand passes took, so a block
    /// that keeps being read keeps climbing; how fast is the slot's bump, the
    /// lever [`reinforce`](Self::reinforce) pulls for last-copy blocks.
    pub fn touch(&self, slot: usize) {
        let bump = self.slots[slot].bump.load(Relaxed);
        let _ = self.slots[slot]
            .refs
            .fetch_update(Relaxed, Relaxed, |refs| {
                (refs < *MAX_LIVES).then(|| refs.saturating_add(bump).min(*MAX_LIVES))
            });
    }

    /// The slot's other-tier copy of the same bytes died: raise its per-touch
    /// bump to [`REINFORCE_BUMP`] (and grant one bump immediately). While the
    /// block keeps being read, its lives now climb faster than eviction
    /// pressure drains them, so the last in-memory copy of hot bytes
    /// effectively stops being a victim; once it goes cold, it drains like
    /// anything else. Fire-and-forget - if the slot was concurrently rebound,
    /// [`bind`](Self::bind) resets the bump, and the stray lives merely delay
    /// an unrelated block, an imprecision this CLOCK tolerates everywhere.
    pub fn reinforce(&self, slot: usize) {
        self.slots[slot].bump.store(*REINFORCE_BUMP, Relaxed);
        self.touch(slot);
    }

    /// Return a slot to the free pool: no owner, never a victim.
    pub fn release(&self, slot: usize) {
        if let Some(owner) = self.owner(slot) {
            self.counts_of(owner)[self.region_of(slot)].fetch_sub(1, Relaxed);
        }
        self.slots[slot].owner.store(FREE, Relaxed);
    }

    pub fn owner(&self, slot: usize) -> Option<Owner> {
        Owner::decode(self.slots[slot].owner.load(Relaxed))
    }

    /// Slots currently bound to `tier`, over the whole ring.
    pub fn owned(&self, tier: Owner) -> usize {
        self.counts_of(tier)
            .iter()
            .map(|count| count.load(Relaxed))
            .sum()
    }

    /// Slots currently bound to `tier` in `region`.
    pub fn owned_in_region(&self, tier: Owner, region: usize) -> usize {
        self.counts_of(tier)[region].load(Relaxed)
    }

    /// The most sweeps of its hand any slot can survive (the refs ceiling).
    /// Sizes the evictor's patience thresholds.
    pub fn max_lives(&self) -> u8 {
        *MAX_LIVES
    }

    /// The tier eviction in `region` should take from next: compressed while
    /// its share of the region's cached slots exceeds the target percentage,
    /// decompressed otherwise (strictly greater, so at the boundary the
    /// decompressed tier gives way - no oscillation). The zero cases fall out
    /// of the arithmetic: an empty compressed tier is never above target, and
    /// an empty decompressed tier puts compressed at 100%. Judged per region
    /// because that is all the region's hands can act on: a tier over target
    /// ring-wide but absent here has nothing to give here.
    pub fn preferred_victim_tier(&self, region: usize) -> Owner {
        let compressed = self.owned_in_region(Owner::Compressed, region);
        let decompressed = self.owned_in_region(Owner::Decompressed, region);
        if compressed * 100 > *COMPRESSED_SHARE_PCT * (compressed + decompressed) {
            Owner::Compressed
        } else {
            Owner::Decompressed
        }
    }

    /// How many steps `tier`'s hand in `region` has taken in total - test-only
    /// visibility for bounding an eviction's sweep.
    #[cfg(test)]
    pub fn hand_position(&self, tier: Owner, region: usize) -> usize {
        let hands = match tier {
            Owner::Compressed => &self.compressed_hands,
            Owner::Decompressed => &self.decompressed_hands,
        };
        hands[region].load(Relaxed)
    }

    /// The slot's current remaining lives - test-only visibility for asserting
    /// reinforcement and aging behavior.
    #[cfg(test)]
    pub fn refs(&self, slot: usize) -> u8 {
        self.slots[slot].refs.load(Relaxed)
    }

    /// The configured reinforce bump - test-only, so assertions track the
    /// default without re-stating it.
    #[cfg(test)]
    pub fn reinforce_bump(&self) -> u8 {
        *REINFORCE_BUMP
    }

    /// Advance `tier`'s hand for `region` one slot and age it. Returns the
    /// slot if it is an eviction candidate this pass (owned by `tier`, counter
    /// at zero); `None` when the slot belongs to anything else (stepped over,
    /// untouched) or still had a life to spend. Only sweeps `region`'s slots,
    /// so the caller can never be handed another node's memory.
    pub fn advance(&self, tier: Owner, region: usize) -> Option<usize> {
        match self.sweep(tier, region) {
            Sweep::Candidate(slot) => Some(slot),
            Sweep::Skipped | Sweep::Aged => None,
        }
    }

    /// [`advance`](Self::advance), telling apart a step that aged one of the
    /// tier's slots from one that stepped over another tier's or a free slot:
    /// the evictor reads the difference as whether the tier can still produce
    /// a victim by sweeping on.
    pub fn sweep(&self, tier: Owner, region: usize) -> Sweep {
        let hands = match tier {
            Owner::Compressed => &self.compressed_hands,
            Owner::Decompressed => &self.decompressed_hands,
        };
        let offset = hands[region].fetch_add(1, Relaxed) % self.slots_per_region;
        let slot = region * self.slots_per_region + offset;
        if self.owner(slot) != Some(tier) {
            return Sweep::Skipped;
        }
        // Decrement while positive; an already-zero counter is the victim.
        let had_second_chance = self.slots[slot]
            .refs
            .fetch_update(Relaxed, Relaxed, |refs| (refs > 0).then(|| refs - 1))
            .is_ok();
        if had_second_chance {
            Sweep::Aged
        } else {
            Sweep::Candidate(slot)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sweep `tier`'s region-0 hand until it reports a victim, returning that
    /// slot.
    fn evict(clock: &Clock, tier: Owner) -> usize {
        loop {
            if let Some(slot) = clock.advance(tier, 0) {
                return slot;
            }
        }
    }

    #[test]
    fn a_free_slot_is_never_a_victim() {
        let clock = Clock::new(1);

        assert_eq!(clock.advance(Owner::Compressed, 0), None);
        assert_eq!(clock.advance(Owner::Decompressed, 0), None);
    }

    #[test]
    fn a_hand_only_evicts_its_own_tier() {
        let clock = Clock::new(2);
        clock.bind(0, Owner::Compressed);
        clock.bind(1, Owner::Decompressed);

        assert_eq!(evict(&clock, Owner::Decompressed), 1);
        assert_eq!(evict(&clock, Owner::Compressed), 0);
    }

    #[test]
    fn a_hand_does_not_age_the_other_tier() {
        let clock = Clock::new(2);
        clock.bind(0, Owner::Compressed);
        clock.bind(1, Owner::Decompressed);

        for _ in 0..10 {
            clock.advance(Owner::Decompressed, 0);
        }

        assert_eq!(clock.refs(0), 1, "aged by the other tier's hand");
    }

    #[test]
    fn the_victim_tier_is_judged_by_the_regions_own_share() {
        let clock = Clock::with_regions(2, 4);
        for slot in 0..3 {
            clock.bind(slot, Owner::Compressed); // region 0: 3 compressed, 1 decompressed
        }
        clock.bind(3, Owner::Decompressed);
        clock.bind(4, Owner::Compressed); // region 1: 1 compressed, 3 decompressed
        for slot in 5..8 {
            clock.bind(slot, Owner::Decompressed);
        }

        assert_eq!(
            clock.owned(Owner::Compressed),
            4,
            "ring-wide the tiers are level"
        );
        assert_eq!(clock.preferred_victim_tier(0), Owner::Compressed);
        assert_eq!(clock.preferred_victim_tier(1), Owner::Decompressed);
    }

    #[test]
    fn binding_counts_a_slot_toward_its_tier() {
        let clock = Clock::new(2);

        clock.bind(0, Owner::Compressed);
        clock.bind(1, Owner::Decompressed);

        assert_eq!(clock.owned(Owner::Compressed), 1);
        assert_eq!(clock.owned(Owner::Decompressed), 1);
    }

    #[test]
    fn releasing_uncounts_a_slot_from_its_tier() {
        let clock = Clock::new(2);
        clock.bind(0, Owner::Compressed);
        clock.bind(1, Owner::Decompressed);

        clock.release(0);
        clock.release(1);

        assert_eq!(clock.owned(Owner::Compressed), 0);
        assert_eq!(clock.owned(Owner::Decompressed), 0);
    }

    #[test]
    fn reinforce_grants_extra_sweeps_immediately() {
        let clock = Clock::new(1);
        clock.bind(0, Owner::Decompressed);
        clock.advance(Owner::Decompressed, 0); // its one insert-time life, spent

        clock.reinforce(0); // one immediate REINFORCE_BUMP worth of lives

        for _ in 0..clock.reinforce_bump() {
            assert_eq!(clock.advance(Owner::Decompressed, 0), None);
        }
        assert_eq!(clock.advance(Owner::Decompressed, 0), Some(0)); // now the victim
    }

    #[test]
    fn touches_accumulate_one_life_up_to_the_max() {
        let clock = Clock::new(1);
        clock.bind(0, Owner::Decompressed); // 1 life

        for _ in 0..20 {
            clock.touch(0);
        }

        assert_eq!(clock.refs(0), 16);
    }

    #[test]
    fn a_reinforced_slot_climbs_faster_than_an_ordinary_one() {
        let clock = Clock::new(2);
        clock.bind(0, Owner::Decompressed);
        clock.bind(1, Owner::Decompressed);
        clock.reinforce(0); // slot 0 is now the last copy of its bytes

        // One read per scan pass against heavier per-pass hand pressure.
        for _ in 0..4 {
            clock.touch(0);
            clock.touch(1);
            clock.advance(Owner::Decompressed, 0);
            clock.advance(Owner::Decompressed, 0);
        }

        assert!(clock.refs(0) > 8, "last copy should climb");
        assert!(clock.refs(1) <= 2, "ordinary block must not climb");
    }

    #[test]
    fn rebinding_resets_the_reinforced_bump() {
        let clock = Clock::new(1);
        clock.bind(0, Owner::Decompressed);
        clock.reinforce(0);

        clock.release(0);
        clock.bind(0, Owner::Decompressed);
        clock.touch(0);

        assert_eq!(clock.refs(0), 2, "fresh block must be back to bump 1");
    }

    #[test]
    fn decompressed_is_preferred_while_compressed_is_under_its_target_share() {
        let clock = Clock::new(16);

        // 1 compressed / 11 total = 9%, under the 10% default target.
        clock.bind(0, Owner::Compressed);
        for slot in 1..11 {
            clock.bind(slot, Owner::Decompressed);
        }

        assert_eq!(clock.preferred_victim_tier(0), Owner::Decompressed);
    }

    #[test]
    fn compressed_is_preferred_above_its_target_share() {
        let clock = Clock::new(8);

        // 3 compressed / 6 total = 50%, above the target share.
        for slot in 0..3 {
            clock.bind(slot, Owner::Compressed);
        }
        for slot in 3..6 {
            clock.bind(slot, Owner::Decompressed);
        }

        assert_eq!(clock.preferred_victim_tier(0), Owner::Compressed);
    }

    #[test]
    fn an_empty_tier_is_never_preferred() {
        let clock = Clock::new(2);

        clock.bind(0, Owner::Decompressed);
        assert_eq!(clock.preferred_victim_tier(0), Owner::Decompressed);

        clock.release(0);
        clock.bind(1, Owner::Compressed);
        assert_eq!(clock.preferred_victim_tier(0), Owner::Compressed);
    }

    #[test]
    fn a_released_slot_stops_being_a_victim() {
        let clock = Clock::new(1);
        clock.bind(0, Owner::Decompressed);
        clock.release(0);

        assert_eq!(clock.advance(Owner::Decompressed, 0), None);
    }

    #[test]
    fn a_regions_sweep_never_touches_another_regions_slots() {
        let clock = Clock::with_regions(2, 2);
        clock.bind(0, Owner::Decompressed);
        clock.bind(3, Owner::Decompressed);

        // Sweeping region 1 repeatedly evicts only its own slot; region 0's
        // victim stays for region 0's own sweep.
        for _ in 0..8 {
            if let Some(slot) = clock.advance(Owner::Decompressed, 1) {
                assert!(slot >= 2);
            }
        }
        assert_eq!(evict(&clock, Owner::Decompressed), 0);
    }
}
