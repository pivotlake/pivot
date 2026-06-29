//! A CLOCK eviction policy shared by the compressed
//! [`CompressedCache`](super::compressed_cache::CompressedCache) and the
//! [`DecompressedCache`](super::decompressed_cache::DecompressedCache), indexed
//! by ring slot.
//!
//! Each slot has an `owner` and a small "recently used" counter. A hit refreshes
//! the counter to the owner's tier max - **2 for compressed, 1 for
//! decompressed** - and the sweep decrements it, evicting at zero. So
//! decompressed slots age out twice as fast (given up first under pressure), yet
//! a just-used decompressed page still outlives a cold compressed slot.
//!
//! Everything is lock-free; a little racing between the two atomics is fine
//! because eviction only needs to be approximate.

use std::sync::atomic::AtomicU8;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;

/// Which cache (if any) owns a ring slot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Owner {
    Free,
    Compressed,
    Decompressed,
}

impl Owner {
    /// How many sweeps a freshly-used slot of this owner survives.
    fn lives(self) -> u8 {
        match self {
            Owner::Free => 0,
            Owner::Compressed => 2,
            Owner::Decompressed => 1,
        }
    }

    fn decode(byte: u8) -> Owner {
        match byte {
            1 => Owner::Compressed,
            2 => Owner::Decompressed,
            _ => Owner::Free,
        }
    }
}

struct Slot {
    owner: AtomicU8,
    refs: AtomicU8,
}

/// One CLOCK over every ring slot, shared by both caches.
pub struct Clock {
    slots: Box<[Slot]>,
    hand: AtomicUsize,
}

impl Clock {
    pub fn new(len: usize) -> Self {
        Self {
            slots: (0..len)
                .map(|_| Slot {
                    owner: AtomicU8::new(Owner::Free as u8),
                    refs: AtomicU8::new(0),
                })
                .collect(),
            hand: AtomicUsize::new(0),
        }
    }

    /// Give a free slot to `owner` and mark it freshly used.
    pub fn bind(&self, slot: usize, owner: Owner) {
        self.slots[slot].refs.store(owner.lives(), Relaxed);
        self.slots[slot].owner.store(owner as u8, Relaxed);
    }

    /// Mark a slot used again, refreshing its counter to its tier max.
    pub fn touch(&self, slot: usize) {
        self.slots[slot]
            .refs
            .store(self.owner(slot).lives(), Relaxed);
    }

    /// Return a slot to the free pool: no owner, never a victim.
    pub fn release(&self, slot: usize) {
        self.slots[slot].owner.store(Owner::Free as u8, Relaxed);
    }

    pub fn owner(&self, slot: usize) -> Owner {
        Owner::decode(self.slots[slot].owner.load(Relaxed))
    }

    /// Advance the hand one slot and age it. Returns the slot index plus its
    /// owner *if* the slot is an eviction candidate this pass (owned, and its
    /// counter has reached zero); `None` when it is free or still had a second
    /// chance (now spent).
    pub fn advance(&self) -> (usize, Option<Owner>) {
        let slot = self.hand.fetch_add(1, Relaxed) % self.slots.len();
        let owner = self.owner(slot);
        if owner == Owner::Free {
            return (slot, None);
        }
        // Decrement while positive; an already-zero counter is the victim.
        let had_second_chance = self.slots[slot]
            .refs
            .fetch_update(Relaxed, Relaxed, |r| (r > 0).then(|| r - 1))
            .is_ok();
        (slot, (!had_second_chance).then_some(owner))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sweep the hand until it reports a victim, returning that slot.
    fn evict(clock: &Clock) -> usize {
        loop {
            if let (slot, Some(_)) = clock.advance() {
                return slot;
            }
        }
    }

    #[test]
    fn a_free_slot_is_never_a_victim() {
        let clock = Clock::new(1);

        assert_eq!(clock.advance(), (0, None));
    }

    #[test]
    fn a_compressed_slot_survives_two_sweeps_a_decompressed_one() {
        let clock = Clock::new(2);
        clock.bind(0, Owner::Compressed);
        clock.bind(1, Owner::Decompressed);

        // Decompressed (slot 1) reaches zero first: it is the earlier victim.
        assert_eq!(evict(&clock), 1);
    }

    #[test]
    fn touch_refreshes_a_slots_second_chance() {
        let clock = Clock::new(1);
        clock.bind(0, Owner::Decompressed);

        assert_eq!(clock.advance(), (0, None)); // ref 1 -> 0, life spent
        clock.touch(0); // refreshed back to 1
        assert_eq!(clock.advance(), (0, None)); // ref 1 -> 0 again, still spared
        assert_eq!(clock.advance(), (0, Some(Owner::Decompressed))); // now evictable
    }

    #[test]
    fn a_released_slot_stops_being_a_victim() {
        let clock = Clock::new(1);
        clock.bind(0, Owner::Decompressed);
        clock.release(0);

        assert_eq!(clock.advance(), (0, None));
    }
}
