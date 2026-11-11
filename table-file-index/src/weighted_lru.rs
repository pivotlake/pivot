//! A map that holds its values within a budget of bytes, letting go of the
//! ones used longest ago.
//!
//! Every value is held with a weight, the bytes it counts for in the
//! budget, which whoever inserts it names. The entries stand in one order
//! of last use. Past the budget the map hands back the entries used longest
//! ago, down to the one used last, which it keeps whatever it weighs: so an
//! entry just inserted is always held.
//!
//! The map lets go of a value the moment it leaves: one removed, evicted or
//! replaced is handed back or dropped then, never kept for later. The slots
//! the entries stand in are let go of as well, once fewer than a quarter of
//! them hold an entry.

use std::hash::Hash;

use hashbrown::{Equivalent, HashMap};

/// The fewest slots a map gives any back from: fewer take too little memory
/// to be worth moving the entries for.
const MIN_COMPACTED_SLOTS: usize = 1024;

/// How many slots a map keeps to an entry before it gives any back: the
/// slots are given back once fewer than one in this many holds an entry.
const SPARSE_SLOTS_PER_ENTRY: usize = 4;

/// Values by key, within a budget of bytes, in the order they were last
/// used.
pub struct WeightedLru<Key, Value> {
    capacity_bytes: u64,
    /// The slot each held entry stands in.
    slots_by_key: HashMap<Key, usize, ahash::RandomState>,
    /// The entries, in a list through numbered slots by when each was last
    /// used: counting an entry as used moves it to the end without anything
    /// being searched.
    slots: Vec<Slot<Key, Value>>,
    /// The slots no entry stands in.
    free_slots: Vec<usize>,
    /// The slot of the entry used longest ago, and of the one used last.
    oldest: Option<usize>,
    newest: Option<usize>,
    /// The sum of the held entries' weights.
    held_bytes: u64,
}

struct Slot<Key, Value> {
    /// The slots of the entries used just before and just after this one.
    earlier: Option<usize>,
    later: Option<usize>,
    /// The entry standing here; `None` for a free slot.
    entry: Option<Entry<Key, Value>>,
}

struct Entry<Key, Value> {
    key: Key,
    value: Value,
    /// The entry's weight, as it was counted into the total.
    weight_bytes: u64,
}

impl<Key: Hash + Eq + Clone, Value> WeightedLru<Key, Value> {
    /// What the map itself spends on one entry, for whoever names the
    /// entries' weights to count in: the slot the entry stands in, which
    /// holds its key, its value and its weight, and the key and slot number
    /// the map finds it by. What a key or a value holds outside itself is
    /// not counted, and neither are the slots no entry stands in or the
    /// room the map by key keeps spare.
    pub const ENTRY_OVERHEAD_BYTES: usize =
        size_of::<Slot<Key, Value>>() + size_of::<(Key, usize)>();

    /// A map that holds values of up to `capacity_bytes` of weight between
    /// them.
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            slots_by_key: HashMap::default(),
            slots: Vec::new(),
            free_slots: Vec::new(),
            oldest: None,
            newest: None,
            held_bytes: 0,
        }
    }

    /// The value held under `key`, counted as used now.
    pub fn get<Asked>(&mut self, key: &Asked) -> Option<&Value>
    where
        Asked: Hash + Equivalent<Key> + ?Sized,
    {
        let position = *self.slots_by_key.get(key)?;
        self.move_to_newest(position);
        Some(&self.read_entry(position).value)
    }

    /// The value held under `key`, which stays where it stood in the order
    /// of last use.
    pub fn peek<Asked>(&self, key: &Asked) -> Option<&Value>
    where
        Asked: Hash + Equivalent<Key> + ?Sized,
    {
        let position = *self.slots_by_key.get(key)?;
        Some(&self.read_entry(position).value)
    }

    /// Count the value held under `key`, if one is, as used now.
    pub fn mark_used<Asked>(&mut self, key: &Asked)
    where
        Asked: Hash + Equivalent<Key> + ?Sized,
    {
        if let Some(&position) = self.slots_by_key.get(key) {
            self.move_to_newest(position);
        }
    }

    /// Hold `value` under `key` for `weight_bytes` of the budget, used now,
    /// dropping the value already held under `key`. Hands back the entries
    /// that left to keep the map within its budget, the one used longest
    /// ago first; the entry just inserted is never one of them.
    pub fn insert(&mut self, key: Key, value: Value, weight_bytes: u64) -> Vec<(Key, Value)> {
        if let Some(position) = self.slots_by_key.remove(&key) {
            self.vacate_slot(position);
        }
        let slot = Slot {
            earlier: None,
            later: None,
            entry: Some(Entry {
                key: key.clone(),
                value,
                weight_bytes,
            }),
        };
        let position = match self.free_slots.pop() {
            Some(position) => {
                self.slots[position] = slot;
                position
            }
            None => {
                self.slots.push(slot);
                self.slots.len() - 1
            }
        };
        self.link_newest(position);
        self.slots_by_key.insert(key, position);
        self.held_bytes += weight_bytes;
        self.evict_past_capacity()
    }

    /// Let go of the value held under `key`, handing it back with its key.
    pub fn remove<Asked>(&mut self, key: &Asked) -> Option<(Key, Value)>
    where
        Asked: Hash + Equivalent<Key> + ?Sized,
    {
        let position = self.slots_by_key.remove(key)?;
        let removed = self.vacate_slot(position);
        self.compact_sparse_slots();
        Some(removed)
    }

    /// The slots the map has memory for, whether or not an entry stands in
    /// each: what the entries' places cost on top of the entries themselves.
    pub fn count_allocated_slots(&self) -> usize {
        self.slots.capacity()
    }

    /// While more than the budget is held, let go of the entry used longest
    /// ago, but never of the one used last.
    fn evict_past_capacity(&mut self) -> Vec<(Key, Value)> {
        let mut evicted = Vec::new();
        while self.held_bytes > self.capacity_bytes && self.slots_by_key.len() > 1 {
            let oldest = self.oldest.expect("more than one entry is held");
            let (key, value) = self.vacate_slot(oldest);
            self.slots_by_key.remove(&key);
            evicted.push((key, value));
        }
        self.compact_sparse_slots();
        evicted
    }

    /// When fewer than a quarter of the slots hold an entry, move the
    /// entries into as many slots as there are entries, in the order they
    /// were last used, and give the rest of the memory back. A slot is only
    /// added when every slot holds an entry, so three quarters of the
    /// entries leave between one move and the next: the moves cost no more
    /// than the removals that led to them.
    fn compact_sparse_slots(&mut self) {
        let held = self.slots_by_key.len();
        if self.slots.len() < MIN_COMPACTED_SLOTS
            || held * SPARSE_SLOTS_PER_ENTRY >= self.slots.len()
        {
            return;
        }
        let mut compacted = Vec::with_capacity(held);
        let mut next = self.oldest;
        while let Some(position) = next {
            next = self.slots[position].later;
            let entry = self.slots[position]
                .entry
                .take()
                .expect("an entry held stands in its slot");
            let moved_to = compacted.len();
            *self
                .slots_by_key
                .get_mut(&entry.key)
                .expect("an entry held is found by its key") = moved_to;
            compacted.push(Slot {
                earlier: moved_to.checked_sub(1),
                later: (moved_to + 1 < held).then_some(moved_to + 1),
                entry: Some(entry),
            });
        }
        self.slots = compacted;
        self.free_slots = Vec::new();
        self.oldest = (held > 0).then_some(0);
        self.newest = held.checked_sub(1);
        self.slots_by_key.shrink_to_fit();
    }

    /// Take the entry standing in `position` out of the order of last use
    /// and out of the total, leaving its slot free.
    fn vacate_slot(&mut self, position: usize) -> (Key, Value) {
        self.unlink(position);
        self.free_slots.push(position);
        let entry = self.slots[position]
            .entry
            .take()
            .expect("an entry held stands in its slot");
        self.held_bytes -= entry.weight_bytes;
        (entry.key, entry.value)
    }

    fn read_entry(&self, position: usize) -> &Entry<Key, Value> {
        self.slots[position]
            .entry
            .as_ref()
            .expect("an entry held stands in its slot")
    }

    /// Count the entry standing in `position` as used now.
    fn move_to_newest(&mut self, position: usize) {
        if self.newest == Some(position) {
            return;
        }
        self.unlink(position);
        self.link_newest(position);
    }

    fn link_newest(&mut self, position: usize) {
        self.slots[position].earlier = self.newest;
        self.slots[position].later = None;
        match self.newest {
            Some(newest) => self.slots[newest].later = Some(position),
            None => self.oldest = Some(position),
        }
        self.newest = Some(position);
    }

    fn unlink(&mut self, position: usize) {
        let (earlier, later) = (self.slots[position].earlier, self.slots[position].later);
        match earlier {
            Some(earlier) => self.slots[earlier].later = later,
            None => self.oldest = later,
        }
        match later {
            Some(later) => self.slots[later].earlier = earlier,
            None => self.newest = earlier,
        }
    }
}
