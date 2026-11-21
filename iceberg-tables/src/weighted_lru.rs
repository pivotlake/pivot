//! A map of groups of values that holds its values within a budget of
//! bytes, letting go of the ones used longest ago.
//!
//! A value is found by its group and then by its key within the group. A
//! group's own map of keys is the list of what the group holds, and a group
//! that holds nothing is not kept.
//!
//! Every value is held with a weight, the bytes it counts for in the
//! budget, which whoever inserts it names. The entries of every group stand
//! in one order of last use. Past the budget the map hands back the entries
//! used longest ago, down to the one used last, which it keeps whatever it
//! weighs: so an entry just inserted is always held.
//!
//! A value nobody asked for is inserted as the entry used longest ago
//! ([`WeightedLru::insert_as_oldest`]): it is held only while the budget has
//! room for it, and is the first to leave, so it never costs the map an
//! entry that was used.
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

/// The slot each entry of one group stands in, by the entry's key.
type GroupSlots<Key> = HashMap<Key, usize, ahash::RandomState>;

/// Values by group and then by key, within a budget of bytes, in the order
/// they were last used.
pub struct WeightedLru<Group, Key, Value> {
    capacity_bytes: u64,
    /// The slot each held entry stands in, by its group and then by its
    /// key. A group is here exactly while it holds an entry.
    slots_by_group: HashMap<Group, GroupSlots<Key>, ahash::RandomState>,
    /// The entries, in a list through numbered slots by when each was last
    /// used: counting an entry as used moves it to the end without anything
    /// being searched.
    slots: Vec<Slot<Group, Key, Value>>,
    /// The slots no entry stands in.
    free_slots: Vec<usize>,
    /// The slot of the entry used longest ago, and of the one used last.
    oldest: Option<usize>,
    newest: Option<usize>,
    /// The sum of the held entries' weights.
    held_bytes: u64,
}

struct Slot<Group, Key, Value> {
    /// The slots of the entries used just before and just after this one.
    earlier: Option<usize>,
    later: Option<usize>,
    /// The entry standing here; `None` for a free slot.
    entry: Option<Entry<Group, Key, Value>>,
}

struct Entry<Group, Key, Value> {
    /// The group and the key the entry is found by, so an entry that leaves
    /// for the budget is taken out of its group's map.
    group: Group,
    key: Key,
    value: Value,
    /// The entry's weight, as it was counted into the total.
    weight_bytes: u64,
}

impl<Group: Hash + Eq + Clone, Key: Hash + Eq + Clone, Value> WeightedLru<Group, Key, Value> {
    /// What the map itself spends on one entry, for whoever names the
    /// entries' weights to count in: the slot the entry stands in, which
    /// holds its group, its key, its value and its weight, and the key and
    /// slot number its group finds it by. What a group, a key or a value
    /// holds outside itself is not counted, and neither are the slots no
    /// entry stands in or the room the maps keep spare.
    pub const ENTRY_OVERHEAD_BYTES: usize =
        size_of::<Slot<Group, Key, Value>>() + size_of::<(Key, usize)>();

    /// A map that holds values of up to `capacity_bytes` of weight between
    /// them.
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            slots_by_group: HashMap::default(),
            slots: Vec::new(),
            free_slots: Vec::new(),
            oldest: None,
            newest: None,
            held_bytes: 0,
        }
    }

    /// The value `group` holds under `key`, counted as used now.
    pub fn get<Asked>(&mut self, group: &Group, key: &Asked) -> Option<&Value>
    where
        Asked: Hash + Equivalent<Key> + ?Sized,
    {
        let position = self.find_slot(group, key)?;
        self.move_to_newest(position);
        Some(&self.read_entry(position).value)
    }

    /// The value `group` holds under `key`, which stays where it stood in
    /// the order of last use.
    pub fn peek<Asked>(&self, group: &Group, key: &Asked) -> Option<&Value>
    where
        Asked: Hash + Equivalent<Key> + ?Sized,
    {
        let position = self.find_slot(group, key)?;
        Some(&self.read_entry(position).value)
    }

    /// Count the value `group` holds under `key`, if it holds one, as used
    /// now.
    pub fn mark_used<Asked>(&mut self, group: &Group, key: &Asked)
    where
        Asked: Hash + Equivalent<Key> + ?Sized,
    {
        if let Some(position) = self.find_slot(group, key) {
            self.move_to_newest(position);
        }
    }

    /// Hold `value` in `group` under `key` for `weight_bytes` of the
    /// budget, used now, dropping the value the group already held under
    /// `key`. Hands back the entries, of whichever group, that left to keep
    /// the map within its budget, the one used longest ago first; the entry
    /// just inserted is never one of them.
    pub fn insert(
        &mut self,
        group: Group,
        key: Key,
        value: Value,
        weight_bytes: u64,
    ) -> Vec<(Key, Value)> {
        if let Some(position) = self.find_slot(&group, &key) {
            self.vacate_slot(position);
        }
        let slot = Slot {
            earlier: None,
            later: None,
            entry: Some(Entry {
                group: group.clone(),
                key: key.clone(),
                value,
                weight_bytes,
            }),
        };
        let position = self.take_free_slot(slot);
        self.link_newest(position);
        self.slots_by_group
            .entry(group)
            .or_default()
            .insert(key, position);
        self.held_bytes += weight_bytes;
        self.evict_past_capacity()
    }

    /// Hold `value` in `group` under `key` for `weight_bytes` of the
    /// budget, as the entry used longest ago, the first the map lets go of,
    /// when the budget has room for it: nothing leaves for it. A value the
    /// group already holds under `key` stays as it is, where it stood in
    /// the order of last use. Whether the group holds a value under `key`
    /// afterwards.
    pub fn insert_as_oldest(
        &mut self,
        group: Group,
        key: Key,
        value: Value,
        weight_bytes: u64,
    ) -> bool {
        if self.find_slot(&group, &key).is_some() {
            return true;
        }
        if self.held_bytes.saturating_add(weight_bytes) > self.capacity_bytes {
            return false;
        }
        let slot = Slot {
            earlier: None,
            later: None,
            entry: Some(Entry {
                group: group.clone(),
                key: key.clone(),
                value,
                weight_bytes,
            }),
        };
        let position = self.take_free_slot(slot);
        self.link_oldest(position);
        self.slots_by_group
            .entry(group)
            .or_default()
            .insert(key, position);
        self.held_bytes += weight_bytes;
        true
    }

    /// How much of the budget the entries held leave.
    pub fn free_bytes(&self) -> u64 {
        self.capacity_bytes.saturating_sub(self.held_bytes)
    }

    /// Let go of the value `group` holds under `key`, handing it back with
    /// its key.
    pub fn remove<Asked>(&mut self, group: &Group, key: &Asked) -> Option<(Key, Value)>
    where
        Asked: Hash + Equivalent<Key> + ?Sized,
    {
        let position = self.find_slot(group, key)?;
        let entry = self.vacate_slot(position);
        self.forget_key(&entry.group, &entry.key);
        self.compact_sparse_slots();
        Some((entry.key, entry.value))
    }

    /// Let go of every value `group` holds.
    pub fn remove_group(&mut self, group: &Group) {
        let Some(group_slots) = self.slots_by_group.remove(group) else {
            return;
        };
        for position in group_slots.into_values() {
            self.vacate_slot(position);
        }
        self.compact_sparse_slots();
    }

    /// The groups that hold a value, in no order.
    pub fn list_groups(&self) -> impl Iterator<Item = &Group> {
        self.slots_by_group.keys()
    }

    /// Stand `slot` in a slot no entry stands in, a new one when there is
    /// none, and hand back where.
    fn take_free_slot(&mut self, slot: Slot<Group, Key, Value>) -> usize {
        match self.free_slots.pop() {
            Some(position) => {
                self.slots[position] = slot;
                position
            }
            None => {
                self.slots.push(slot);
                self.slots.len() - 1
            }
        }
    }

    /// The slot the entry `group` holds under `key` stands in.
    fn find_slot<Asked>(&self, group: &Group, key: &Asked) -> Option<usize>
    where
        Asked: Hash + Equivalent<Key> + ?Sized,
    {
        self.slots_by_group.get(group)?.get(key).copied()
    }

    /// How many entries are held: every slot but the free ones holds one.
    fn count_entries(&self) -> usize {
        self.slots.len() - self.free_slots.len()
    }

    /// Take `key` out of the map of `group`, whose entry left its slot, and
    /// the group with it once it holds nothing.
    fn forget_key(&mut self, group: &Group, key: &Key) {
        let group_slots = self
            .slots_by_group
            .get_mut(group)
            .expect("the group of an entry held is kept");
        group_slots.remove(key);
        if group_slots.is_empty() {
            self.slots_by_group.remove(group);
        }
    }

    /// While more than the budget is held, let go of the entry used longest
    /// ago, but never of the one used last.
    fn evict_past_capacity(&mut self) -> Vec<(Key, Value)> {
        let mut evicted = Vec::new();
        while self.held_bytes > self.capacity_bytes && self.count_entries() > 1 {
            let oldest = self.oldest.expect("more than one entry is held");
            let entry = self.vacate_slot(oldest);
            self.forget_key(&entry.group, &entry.key);
            evicted.push((entry.key, entry.value));
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
        let held = self.count_entries();
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
                .slots_by_group
                .get_mut(&entry.group)
                .and_then(|group_slots| group_slots.get_mut(&entry.key))
                .expect("an entry held is found by its group and key") = moved_to;
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
        self.slots_by_group.shrink_to_fit();
        for group_slots in self.slots_by_group.values_mut() {
            group_slots.shrink_to_fit();
        }
    }

    /// Take the entry standing in `position` out of the order of last use
    /// and out of the total, leaving its slot free. Its group's map still
    /// names the slot.
    fn vacate_slot(&mut self, position: usize) -> Entry<Group, Key, Value> {
        self.unlink(position);
        self.free_slots.push(position);
        let entry = self.slots[position]
            .entry
            .take()
            .expect("an entry held stands in its slot");
        self.held_bytes -= entry.weight_bytes;
        entry
    }

    fn read_entry(&self, position: usize) -> &Entry<Group, Key, Value> {
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

    fn link_oldest(&mut self, position: usize) {
        self.slots[position].earlier = None;
        self.slots[position].later = self.oldest;
        match self.oldest {
            Some(oldest) => self.slots[oldest].earlier = Some(position),
            None => self.newest = Some(position),
        }
        self.oldest = Some(position);
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
