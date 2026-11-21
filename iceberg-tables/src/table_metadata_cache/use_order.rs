//! The order the cache's entries were last used in, footers and manifest
//! parts alike.
//!
//! Each entry stands in a numbered slot with its key, which names what
//! the cache holds it under, and with what it counts for in the budget. The
//! slots are linked in the order of last use, so counting an entry as used
//! moves it to the end with nothing searched, and the entry used longest
//! ago is always at hand with the key to take it out of the cache by.
//! The cache keeps the number of each entry's slot beside the entry.
//!
//! The slots are given back once fewer than a quarter of them hold an
//! entry: the entries move into as many slots as there are entries, and
//! whoever keeps the slot numbers is told each entry's new one.

use std::sync::Arc;

use iceberg::spec::FormatVersion;
use uuid::Uuid;

use super::FooterSchema;
use super::manifest_part::CachedPartKey;

/// The fewest slots any are given back from: fewer take too little memory
/// to be worth moving the entries for.
const MIN_COMPACTED_SLOTS: usize = 1024;

/// How many slots are kept to an entry before any are given back: the
/// slots are given back once fewer than one in this many holds an entry.
const SPARSE_SLOTS_PER_ENTRY: usize = 4;

/// Where an entry of the cache is held: under which names, which are the
/// allocations the cache holds the entry under, what a table's footers
/// were parsed under shared by every one of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum EntryKey {
    /// The footer of the file at `path`, among those of the table
    /// `table_id` parsed under `schema`.
    Footer {
        table_id: Uuid,
        schema: Arc<FooterSchema>,
        path: Arc<str>,
    },
    /// The part at `key`, among those of the manifests of the table
    /// `table_id` read under `format_version`.
    Part {
        table_id: Uuid,
        format_version: FormatVersion,
        key: CachedPartKey,
    },
}

/// An entry as the order knows it.
pub(super) struct OrderedEntry {
    pub(super) key: EntryKey,
    /// What the entry counts for in the budget.
    pub(super) weight_bytes: u64,
}

struct Slot {
    /// The slots of the entries used just before and just after this one.
    earlier: Option<usize>,
    later: Option<usize>,
    /// The entry standing here; `None` for a free slot.
    entry: Option<OrderedEntry>,
}

/// The entries' keys, in the order the entries were last used.
pub(super) struct UseOrder {
    slots: Vec<Slot>,
    /// The slots no entry stands in.
    free_slots: Vec<usize>,
    /// The slot of the entry used longest ago, and of the one used last.
    oldest: Option<usize>,
    newest: Option<usize>,
}

impl UseOrder {
    /// What the order spends on one entry: the slot it stands in. The
    /// slots no entry stands in are not counted.
    pub(super) const SLOT_BYTES: usize = size_of::<Slot>();

    pub(super) fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_slots: Vec::new(),
            oldest: None,
            newest: None,
        }
    }

    /// Stand `entry` in a slot as the entry used last, and hand back the
    /// slot's number.
    pub(super) fn push_newest(&mut self, entry: OrderedEntry) -> usize {
        let position = self.take_free_slot(entry);
        self.link_newest(position);
        position
    }

    /// Stand `entry` in a slot as the entry used longest ago, and hand
    /// back the slot's number.
    pub(super) fn push_oldest(&mut self, entry: OrderedEntry) -> usize {
        let position = self.take_free_slot(entry);
        self.link_oldest(position);
        position
    }

    /// Count the entry standing in `position` as used now.
    pub(super) fn mark_used(&mut self, position: usize) {
        if self.newest == Some(position) {
            return;
        }
        self.unlink(position);
        self.link_newest(position);
    }

    /// Take the entry standing in `position` out of the order, leaving its
    /// slot free.
    pub(super) fn remove(&mut self, position: usize) -> OrderedEntry {
        self.unlink(position);
        self.free_slots.push(position);
        self.slots[position]
            .entry
            .take()
            .expect("an entry held stands in its slot")
    }

    /// The slot of the entry used longest ago, with the entry's key.
    pub(super) fn find_oldest(&self) -> Option<(usize, &EntryKey)> {
        let position = self.oldest?;
        let entry = self.slots[position].entry.as_ref();
        let entry = entry.expect("an entry held stands in its slot");
        Some((position, &entry.key))
    }

    /// How many entries stand in the order: every slot but the free ones
    /// holds one.
    pub(super) fn count_entries(&self) -> usize {
        self.slots.len() - self.free_slots.len()
    }

    /// When fewer than a quarter of the slots hold an entry, move the
    /// entries into as many slots as there are entries, in the order they
    /// were last used, give the rest of the memory back, and tell
    /// `renumber` the slot each entry stands in now. Whether the entries
    /// moved. A slot is only added when every slot holds an entry, so three
    /// quarters of the entries leave between one move and the next: the
    /// moves cost no more than the removals that led to them.
    pub(super) fn compact_sparse_slots(
        &mut self,
        mut renumber: impl FnMut(&EntryKey, usize),
    ) -> bool {
        let held = self.count_entries();
        if self.slots.len() < MIN_COMPACTED_SLOTS
            || held * SPARSE_SLOTS_PER_ENTRY >= self.slots.len()
        {
            return false;
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
            renumber(&entry.key, moved_to);
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
        true
    }

    /// Stand `entry` in a slot no entry stands in, a new one when there is
    /// none, and hand back where.
    fn take_free_slot(&mut self, entry: OrderedEntry) -> usize {
        let slot = Slot {
            earlier: None,
            later: None,
            entry: Some(entry),
        };
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
