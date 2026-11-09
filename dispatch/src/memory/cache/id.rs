//! Naming a cache: its [`CacheId`], the [`CachePolicy`] that says how hard its
//! slots are to drop, and [`CacheSet`], a small set of ids the evictor passes
//! around.

/// Index of one cache in the [`Caches`](super::Caches) registry.
///
/// Stored per ring slot in the clock's owner byte as `index + 1`, with 0
/// meaning "no cache owns this slot", so it is deliberately one byte: a
/// revolution of the clock walks every slot's record.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CacheId(u8);

impl CacheId {
    /// At most this many caches can be registered: one owner byte per slot,
    /// minus the free marker.
    pub const MAX_CACHES: usize = u8::MAX as usize;

    pub(crate) const fn new(index: u8) -> Self {
        Self(index)
    }

    /// Position in the registry's cache list.
    pub fn index(self) -> usize {
        self.0 as usize
    }

    /// How the clock stores this id in a slot's owner byte.
    pub(crate) fn to_owner_byte(self) -> u8 {
        self.0 + 1
    }

    /// The id a slot's owner byte names, `None` for the free marker.
    pub(crate) fn from_owner_byte(byte: u8) -> Option<Self> {
        byte.checked_sub(1).map(Self)
    }
}

/// How a cache's slots behave in the shared CLOCK.
///
/// Every slot loses one life per hand pass regardless of cache; a cache whose
/// entries should outlive others' at equal recency gets more lives to spend.
#[derive(Clone, Debug)]
pub struct CachePolicy {
    /// What the cache is called in system tables and logs.
    pub name: &'static str,
    /// Lives a slot starts with when the cache binds it.
    pub lives_at_insert: u8,
    /// Lives a slot gains each time one of its entries is read.
    pub lives_per_touch: u8,
    /// Lives a slot gains per read once a sibling cache dropped its copy of
    /// the same bytes, granted once at that moment too. Higher than
    /// `lives_per_touch`, so a copy that is still being read and is closer to
    /// being the last one climbs while equally hot backed copies sink. Only
    /// meaningful for a cache with siblings.
    pub lives_at_reinforce: u8,
}

/// A set of [`CacheId`]s, as a bitmask.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct CacheSet(u128);

impl CacheSet {
    pub const EMPTY: Self = Self(0);

    pub fn insert(&mut self, id: CacheId) {
        self.0 |= 1 << id.index();
    }

    pub fn contains(&self, id: CacheId) -> bool {
        self.0 & (1 << id.index()) != 0
    }

    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_owner_byte_round_trips_and_keeps_zero_for_free() {
        let id = CacheId::new(3);

        let byte = id.to_owner_byte();

        assert_eq!(CacheId::from_owner_byte(byte), Some(id));
        assert_eq!(CacheId::from_owner_byte(0), None);
    }

    #[test]
    fn a_set_reports_only_the_ids_inserted() {
        let mut set = CacheSet::EMPTY;

        set.insert(CacheId::new(2));

        assert!(set.contains(CacheId::new(2)));
        assert!(!set.contains(CacheId::new(1)));
        assert!(!set.is_empty());
    }
}
