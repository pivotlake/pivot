//! Live string keys: the transient key forms that borrow the input batch (or a
//! persisted [`ArenaKey`]) and know how to compare against / persist into the
//! arena. See [`super::arena_key::ArenaKey`] for the stored representation.

use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::LiveKey;
use crate::operations::unary::group::keys::string::arena_key::ArenaKey;
use std::hash::{Hash, Hasher};

/// A live string key holding the raw `&str` and a mutable reference to the
/// worker arena. `eq_persisted` uses the shared arena for comparison;
/// `persist` pushes the string into the arena only when a new entry is needed.
pub struct StringKey<'a, 'b> {
    arena: &'a mut WorkerArena,
    value: &'b str,
}

impl<'a, 'b> StringKey<'a, 'b> {
    pub fn new(arena: &'a mut WorkerArena, value: &'b str) -> Self {
        Self { arena, value }
    }
}

impl Hash for StringKey<'_, '_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl LiveKey for StringKey<'_, '_> {
    type Persisted = ArenaKey;

    #[inline(always)]
    fn eq_persisted(&self, other: &ArenaKey) -> bool {
        self.value.as_bytes() == other.resolve(self.arena.shared())
    }

    #[inline(always)]
    fn persist(self) -> ArenaKey {
        self.arena.push(self.value)
    }
}

/// A previously-persisted [`ArenaKey`] paired with the arena needed to
/// resolve it. Used during the merge phase to re-insert entries from
/// source tables into the target table without re-hashing the raw bytes.
pub struct ResolvedKey<'a> {
    pub key: ArenaKey,
    pub arena: &'a SharedArena,
}

impl<'a> ResolvedKey<'a> {
    pub fn new(key: ArenaKey, arena: &'a SharedArena) -> Self {
        Self { key, arena }
    }
}

impl Hash for ResolvedKey<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key.resolve(self.arena).hash(state);
    }
}

impl LiveKey for ResolvedKey<'_> {
    type Persisted = ArenaKey;

    #[inline(always)]
    fn eq_persisted(&self, other: &ArenaKey) -> bool {
        self.key.resolve(self.arena) == other.resolve(self.arena)
    }

    fn persist(self) -> ArenaKey {
        self.key
    }
}
