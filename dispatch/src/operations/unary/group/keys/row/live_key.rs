//! The transient row key probed against the hash table.

use crate::operations::unary::group::arena::WorkerArena;
use crate::operations::unary::group::hashtables::LiveKey;
use crate::operations::unary::group::keys::ArenaKey;
use std::hash::{Hash, Hasher};

/// A live row key: the encoded tuple bytes plus the worker arena to compare
/// against / persist into. The byte-blob analogue of `StringKey`.
pub struct RowKey<'a, 'b> {
    arena: &'a mut WorkerArena,
    value: &'b [u8],
}

impl<'a, 'b> RowKey<'a, 'b> {
    #[inline(always)]
    pub(super) fn new(arena: &'a mut WorkerArena, value: &'b [u8]) -> Self {
        Self { arena, value }
    }
}

impl Hash for RowKey<'_, '_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl LiveKey for RowKey<'_, '_> {
    type Persisted = ArenaKey;

    #[inline(always)]
    fn eq_persisted(&self, other: &ArenaKey) -> bool {
        self.value == other.resolve(self.arena.shared())
    }

    #[inline(always)]
    fn persist(self) -> ArenaKey {
        self.arena.push_bytes(self.value)
    }
}
