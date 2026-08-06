//! The transient row key probed against the hash table.

use super::reader::RowReader;
use crate::operations::unary::group::arena::WorkerArena;
use crate::operations::unary::group::hashtables::LiveKey;
use crate::operations::unary::group::keys::ArenaKey;
use std::hash::{Hash, Hasher};

/// A live row key: a row of the reader's bound key columns, compared and
/// persisted lazily. Equality against a stored key walks the columns against
/// the persisted blob, and only persisting a *new* group ever encodes the
/// row's bytes — a probe hit materialises nothing.
pub struct RowKey<'a, 'r> {
    arena: &'a mut WorkerArena,
    reader: &'r RowReader<'r>,
    idx: usize,
}

impl<'a, 'r> RowKey<'a, 'r> {
    #[inline(always)]
    pub(super) fn new(arena: &'a mut WorkerArena, reader: &'r RowReader<'r>, idx: usize) -> Self {
        Self { arena, reader, idx }
    }
}

impl Hash for RowKey<'_, '_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.reader.hash_row_into(self.idx, state);
    }
}

impl LiveKey for RowKey<'_, '_> {
    type Persisted = ArenaKey;

    #[inline(always)]
    fn eq_persisted(&self, other: &ArenaKey) -> bool {
        self.reader
            .eq_row(self.idx, other.resolve(self.arena.shared()))
    }

    #[inline(always)]
    fn persist(self) -> ArenaKey {
        let Self { arena, reader, idx } = self;
        reader.encode_row(idx, |blob| arena.push_bytes(blob))
    }
}
