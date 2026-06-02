//! Key extraction strategies for GROUP BY operations.
//!
//! A [`KeyExtractor`] defines how to read group keys from an input batch,
//! persist/compare them in the hash table, and emit the key columns of the
//! result. The per-row aggregate *value* is the separate concern of a
//! [`ValueExtractor`](super::values::ValueExtractor); the two are
//! mixed freely (any key shape × any aggregate shape).
//!
//! ## Reader-based consume
//!
//! Extraction is driven through a per-batch [`Reader`](KeyExtractor::Reader):
//! [`make_reader`](KeyExtractor::make_reader) downcasts the key columns once,
//! then [`hash`](KeyExtractor::hash) and [`live_key`](KeyExtractor::live_key)
//! read row `idx` cheaply.
//!
//! ## Live vs Persisted keys
//!
//! - **Live key** — a transient reference into the input batch (e.g. `&str`).
//! - **Persisted key** — an owned, `Copy` value stored in the table (an
//!   [`ArenaKey`] for strings, an integer / packed integer for
//!   numeric keys). Only created when the key is genuinely new.

use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::{LiveKey, PersistedKey};
use ahash::RandomState;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use std::sync::Arc;

mod int;
pub use int::IntKeyExtractor;

mod int_pair;
pub use int_pair::IntPairKeyExtractor;

mod string;
pub use string::{ArenaKey, StringKeyExtractor};

/// Defines how to extract, compare, and output group keys for a particular key
/// shape.
pub trait KeyExtractor: Send + 'static {
    /// The `Copy` key representation stored inside hash table entries.
    type Persisted: PersistedKey;
    /// A transient key that borrows from the input batch and/or the worker arena.
    type LiveKey<'a, 'b>: LiveKey<Persisted = Self::Persisted>;
    /// A live key reconstructed from an already-persisted key (used during merge).
    type PersistedLiveKey<'a>: LiveKey<Persisted = Self::Persisted>;
    /// Per-batch reader holding downcast key-column accessors.
    type Reader<'b>;
    /// Accumulates persisted keys into the result's leading key column(s).
    type Columns: KeyColumns<Key = Self::Persisted>;

    /// Build a reader over `batch` for the given key columns.
    fn make_reader<'b>(batch: &'b RecordBatch, key_cols: &[usize]) -> Self::Reader<'b>;

    /// Hash the key at row `idx`.
    fn hash(reader: &Self::Reader<'_>, idx: usize, state: &RandomState) -> u64;

    /// Extract a live key from row `idx` (may borrow the arena to persist).
    fn live_key<'a, 'b>(
        reader: &Self::Reader<'b>,
        idx: usize,
        arena: &'a mut WorkerArena,
    ) -> Self::LiveKey<'a, 'b>;

    /// Reconstruct a live key from a persisted key, borrowing from the shared arena.
    fn resolve_persisted(
        arena: &SharedArena,
        persisted: Self::Persisted,
    ) -> Self::PersistedLiveKey<'_>;
}

/// Builds the leading key column(s) of a GROUP BY result, one group at a time.
///
/// The output combinator pushes each surviving group's persisted key, then
/// `finish` materialises the Arrow columns and their fields. `finish` takes the
/// [`SharedArena`] so arena-backed keys (strings) can emit zero-copy views into
/// the ring buffers; non-arena keys ignore it.
pub trait KeyColumns {
    type Key;

    /// Allocate key-column builders over engine memory, sized for `rows` (one
    /// output chunk; must fit a single 2MB slab). Arena-backed keys (strings)
    /// ignore the allocator and pull their data from the shared arena at finish.
    fn with_capacity(allocator: &mut SlabAllocator, rows: usize) -> Self;
    fn push(&mut self, key: &Self::Key);
    fn finish(self, arena: &Arc<SharedArena>) -> (Vec<Field>, Vec<ArrayRef>);
}
