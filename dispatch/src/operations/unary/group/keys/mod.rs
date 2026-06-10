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

mod hash_only_int;
pub use hash_only_int::HashOnlyIntKeyExtractor;

mod string;
pub use string::{ArenaKey, StringKeyExtractor};

mod row;
pub use row::{RowKeyExtractor, RowKeySchema};

/// Defines how to extract, compare, and output group keys for a particular key
/// shape.
pub trait KeyExtractor: Send + 'static {
    /// Whether this key may switch from in-place aggregation to radix scatter at
    /// high cardinality. Strings stay in-place (deferred dedup would store every
    /// occurrence un-deduped); fixed-width integer keys switch.
    const SUPPORTS_RADIX: bool = false;

    /// Multiplier on [`RadixConfig::switch_threshold`] for this key shape.
    ///
    /// The threshold is slot-count-based and tuned for the typical ~24–32-byte
    /// entry (hash + integer key + aggregate), but what it really guards is the
    /// in-place table's *byte* footprint staying L2-resident. An extractor with
    /// narrower entries can therefore grow to proportionally more slots before
    /// its probes start missing cache; scaling the threshold keeps the switch at
    /// the same byte budget, so a medium-cardinality build the in-place table
    /// still serves from L2 doesn't pay scatter + a 4096-way merge for nothing.
    /// The default of 1 leaves every existing key shape's behavior unchanged.
    ///
    /// [`RadixConfig::switch_threshold`]: super::hashtables::RadixConfig::switch_threshold
    const RADIX_SWITCH_SCALE: usize = 1;

    /// When `true`, the persisted key is a zero-sized `()` and dedup is purely by
    /// the (bijective) hash, so a hash of 0 — which the table reserves as its
    /// empty-slot sentinel — cannot be remapped without aliasing a real key.
    /// [`AggregatedTable`](super::hashtables::AggregatedTable) instead counts the
    /// single 0-hash key out of band: it never reaches the in-place table *or*
    /// the radix scatter buffers (the post-switch scatter skips it the same
    /// way), so a stored hash of 0 always means "empty" and the merge's
    /// insert-by-stored-hash never aliases it onto a real key.
    const DEDUP_BY_HASH: bool = false;

    /// Runtime configuration threaded from the operator spec to the per-batch
    /// reader and the output key columns. `()` for extractors whose shape is
    /// fully determined by their type; [`RowKeyExtractor`] carries its key
    /// schema here (the one piece the output decode can't derive on its own).
    type Config: Clone + Send + Sync + 'static;
    /// The `Copy` key representation stored inside hash table entries.
    type Persisted: PersistedKey;
    /// A transient key that borrows from the input batch and/or the worker arena.
    type LiveKey<'a, 'b>: LiveKey<Persisted = Self::Persisted>;
    /// A live key reconstructed from an already-persisted key (used during merge).
    type PersistedLiveKey<'a>: LiveKey<Persisted = Self::Persisted>;
    /// Per-batch reader holding downcast key-column accessors (or, for the row
    /// extractor, the batch's pre-encoded key rows).
    type Reader<'b>;
    /// Accumulates persisted keys into the result's leading key column(s).
    type Columns: KeyColumns<Key = Self::Persisted, Config = Self::Config>;

    /// Build a reader over `batch` for the given key columns.
    fn make_reader<'b>(
        batch: &'b RecordBatch,
        key_cols: &[usize],
        config: &Self::Config,
    ) -> Self::Reader<'b>;

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
    /// The extractor's runtime configuration (see [`KeyExtractor::Config`]).
    type Config;

    /// Allocate key-column builders over engine memory, sized for `rows` (one
    /// output chunk; must fit a single 2MB slab). Arena-backed keys (strings)
    /// ignore the allocator and pull their data from the shared arena at finish.
    fn with_capacity(allocator: &mut SlabAllocator, rows: usize, config: &Self::Config) -> Self;
    fn push(&mut self, key: &Self::Key);
    /// Materialise the key columns. Takes the arena (for zero-copy string
    /// views) and the chunk's allocator (for builders whose size is only
    /// known at decode time, e.g. the row extractor's decoded columns).
    fn finish(
        self,
        arena: &Arc<SharedArena>,
        allocator: &mut SlabAllocator,
    ) -> (Vec<Field>, Vec<ArrayRef>);
}
