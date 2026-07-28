//! Key extraction strategies for GROUP BY operations.
//!
//! A [`KeyExtractor`] defines how to read group keys from an input batch,
//! persist/compare them in the hash table, and emit the key columns of the
//! result. The per-row aggregate *value* is the separate concern of a
//! [`AggregationValue`](super::values::AggregationValue); the two are
//! mixed freely (any key shape × any aggregate shape).
//!
//! ## Reader-based consume
//!
//! Extraction is driven through a per-batch [`Reader`](KeyExtractor::Reader):
//! [`make_reader`](KeyExtractor::make_reader) binds the key columns,
//! [`prepare_and_hash`](KeyExtractor::prepare_and_hash) fills the batch's hash
//! buffer (and materialises keys, for the row extractor), then
//! [`live_key`](KeyExtractor::live_key) reads row `idx` cheaply during the probe.
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
use arrow_buffer::Buffer;
use arrow_schema::Field;
use std::sync::Arc;

mod int;
pub use int::IntKeyExtractor;

mod int_pair;
pub use int_pair::IntPairKeyExtractor;

mod int_string;
pub use int_string::IntStrKeyExtractor;

mod hash_only_int;
pub use hash_only_int::HashOnlyIntKeyExtractor;

mod string;
pub use string::{ArenaKey, StringKeyExtractor};

mod row;
pub use row::{RowKeyExtractor, RowKeySchema};

/// Defines how to extract, compare, and output group keys for a particular key
/// shape.
pub trait KeyExtractor: Send + 'static {
    /// Which radix route a key takes once its in-place table outgrows the
    /// cache-resident size: `true` = *abandon* (keep deduplicating in the bounded
    /// table, flushing each window's distinct partials to the partition buffers),
    /// `false` = *scatter* (append raw rows to the partition buffers, dedup only
    /// in the merge). Abandon pays off only when it avoids a per-occurrence
    /// out-of-line copy (long string keys); fixed-width keys scatter's cheap
    /// append wins, so this stays `false`.
    ///
    /// Radix applies to every key with real bytes to scatter. A key that dedups
    /// purely by hash ([`DEDUP_BY_HASH`](Self::DEDUP_BY_HASH)) has none, so it
    /// stays fully in-place regardless of this flag.
    const RADIX_ABANDON: bool = false;

    /// When `true`, the persisted key is a zero-sized `()` and dedup is purely by
    /// the (bijective) hash, so a hash of 0 — which the table reserves as its
    /// empty-slot sentinel — cannot be remapped without aliasing a real key.
    /// [`AggregatedTable`](super::hashtables::AggregatedTable) instead counts the
    /// single 0-hash key out of band (it never reaches the table). Such a key has
    /// no bytes to scatter, so it never takes the radix path.
    const DEDUP_BY_HASH: bool = false;

    /// Runtime configuration threaded from the operator spec to the per-batch
    /// reader and the output columns. Most extractors are fully determined by
    /// their type and use `()`; [`RowKeyExtractor`] carries its key schema here
    /// — the one thing neither the reader nor the output decode can recover on
    /// their own.
    type Config: Clone + Send + Sync + 'static;
    /// The `Copy` key representation stored inside hash table entries.
    type Persisted: PersistedKey;
    /// A transient key that borrows from the input batch and/or the worker arena.
    type LiveKey<'a, 'b>: LiveKey<Persisted = Self::Persisted>;
    /// A live key reconstructed from an already-persisted key (used during merge).
    type PersistedLiveKey<'a>: LiveKey<Persisted = Self::Persisted>;
    /// Per-batch reader holding downcast key-column accessors (or, for the row
    /// extractor, a borrow of the worker [`Scratch`](Self::Scratch) it encodes
    /// into).
    type Reader<'b>;
    /// Accumulates persisted keys into the result's leading key column(s).
    type ColumnBuilder: KeyColumnBuilder<Key = Self::Persisted, Config = Self::Config>;
    /// Per-worker reusable scratch, owned by the table and reused across batches.
    /// `()` for extractors that read columns directly; the row extractor uses it
    /// to hold the batch's encoded key bytes so nothing is reallocated per batch.
    type Scratch: Default + Send;

    /// Bind `batch`'s key columns into a reader. Cheap and side-effect-free —
    /// just downcasts (and, for the row extractor, casts) the columns and borrows
    /// `scratch`. The actual per-batch work happens in
    /// [`prepare_and_hash`](Self::prepare_and_hash).
    fn make_reader<'b>(
        batch: &'b RecordBatch,
        key_cols: &[usize],
        config: &Self::Config,
        scratch: &'b mut Self::Scratch,
    ) -> Self::Reader<'b>;

    /// Process one batch: write each row's hash into `hashes` (which the caller
    /// sized to the batch length). An extractor that materialises keys (the row
    /// extractor) also encodes them into its scratch here, hashing each as it is
    /// written; the others hash straight from the columns.
    fn prepare_and_hash(reader: &mut Self::Reader<'_>, state: &RandomState, hashes: &mut [u64]);

    /// Extract a live key from row `idx` (may borrow the arena to persist).
    ///
    /// The live key borrows the *reader* (lifetime `'r`), not the batch directly.
    /// That covers both a key pointing into the input columns (string — whose
    /// longer batch borrow simply shortens to `'r`) and one pointing into the
    /// reader's own scratch (row). It is always consumed — `eq_persisted` or
    /// `persist` — within the probe iteration, well inside `'r`.
    fn live_key<'a, 'r>(
        reader: &'r Self::Reader<'_>,
        idx: usize,
        arena: &'a mut WorkerArena,
    ) -> Self::LiveKey<'a, 'r>;

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
pub trait KeyColumnBuilder {
    type Key;
    /// The owning extractor's runtime configuration (see [`KeyExtractor::Config`]).
    type Config;

    /// Allocate key-column builders over engine memory, sized for `rows` (one
    /// output chunk; must fit a single 2MB slab). Arena-backed keys (strings)
    /// ignore the allocator and pull their data from the shared arena at finish.
    fn with_capacity(allocator: &mut SlabAllocator, rows: usize, config: &Self::Config) -> Self;
    fn push(&mut self, key: &Self::Key);
    /// Materialise the key columns. Takes the arena (for zero-copy string views)
    /// and the chunk's allocator (for builders whose size is only known at
    /// decode time, e.g. the row extractor's per-field decoded columns).
    /// `output_buffers` is the arena's ring buffers wrapped as Arrow `Buffer`s,
    /// built once for the whole output phase and shared by every batch; string
    /// keys emit zero-copy views into it (cloning the `Arc` is one bump); keys
    /// with no out-of-line bytes ignore it.
    fn finish(
        self,
        arena: &Arc<SharedArena>,
        output_buffers: &Arc<[Buffer]>,
        allocator: &mut SlabAllocator,
    ) -> (Vec<Field>, Vec<ArrayRef>);
}
