//! Key extraction strategies for GROUP BY operations.
//!
//! A [`KeyExtractor`] defines how to read group keys and per-row aggregate
//! values from an input batch, persist/compare keys in the hash table, and
//! emit the finished table as an Arrow [`RecordBatch`].
//!
//! ## Reader-based consume
//!
//! Extraction is driven through a per-batch [`Reader`](KeyExtractor::Reader):
//! [`make_reader`](KeyExtractor::make_reader) downcasts the configured key
//! columns (one or more) and value columns once, then
//! [`hash`](KeyExtractor::hash), [`live_key`](KeyExtractor::live_key) and
//! [`value`](KeyExtractor::value) read row `idx` cheaply. This lets a single
//! trait cover single-column count grouping, multi-column keys, and multi-slot
//! sum/count/avg values.
//!
//! ## Live vs Persisted keys
//!
//! - **Live key** — a transient reference into the input batch (e.g. `&str`).
//! - **Persisted key** — an owned, `Copy` value stored in the table (an
//!   [`ArenaKey`](super::ArenaKey) for strings, an integer / packed integer for
//!   numeric keys). Only created when the key is genuinely new.

use std::sync::Arc;

use crate::operations::unary::group::aggregations::GroupAggSlot;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::{
    LiveKey, PersistedKey, Table, TableStorage, Value,
};
use ahash::RandomState;
use arrow_array::RecordBatch;
use arrow_schema::ArrowError;

mod int_key_extractor;
pub use int_key_extractor::IntKeyExtractor;

mod int_pair_extractor;
pub use int_pair_extractor::IntPairAggExtractor;

mod string_extractor;
pub use string_extractor::StringKeyExtractor;

/// Defines how to extract, compare, and output group keys (and per-row
/// aggregate values) for a particular key/value shape.
pub trait KeyExtractor: Send + 'static {
    /// The `Copy` key representation stored inside hash table entries.
    type Persisted: PersistedKey;
    /// A transient key that borrows from the input batch and/or the worker arena.
    type LiveKey<'a, 'b>: LiveKey<Persisted = Self::Persisted>;
    /// A live key reconstructed from an already-persisted key (used during merge).
    type PersistedLiveKey<'a>: LiveKey<Persisted = Self::Persisted>;
    /// The aggregation value stored alongside each key.
    type Value: Value + Send;
    /// Per-batch reader holding downcast key/value column accessors.
    type Reader<'b>;

    /// Build a reader over `batch` for the given key columns and value slots.
    fn make_reader<'b>(
        batch: &'b RecordBatch,
        key_cols: &[usize],
        value_slots: &[GroupAggSlot],
    ) -> Self::Reader<'b>;

    /// Number of rows the reader spans.
    fn rows(reader: &Self::Reader<'_>) -> usize;

    /// Hash the key at row `idx`.
    fn hash(reader: &Self::Reader<'_>, idx: usize, state: &RandomState) -> u64;

    /// Extract a live key from row `idx` (may borrow the arena to persist).
    fn live_key<'a, 'b>(
        reader: &Self::Reader<'b>,
        idx: usize,
        arena: &'a mut WorkerArena,
    ) -> Self::LiveKey<'a, 'b>;

    /// Build the per-row aggregate value at row `idx`.
    fn value(reader: &Self::Reader<'_>, idx: usize) -> Self::Value;

    /// Reconstruct a live key from a persisted key, borrowing from the shared arena.
    fn resolve_persisted(
        arena: &SharedArena,
        persisted: Self::Persisted,
    ) -> Self::PersistedLiveKey<'_>;

    /// Convert a completed hash table into an Arrow `RecordBatch` of key +
    /// value columns.
    fn create_record_batch<S: TableStorage<Self>>(
        table: Table<Self, S>,
        arena: &Arc<SharedArena>,
    ) -> Result<RecordBatch, ArrowError>;
}
