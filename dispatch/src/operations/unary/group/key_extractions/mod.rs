//! Key extraction strategies for GROUP BY operations.
//!
//! A [`KeyExtractor`] defines how to extract, compare, persist, and output
//! group keys for a particular data type. It bridges Arrow arrays (the input)
//! with the hash table's key/value storage (the engine) and Arrow record
//! batches (the output).
//!
//! TODO: Right now the KeyExtractor includes definitions for the value- when we have different
//! group by values this will need to move
//!
//! ## Live vs Persisted keys
//!
//! Each extractor works with two representations of the same key:
//!
//! - **Live key** — a transient reference into the input array (e.g. `&str`
//!   from a `StringViewArray`). Cheap to create and compare, but borrows
//!   from the input batch which will be dropped.
//! - **Persisted key** — an owned, `Copy` value stored inside the hash table
//!   (e.g. an [`ArenaKey`](super::ArenaKey) pointing into the shared arena).
//!   Only created when the key is actually new.
//!
//! This split avoids copying strings into the arena on every row — only
//! genuinely new keys pay the persist cost.
//!
//! ## Provided extractors
//!
//! - [`IntKeyExtractor<T>`] — for Arrow primitive types (i32, u64, etc.).
//!   Keys are `Copy` integers, so live and persisted forms are identical.
//! - [`StringKeyExtractor`] — for `StringViewArray`. Live keys borrow
//!   the raw `&str`; persisted keys are arena-backed `ArenaKey`s that
//!   share the same u128 layout as Arrow's StringView.

use std::hash::Hash;
use std::sync::Arc;

use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::{
    LiveKey, PersistedKey, Table, TableStorage, Value,
};
use arrow_array::{Array, RecordBatch};
use arrow_schema::ArrowError;

mod int_key_extractor;
pub use int_key_extractor::IntKeyExtractor;

mod string_extractor;
pub use string_extractor::StringKeyExtractor;

/// Defines how to extract, compare, and output group keys for a data type.
///
/// Implementors specify:
/// - How to read a key from an Arrow array ([`live_key`](Self::live_key))
/// - How to reconstruct a key from its persisted form ([`resolve_persisted`](Self::resolve_persisted))
/// - How to downcast an `&dyn Array` to the concrete array type ([`downcast_column`](Self::downcast_column))
/// - How to convert a finished hash table into an Arrow [`RecordBatch`] ([`create_record_batch`](Self::create_record_batch))
pub trait KeyExtractor: Send + 'static {
    /// The concrete Arrow array type this extractor reads from.
    type ArrayRef<'a>: arrow::array::ArrayAccessor<Item: Hash + Eq> + arrow::array::Array + Copy;
    /// The `Copy` key representation stored inside hash table entries.
    type Persisted: PersistedKey;
    /// A transient key that borrows from the input array and/or the worker arena.
    type LiveKey<'a, 'b>: LiveKey<Persisted = Self::Persisted>;
    /// A live key reconstructed from an already-persisted key (used during merge).
    type PersistedLiveKey<'a>: LiveKey<Persisted = Self::Persisted>;
    /// The aggregation value stored alongside each key (e.g. `Count`).
    type Value: Value + Send;

    /// Extract a live key from `column` at position `idx`.
    fn live_key<'a, 'b>(
        column: &Self::ArrayRef<'b>,
        idx: usize,
        arena: &'a mut WorkerArena,
    ) -> Self::LiveKey<'a, 'b>;

    /// Reconstruct a live key from a persisted key, borrowing from the shared arena.
    fn resolve_persisted(
        arena: &SharedArena,
        persisted: Self::Persisted,
    ) -> Self::PersistedLiveKey<'_>;

    /// Downcast a type-erased `&dyn Array` to this extractor's concrete array type.
    fn downcast_column<'a>(column: &'a dyn Array) -> Option<Self::ArrayRef<'a>>;

    /// Convert a completed hash table into an Arrow `RecordBatch` with key and value columns.
    fn create_record_batch<S: TableStorage<Self>>(
        table: Table<Self, S>,
        arena: &Arc<SharedArena>,
    ) -> Result<RecordBatch, ArrowError>;
}
