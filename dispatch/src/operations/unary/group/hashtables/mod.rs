//! Hash table infrastructure for GROUP BY aggregation.
//!
//! This module wires together the generic [`BaseHashTable`] with the
//! [`KeyExtractor`] trait to produce concrete table types parameterized
//! by key extraction strategy.
//!
//! - [`Table<K, S>`] — a `BaseHashTable` whose key/value types are derived
//!   from `K: KeyExtractor`, generic over the storage backend `S`.
//! - [`MultiSlabTable<K>`] — a concrete [`Table`] backed by a
//!   [`MultiSlabBuffer`] (supports tables larger than one slab).
//! - [`AggregatedTable<K>`] — the per-worker accumulator used during the
//!   consume phase.

use std::ops::{Index, IndexMut};

use crate::memory::MultiSlabBuffer;
use crate::operations::unary::group::hashtables::hash_table::BaseHashTable;
pub use crate::operations::unary::group::key_extractions::KeyExtractor;

mod hash_table;
pub use hash_table::{Entry, LiveKey, PersistedKey, Value};

mod aggregated_table;
pub use aggregated_table::AggregatedTable;

/// Initial number of slots for a new per-worker hash table.
pub const DEFAULT_CAPACITY: usize = 128;

/// Convenience alias: the persisted key type for a given [`KeyExtractor`].
pub type ExtractorPersisted<K> = <K as KeyExtractor>::Persisted;

/// Marker trait for any backing storage that can index `Entry` by `usize`.
///
/// Automatically implemented for anything that implements
/// `Index<usize> + IndexMut<usize>` with the right output type, so
/// `Vec`, `SlabBuffer`, and `MultiSlabBuffer` all qualify.
pub trait TableStorage<K: KeyExtractor + ?Sized>:
    Index<usize, Output = Entry<ExtractorPersisted<K>, <K as KeyExtractor>::Value>> + IndexMut<usize>
{
}

impl<K: KeyExtractor + ?Sized, T> TableStorage<K> for T where
    T: Index<usize, Output = Entry<ExtractorPersisted<K>, <K as KeyExtractor>::Value>>
        + IndexMut<usize>
{
}

/// A [`BaseHashTable`] parameterized by a [`KeyExtractor`] and storage backend.
pub type Table<K, S> = BaseHashTable<ExtractorPersisted<K>, <K as KeyExtractor>::Value, S>;

/// A [`Table`] backed by multiple slab buffers (for tables exceeding one slab).
pub type MultiSlabTable<K> =
    Table<K, MultiSlabBuffer<Entry<ExtractorPersisted<K>, <K as KeyExtractor>::Value>>>;
