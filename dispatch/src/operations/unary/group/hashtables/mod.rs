//! Hash table infrastructure for GROUP BY aggregation.
//!
//! This module wires together the generic [`BaseHashTable`] with the
//! [`KeyExtractor`] trait to produce concrete table types parameterized
//! by key extraction strategy.
//!
//! - [`Table<K, V, S>`] — a `BaseHashTable` whose key/value types are derived
//!   from `K: KeyExtractor`, generic over the storage backend `S`.
//! - [`MultiSlabTable<K, V>`] — a concrete [`Table`] backed by a
//!   [`MultiSlabBuffer`] (supports tables larger than one slab).
//! - [`AggregatedTable<K, V>`] — the per-worker accumulator used during the
//!   consume phase.

use std::ops::{Index, IndexMut};

use crate::memory::MultiSlabBuffer;
use crate::operations::unary::group::hashtables::hash_table::BaseHashTable;
pub use crate::operations::unary::group::keys::KeyExtractor;
pub use crate::operations::unary::group::values::AggregationValue;

mod hash_table;
pub use hash_table::{Entry, LiveKey, MAX_LOAD_FACTOR, PersistedKey};

mod aggregated_table;
pub use aggregated_table::{AggregatedTable, AggregatedTableOutput, PartitionBuffers, RadixConfig};

/// Initial number of slots for a new per-worker hash table.
pub const DEFAULT_CAPACITY: usize = 128;

/// The `Entry` stored in a table for a given key extractor and aggregation value.
pub type ExtractorEntry<K, V> = Entry<<K as KeyExtractor>::Persisted, V>;

/// Marker trait for any backing storage that can index a key/value `Entry` by
/// `usize`.
///
/// Automatically implemented for anything that implements
/// `Index<usize> + IndexMut<usize>` with the right output type, so
/// `Vec`, `SlabBuffer`, and `MultiSlabBuffer` all qualify.
pub trait TableStorage<K: KeyExtractor + ?Sized, V: AggregationValue>:
    Index<usize, Output = ExtractorEntry<K, V>> + IndexMut<usize>
{
}

impl<K: KeyExtractor + ?Sized, V: AggregationValue, T> TableStorage<K, V> for T where
    T: Index<usize, Output = ExtractorEntry<K, V>> + IndexMut<usize>
{
}

/// A [`BaseHashTable`] parameterized by a key extractor, an aggregation value,
/// and a storage backend.
pub type Table<K, V, S> = BaseHashTable<<K as KeyExtractor>::Persisted, V, S>;

/// A [`Table`] backed by multiple slab buffers (for tables exceeding one slab).
pub type MultiSlabTable<K, V> = Table<K, V, MultiSlabBuffer<ExtractorEntry<K, V>>>;
