//! Hash table infrastructure for GROUP BY aggregation.
//!
//! This module wires together the generic [`BaseHashTable`] with the
//! [`KeyExtractor`] trait to produce concrete table types parameterized
//! by key extraction strategy.
//!
//! - [`Table<K, V>`] — a `BaseHashTable` whose key type is derived from
//!   `K: KeyExtractor`.
//! - [`AggregatedTable<K, V>`] — the per-worker accumulator used during the
//!   consume phase.

use crate::operations::unary::group::hashtables::hash_table::BaseHashTable;
pub use crate::operations::unary::group::keys::KeyExtractor;
pub use crate::operations::unary::group::values::AggregationValue;

mod hash_table;
pub use hash_table::{LiveKey, MAX_LOAD_FACTOR, PersistedKey, Prober};

mod aggregated_table;
pub use aggregated_table::{AggregatedTable, AggregatedTableOutput, PartitionBuffers, RadixConfig};

/// Initial number of slots for a new per-worker hash table.
pub const DEFAULT_CAPACITY: usize = 128;

/// A [`BaseHashTable`] parameterized by a key extractor and an aggregation
/// value.
pub type Table<K, V> = BaseHashTable<<K as KeyExtractor>::Persisted, V>;

/// The per-worker and merge-phase table type (slab-backed, may span slabs).
pub type MultiSlabTable<K, V> = Table<K, V>;
