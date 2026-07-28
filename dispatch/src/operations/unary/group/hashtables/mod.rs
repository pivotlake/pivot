//! Hash tables used by grouped aggregation.
//!
//! [`Table`] stores one key and aggregation value per group.
//! [`AggregatedTable`] owns the tables built by a single worker and decides
//! when to partition them for parallel merging.

use crate::operations::unary::group::hashtables::hash_table::BaseHashTable;
pub use crate::operations::unary::group::keys::KeyExtractor;
pub use crate::operations::unary::group::values::AggregationValue;

mod hash_table;
pub use hash_table::{LiveKey, MAX_LOAD_FACTOR, PersistedKey, Prober, entry_stride};

mod table_layout;
pub(crate) use table_layout::TableLayout;

mod scatter;
pub use scatter::{ScatterRows, SizedScatterRows, StridedScatterRows};

mod aggregated_table;
pub use aggregated_table::{AggregatedTable, AggregatedTableOutput, PartitionBuffers, RadixConfig};

/// Initial number of slots for a new per-worker hash table.
pub const DEFAULT_CAPACITY: usize = 128;

/// A table whose persisted key type comes from `K`.
pub type Table<K, V> = BaseHashTable<<K as KeyExtractor>::Persisted, V>;

/// A slab-backed table used both while consuming rows and while merging.
pub type MultiSlabTable<K, V> = Table<K, V>;
