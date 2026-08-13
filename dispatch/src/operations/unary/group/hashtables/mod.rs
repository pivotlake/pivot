//! Hash tables used by grouped aggregation.
//!
//! [`Table`] stores one key and aggregation value per group.
//! [`AggregatedTable`] owns the stack of tables one worker builds and decides
//! when to retire the active one.

use crate::operations::unary::group::hashtables::hash_table::BaseHashTable;
pub use crate::operations::unary::group::keys::KeyExtractor;
pub use crate::operations::unary::group::values::AggregationValue;

mod hash_table;
pub(crate) use hash_table::value_offset_for;
pub use hash_table::{LiveKey, MAX_LOAD_FACTOR, PersistedKey, Prober};

mod table_reader;
pub(crate) use table_reader::TableReader;

mod aggregated_table;
pub use aggregated_table::{AggregatedTable, AggregatedTableOutput, SpillConfig};

/// Initial number of slots for a new per-worker hash table.
pub const DEFAULT_CAPACITY: usize = 128;

/// A table whose persisted key type comes from `K`.
pub type Table<KP, V> = BaseHashTable<KP, V>;

/// A slab-backed table used both while consuming rows and while merging.
pub type MultiSlabTable<KP, V> = Table<KP, V>;
