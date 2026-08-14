//! Hash tables used by grouped aggregation.
//!
//! [`Table`] stores one key and aggregation value per group.
//! [`AggregatedTable`] owns the stack of tables one worker builds and decides
//! when to retire the active one.

use crate::operations::unary::group::hashtables::hash_table::BaseHashTable;
pub use crate::operations::unary::group::keys::KeyExtractor;
pub use crate::operations::unary::group::values::AggregationValue;

mod hash_table;
pub use hash_table::{LiveKey, MAX_LOAD_FACTOR, PersistedKey};
pub(crate) use hash_table::{prefetch_l1_line, value_offset_for};

mod table_reader;
pub(crate) use table_reader::TableReader;

mod sorted_run;
pub use sorted_run::{BUCKET_BITS, SortedRun};

mod stub_run;
pub(crate) use stub_run::Stub;
pub use stub_run::StubRun;

mod aggregated_table;
pub use aggregated_table::{AggregatedTable, AggregatedTableOutput, MergeSource, SpillConfig};

/// Initial number of slots for a new per-worker hash table.
pub const DEFAULT_CAPACITY: usize = 128;

/// A table whose persisted key type comes from `K`.
pub type Table<KP, V> = BaseHashTable<KP, V>;

/// A slab-backed table used both while consuming rows and while merging.
pub type MultiSlabTable<KP, V> = Table<KP, V>;
