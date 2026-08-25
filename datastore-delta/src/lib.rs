//! PivotDB's Delta datastore implementation, backed by Parquet data files.
//!
//! - [`DeltaDatastore`], the tables: a durable Pivot table index,
//!   per-table Delta Lake snapshots, and live row-group state in memory.
//!
//! The reusable object-store and Parquet engines live in `object-storage` and
//! `parquet-engine`; this crate owns only Delta catalog and transaction policy.

mod catalog;
mod compact;
mod delta;
mod manifest;
/// A Docker-backed object-store test harness (MinIO). Gated
/// behind the `test-support` feature so it, and its heavy testcontainers deps,
/// never enter a normal build.
#[cfg(feature = "test-support")]
pub mod test_support;
mod vacuum;

pub use catalog::{
    CatalogTable, DataFileInfo, DeltaDatastore, DeltaSnapshot, DeltaTransaction, Error, Result,
    TableBinding,
};
pub use compact::{
    CompacterHandle, CompactionConfig, DEFAULT_COMPACT_BYTES, DEFAULT_COMPACT_POLL,
    DEFAULT_MIN_FILES_TO_MERGE, DEFAULT_REFRESH_INTERVAL, MaintenanceConfig, compact_table_files,
    default_merge_target_bytes,
};
pub use manifest::{
    ColumnStatFilter, DeltaFileEntry, PartitionEqFilter, PartitionValues, pivot_scalar,
    scalar_values_equal, scalar_values_from_row,
};
pub use object_storage::FileRef;
pub use vacuum::{DEFAULT_VACUUM_POLL, VacuumConfig, Vacuumer};
