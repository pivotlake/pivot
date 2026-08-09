//! PivotDB's Delta datastore implementation, backed by Parquet data files.
//!
//! - [`DeltaDatastore`], the tables: a durable Pivot table index,
//!   per-table Delta Lake snapshots, and live row-group state in memory.
//! - [`parquet`], the engines: the per-query scan pipeline and the
//!   metadata-fetch (table load) pipeline, both dataflows over the dispatch
//!   worker pool.
//! - [`store`], the object-store backends (local filesystem, S3 and GCS)
//!   everything above persists through.

// Internal engine crate: the Parquet pipeline's public factories document their
// behaviour by linking to the private operators they build (e.g.
// `DecompressorFactory` → `Decompressor`). That's intentional here, we're not a
// published API, so allow public docs to reference private items.
#![allow(rustdoc::private_intra_doc_links)]

mod catalog;
mod compact;
mod delta;
mod manifest;
pub mod parquet;
pub mod store;
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
    Compacter, CompactionConfig, DEFAULT_COMPACT_BYTES, DEFAULT_COMPACT_POLL,
    DEFAULT_MIN_FILES_TO_MERGE, DEFAULT_REFRESH_INTERVAL, MaintenanceConfig, compact_table_files,
};
pub use manifest::{
    ColumnStatFilter, DeltaFileEntry, PartitionEqFilter, PartitionValues, pivot_scalar,
    scalar_values_equal, scalar_values_from_row,
};
pub use store::FileRef;
pub use vacuum::{DEFAULT_VACUUM_POLL, VacuumConfig, Vacuumer};
