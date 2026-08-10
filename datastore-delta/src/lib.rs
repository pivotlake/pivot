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

/// Run one blocking unit of work (store I/O, a Delta log write, a dataflow
/// drive) on tokio's blocking pool and await it. A panic in the closure is
/// resumed on the caller rather than swallowed; a cancelled task (runtime
/// shutdown) panics too, because the caller cannot distinguish "not done" from
/// "done" and must not carry on as if the work happened.
pub(crate) async fn run_blocking<T, F>(work: F) -> T
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    match tokio::task::spawn_blocking(work).await {
        Ok(value) => value,
        Err(join_error) => match join_error.try_into_panic() {
            Ok(panic) => std::panic::resume_unwind(panic),
            Err(join_error) => panic!("blocking task cancelled: {join_error}"),
        },
    }
}
