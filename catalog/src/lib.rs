//! catalog: pivotdb's table layer over Parquet.
//!
//! - [`catalog`](ParquetCatalog) — the tables: a durable Pivot table index,
//!   per-table Delta Lake snapshots, and live row-group state in memory.
//! - [`parquet`] — the engines: the per-query scan pipeline and the
//!   metadata-fetch (table load) pipeline, both dataflows over the dispatch
//!   worker pool.
//! - [`store`] — the object-store backends (local fs, S3, GCS, in-memory)
//!   everything above persists through.

// Internal engine crate: the Parquet pipeline's public factories document their
// behaviour by linking to the private operators they build (e.g.
// `DecompressorFactory` → `Decompressor`). That's intentional here — we're not a
// published API — so allow public docs to reference private items.
#![allow(rustdoc::private_intra_doc_links)]

mod catalog;
mod delta;
mod manifest;
pub mod parquet;
pub mod parquet_writing;
pub mod store;
/// A Docker-backed object-store test harness (MinIO / fake-gcs-server). Gated
/// behind the `test-support` feature so it — and its heavy testcontainers deps —
/// never enter a normal build.
#[cfg(feature = "test-support")]
pub mod test_support;

pub use catalog::{
    CatalogSnapshot, CatalogTable, Error, ParquetCatalog, ParquetTransaction, Result, TableBinding,
};
pub use manifest::{
    ManifestEntry, PartitionEqFilter, SortBounds, pivot_scalar, scalar_values_equal,
    scalar_values_from_row,
};
pub use store::FileRef;
