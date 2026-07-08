//! catalog: pivotdb's table layer over Parquet, with Delta Lake as the table
//! format.
//!
//! - [`catalog`](ParquetCatalog), the tables: each one a Delta table whose
//!   transaction log ([`delta`]) holds its schema, specs, and file list, plus
//!   live row-group state in memory, reloaded up to the latest log version at
//!   every query bind. The [`db_index`] names the tables of one database.
//! - [`parquet`] — the engines: the per-query scan pipeline and the
//!   metadata-fetch (table load) pipeline, both dataflows over the dispatch
//!   worker pool.
//! - [`parquet_writing`] - the write-side engine: the streaming pipeline that
//!   encodes `RecordBatch` dataflows into Parquet files, feeding `INSERT`
//!   ([`ParquetCatalog`]'s `insert`) and external compaction alike.
//! - [`store`], the object-store backends (local fs, S3, GCS) everything
//!   above persists through.

// Internal engine crate: the Parquet pipeline's public factories document their
// behaviour by linking to the private operators they build (e.g.
// `DecompressorFactory` → `Decompressor`). That's intentional here — we're not a
// published API — so allow public docs to reference private items.
#![allow(rustdoc::private_intra_doc_links)]

mod catalog;
mod db_index;
mod delta;
pub mod parquet;
pub mod parquet_writing;
pub mod store;
/// A Docker-backed object-store test harness (MinIO / fake-gcs-server). Gated
/// behind the `test-support` feature so it — and its heavy testcontainers deps —
/// never enter a normal build.
#[cfg(feature = "test-support")]
pub mod test_support;

pub use catalog::{CatalogTable, Error, ParquetCatalog, Result, TableBinding};
pub use delta::{ManifestEntry, PartitionEqFilter, SortBounds};
pub use store::FileRef;
