//! goose: pivotdb's table layer over Parquet.
//!
//! - [`catalog`](ParquetCatalog) — the tables: durable definitions in the
//!   [`manifest`], durable per-table file lists in the [`table_log`], live
//!   row-group state in memory, reloaded up to the latest log version at
//!   every query bind.
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
pub mod manifest;
pub mod parquet;
mod sql_type;
pub mod store;
pub mod table_log;

pub use catalog::{Error, ParquetCatalog, RegisterOutcome, Result, TableBinding, TableStore};
pub use manifest::ManifestEntry;
pub use table_log::LoggedFile;
