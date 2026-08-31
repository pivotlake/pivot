//! Read-only Apache Iceberg datastore support.
//!
//! This crate joins REST catalog discovery, Iceberg snapshot planning, and
//! Pivot's Parquet execution so Iceberg tables can be queried like any other
//! datastore.

mod rest;
mod storage;
mod table;

pub use storage::PivotStorageFactory;
pub use table::{IcebergAuth, IcebergConfig, IcebergDatastore};
