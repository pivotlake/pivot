//! An Iceberg REST catalog backend for the planner: resolve table names against
//! an existing Iceberg REST catalog and scan their Parquet data files through
//! the `catalog` crate's read pipeline.
//!
//! The moving parts, one module each:
//!
//! - [`client`] - the REST protocol (`/v1/config`, `loadTable`) over blocking
//!   HTTP, mirroring `catalog::store`'s no-async-runtime convention.
//! - [`metadata`] - the table-metadata JSON (schema, current snapshot) and the
//!   mapping from Iceberg column types to planner [`Type`](planner::types::Type)s.
//! - [`manifest`] - the Avro manifest list and manifest files a snapshot's data
//!   file set is recorded in.
//! - [`warehouse`] - resolving the absolute `s3://`/`gs://`/`file://` URIs those
//!   files sit at onto `catalog::store` backends the engine can range-read.
//! - [`rest_catalog`] - the [`IcebergRestCatalog`] tying it together: a
//!   [`planner::catalog::Catalog`] whose tables pin one snapshot per query.
//!
//! This backend is read-only: `CREATE TABLE` is rejected, and a table carrying
//! delete files (Iceberg row-level deletes) fails its scan rather than
//! returning undeleted rows.

// The crate overview above documents the internal layout by linking to the
// private modules; this is not a published API, so allow it (as catalog does).
#![allow(rustdoc::private_intra_doc_links)]

mod client;
mod manifest;
mod metadata;
mod rest_catalog;
mod warehouse;

pub use rest_catalog::{IcebergRestCatalog, IcebergRestConfig};

/// Everything that can go wrong resolving or scanning an Iceberg table. All of
/// it surfaces to the caller (a failed resolve is logged and binds as "no such
/// table"; a failed scan fails the query) - never a silent empty result.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("iceberg rest request to `{url}` failed: {message}")]
    Http { url: String, message: String },
    #[error("unexpected response from `{url}`: {message}")]
    UnexpectedResponse { url: String, message: String },
    #[error(
        "column `{column}` of table `{table}` has iceberg type `{iceberg_type}`, which pivot does not support"
    )]
    UnsupportedColumnType {
        table: String,
        column: String,
        iceberg_type: String,
    },
    #[error("unsupported iceberg feature: {0}")]
    Unsupported(String),
    #[error(
        "a data file of table `{table}` does not match its schema ({detail}); iceberg schema evolution is not supported"
    )]
    DataFileSchemaMismatch { table: String, detail: String },
    #[error("invalid table metadata: {0}")]
    Metadata(String),
    #[error(
        "unsupported data file uri `{0}` (expected s3://, gs://, file://, or an absolute path)"
    )]
    UnsupportedUri(String),
    #[error("`{uri}` does not exist in the object store")]
    MissingObject { uri: String },
    #[error("parsing avro manifest: {0}")]
    Avro(#[from] apache_avro::Error),
    #[error(transparent)]
    Store(#[from] catalog::store::StoreError),
    #[error("loading data file footers: {0}")]
    Load(#[from] dispatch::DataFlowError),
    #[error("CREATE TABLE is not supported on an iceberg rest catalog (it is read-only)")]
    CreateTableUnsupported,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl From<Error> for planner::catalog::Error {
    fn from(value: Error) -> Self {
        planner::catalog::Error::Other(Box::new(value))
    }
}
