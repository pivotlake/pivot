//! The **metastore**: the server's source of configuration about *where* its
//! data lives and how to reach it. Its first responsibility is defining the set
//! of named **datastores** the server serves — each a local directory or S3
//! bucket, opened as a [`ParquetCatalog`] and attached to DuckDB
//! as its own database. The datastore named [`DEFAULT_DATASTORE_NAME`] is the
//! current database, so unqualified table names resolve against it.
//!
//! [`Metastore`] is the trait the server holds; the datastores it returns are
//! resolved however the implementation likes. The first (and only) implementation
//! is [`TomlMetastore`], which reads them from a TOML file on disk — the metastore
//! itself lives on disk even when the data it points at sits on S3.

mod toml_store;

use std::collections::HashMap;
use std::sync::Arc;

use catalog::Datastore;
use dispatch::DataFlowDispatcher;

pub use catalog::DEFAULT_DATASTORE_NAME;
pub use toml_store::TomlMetastore;

/// The server's configuration source. Today it defines the datastores to serve;
/// it is the seam future server configuration (users, secrets) would extend.
pub trait Metastore: Send + Sync {
    /// Open every configured datastore, keyed by name. Each is a
    /// [`Datastore`] over the datastore's object store; opening reads the
    /// tables' footers over `dispatcher`. A datastore named
    /// [`DEFAULT_DATASTORE_NAME`] must be present.
    fn datastores(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<HashMap<String, Arc<dyn Datastore>>>;
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("reading metastore file `{path}`: {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },
    #[error("parsing metastore file `{path}`: {source}")]
    Parse {
        path: String,
        source: toml::de::Error,
    },
    #[error("datastore `{name}`: {message}")]
    Datastore { name: String, message: String },
    #[error(
        "no `{DEFAULT_DATASTORE_NAME}` datastore is defined; one is required (it is the default database)"
    )]
    MissingDefault,
    #[error(transparent)]
    Store(#[from] datastore_delta::store::StoreError),
    #[error(transparent)]
    Catalog(#[from] datastore_delta::Error),
}
