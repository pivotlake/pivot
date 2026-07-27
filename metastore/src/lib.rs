//! The provider-neutral **metastore** interface: the server's source of
//! configuration about which named datastores it serves.
//!
//! [`Metastore`] is the trait the server holds; the datastores it returns are
//! resolved however the implementation likes. Concrete providers live in sibling
//! crates so a TOML file, PostgreSQL, or another backend can be selected without
//! coupling this interface to its configuration format or datastore implementation.

use catalog::Datastore;
use dispatch::DataFlowDispatcher;
use std::collections::HashMap;
use std::sync::Arc;

/// The conventional name for a standalone datastore; see
/// [`planner::DEFAULT_DATASTORE_NAME`].
pub use catalog::DEFAULT_DATASTORE_NAME;

/// The server's configuration source. Today it defines the datastores to serve;
/// it is the seam future server configuration (users, secrets) would extend.
pub trait Metastore: Send + Sync {
    /// Open every configured datastore, keyed by name. Each is a
    /// [`Datastore`] over the datastore's object store; opening reads the
    /// tables' footers over `dispatcher`. The datastore named by
    /// [`default_datastore_name`](Self::default_datastore_name) must be present.
    fn open_datastores(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<HashMap<String, Arc<dyn Datastore>>>;

    /// The name of the default datastore: DuckDB's current database, the target
    /// of unqualified table names. Known from the parsed configuration, so this
    /// does no I/O.
    fn default_datastore_name(&self) -> &str;
}

/// Provider errors cross the trait boundary without making this crate depend on
/// a concrete metastore implementation.
pub type Error = Box<dyn std::error::Error + Send + Sync + 'static>;
pub type Result<T> = std::result::Result<T, Error>;
