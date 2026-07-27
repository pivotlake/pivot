//! The **cross-datastore catalog**: the layer that presents the set of named
//! datastores a server serves to the planner as one [`planner::catalog::Catalog`].
//!
//! [`PivotCatalog`] holds the datastores keyed by name as `Arc<dyn Datastore>`
//! and implements `Catalog`. Its `begin_transaction` returns a
//! [`PivotTransaction`] that opens a datastore's sub-transaction **lazily**, the
//! first time a query touches that datastore (each datastore is independently
//! snapshot-isolated); there is no cross-datastore atomic transaction. Query
//! resolution routes by the datastore qualifier of a `datastore.schema.table`
//! reference (or the default datastore when unqualified), and the transaction's
//! `create_table` routes DDL to the datastore a `CREATE TABLE db.t` names.
//!
//! Commit iterates only the datastores the query actually touched, awaiting each
//! one's own commit; a datastore decides for itself whether that commit does
//! blocking store I/O or is an in-memory no-op.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use datastore::DatastoreTransaction;
use planner::TableFunction;
use planner::catalog::{
    Catalog, CatalogTransaction, CreateTableRequest, Error as CatalogError,
    Result as CatalogResult, Table, TableCreation,
};

/// One named data source served by pivotdb. Re-exported from `datastore`, where
/// the trait lives; concrete backends (e.g. `datastore_delta::DeltaDatastore`)
/// implement it and are held here behind `Arc<dyn Datastore>`.
pub use datastore::Datastore;
/// The datastore name a datastore opened without an explicit one takes: the
/// database DuckDB attaches it as by default and the key it registers under. A
/// datastore with this name is required; it is DuckDB's current database, so
/// unqualified table names and DDL resolve against it. Defined by `planner` (the
/// lowest crate that names it) and re-exported here.
pub use planner::DEFAULT_DATASTORE_NAME;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no datastore named `{0}` (the default datastore) among the configured datastores")]
    MissingDefaultDatastore(String),
    #[error("no datastore named `{0}`")]
    UnknownDatastore(String),
}

/// A set of named datastores presented to the planner as one [`Catalog`]. The
/// datastore named `default_name` is DuckDB's current database, so unqualified
/// names resolve against it.
#[derive(Debug, Clone)]
pub struct PivotCatalog {
    /// Shared (`Arc`) so a [`PivotTransaction`] can hold the same map and open a
    /// datastore's sub-transaction lazily, the first time a query touches it.
    datastores: Arc<HashMap<String, Arc<dyn Datastore>>>,
    default_name: String,
}

impl PivotCatalog {
    /// Build the composite from datastores keyed by name. Errors if no datastore
    /// is named `default_name`: that datastore is DuckDB's current database and
    /// the target of unqualified DDL, so it must exist.
    pub fn new(
        datastores: HashMap<String, Arc<dyn Datastore>>,
        default_name: String,
    ) -> Result<Self> {
        if !datastores.contains_key(&default_name) {
            return Err(Error::MissingDefaultDatastore(default_name));
        }
        Ok(Self {
            datastores: Arc::new(datastores),
            default_name,
        })
    }

    /// The datastore named `name`, or `None`.
    pub fn get_datastore(&self, name: &str) -> Option<&Arc<dyn Datastore>> {
        self.datastores.get(name)
    }

    /// The default datastore: DuckDB's current database.
    pub fn default_datastore(&self) -> &Arc<dyn Datastore> {
        &self.datastores[&self.default_name]
    }

    /// The name of the default datastore.
    pub fn default_datastore_name(&self) -> &str {
        &self.default_name
    }

    /// Every datastore, name and handle: for wiring the planner's attach list
    /// and for introspection.
    pub fn iter_datastores(&self) -> impl Iterator<Item = (&String, &Arc<dyn Datastore>)> {
        self.datastores.iter()
    }

    /// Start every datastore's background maintenance. Called once when the
    /// server begins serving, so maintenance spawns onto the serving runtime.
    pub fn start(&self) {
        for datastore in self.datastores.values() {
            datastore.clone().start();
        }
    }

    /// Stop every datastore's background maintenance. Called on shutdown before
    /// the worker pool is torn down, so no maintenance sweep races the teardown.
    pub fn abort(&self) {
        for datastore in self.datastores.values() {
            datastore.abort();
        }
    }

    /// The datastore sub-transactions a [`PivotTransaction`] lazily opened, as
    /// `(name, child)` pairs, with the lock dropped before the caller uses them:
    /// commit and rollback both iterate these and must not hold the transaction's
    /// lock across a datastore's own commit.
    fn touched_children(
        transaction: &Arc<dyn CatalogTransaction>,
    ) -> Vec<(String, Arc<dyn DatastoreTransaction>)> {
        let Some(named) = transaction.as_any().downcast_ref::<PivotTransaction>() else {
            return Vec::new();
        };
        named
            .children
            .lock()
            .unwrap()
            .iter()
            .map(|(name, child)| (name.clone(), child.clone()))
            .collect()
    }
}

#[async_trait]
impl Catalog for PivotCatalog {
    fn begin_transaction(&self) -> Arc<dyn CatalogTransaction> {
        Arc::new(PivotTransaction {
            datastores: self.datastores.clone(),
            default_name: self.default_name.clone(),
            children: Mutex::new(HashMap::new()),
        })
    }

    async fn commit_transaction(
        &self,
        transaction: Arc<dyn CatalogTransaction>,
    ) -> CatalogResult<()> {
        // Commit only the datastores this query touched, awaiting each one's own
        // commit. A datastore decides for itself whether that commit does blocking
        // store I/O (hopping to the blocking pool) or is an in-memory no-op it can
        // finish inline, so there is no offload gate here.
        for (name, child) in Self::touched_children(&transaction) {
            self.datastores[&name].commit_transaction(child).await?;
        }
        Ok(())
    }

    fn rollback_transaction(&self, transaction: Arc<dyn CatalogTransaction>) {
        for (name, child) in Self::touched_children(&transaction) {
            self.datastores[&name].rollback_transaction(child);
        }
    }
}

/// The [`CatalogTransaction`] a [`PivotCatalog`] opens. It holds the datastore
/// map and opens a datastore's child [`DatastoreTransaction`] **lazily**, the
/// first time the query touches that datastore (reusing it thereafter), so a
/// query that reads one datastore never snapshots the others. The `bind_table`
/// / `bind_table_function` resolutions route by name (the path DuckDB's
/// per-database binding takes); the unqualified `bind_default_table_function`
/// and `create_table` fall to the default datastore. Each resolved table binding
/// is self-contained: it captures its datastore's snapshot at bind time, so the
/// composite needs no downcast back to a per-datastore transaction at compile.
#[derive(Debug)]
pub struct PivotTransaction {
    datastores: Arc<HashMap<String, Arc<dyn Datastore>>>,
    default_name: String,
    /// The child transactions opened so far, keyed by datastore name. Populated
    /// on first touch during binding (`&self`, hence the lock) and read back at
    /// commit to publish only the datastores the query used.
    children: Mutex<HashMap<String, Arc<dyn DatastoreTransaction>>>,
}

impl PivotTransaction {
    /// The child transaction for `datastore`, opening (and remembering) one on
    /// first touch and reusing it thereafter, or `None` if no such datastore
    /// exists. Every table a query binds in one datastore shares this single
    /// child, so they read one consistent snapshot of it.
    fn child(&self, datastore: &str) -> Option<Arc<dyn DatastoreTransaction>> {
        let mut children = self.children.lock().unwrap();
        if let Some(existing) = children.get(datastore) {
            return Some(existing.clone());
        }
        let child = self.datastores.get(datastore)?.begin_transaction();
        children.insert(datastore.to_string(), child.clone());
        Some(child)
    }
}

impl CatalogTransaction for PivotTransaction {
    fn bind_table(&self, datastore: &str, name: &str) -> Option<Box<dyn Table>> {
        self.child(datastore)?.bind_table(name)
    }

    fn bind_table_function(&self, datastore: &str, name: &str) -> Option<Box<dyn TableFunction>> {
        self.child(datastore)?.bind_table_function(name)
    }

    fn bind_default_table_function(&self, name: &str) -> Option<Box<dyn TableFunction>> {
        self.child(&self.default_name)?.bind_table_function(name)
    }

    fn bind_create_table(
        &self,
        request: CreateTableRequest,
    ) -> CatalogResult<Box<dyn TableCreation>> {
        // Route to the datastore the statement named (`CREATE TABLE db.t`), or the
        // default when unqualified. An explicit name that matches no datastore is
        // an internal inconsistency.
        let target = request
            .datastore_name
            .clone()
            .unwrap_or_else(|| self.default_name.clone());
        let child = self.child(&target).ok_or_else(|| {
            CatalogError::Other(Box::new(Error::UnknownDatastore(target.clone())))
        })?;
        child.bind_create_table(request)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datastore_delta::DeltaDatastore;
    use dispatch::Dispatch;

    #[test]
    fn requires_the_default_datastore() {
        let error =
            PivotCatalog::new(HashMap::new(), DEFAULT_DATASTORE_NAME.to_string()).unwrap_err();

        assert!(
            matches!(error, Error::MissingDefaultDatastore(name) if name == DEFAULT_DATASTORE_NAME)
        );
    }

    #[test]
    fn one_datastore_catalog_exposes_the_datastore() {
        let dispatch = Dispatch::spin_up(1, 32, None);
        let directory = tempfile::tempdir().unwrap();
        let datastore: Arc<dyn Datastore> =
            DeltaDatastore::open_local(directory.path(), dispatch.dispatcher()).unwrap();

        assert_eq!(datastore.name(), DEFAULT_DATASTORE_NAME);
        assert!(!datastore.refresh().unwrap());

        let catalog = PivotCatalog::new(
            HashMap::from([(DEFAULT_DATASTORE_NAME.to_string(), datastore)]),
            DEFAULT_DATASTORE_NAME.to_string(),
        )
        .unwrap();

        assert_eq!(catalog.default_name(), DEFAULT_DATASTORE_NAME);
        assert!(catalog.get_datastore(DEFAULT_DATASTORE_NAME).is_some());
        dispatch.exit();
    }
}
