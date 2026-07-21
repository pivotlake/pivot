//! The **cross-datastore catalog**: the layer that presents the set of named
//! datastores a server serves to the planner as one [`planner::catalog::Catalog`].
//!
//! A [`Datastore`] is one named data source (today a Delta table store; an
//! Iceberg or Unity store tomorrow), which each backend crate implements. This
//! crate does not know any concrete datastore kind; it works entirely over
//! `Arc<dyn Datastore>`.
//!
//! [`PivotCatalog`] holds the datastores keyed by name and implements
//! `Catalog`. Its `begin_transaction` opens one **sub-transaction per datastore**
//! (each datastore is independently snapshot-isolated) and bundles them into a
//! [`PivotTransaction`]; there is no cross-datastore atomic transaction. Query
//! resolution routes by the datastore qualifier of a `catalog.schema.table`
//! reference (or the default datastore when unqualified), and `create_table`
//! routes DDL to the datastore a `CREATE TABLE db.t` names.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use planner::TableFunction;
use planner::catalog::{
    Catalog, CatalogTransaction, CreateTableRequest, Error as CatalogError,
    Result as CatalogResult, Table,
};

/// The datastore name a catalog opened without an explicit one takes: the
/// database DuckDB attaches it as by default and the key it registers under. A
/// datastore with this name is required; it is DuckDB's current database, so
/// unqualified table names and DDL resolve against it.
pub const DEFAULT_DATASTORE_NAME: &str = "default";

/// One named data source the server serves: a Delta store, an Iceberg store, a
/// Unity store, and so on. Every kind is a [`Catalog`] (so the planner can bind
/// and scan its tables) plus the server-side lifecycle the composite needs.
///
/// Implemented by each backend crate (e.g. `datastore-delta`); this crate only
/// ever holds `Arc<dyn Datastore>`.
pub trait Datastore: Catalog {
    /// This datastore's name: its key in the metastore and the database DuckDB
    /// attaches it as.
    fn name(&self) -> &str;

    /// Bring the in-memory table set up to date with the backing store (new
    /// versions, new files, tables committed by other processes). Returns whether
    /// anything changed. A static datastore may be a no-op returning `Ok(false)`.
    fn refresh(&self) -> CatalogResult<bool>;

    /// Recover the concrete implementation. Server subsystems that are inherently
    /// format-specific (ingest, compaction, the dashboard) downcast the default
    /// datastore to its concrete type through this; a datastore kind that does
    /// not support them is simply not recovered.
    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no datastore named `{0}` (the default datastore) among the configured datastores")]
    MissingDefault(String),
    #[error("no datastore named `{0}`")]
    Unknown(String),
}

/// A set of named datastores presented to the planner as one [`Catalog`]. The
/// datastore named `default_name` is DuckDB's current database, so unqualified
/// names resolve against it.
#[derive(Debug)]
pub struct PivotCatalog {
    datastores: HashMap<String, Arc<dyn Datastore>>,
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
            return Err(Error::MissingDefault(default_name));
        }
        Ok(Self {
            datastores,
            default_name,
        })
    }

    /// A composite over a single datastore, which is therefore the default,
    /// keyed by the datastore's own name. For embedding one datastore (an
    /// ephemeral server, tests) without spelling out the map.
    pub fn single(datastore: Arc<dyn Datastore>) -> Self {
        let name = datastore.name().to_string();
        Self {
            datastores: HashMap::from([(name.clone(), datastore)]),
            default_name: name,
        }
    }

    /// The datastore named `name`, or `None`.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Datastore>> {
        self.datastores.get(name)
    }

    /// The default datastore: DuckDB's current database.
    pub fn default_datastore(&self) -> &Arc<dyn Datastore> {
        &self.datastores[&self.default_name]
    }

    /// The name of the default datastore.
    pub fn default_name(&self) -> &str {
        &self.default_name
    }

    /// Every datastore, name and handle: for wiring the planner's attach list
    /// and for introspection.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Arc<dyn Datastore>)> {
        self.datastores.iter()
    }

    /// Refresh every datastore to its latest committed state; returns whether any
    /// of them changed.
    pub fn refresh_all(&self) -> CatalogResult<bool> {
        let mut changed = false;
        for datastore in self.datastores.values() {
            changed |= datastore.refresh()?;
        }
        Ok(changed)
    }
}

impl Catalog for PivotCatalog {
    fn begin_transaction(&self) -> Arc<dyn CatalogTransaction> {
        let children = self
            .datastores
            .iter()
            .map(|(name, datastore)| (name.clone(), datastore.begin_transaction()))
            .collect();
        Arc::new(PivotTransaction {
            children,
            default_name: self.default_name.clone(),
        })
    }

    fn create_table(
        &self,
        request: CreateTableRequest,
        dispatcher: &DataFlowDispatcher,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        // Route to the datastore the statement named (`CREATE TABLE db.t`), or the
        // default when unqualified. DuckDB fills `catalog` with the resolved
        // database, so an unknown name here is an internal inconsistency.
        let target = request.catalog.as_deref().unwrap_or(&self.default_name);
        let datastore = self
            .datastores
            .get(target)
            .ok_or_else(|| CatalogError::Other(Box::new(Error::Unknown(target.to_string()))))?;
        datastore.create_table(request, dispatcher)
    }

    fn commit_transaction(&self, transaction: Arc<dyn CatalogTransaction>) -> CatalogResult<()> {
        // Route the commit to each datastore's own child transaction (a datastore
        // publishes the files an INSERT injected into its transaction).
        let named = transaction
            .as_any()
            .downcast_ref::<PivotTransaction>()
            .ok_or_else(|| {
                CatalogError::Other("commit_transaction on a non-PivotTransaction".into())
            })?;
        for (name, child) in &named.children {
            if let Some(datastore) = self.datastores.get(name) {
                datastore.commit_transaction(child.clone())?;
            }
        }
        Ok(())
    }

    fn rollback_transaction(&self, transaction: Arc<dyn CatalogTransaction>) {
        if let Some(named) = transaction.as_any().downcast_ref::<PivotTransaction>() {
            for (name, child) in &named.children {
                if let Some(datastore) = self.datastores.get(name) {
                    datastore.rollback_transaction(child.clone());
                }
            }
        }
    }
}

/// The [`CatalogTransaction`] a [`PivotCatalog`] opens: one child transaction
/// per datastore, keyed by datastore name. The unqualified `table`/
/// `table_function` resolve against the default datastore; the `*_in` variants
/// route by name (the path DuckDB's per-database binding takes). A backend's
/// table recovers its own datastore's transaction through
/// [`transaction_in`](CatalogTransaction::transaction_in).
#[derive(Debug)]
pub struct PivotTransaction {
    children: HashMap<String, Arc<dyn CatalogTransaction>>,
    default_name: String,
}

impl PivotTransaction {
    fn default_child(&self) -> &Arc<dyn CatalogTransaction> {
        &self.children[&self.default_name]
    }
}

impl CatalogTransaction for PivotTransaction {
    fn table(&self, name: &str) -> Option<Box<dyn Table>> {
        self.default_child().table(name)
    }

    fn table_in(&self, catalog: &str, name: &str) -> Option<Box<dyn Table>> {
        self.children.get(catalog)?.table(name)
    }

    fn table_function(&self, name: &str) -> Option<Box<dyn TableFunction>> {
        self.default_child().table_function(name)
    }

    fn table_function_in(&self, catalog: &str, name: &str) -> Option<Box<dyn TableFunction>> {
        self.children.get(catalog)?.table_function(name)
    }

    fn transaction_in(&self, catalog: &str) -> Option<&dyn CatalogTransaction> {
        self.children.get(catalog).map(|child| child.as_ref())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
