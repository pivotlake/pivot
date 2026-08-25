//! The **cross-datastore catalog**: the layer that presents the set of named
//! datastores a server serves to the planner.
//!
//! [`PivotCatalog`] holds the datastores keyed by name as `Arc<dyn Datastore>`.
//! Its `begin_transaction` returns a
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

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use crossbeam_deque::{Injector, Steal};
use datastore::DatastoreTransaction;
use datastore_system::{DatastoreEntry, SystemTransaction};
use dispatch::{DataFlowDispatcher, OneShotNullaryFactory, RecordBatchOperatorSpec};
use metastore::Metastore;
use object_storage::ExternalStoreFactory;
use planner::catalog::{
    BoundTable, CatalogTransaction, CreateSchemaRequest, CreateTableRequest, CreateUserRequest,
    DropTableRequest, Error as CatalogError, Result as CatalogResult, SchemaCreation,
    TableCreation, TableDrop, TableReference, TableRevision, UserCreation,
};

/// One named data source served by pivotdb. Re-exported from `datastore`, where
/// the trait lives; concrete backends (e.g. `datastore_delta::DeltaDatastore`)
/// implement it and are held here behind `Arc<dyn Datastore>`.
pub use datastore::Datastore;

/// The conventional name for a single-datastore configuration's datastore; see
/// [`planner::DEFAULT_DATASTORE_NAME`].
pub use planner::DEFAULT_DATASTORE_NAME;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no datastore named `{0}` (the default datastore) among the configured datastores")]
    MissingDefaultDatastore(String),
    #[error("no datastore named `{0}`")]
    UnknownDatastore(String),
}

/// Process-level dependencies for reading files outside the configured
/// datastores. Kept on the composite catalog because external Parquet is a
/// global table function, while the chosen credential policy belongs to the
/// embedding process.
#[derive(Clone)]
struct ExternalParquetContext {
    dispatcher: DataFlowDispatcher,
    store_factory: Arc<dyn ExternalStoreFactory>,
}

impl std::fmt::Debug for ExternalParquetContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExternalParquetContext")
            .field("store_factory", &self.store_factory)
            .finish_non_exhaustive()
    }
}

/// A set of named datastores presented to the planner. The
/// datastore named `default_name` is DuckDB's current database, so unqualified
/// names resolve against it.
#[derive(Debug, Clone)]
pub struct PivotCatalog {
    /// Shared (`Arc`) so a [`PivotTransaction`] can hold the same map and open a
    /// datastore's sub-transaction lazily, the first time a query touches it.
    datastores: Arc<HashMap<String, Arc<dyn Datastore>>>,
    default_name: String,
    /// Where metastore-changing statements route. The metastore's definitions
    /// are server-wide, not a datastore's, so changing them doesn't ride a
    /// sub-transaction.
    metastore: Arc<dyn Metastore>,
    /// Process-wide external parquet access. This is global to the composite
    /// catalog rather than associated with one datastore.
    external_parquet_read_context: Option<ExternalParquetContext>,
}

impl PivotCatalog {
    /// Build the composite from datastores keyed by name. Errors if no datastore
    /// is named `default_name`: that datastore is DuckDB's current database and
    /// the target of unqualified DDL, so it must exist.
    pub fn new(
        datastores: HashMap<String, Arc<dyn Datastore>>,
        default_name: String,
        metastore: Arc<dyn Metastore>,
    ) -> Result<Self> {
        if !datastores.contains_key(&default_name) {
            return Err(Error::MissingDefaultDatastore(default_name));
        }
        Ok(Self {
            datastores: Arc::new(datastores),
            default_name,
            metastore,
            external_parquet_read_context: None,
        })
    }

    /// Enable the planner-owned `read_parquet(path)` function with this
    /// process's dispatcher and external-store credential policy.
    pub fn with_external_parquet_read_context(
        mut self,
        dispatcher: &DataFlowDispatcher,
        store_factory: Arc<dyn ExternalStoreFactory>,
    ) -> Self {
        self.external_parquet_read_context = Some(ExternalParquetContext {
            dispatcher: dispatcher.clone(),
            store_factory,
        });
        self
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

    /// Every datastore, name and handle: for reaching one by name and for
    /// introspection.
    pub fn iter_datastores(&self) -> impl Iterator<Item = (&String, &Arc<dyn Datastore>)> {
        self.datastores.iter()
    }

    pub fn datastore_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.datastores.keys().cloned().collect();
        names.push(datastore_system::DATASTORE_NAME.to_string());
        names
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

    /// Open a transaction over these datastores: a [`PivotTransaction`] that opens
    /// each datastore's sub-transaction lazily, the first time a query touches it.
    /// Commit and rollback are methods on the returned transaction itself.
    pub fn begin_transaction(&self) -> Arc<dyn CatalogTransaction> {
        Arc::new(PivotTransaction {
            datastores: self.datastores.clone(),
            default_name: self.default_name.clone(),
            sub_transactions: Mutex::new(HashMap::new()),
            metastore: self.metastore.clone(),
            external_parquet_read_context: self.external_parquet_read_context.clone(),
            staged_users: Arc::new(Injector::new()),
        })
    }
}

/// The [`CatalogTransaction`] a [`PivotCatalog`] opens. It holds the datastore
/// map and opens a datastore's sub-transaction [`DatastoreTransaction`] **lazily**, the
/// first time the query touches that datastore (reusing it thereafter), so an
/// ordinary query that reads one datastore never snapshots the others. Global
/// virtual catalog relations deliberately request every datastore transaction.
/// The `bind_table` resolution routes by name (the path DuckDB's per-database
/// binding takes); the unqualified `create_table` falls to the default
/// datastore. Each resolved table binding is self-contained: it captures its
/// datastore's snapshot at bind time, so the composite needs no downcast back
/// to a per-datastore transaction at compile.
#[derive(Debug)]
pub struct PivotTransaction {
    datastores: Arc<HashMap<String, Arc<dyn Datastore>>>,
    default_name: String,
    /// The datastore sub-transactions opened so far, keyed by datastore name.
    /// Populated on first touch during binding (`&self`, hence the lock) and read
    /// back at commit to publish only the datastores the query used.
    sub_transactions: Mutex<HashMap<String, Arc<dyn DatastoreTransaction>>>,
    /// Where staged users land at commit.
    metastore: Arc<dyn Metastore>,
    external_parquet_read_context: Option<ExternalParquetContext>,
    /// The users this transaction's dataflows staged (see
    /// [`PivotUserCreation`]), applied to the metastore at commit and dropped
    /// on rollback. Shared (`Arc`) with the staging dataflow's workers.
    staged_users: Arc<Injector<CreateUserRequest>>,
}

impl PivotTransaction {
    /// The sub-transaction for `datastore`, opening (and remembering) one on first
    /// touch and reusing it thereafter, or `None` if no such datastore exists.
    /// Every table a query binds in one datastore shares this single
    /// sub-transaction, so they read one consistent snapshot of it.
    fn find_or_create_sub_transaction(
        &self,
        datastore: &str,
    ) -> Option<Arc<dyn DatastoreTransaction>> {
        if datastore == datastore_system::DATASTORE_NAME {
            return Some(self.find_or_create_system_transaction());
        }
        let mut sub_transactions = self.sub_transactions.lock().unwrap();
        if let Some(existing) = sub_transactions.get(datastore) {
            return Some(existing.clone());
        }
        let sub_transaction = self.datastores.get(datastore)?.clone().begin_transaction();
        sub_transactions.insert(datastore.to_string(), sub_transaction.clone());
        Some(sub_transaction)
    }

    /// The `system` sub-transaction, which reads every other datastore and so
    /// opens them all. They are opened before the map is locked: opening one
    /// locks it too, and this lock is not reentrant.
    fn find_or_create_system_transaction(&self) -> Arc<dyn DatastoreTransaction> {
        if let Some(existing) = self
            .sub_transactions
            .lock()
            .unwrap()
            .get(datastore_system::DATASTORE_NAME)
        {
            return existing.clone();
        }
        let datastores = self
            .list_datastores()
            .into_iter()
            .map(|datastore_name| {
                let transaction = self
                    .find_or_create_sub_transaction(&datastore_name)
                    .expect("a configured datastore can always open a transaction");
                let datastore = &self.datastores[&datastore_name];
                DatastoreEntry {
                    name: datastore_name,
                    kind: datastore.kind().to_string(),
                    data_path: datastore.data_path(),
                    transaction,
                }
            })
            .collect();
        self.sub_transactions
            .lock()
            .unwrap()
            .entry(datastore_system::DATASTORE_NAME.to_string())
            .or_insert_with(|| Arc::new(SystemTransaction::new(datastores)))
            .clone()
    }

    /// The sub-transactions opened so far, as `(name, sub-transaction)` pairs, with
    /// the lock dropped before returning: commit and rollback iterate these and
    /// must not hold the lock across a datastore's own commit.
    fn opened_sub_transactions(&self) -> Vec<(String, Arc<dyn DatastoreTransaction>)> {
        self.sub_transactions
            .lock()
            .unwrap()
            .iter()
            .map(|(name, sub)| (name.clone(), sub.clone()))
            .collect()
    }

    /// Every configured datastore name in deterministic order.
    fn list_datastores(&self) -> Vec<String> {
        let mut datastore_names: Vec<_> = self.datastores.keys().cloned().collect();
        datastore_names.sort();
        datastore_names
    }
}

#[async_trait]
impl CatalogTransaction for PivotTransaction {
    fn does_schema_exist(&self, datastore: &str, schema: &str) -> bool {
        self.find_or_create_sub_transaction(datastore)
            .is_some_and(|sub_transaction| sub_transaction.does_schema_exist(schema))
    }

    fn bind_table(&self, reference: &TableReference) -> Option<Box<dyn BoundTable>> {
        let sub_transaction = self.find_or_create_sub_transaction(&reference.datastore)?;
        sub_transaction.bind_table(&reference.datastore, &reference.schema_qualified_name())
    }

    fn bind_read_parquet(&self, location: &str) -> CatalogResult<Box<dyn BoundTable>> {
        let context = self.external_parquet_read_context.as_ref().ok_or_else(|| {
            CatalogError::Other("read_parquet is not configured for this catalog".into())
        })?;
        parquet_engine::bind_read_parquet(
            &context.dispatcher,
            context.store_factory.as_ref(),
            location,
        )
    }

    fn table_revision(&self, reference: &TableReference) -> Option<TableRevision> {
        let sub_transaction = self.find_or_create_sub_transaction(&reference.datastore)?;
        sub_transaction.table_revision(&reference.schema_qualified_name())
    }

    async fn compact(
        &self,
        datastore: &str,
        table: &planner::catalog::SchemaQualifiedTableName,
        final_sweep: bool,
    ) -> CatalogResult<u64> {
        let transaction = self
            .find_or_create_sub_transaction(datastore)
            .ok_or_else(|| {
                CatalogError::Other(Box::new(Error::UnknownDatastore(datastore.into())))
            })?;
        transaction.compact(table, final_sweep).await
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
        let sub_transaction = self
            .find_or_create_sub_transaction(&target)
            .ok_or_else(|| {
                CatalogError::Other(Box::new(Error::UnknownDatastore(target.clone())))
            })?;
        sub_transaction.bind_create_table(request)
    }

    fn bind_drop_table(&self, request: DropTableRequest) -> CatalogResult<Box<dyn TableDrop>> {
        // Route to the datastore the statement resolved (`DROP TABLE db.t`), or
        // the default when unqualified.
        let target = request
            .datastore_name
            .clone()
            .unwrap_or_else(|| self.default_name.clone());
        let sub_transaction = self
            .find_or_create_sub_transaction(&target)
            .ok_or_else(|| {
                CatalogError::Other(Box::new(Error::UnknownDatastore(target.clone())))
            })?;
        sub_transaction.bind_drop_table(request)
    }

    fn bind_create_schema(
        &self,
        request: CreateSchemaRequest,
    ) -> CatalogResult<Box<dyn SchemaCreation>> {
        // DuckDB keeps an unqualified CREATE SCHEMA's catalog unresolved in the
        // logical operator and normally applies the current database during
        // physical execution. Pivot executes the logical operator itself, so it
        // applies the same default here.
        let target = request
            .datastore_name
            .clone()
            .unwrap_or_else(|| self.default_name.clone());
        let sub_transaction = self
            .find_or_create_sub_transaction(&target)
            .ok_or_else(|| {
                CatalogError::Other(Box::new(Error::UnknownDatastore(target.clone())))
            })?;
        sub_transaction.bind_create_schema(request)
    }

    fn bind_create_user(&self, request: CreateUserRequest) -> CatalogResult<Box<dyn UserCreation>> {
        if self.metastore.user_auth(&request.name).is_some() {
            return Err(CatalogError::Other(
                format!("user `{}` already exists", request.name).into(),
            ));
        }
        Ok(Box::new(PivotUserCreation {
            staged_users: self.staged_users.clone(),
            request,
        }))
    }

    /// Commit every sub-transaction the query opened, awaiting each datastore's own
    /// commit. Each datastore decides whether its commit does blocking store I/O
    /// (hopping to the blocking pool) or is an in-memory no-op it finishes inline.
    /// Staged users land in the metastore last: its file write is a few
    /// kilobytes, small enough to finish inline.
    async fn commit(&self) -> CatalogResult<()> {
        for (_name, sub_transaction) in self.opened_sub_transactions() {
            sub_transaction.commit().await?;
        }
        for request in drain_injector(&self.staged_users) {
            self.metastore
                .create_user(&request.name, request.password.as_deref())
                .map_err(CatalogError::Other)?;
        }
        Ok(())
    }

    fn rollback(&self) {
        for (_name, sub_transaction) in self.opened_sub_transactions() {
            sub_transaction.rollback();
        }
        drain_injector(&self.staged_users);
    }
}

fn drain_injector<T>(injector: &Injector<T>) -> Vec<T> {
    let mut items = Vec::new();
    loop {
        match injector.steal() {
            Steal::Success(item) => items.push(item),
            Steal::Retry => continue,
            Steal::Empty => return items,
        }
    }
}

/// A resolved `CREATE USER`: the request plus the transaction-owned staging
/// list it will land in. Compiling it builds a dataflow that stages the
/// request and emits no rows, so the user is created only once the statement
/// runs and its transaction commits — merely planning one (to report an
/// error, to render `EXPLAIN`) creates nothing.
struct PivotUserCreation {
    staged_users: Arc<Injector<CreateUserRequest>>,
    request: CreateUserRequest,
}

impl UserCreation for PivotUserCreation {
    fn compile(&self, dispatcher: &DataFlowDispatcher) -> CatalogResult<RecordBatchOperatorSpec> {
        // One nullary per worker, but only the first carries the request; the
        // rest no-op.
        let mut request = Some(self.request.clone());
        let factories: Vec<_> = (0..dispatcher.worker_count())
            .map(|_| {
                let staged = request.take();
                let staged_users = self.staged_users.clone();
                OneShotNullaryFactory::new(move || {
                    if let Some(request) = staged {
                        staged_users.push(request);
                    }
                    None
                })
            })
            .collect();
        Ok(RecordBatchOperatorSpec::from_nullary(dispatcher, factories))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datastore_delta::DeltaDatastore;
    use dispatch::Dispatch;
    use metastore::{DEFAULT_USER_NAME, UserAuth};

    /// A metastore serving no datastores and only the built-in trusted user:
    /// the catalogs here get their datastores handed in directly.
    #[derive(Debug)]
    struct TrustMetastore;

    impl Metastore for TrustMetastore {
        fn open_datastores(
            &self,
            _dispatcher: &DataFlowDispatcher,
        ) -> metastore::Result<HashMap<String, Arc<dyn Datastore>>> {
            Ok(HashMap::new())
        }

        fn default_datastore_name(&self) -> &str {
            DEFAULT_DATASTORE_NAME
        }

        fn user_auth(&self, username: &str) -> Option<UserAuth> {
            (username == DEFAULT_USER_NAME).then_some(UserAuth::Trust)
        }
    }

    #[test]
    fn requires_the_default_datastore() {
        let error = PivotCatalog::new(
            HashMap::new(),
            DEFAULT_DATASTORE_NAME.to_string(),
            Arc::new(TrustMetastore),
        )
        .unwrap_err();

        assert!(
            matches!(error, Error::MissingDefaultDatastore(name) if name == DEFAULT_DATASTORE_NAME)
        );
    }

    #[test]
    fn one_datastore_catalog_exposes_the_datastore() {
        let dispatch = Dispatch::spin_up(1, 32, None);
        let directory = tempfile::tempdir().unwrap();
        let datastore: Arc<dyn Datastore> =
            DeltaDatastore::open(&directory.path().to_string_lossy(), dispatch.dispatcher())
                .unwrap();

        let catalog = PivotCatalog::new(
            HashMap::from([(DEFAULT_DATASTORE_NAME.to_string(), datastore)]),
            DEFAULT_DATASTORE_NAME.to_string(),
            Arc::new(TrustMetastore),
        )
        .unwrap();

        assert_eq!(catalog.default_datastore_name(), DEFAULT_DATASTORE_NAME);
        assert!(catalog.get_datastore(DEFAULT_DATASTORE_NAME).is_some());
        dispatch.exit();
    }
}
