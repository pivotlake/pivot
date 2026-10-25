//! A [`catalog::datastore::Datastore`] over the tables of an Iceberg REST
//! catalog: every table the catalog lists is read, and written to by INSERT,
//! COPY, CREATE TABLE and DROP TABLE. The Iceberg tables Pivot keeps in its
//! own directories live in the `pivotlake` datastore; both are built on the
//! `iceberg-tables` crate.
//!
//! The catalog is the only source of table identity: namespaces are the
//! datastore's schemas (single-level namespaces only, as a Pivot schema is one
//! identifier), and a table is what the catalog last reported it to be.
//!
//! # Writing
//!
//! An INSERT writes its Parquet files under the table's data path through
//! Pivot's own store, with the credentials the catalog vended for the table
//! when the statement was bound, cut by the table's partition spec (a
//! transform of a column, a day or a bucket, is computed per row and
//! recorded as the file's partition value), and its commit writes the
//! snapshot's manifests the same way; only the last step is the catalog's,
//! an `updateTable` that moves `main` onto the new snapshot provided nothing
//! else moved it first. A commit that loses that race is built again on the
//! winner. The table's entry in the index is replaced with the version that
//! landed, so the session reads its own writes before the next refresh.
//! CREATE TABLE registers a format version 3 table through the catalog, which
//! assigns its location; DROP TABLE unregisters it and leaves its files, as
//! `DROP TABLE` without `PURGE` does in every engine. Nothing here compacts,
//! expires or vacuums a catalog table: that is the catalog's own maintenance.
//!
//! A write is refused, naming the reason, where this writer cannot honour the
//! table's shape: a partition field whose key cannot be computed (a source
//! that is not a column, a transform the spec crate cannot evaluate for its
//! type), a format version below 2, delete files in the current snapshot, a
//! column type Pivot cannot represent.
//!
//! # Freshness
//!
//! The datastore holds the catalog's answer for every table: the load-table
//! response, which is the table's metadata and where it lives. That is fetched
//! whole when the datastore opens and again every refresh interval in the
//! background, the way the pivotlake datastore refreshes its own tables, so a
//! query never waits on the catalog and reads a snapshot at most one interval
//! old. A table created or committed to between refreshes is seen at the next
//! one. What is indexed is small (kilobytes per table). Binding reads the
//! manifest list; preparing a scan lazily reads selected manifests and footers
//! through the ring's caches. Those objects are not indexed.
//!
//! Every bind of one query resolves against the same index, so a refresh landing
//! mid-query does not move a table under it.
//!
//! # Reading through the ring
//!
//! A query reads a table through the `iceberg-tables` crate's
//! [`Table`](iceberg_tables::Table), the same code the pivotlake datastore
//! reads its own Iceberg tables with; this crate supplies the catalog it
//! loads the table's metadata from. Manifest lists and manifests are read as
//! whole objects over the io_uring ring ([`object_storage::load_objects`]),
//! and deletion vectors as their blob's range of its Puffin file, so they
//! pass through the compressed cache and the disk cache exactly as data
//! does: a restarted server finds them on local disk. A filtered query reads
//! only the manifests whose partition summaries admit its predicates, and
//! only the footers of the files whose partition values and column bounds
//! do. Data files are Parquet, read by the same engine that reads every other
//! datastore, with row-group pruning, late materialization and dynamic
//! filters. Columns are matched to each file by Parquet field id, so renamed
//! columns read correctly and a column added after a file was written reads
//! as NULL for that file.
//!
//! Bounds are taken as recorded: Iceberg and Parquet leave NaNs out of theirs,
//! so a range filter on a floating-point column can skip a file or row group
//! whose only matching rows are NaNs.
//!
//! # Credentials
//!
//! Every table load asks the catalog to vend storage credentials
//! (`X-Iceberg-Access-Delegation: vended-credentials`). A catalog that does
//! answers with S3 keys in the table's storage properties, and the table's
//! store signs its requests with them. Those keys are usually an STS
//! session's, short-lived, and scoped to the table, so every refresh loads
//! the table again and renews the keys of its store in place: a statement
//! that outlives the keys it started with signs its next request with the
//! new ones. When the catalog says when the keys expire
//! (`s3.session-token-expires-at-ms`), the refresh runs early enough to renew
//! them first. A catalog that vends nothing leaves the store to the factory
//! the datastore was opened with, the process's own credential policy. Only
//! the flat `s3.*` form is read; the per-prefix `storage-credentials` list is
//! not.
//!
//! A table's files are read through one store, the bucket its metadata file
//! is in. The managed catalogs keep a table in one bucket; a table split
//! across buckets is refused rather than read through the wrong store.
//!
//! # What is refused
//!
//! A table is refused, with an error naming the reason, rather than served
//! partially or wrongly: encryption, a snapshot that carries delete files
//! other than deletion vectors (position-delete and equality-delete files are
//! not applied), data files that are not Parquet, and a column whose Iceberg
//! type has no Pivot type or whose initial default would require synthesizing
//! values for older files.

mod binding;
mod describe;
pub mod env;
mod rest_catalog;
mod table;
/// The object-store test harness, re-exported for this crate's integration
/// tests. Gated behind the `test-support` feature so it never enters a
/// normal build.
#[cfg(feature = "test-support")]
pub mod test_support;
mod write;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use catalog::datastore::{Datastore, DatastoreTableMetadata, DatastoreTransaction};
use crossbeam_deque::{Injector, Steal};
use dispatch::{DataFlowDispatcher, OneShotNullaryFactory, RecordBatchOperatorSpec};
use iceberg::spec::TableMetadataRef;
use iceberg::table::Table;
use iceberg::{Catalog, ErrorKind};
use iceberg_catalog_rest::RestCatalog;
use object_storage::ExternalStoreFactory;
use planner::catalog::{
    BoundTable, CreateTableRequest, DropTableRequest, Error as CatalogError,
    Result as CatalogResult, SchemaQualifiedTableName, TableCreation, TableDrop, TableReference,
    TableRevision,
};

use crate::binding::IcebergTableBinding;
use crate::write::RestTableCatalog;

use crate::rest_catalog::build_rest_catalog;
pub use iceberg_tables::block_on;
use iceberg_tables::{CurrentMetadata, Table as LoadedTable, TableCatalog, TableStore};

use crate::table::{fetch_manifest_list_size, open_table_store, revision};

/// How to reach the REST catalog. The fields mirror the properties the
/// Iceberg REST client is configured with; `properties` passes any further
/// ones (a `prefix`, a `header.*`) through verbatim.
#[derive(Clone, Debug, Default)]
pub struct IcebergCatalogConfig {
    /// The catalog's base URI, such as `https://catalog.example.com/api`.
    pub uri: String,
    /// The warehouse the catalog serves, when it serves several.
    pub warehouse: Option<String>,
    /// What the catalog is authenticated to with; `None` for a catalog that
    /// requires nothing.
    pub auth: Option<IcebergCatalogAuth>,
    /// Further client properties, passed through as written.
    pub properties: HashMap<String, String>,
}

/// What a REST catalog is authenticated to with.
#[derive(Clone)]
pub enum IcebergCatalogAuth {
    /// A bearer token sent on every request.
    Token(String),
    /// An OAuth2 client credential (`client_id:client_secret`, or a bare
    /// secret) exchanged for a token at `server_uri`, or at the catalog's own
    /// token endpoint when there is none, for `scope` when one is given.
    OAuth2 {
        credential: String,
        server_uri: Option<String>,
        scope: Option<String>,
    },
}

/// By hand with the secret redacted: a token or credential must not leak
/// into a log through a `{:?}` of the config that holds it.
impl std::fmt::Debug for IcebergCatalogAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Token(_) => f.write_str("Token(redacted)"),
            Self::OAuth2 {
                server_uri, scope, ..
            } => f
                .debug_struct("OAuth2")
                .field("credential", &"redacted")
                .field("server_uri", server_uri)
                .field("scope", scope)
                .finish(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Iceberg REST catalog: {0}")]
    Catalog(#[from] Box<iceberg::Error>),
    #[error("table `{table}` is staged in the catalog and has no metadata location yet")]
    StagedTable { table: String },
    #[error("table `{table}` could not be indexed at the last refresh: {message}")]
    TableUnavailable { table: String, message: String },
    #[error("table `{0}` already exists")]
    TableExists(String),
    #[error("table `{0}` does not exist")]
    TableNotFound(String),
    #[error("schema `{0}` does not exist")]
    SchemaNotFound(String),
    #[error("`{option}` is not a table option; a catalog table takes {known}")]
    UnknownOption { option: String, known: String },
    #[error("`{option}` column `{column}` is not a declared column of the table")]
    UnknownSpecColumn { option: String, column: String },
    /// An error from the shared table crate: mapping a schema's field ids
    /// onto the declared columns, locating a file under the table's store.
    #[error(transparent)]
    Tables(#[from] iceberg_tables::Error),
    #[error("table `{table}` metadata object `{path}` is not in the store")]
    MissingMetadataObject { table: String, path: String },
    #[error(transparent)]
    Store(#[from] object_storage::StoreError),
    #[error("reading table metadata over the pool: {0}")]
    Load(#[from] dispatch::DataFlowError),
    #[error("metadata json: {0}")]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    ParquetTable(#[from] parquet_engine::ParquetTableError),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl From<Error> for CatalogError {
    fn from(error: Error) -> Self {
        CatalogError::Other(Box::new(error))
    }
}

/// The tables of one REST catalog, as the catalog last reported them.
pub struct IcebergDatastore {
    /// What the REST catalog client is built from: the catalog is the source
    /// of truth for what exists and where each table's current metadata
    /// lives, asked at open and at every refresh, each through a client of
    /// its own. A client exchanges an OAuth2 credential for a token on its
    /// first request and never renews it, so none is kept past its refresh,
    /// and the catalog's tokens never expire on one.
    config: IcebergCatalogConfig,
    /// The catalog's URI, which stands in for a data path: every file location
    /// a table reports is absolute.
    catalog_uri: String,
    /// Opens the object store a table's files live in (a bucket, the local
    /// filesystem), applying the process's credential policy. Each refresh
    /// opens every table's store through it, so credentials are resolved every
    /// interval rather than pinned for the process, and the catalog's vended
    /// credentials take its place for a table it vended for.
    store_factory: Arc<dyn ExternalStoreFactory>,
    /// The catalog client the last refresh was made with, which the tables a
    /// query loads carry for the catalog calls they never make: a read asks
    /// the catalog nothing, and a write builds its own client.
    catalog: Mutex<Arc<RestCatalog>>,
    /// The worker pool a table's metadata (manifest list, manifests, footers)
    /// is read through when it loads. Held by the datastore because a resolve
    /// drives its own dataflows, with no dispatcher passed in.
    dispatcher: DataFlowDispatcher,
    /// The catalog's schemas and tables as of the last refresh: what every
    /// bind resolves against. Replaced whole by each refresh; a query holds
    /// the `Arc` it started with, so a refresh never moves a table under a
    /// query in flight.
    index: RwLock<Arc<DatastoreIndex>>,
    /// How often the background refresh asks the catalog again.
    refresh_interval: Duration,
    /// The refresh task [`start`](Datastore::start) spawned, so
    /// [`abort`](Datastore::abort) can stop it on shutdown.
    refresh_task: Mutex<Option<tokio::task::AbortHandle>>,
}

/// The datastore's index, built from one answer of the catalog and shaped as
/// the catalog is: its schemas (the catalog's top-level namespaces), each
/// holding its tables by name.
#[derive(Default)]
struct DatastoreIndex {
    schemas: HashMap<String, SchemaEntry>,
}

impl DatastoreIndex {
    /// The entry of the table `name`, or `None` when the catalog listed no
    /// such schema or table.
    fn table_entry(&self, name: &SchemaQualifiedTableName) -> Option<&Result<TableEntry, String>> {
        self.schemas.get(&name.schema)?.tables.get(&name.table)
    }

    /// The soonest any table's vended keys stop working, when some catalog
    /// said when its keys expire.
    fn earliest_key_expiry(&self) -> Option<SystemTime> {
        self.schemas
            .values()
            .flat_map(|schema| schema.tables.values())
            .filter_map(|entry| entry.as_ref().ok()?.store.keys_expire_at())
            .min()
    }

    /// Whether the catalog lists the table `name`, loadable or not.
    fn contains_table(&self, name: &SchemaQualifiedTableName) -> bool {
        self.table_entry(name).is_some()
    }

    /// This index with `entry` as the table `name`'s, unless the entry it
    /// holds is newer. A commit publishes the version it landed this way,
    /// and a refresh that started before the commit and answered after it
    /// is merged the same way, so neither overwrites the other with an older
    /// version.
    fn with_entry_if_newer(&self, name: &SchemaQualifiedTableName, entry: TableEntry) -> Self {
        let mut schemas = self.schemas.clone();
        let schema = schemas.entry(name.schema.clone()).or_default();
        let is_older = schema
            .tables
            .get(&name.table)
            .and_then(|held| held.as_ref().ok())
            .is_some_and(|held| held.version() > entry.version());
        if !is_older {
            schema.tables.insert(name.table.clone(), Ok(entry));
        }
        Self { schemas }
    }

    /// `fetched`, except for every table whose entry in this index is newer
    /// than the fetched one: a commit of this process that landed while the
    /// fetch was in flight.
    fn merged_with_fetched(&self, fetched: Self) -> Self {
        let mut merged = fetched;
        for (schema_name, schema) in &self.schemas {
            for (table_name, held) in &schema.tables {
                let Ok(held) = held else { continue };
                let name = SchemaQualifiedTableName::new(schema_name.clone(), table_name.clone());
                let fetched_is_older = merged
                    .table_entry(&name)
                    .and_then(|fetched| fetched.as_ref().ok())
                    .is_some_and(|fetched| fetched.version() < held.version());
                if fetched_is_older {
                    merged = merged.with_entry_if_newer(&name, held.clone());
                }
            }
        }
        merged
    }
}

/// How long before vended keys expire the refresh that renews them runs: long
/// enough for a refresh over many tables to reach every one of them first.
const KEY_RENEWAL_LEAD: Duration = Duration::from_secs(5 * 60);

/// The least a refresh waits for the next, so keys vended for less than the
/// renewal lead do not make the refresh spin.
const MIN_REFRESH_DELAY: Duration = Duration::from_secs(10);

/// How long the refresh waits before running again: `interval`, or less when
/// some table's vended keys expire sooner than that allows renewing them.
fn wait_before_refresh(
    interval: Duration,
    earliest_key_expiry: Option<SystemTime>,
    now: SystemTime,
) -> Duration {
    let Some(expires_at) = earliest_key_expiry else {
        return interval;
    };
    let until_renewal = expires_at
        .duration_since(now)
        .unwrap_or_default()
        .saturating_sub(KEY_RENEWAL_LEAD);
    until_renewal.max(MIN_REFRESH_DELAY).min(interval)
}

/// One schema's tables. A table the catalog listed but that could not be
/// indexed (it would not load, or its manifest list could not be sized) is
/// entered with the reason, so a query over it reports that reason rather
/// than "no such table".
#[derive(Clone, Default)]
struct SchemaEntry {
    tables: HashMap<String, Result<TableEntry, String>>,
}

/// One table's entry in the index: the table's metadata and where it is, as
/// the catalog last answered or as a commit of this process landed it, with
/// what a load needs that the answer does not say: the store its files are
/// read through, signing with the credentials the catalog vended, and the
/// size of the current snapshot's manifest list.
#[derive(Clone)]
struct TableEntry {
    metadata: TableMetadataRef,
    metadata_location: String,
    store: TableStore,
    /// Iceberg names the manifest list without a length, and a ring read needs
    /// one, so the store is asked once, at refresh, and the answer kept here
    /// rather than asked again by every query. Ugly, but off the query path.
    /// `None` for a table with no current snapshot.
    manifest_list_size: Option<u64>,
}

impl TableEntry {
    fn manifest_list_path(&self) -> Option<&str> {
        self.metadata
            .current_snapshot()
            .map(|snapshot| snapshot.manifest_list())
    }

    /// Where this entry's version stands among the table's, which orders two
    /// answers of the catalog about it: a later commit has a higher sequence
    /// number, and a metadata change without a commit a later update time.
    fn version(&self) -> (i64, i64) {
        (
            self.metadata.last_sequence_number(),
            self.metadata.last_updated_ms(),
        )
    }
}

impl std::fmt::Debug for IcebergDatastore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IcebergDatastore")
            .field("catalog_uri", &self.catalog_uri)
            .finish_non_exhaustive()
    }
}

impl IcebergDatastore {
    /// Open the datastore over the catalog `config` describes, fetching every
    /// table it holds; a catalog that cannot be reached fails the open, as a
    /// store that cannot be read fails a pivotlake datastore's.
    /// `store_factory` opens the object stores the tables' files live in, so
    /// their credentials follow the same policy as every other datastore's;
    /// `refresh_interval` is how often the catalog is asked again once
    /// [`start`](Datastore::start) is called.
    pub fn open(
        config: &IcebergCatalogConfig,
        store_factory: Arc<dyn ExternalStoreFactory>,
        dispatcher: &DataFlowDispatcher,
        refresh_interval: Duration,
    ) -> Result<Arc<Self>> {
        let datastore = Self {
            config: config.clone(),
            catalog_uri: config.uri.clone(),
            store_factory,
            catalog: Mutex::new(Arc::new(build_rest_catalog(config)?)),
            dispatcher: dispatcher.clone(),
            index: RwLock::new(Arc::new(DatastoreIndex::default())),
            refresh_interval,
            refresh_task: Mutex::new(None),
        };
        datastore.refresh()?;
        Ok(Arc::new(datastore))
    }

    /// Ask the catalog, through a client built for the occasion, for
    /// everything it holds and replace the index with the answer, keeping any
    /// table a commit of this process moved past the answer while it was in
    /// flight. A failure leaves the index as it was.
    pub fn refresh(&self) -> Result<()> {
        let catalog = Arc::new(build_rest_catalog(&self.config)?);
        let fetched = self.fetch_index(&catalog)?;
        *self.catalog.lock().unwrap() = catalog;
        let mut index = self.index.write().unwrap();
        *index = Arc::new(index.merged_with_fetched(fetched));
        Ok(())
    }

    /// Replace the table `name`'s entry with `entry`, the version a commit
    /// just landed, so the next query sees it before the next refresh.
    pub(crate) fn publish_entry(&self, name: &SchemaQualifiedTableName, entry: TableEntry) {
        let mut index = self.index.write().unwrap();
        *index = Arc::new(index.with_entry_if_newer(name, entry));
    }

    /// The table `name`'s versions as the catalog keeps them, through the
    /// client the last refresh was made with: what a write's commit lands
    /// through.
    pub(crate) fn rest_table_catalog(
        &self,
        name: &SchemaQualifiedTableName,
    ) -> Arc<RestTableCatalog> {
        let catalog = Arc::clone(&self.catalog.lock().unwrap());
        Arc::new(RestTableCatalog::new(catalog, name))
    }

    /// The index as `catalog` would have it now: a schema per top-level
    /// namespace, and every table of theirs. Namespaces nested below the top
    /// level are not schemas of this datastore and are not listed.
    fn fetch_index(&self, catalog: &RestCatalog) -> Result<DatastoreIndex> {
        let previous = self.index();
        let mut index = DatastoreIndex::default();
        let namespaces = block_on(catalog.list_namespaces(None)).map_err(Box::new)?;
        for namespace in namespaces {
            let listed = block_on(catalog.list_tables(&namespace)).map_err(Box::new)?;
            // Listed under no parent, the namespace is a top-level one: a
            // single identifier, which is its schema name.
            let schema = namespace
                .inner()
                .pop()
                .expect("a namespace has at least one level");
            let mut tables = HashMap::new();
            for ident in listed {
                let name = SchemaQualifiedTableName::new(schema.clone(), ident.name.clone());
                let entry = match block_on(catalog.load_table(&ident)) {
                    Ok(table) => self
                        .build_table_entry(&name, table, previous.table_entry(&name))
                        .map_err(|error| error.to_string()),
                    // A table dropped between the listing and its load is
                    // simply not there any more.
                    Err(error)
                        if matches!(
                            error.kind(),
                            ErrorKind::TableNotFound | ErrorKind::NamespaceNotFound
                        ) =>
                    {
                        continue;
                    }
                    // A table the catalog lists but will not hand over (a
                    // permission it lacks; a view it lists among tables) is
                    // entered with the catalog's reason, so a query naming it
                    // reports that rather than "no such table", and so it
                    // does not keep every other table from refreshing.
                    Err(error) => {
                        tracing::warn!(
                            table = %name,
                            error = %error,
                            "a table the catalog lists could not be loaded from it"
                        );
                        Err(error.to_string())
                    }
                };
                tables.insert(ident.name, entry);
            }
            index.schemas.insert(schema, SchemaEntry { tables });
        }
        Ok(index)
    }

    /// `table`'s entry: with its store, opened now from the credentials the
    /// catalog just vended, its manifest list's size, the one `previous`
    /// learned when the snapshot is the one it entered, else asked of the
    /// store now, and the manifests `previous` had decoded.
    fn build_table_entry(
        &self,
        name: &SchemaQualifiedTableName,
        table: Table,
        previous: Option<&Result<TableEntry, String>>,
    ) -> Result<TableEntry> {
        let previous = previous.and_then(|previous| previous.as_ref().ok());
        let store = open_table_store(
            name,
            &table,
            &*self.store_factory,
            previous.map(|previous| &previous.store),
        )?;
        let path = table
            .metadata()
            .current_snapshot()
            .map(|snapshot| snapshot.manifest_list());
        let unchanged =
            previous.filter(|previous| path.is_some() && previous.manifest_list_path() == path);
        let manifest_list_size = match unchanged {
            Some(previous) => previous.manifest_list_size,
            None => fetch_manifest_list_size(name, &table, &store)?,
        };
        let metadata_location = table
            .metadata_location()
            .ok_or_else(|| Error::StagedTable {
                table: name.to_string(),
            })?
            .to_string();
        Ok(TableEntry {
            metadata: table.metadata_ref(),
            metadata_location,
            store,
            manifest_list_size,
        })
    }

    /// The index as of now: the `Arc` a query pins for its lifetime, so the
    /// refresh's swap moves nothing under it.
    fn index(&self) -> Arc<DatastoreIndex> {
        self.index.read().unwrap().clone()
    }

    /// The table `name` as `index` has it, its schema being the namespace,
    /// loaded; or `None` when the catalog listed no such table.
    fn load_table(
        &self,
        index: &DatastoreIndex,
        name: &SchemaQualifiedTableName,
    ) -> Result<Option<Arc<LoadedTable>>> {
        let Some(entry) = index.table_entry(name) else {
            return Ok(None);
        };
        let entry = entry.as_ref().map_err(|message| Error::TableUnavailable {
            table: name.to_string(),
            message: message.clone(),
        })?;
        let catalog = Arc::clone(&self.catalog.lock().unwrap());
        let rest_table: Arc<dyn TableCatalog> = Arc::new(RestTableCatalog::new(catalog, name));
        let location = write::table_location(&entry.metadata, &entry.store)?;
        let loaded = LoadedTable::at(
            &self.dispatcher,
            rest_table,
            location,
            entry.store.clone(),
            &name.to_string(),
            CurrentMetadata {
                metadata_location: entry.metadata_location.clone(),
                metadata: Arc::clone(&entry.metadata),
            },
            entry.manifest_list_size,
        )?;
        Ok(Some(Arc::new(loaded)))
    }

    /// Every table `index` has that the catalog handed over, loaded. A table
    /// the catalog lists but would not hand over (Unity Catalog lists its
    /// `information_schema` views, which are not Iceberg tables) is not
    /// served, so it is not listed; a query naming it reports why.
    fn load_all_tables(
        &self,
        index: &DatastoreIndex,
    ) -> Result<Vec<(SchemaQualifiedTableName, Arc<LoadedTable>)>> {
        let mut tables = Vec::new();
        for (schema, schema_entry) in &index.schemas {
            for (table, entry) in &schema_entry.tables {
                if entry.is_err() {
                    continue;
                }
                let name = SchemaQualifiedTableName::new(schema.clone(), table.clone());
                if let Some(loaded) = self.load_table(index, &name)? {
                    tables.push((name, loaded));
                }
            }
        }
        Ok(tables)
    }
}

impl Datastore for IcebergDatastore {
    fn begin_transaction(self: Arc<Self>) -> Arc<dyn DatastoreTransaction> {
        let index = self.index();
        Arc::new(IcebergTransaction {
            datastore: self,
            index,
            loaded_tables: Mutex::new(HashMap::new()),
            commit_actions: Arc::new(Injector::new()),
            committed: AtomicBool::new(false),
        })
    }

    fn kind(&self) -> &'static str {
        "iceberg"
    }

    /// The catalog's URI: every file path a table reports is a full location
    /// of its own, since Iceberg names files absolutely.
    fn data_path(&self) -> String {
        self.catalog_uri.clone()
    }

    /// Refresh the index every `refresh_interval`, on the ambient runtime, and
    /// sooner when some table's vended keys would expire before the next
    /// refresh could renew them. A refresh that fails is logged and the index
    /// kept, so an unreachable catalog degrades to stale tables rather than
    /// none.
    ///
    /// The task holds the datastore weakly: the index keeps the statistics
    /// of the files this process committed, which live in the workers' ring
    /// and must be released by their owner's thread, never by an aborted
    /// task's drop on a runtime thread.
    fn start(self: Arc<Self>) {
        let datastore = Arc::downgrade(&self);
        let interval = self.refresh_interval;
        let task = tokio::spawn(async move {
            loop {
                // The catalog was fetched at open, and by the refresh before
                // this one, so the wait starts from now. The datastore is let
                // go of before the wait, so the task never keeps it alive.
                let delay = match datastore.upgrade() {
                    Some(datastore) => wait_before_refresh(
                        interval,
                        datastore.index().earliest_key_expiry(),
                        SystemTime::now(),
                    ),
                    None => return,
                };
                tokio::time::sleep(delay).await;
                let Some(datastore) = datastore.upgrade() else {
                    return;
                };
                // The refresh blocks on catalog calls, so it runs off the
                // reactor.
                match tokio::task::spawn_blocking(move || datastore.refresh()).await {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        tracing::warn!(error = %error, "Iceberg catalog refresh failed")
                    }
                    Err(error) => {
                        tracing::warn!(error = %error, "Iceberg catalog refresh panicked")
                    }
                }
            }
        });
        *self.refresh_task.lock().unwrap() = Some(task.abort_handle());
    }

    fn abort(&self) {
        if let Some(task) = self.refresh_task.lock().unwrap().take() {
            task.abort();
        }
    }
}

/// One durable effect a statement staged on its transaction: the files an
/// INSERT uploaded, a table created or dropped. The transaction's commit
/// applies its actions in the order the statements ran, so a statement sees
/// what the ones before it did; a failure stops the commit and the effects
/// applied before it stay. Blocking catalog and store I/O, so
/// [`IcebergTransaction::commit`] applies them on the blocking pool.
pub(crate) trait CommitAction: Send {
    fn apply(self: Box<Self>, datastore: &IcebergDatastore) -> Result<()>;
}

/// One query's view of the catalog: the index as it was when the query began,
/// which every schema check and bind of the query resolves against, so a
/// refresh landing mid-query moves nothing under it. A table is loaded the
/// first time the query touches it and kept for the query's lifetime, so a
/// plan's scan and its late materialize see the same load, and a table the
/// query names twice is loaded once.
///
/// What the query writes is staged on the transaction as commit actions
/// (the files an INSERT uploaded, the tables it created or dropped) and
/// lands at commit, in the order it was staged.
pub struct IcebergTransaction {
    datastore: Arc<IcebergDatastore>,
    index: Arc<DatastoreIndex>,
    loaded_tables: Mutex<HashMap<SchemaQualifiedTableName, Option<Arc<LoadedTable>>>>,
    commit_actions: Arc<Injector<Box<dyn CommitAction>>>,
    /// Whether commit has run. A second commit is refused, and the drop
    /// safety net discards the staged writes of a transaction its owner
    /// never resolved.
    committed: AtomicBool,
}

impl std::fmt::Debug for IcebergTransaction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IcebergTransaction")
            .field("datastore", &self.datastore)
            .finish_non_exhaustive()
    }
}

/// One table creation resolved by a transaction and waiting for it to commit.
#[derive(Clone)]
struct PendingTableCreation {
    name: SchemaQualifiedTableName,
    columns: Vec<planner::catalog::Column>,
    partition_by: Vec<String>,
    sort_by: Vec<String>,
    /// Whether the statement used `IF NOT EXISTS`, so the commit treats a
    /// table that appeared since resolution as a success rather than an error.
    if_not_exists: bool,
}

impl CommitAction for PendingTableCreation {
    /// Register the table through the catalog, then refresh the index, which
    /// is shaped by what the catalog lists.
    fn apply(self: Box<Self>, datastore: &IcebergDatastore) -> Result<()> {
        let catalog = build_rest_catalog(&datastore.config)?;
        match write::create_table(
            &catalog,
            &self.name,
            &self.columns,
            &self.partition_by,
            &self.sort_by,
        ) {
            Ok(_) => {}
            Err(Error::Catalog(error))
                if self.if_not_exists && error.kind() == ErrorKind::TableAlreadyExists => {}
            Err(error) => return Err(error),
        }
        datastore.refresh()
    }
}

/// One table drop resolved by a transaction and waiting for it to commit.
#[derive(Clone)]
struct PendingTableDrop {
    name: SchemaQualifiedTableName,
    /// Whether the statement used `IF EXISTS`, so the commit treats a table
    /// that vanished since resolution as a success rather than an error.
    if_exists: bool,
}

impl CommitAction for PendingTableDrop {
    /// Unregister the table through the catalog, leaving its files, then
    /// refresh the index, which is shaped by what the catalog lists.
    fn apply(self: Box<Self>, datastore: &IcebergDatastore) -> Result<()> {
        let catalog = build_rest_catalog(&datastore.config)?;
        write::drop_table(&catalog, &self.name, self.if_exists)?;
        datastore.refresh()
    }
}

/// `WITH (partition_by = 'a, b')` and `WITH (sort_by = 'a, b')`: ordered,
/// comma-separated columns. The only options a catalog table takes; a
/// catalog assigns the table's location, and adopting files is not offered.
const PARTITION_BY_OPTION: &str = "partition_by";
const SORT_BY_OPTION: &str = "sort_by";
const TABLE_OPTIONS: [&str; 2] = [PARTITION_BY_OPTION, SORT_BY_OPTION];

/// The columns `option` names in `request`, each a declared column.
fn parse_spec_columns(request: &CreateTableRequest, option: &str) -> Result<Vec<String>> {
    let Some(raw) = request.options.get(option) else {
        return Ok(Vec::new());
    };
    raw.split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(|name| {
            if request.columns.iter().any(|column| column.name == name) {
                Ok(name.to_string())
            } else {
                Err(Error::UnknownSpecColumn {
                    option: option.to_string(),
                    column: name.to_string(),
                })
            }
        })
        .collect()
}

/// Everything an injector holds, in the order it was pushed.
pub(crate) fn drain_injector<T>(injector: &Injector<T>) -> Vec<T> {
    let mut drained = Vec::new();
    loop {
        match injector.steal() {
            Steal::Success(item) => drained.push(item),
            Steal::Retry => continue,
            Steal::Empty => break,
        }
    }
    drained
}

/// A dataflow that stages `action` on the transaction's queue when it runs
/// and emits no rows: one nullary per worker, of which only the first
/// carries the action, so the catalog changes when the statement runs rather
/// than when it is planned.
fn staging_spec<T: CommitAction + Clone + 'static>(
    dispatcher: &DataFlowDispatcher,
    commit_actions: &Arc<Injector<Box<dyn CommitAction>>>,
    action: T,
) -> RecordBatchOperatorSpec {
    let mut carried = Some(action);
    let factories: Vec<_> = (0..dispatcher.worker_count())
        .map(|_| {
            let carried = carried.take();
            let commit_actions = commit_actions.clone();
            OneShotNullaryFactory::new(move || {
                if let Some(action) = carried {
                    commit_actions.push(Box::new(action));
                }
                None
            })
        })
        .collect();
    RecordBatchOperatorSpec::from_nullary(dispatcher, factories)
}

/// A resolved `CREATE TABLE` for a catalog table, staged when its dataflow
/// runs.
struct IcebergTableCreation {
    commit_actions: Arc<Injector<Box<dyn CommitAction>>>,
    creation: PendingTableCreation,
}

impl TableCreation for IcebergTableCreation {
    fn compile(&self, dispatcher: &DataFlowDispatcher) -> CatalogResult<RecordBatchOperatorSpec> {
        Ok(staging_spec(
            dispatcher,
            &self.commit_actions,
            self.creation.clone(),
        ))
    }
}

/// A resolved `DROP TABLE` for a catalog table, staged when its dataflow
/// runs.
struct IcebergTableDrop {
    commit_actions: Arc<Injector<Box<dyn CommitAction>>>,
    drop: PendingTableDrop,
}

impl TableDrop for IcebergTableDrop {
    fn compile(&self, dispatcher: &DataFlowDispatcher) -> CatalogResult<RecordBatchOperatorSpec> {
        Ok(staging_spec(
            dispatcher,
            &self.commit_actions,
            self.drop.clone(),
        ))
    }
}

impl IcebergTransaction {
    fn resolve_table(&self, name: &SchemaQualifiedTableName) -> Result<Option<Arc<LoadedTable>>> {
        if let Some(table) = self.loaded_tables.lock().unwrap().get(name) {
            return Ok(table.clone());
        }
        let table = self.datastore.load_table(&self.index, name)?;
        self.loaded_tables
            .lock()
            .unwrap()
            .insert(name.clone(), table.clone());
        Ok(table)
    }
}

#[async_trait]
impl DatastoreTransaction for IcebergTransaction {
    /// Whether the catalog had a top-level namespace named `schema` when the
    /// query began.
    fn does_schema_exist(&self, schema: &str) -> CatalogResult<bool> {
        Ok(self.index.schemas.contains_key(schema))
    }

    fn bind_table(
        &self,
        datastore: &str,
        name: &SchemaQualifiedTableName,
    ) -> CatalogResult<Option<Box<dyn BoundTable>>> {
        let Some(table) = self.resolve_table(name)? else {
            return Ok(None);
        };
        let reference = TableReference {
            datastore: datastore.to_string(),
            schema: name.schema.clone(),
            table: name.table.clone(),
        };
        Ok(Some(Box::new(IcebergTableBinding::new(
            reference,
            table,
            Arc::clone(&self.datastore),
            Arc::clone(&self.commit_actions),
        ))))
    }

    /// Resolve a `CREATE TABLE` against the catalog as the query sees it: the
    /// namespace must exist, the name must be free unless `IF NOT EXISTS`,
    /// and the options are the partition and sort columns.
    fn bind_create_table(
        &self,
        request: CreateTableRequest,
    ) -> CatalogResult<Box<dyn TableCreation>> {
        let name = request.schema_qualified_name();
        if let Some(option) = request
            .options
            .keys()
            .find(|option| !TABLE_OPTIONS.contains(&option.as_str()))
        {
            return Err(Error::UnknownOption {
                option: option.clone(),
                known: TABLE_OPTIONS.map(|known| format!("`{known}`")).join(", "),
            }
            .into());
        }
        if !self.index.schemas.contains_key(&name.schema) {
            return Err(Error::SchemaNotFound(name.schema).into());
        }
        if !request.if_not_exists && self.index.contains_table(&name) {
            return Err(Error::TableExists(name.to_string()).into());
        }
        let partition_by = parse_spec_columns(&request, PARTITION_BY_OPTION)?;
        let sort_by = parse_spec_columns(&request, SORT_BY_OPTION)?;
        Ok(Box::new(IcebergTableCreation {
            commit_actions: Arc::clone(&self.commit_actions),
            creation: PendingTableCreation {
                name,
                columns: request.columns,
                partition_by,
                sort_by,
                if_not_exists: request.if_not_exists,
            },
        }))
    }

    fn bind_drop_table(&self, request: DropTableRequest) -> CatalogResult<Box<dyn TableDrop>> {
        let name = request.schema_qualified_name();
        if !request.if_exists {
            if !self.index.schemas.contains_key(&name.schema) {
                return Err(Error::SchemaNotFound(name.schema).into());
            }
            if !self.index.contains_table(&name) {
                return Err(Error::TableNotFound(name.to_string()).into());
            }
        }
        Ok(Box::new(IcebergTableDrop {
            commit_actions: Arc::clone(&self.commit_actions),
            drop: PendingTableDrop {
                name,
                if_exists: request.if_exists,
            },
        }))
    }

    /// Land what the query staged, in the order it was staged. A read-only
    /// transaction staged nothing and returns inline; a writing one lands its
    /// changes on the blocking pool, since every step blocks on the catalog
    /// or the store.
    async fn commit(&self) -> CatalogResult<()> {
        // Marked at entry, not on success: the swap refuses a second commit,
        // and a failed commit has already drained the staged state, so the
        // drop net must not "roll back" what is no longer staged.
        if self.committed.swap(true, Ordering::AcqRel) {
            return Err(CatalogError::Other("transaction already committed".into()));
        }
        if self.commit_actions.is_empty() {
            return Ok(());
        }
        let commit_actions = drain_injector(&self.commit_actions);
        let datastore = Arc::clone(&self.datastore);
        tokio::task::spawn_blocking(move || -> CatalogResult<()> {
            for action in commit_actions {
                action.apply(&datastore)?;
            }
            Ok(())
        })
        .await
        .map_err(|error| CatalogError::Other(format!("commit thread panicked: {error}").into()))?
    }

    fn rollback(&self) {
        drain_injector(&self.commit_actions);
    }

    fn table_revision(
        &self,
        name: &SchemaQualifiedTableName,
    ) -> CatalogResult<Option<TableRevision>> {
        Ok(self.resolve_table(name)?.map(|table| revision(&table)))
    }

    fn tables(&self) -> CatalogResult<Vec<DatastoreTableMetadata>> {
        let tables = self.datastore.load_all_tables(&self.index)?;
        let mut memo = self.loaded_tables.lock().unwrap();
        let described = tables
            .into_iter()
            .map(|(name, table)| {
                memo.insert(name.clone(), Some(table.clone()));
                crate::describe::describe_table(&name, &table)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(described)
    }
}

/// The safety net behind the explicit resolution points: a transaction
/// dropped with staged writes and neither commit nor rollback run discards
/// them here. Reaching this is a bug in the owner, hence the warning.
impl Drop for IcebergTransaction {
    fn drop(&mut self) {
        if self.committed.load(Ordering::Acquire) {
            return;
        }
        if !self.commit_actions.is_empty() {
            tracing::warn!(
                "transaction dropped with staged writes and no resolution; rolling back"
            );
            DatastoreTransaction::rollback(self);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INTERVAL: Duration = Duration::from_secs(30);

    #[test]
    fn the_refresh_waits_the_interval_when_no_keys_expire() {
        let delay = wait_before_refresh(INTERVAL, None, SystemTime::UNIX_EPOCH);

        assert_eq!(delay, INTERVAL);
    }

    #[test]
    fn the_refresh_runs_early_to_renew_keys_before_they_expire() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
        let interval = Duration::from_secs(3_600);
        let expires_at = now + KEY_RENEWAL_LEAD + Duration::from_secs(20);

        let delay = wait_before_refresh(interval, Some(expires_at), now);

        assert_eq!(delay, Duration::from_secs(20));
    }

    #[test]
    fn keys_expiring_within_the_lead_do_not_make_the_refresh_spin() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);

        let delay = wait_before_refresh(INTERVAL, Some(now + Duration::from_secs(1)), now);

        assert_eq!(delay, MIN_REFRESH_DELAY);
    }

    #[test]
    fn a_short_interval_is_never_stretched_by_the_minimum() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
        let interval = Duration::from_millis(500);

        let delay = wait_before_refresh(interval, Some(now), now);

        assert_eq!(delay, interval);
    }
}
