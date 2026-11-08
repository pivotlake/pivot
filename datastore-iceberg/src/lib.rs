//! A [`catalog::datastore::Datastore`] over the tables of an Iceberg REST
//! catalog: it reads them, and appends to them with `INSERT`.
//!
//! The catalog is the only source of table identity: namespaces are the
//! datastore's schemas (single-level namespaces only, as a Pivot schema is one
//! identifier), and a table is what the catalog last reported it to be.
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
//! Manifest lists and manifests are read as whole objects over the io_uring
//! ring ([`object_storage::load_objects`]), and deletion vectors as their
//! blob's range of its Puffin file, so they pass through the compressed cache
//! and the disk cache exactly as data does: a restarted server finds them on
//! local disk. A filtered query reads only the manifests whose
//! partition summaries admit its predicates, and only the footers of the files
//! whose partition values and column bounds do. Data files are Parquet, read
//! by the same engine that reads every other datastore, with row-group
//! pruning, late materialization and dynamic filters. Columns are matched to
//! each file by Parquet field id, so renamed columns read correctly and a column added after
//! a file was written reads as NULL for that file.
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
//! store is opened with them; those keys are usually an STS session's,
//! short-lived, and scoped to the table, so the store is opened again at every
//! refresh from the keys the catalog vended then. A catalog that vends nothing
//! leaves the store to the factory the datastore was opened with, the
//! process's own credential policy. Only the flat `s3.*` form is read; the
//! per-prefix `storage-credentials` list is not.
//!
//! A table's files are read through one store, the bucket its metadata file
//! is in. The managed catalogs keep a table in one bucket; a table split
//! across buckets is refused rather than read through the wrong store.
//!
//! # Writing
//!
//! An `INSERT` writes Parquet files under the table's data path through the
//! store the catalog vended credentials for, partitioned the way the table's
//! partition spec says (a transform of a column, a day or a bucket, is
//! computed per row and recorded as the file's partition value), with each
//! column under its Iceberg field id. The transaction's commit records the
//! files in a manifest of their own, writes a manifest list naming it ahead
//! of the current snapshot's manifests, and asks the catalog to add the
//! snapshot and move `main` onto it, on the condition that `main` has not
//! moved since the commit was built; when it has, the commit is built again
//! on the winner. The table at the landed version replaces the index's entry
//! at once, so the next query sees the rows before the background refresh
//! does. Tables are created and dropped through the catalog itself, and
//! maintained by it: nothing here compacts, expires or vacuums.
//!
//! # What is refused
//!
//! A table is refused, with an error naming the reason, rather than served
//! partially or wrongly: encryption, a snapshot that carries delete files
//! other than deletion vectors (position-delete and equality-delete files are
//! not applied), data files that are not Parquet, and a column whose Iceberg
//! type has no Pivot type or whose initial default would require synthesizing
//! values for older files. An `INSERT` is refused into a format version 1
//! table (its manifests carry no sequence numbers), and into a table whose
//! partition spec names a field after a column or a transform its source
//! column's type does not take.

mod binding;
mod columns;
mod datum;
mod deletes;
mod describe;
pub mod env;
mod manifests;
mod pruning;
mod rest_catalog;
mod storage;
mod store;
mod table;
mod write;

pub use crate::rest_catalog::{IcebergCatalogAuth, IcebergCatalogConfig};

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use async_trait::async_trait;
use catalog::StagedCommit;
use catalog::datastore::{Datastore, DatastoreTableMetadata, DatastoreTransaction};
use dispatch::DataFlowDispatcher;
use iceberg::table::Table;
use iceberg::{Catalog, ErrorKind};
use iceberg_catalog_rest::RestCatalog;
use object_storage::ExternalStoreFactory;
use planner::catalog::{
    BoundTable, Error as CatalogError, Result as CatalogResult, SchemaQualifiedTableName,
    TableReference, TableRevision,
};

use crate::binding::IcebergTableBinding;
use crate::rest_catalog::block_on;
use crate::storage::PivotStorageFactory;
use crate::store::TableStore;
use crate::table::{LoadedTable, fetch_manifest_list_size, open_table_store};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Iceberg REST catalog: {0}")]
    Catalog(#[from] Box<iceberg::Error>),
    #[error("table `{table}` is staged in the catalog and has no metadata location yet")]
    StagedTable { table: String },
    #[error("table `{table}` could not be indexed at the last refresh: {message}")]
    TableUnavailable { table: String, message: String },
    #[error("table `{table}` is encrypted; encrypted tables are not read")]
    EncryptedTable { table: String },
    #[error(
        "table `{table}` column `{column}` has Iceberg type `{iceberg_type}`, which Pivot cannot represent"
    )]
    UnsupportedColumnType {
        table: String,
        column: String,
        iceberg_type: String,
    },
    #[error(
        "table `{table}` column `{column}` has initial default {default}; a column absent from a file is only read as NULL, so the table is not served"
    )]
    UnsupportedColumnDefault {
        table: String,
        column: String,
        default: String,
    },
    #[error(
        "delete file `{file}` is not a deletion vector; only deletion vectors are applied, so the table is not served"
    )]
    DeleteFile { file: String },
    #[error("the deletion vector for `{file}` is invalid: {message}")]
    DeletionVector { file: String, message: String },
    #[error("table `{table}` data file `{file}` is {format}; only Parquet data files are read")]
    NonParquetFile {
        table: String,
        file: String,
        format: String,
    },
    #[error("table `{table}` metadata object `{path}` is not in the store")]
    MissingMetadataObject { table: String, path: String },
    #[error("table `{table}` partition metadata is invalid: {source}")]
    InvalidPartitionMetadata {
        table: String,
        #[source]
        source: Box<iceberg::Error>,
    },
    #[error("metadata object `{path}` does not parse: {source}")]
    MalformedMetadataObject {
        path: String,
        #[source]
        source: Box<iceberg::Error>,
    },
    #[error("file location `{path}` names no supported store: {message}")]
    UnsupportedLocation { path: String, message: String },
    #[error(
        "`{location}` is outside `{root}`, the root the table's file IO was opened on; a table's files must all live in one bucket"
    )]
    LocationOutsideRoot { location: String, root: String },
    #[error(
        "table `{table}` file `{file}` is outside `{root}`, where the table's metadata lives; a table's files must all live in one bucket"
    )]
    FileOutsideTableStore {
        table: String,
        file: String,
        root: String,
    },
    #[error("table `{table}` was vended an S3 access key without its secret key, or the reverse")]
    IncompleteVendedCredentials { table: String },
    #[error(transparent)]
    Store(#[from] object_storage::StoreError),
    #[error("reading table metadata over the pool: {0}")]
    Load(#[from] dispatch::DataFlowError),
    #[error("Iceberg metadata: {0}")]
    Metadata(Box<iceberg::Error>),
    #[error(
        "table `{table}` is Iceberg format version {version}; only format versions 2 and 3 are written"
    )]
    UnsupportedFormatVersion { table: String, version: u8 },
    #[error(
        "table `{table}` partition field `{field}` has no source among the table's columns, so its files cannot be partitioned"
    )]
    UnsupportedPartitionSource { table: String, field: String },
    #[error(
        "table `{table}` partition field `{field}` is named like a column, so its key could not be told from the column's values"
    )]
    PartitionFieldNamedLikeColumn { table: String, field: String },
    #[error(
        "table `{table}` partition field `{field}` applies `{transform}`, which cannot be computed for its source column: {message}"
    )]
    UnsupportedPartitionTransform {
        table: String,
        field: String,
        transform: String,
        message: String,
    },
    #[error(
        "table `{table}` was dropped and created again while the insert ran; its files were written for the previous table"
    )]
    TableRecreated { table: String },
    #[error("data file `{file}` has no statistics to record")]
    MissingFileStats { file: String },
    #[error("data file `{file}` cannot be recorded: {message}")]
    InvalidDataFile { file: String, message: String },
}

impl Error {
    fn metadata(error: iceberg::Error) -> Self {
        Error::Metadata(Box::new(error))
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl From<Error> for CatalogError {
    fn from(error: Error) -> Self {
        CatalogError::Other(Box::new(error))
    }
}

/// The tables of one REST catalog, as the catalog last reported them.
pub struct IcebergDatastore {
    /// The client every refresh and every commit asks the catalog through,
    /// built once: the catalog is the source of truth for what exists and
    /// where each table's current metadata lives, and the client's OAuth2
    /// token, when it has one, is renewed by its own session before it
    /// expires. The file IO of every table it loads goes over the stores
    /// `store_factory` opens or the credentials the catalog vended.
    catalog: Arc<RestCatalog>,
    /// The catalog's URI, which stands in for a data path: every file location
    /// a table reports is absolute.
    catalog_uri: String,
    /// Opens the object store a table's files live in (a bucket, the local
    /// filesystem), applying the process's credential policy. Each refresh
    /// opens every table's store through it, so credentials are resolved every
    /// interval rather than pinned for the process, and the catalog's vended
    /// credentials take its place for a table it vended for.
    store_factory: Arc<dyn ExternalStoreFactory>,
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
#[derive(Clone, Default)]
struct DatastoreIndex {
    schemas: HashMap<String, SchemaEntry>,
}

impl DatastoreIndex {
    /// The entry of the table `name`, or `None` when the catalog listed no
    /// such schema or table.
    fn table_entry(&self, name: &SchemaQualifiedTableName) -> Option<&Result<TableEntry, String>> {
        self.schemas.get(&name.schema)?.tables.get(&name.table)
    }
}

/// One schema's tables. A table the catalog listed but that could not be
/// indexed (it would not load, or its manifest list could not be sized) is
/// entered with the reason, so a query over it reports that reason rather
/// than "no such table".
#[derive(Clone, Default)]
struct SchemaEntry {
    tables: HashMap<String, Result<TableEntry, String>>,
}

/// One table's entry in the index: the table as the catalog returned it, with
/// what a load needs that the response does not say: the store its files are
/// read through, opened with the credentials the catalog vended, and the size
/// of the current snapshot's manifest list.
#[derive(Clone)]
struct TableEntry {
    table: Table,
    store: TableStore,
    /// Iceberg names the manifest list without a length, and a ring read needs
    /// one, so the store is asked once, at refresh, and the answer kept here
    /// rather than asked again by every query. Ugly, but off the query path.
    /// `None` for a table with no current snapshot.
    manifest_list_size: Option<u64>,
}

/// Two entries hold different tables under the same name, one dropped and
/// another created, so their versions are not ordered against each other.
#[derive(Debug, PartialEq, Eq)]
struct DifferentTable;

impl TableEntry {
    /// Where this entry's version stands against `other`'s. The catalog
    /// numbers a table's snapshots in commit order and the number never
    /// decreases, so a higher last sequence number is later, and an equal
    /// one is the same commit or a metadata change on top of it.
    fn compare_version(&self, other: &TableEntry) -> Result<Ordering, DifferentTable> {
        let (mine, theirs) = (self.table.metadata(), other.table.metadata());
        if mine.uuid() != theirs.uuid() {
            return Err(DifferentTable);
        }
        Ok(mine
            .last_sequence_number()
            .cmp(&theirs.last_sequence_number()))
    }

    fn manifest_list_path(&self) -> Option<&str> {
        self.table
            .metadata()
            .current_snapshot()
            .map(|snapshot| snapshot.manifest_list())
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
        let file_io_factory = Arc::new(PivotStorageFactory::new(store_factory.clone()));
        let catalog = Arc::new(config.build_client(file_io_factory)?);
        let datastore = Self {
            catalog,
            catalog_uri: config.uri.clone(),
            store_factory,
            dispatcher: dispatcher.clone(),
            index: RwLock::new(Arc::new(DatastoreIndex::default())),
            refresh_interval,
            refresh_task: Mutex::new(None),
        };
        datastore.refresh()?;
        Ok(Arc::new(datastore))
    }

    /// Ask the catalog for everything it holds and replace the index with the
    /// answer. A failure leaves the index as it was. A table this process
    /// committed to while the answer was in flight keeps the committed
    /// version, which is later than the answer's.
    pub fn refresh(&self) -> Result<()> {
        let mut fetched = self.fetch_index(&self.catalog)?;
        let mut index = self.index.write().unwrap();
        for (schema, held_schema) in &index.schemas {
            let Some(fetched_schema) = fetched.schemas.get_mut(schema) else {
                continue;
            };
            for (table, held_entry) in &held_schema.tables {
                let (Ok(held_entry), Some(Ok(fetched_entry))) =
                    (held_entry, fetched_schema.tables.get_mut(table))
                else {
                    continue;
                };
                match held_entry.compare_version(fetched_entry) {
                    Ok(Ordering::Greater) => *fetched_entry = held_entry.clone(),
                    Ok(Ordering::Less | Ordering::Equal) => {}
                    Err(DifferentTable) => {
                        tracing::info!(%schema, %table, "the table was dropped and created again");
                    }
                }
            }
        }
        *index = Arc::new(fetched);
        Ok(())
    }

    /// The client the catalog is asked through.
    pub(crate) fn catalog(&self) -> Arc<RestCatalog> {
        self.catalog.clone()
    }

    /// The table `name` as the index holds it now, or `None` when the catalog
    /// no longer lists it or would not hand it over.
    pub(crate) fn current_table(&self, name: &SchemaQualifiedTableName) -> Option<Table> {
        let index = self.index();
        let entry = index.table_entry(name)?.as_ref().ok()?;
        Some(entry.table.clone())
    }

    /// Publish `table`, at the version a commit of this process just landed,
    /// into the index with the store the commit wrote through, so the next
    /// query sees the version before the background refresh does. An entry
    /// already at that version or later (a refresh got there first, perhaps
    /// with a metadata change on top of the commit) stands, as does one of a
    /// table created again under the name since the commit was built.
    pub(crate) fn publish_table(
        &self,
        name: &SchemaQualifiedTableName,
        table: Table,
        store: TableStore,
        manifest_list_size: u64,
    ) {
        let entry = TableEntry {
            table,
            store,
            manifest_list_size: Some(manifest_list_size),
        };
        let mut index = self.index.write().unwrap();
        let held = index.table_entry(name).and_then(|held| held.as_ref().ok());
        if held.is_some_and(|held| held.compare_version(&entry) != Ok(Ordering::Less)) {
            return;
        }
        let mut updated = (**index).clone();
        updated
            .schemas
            .entry(name.schema.clone())
            .or_default()
            .tables
            .insert(name.table.clone(), Ok(entry));
        *index = Arc::new(updated);
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
    /// catalog just vended, and its manifest list's size, the one `previous`
    /// learned when the snapshot is the one it entered, else asked of the
    /// store now.
    fn build_table_entry(
        &self,
        name: &SchemaQualifiedTableName,
        table: Table,
        previous: Option<&Result<TableEntry, String>>,
    ) -> Result<TableEntry> {
        let store = open_table_store(name, &table, &*self.store_factory)?;
        let path = table
            .metadata()
            .current_snapshot()
            .map(|snapshot| snapshot.manifest_list());
        let unchanged = previous
            .and_then(|previous| previous.as_ref().ok())
            .filter(|previous| path.is_some() && previous.manifest_list_path() == path);
        let manifest_list_size = match unchanged {
            Some(previous) => previous.manifest_list_size,
            None => fetch_manifest_list_size(name, &table, &store)?,
        };
        Ok(TableEntry {
            table,
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
        let loaded = LoadedTable::load(
            name,
            &entry.table,
            entry.store.clone(),
            entry.manifest_list_size,
            &self.dispatcher,
        )?;
        Ok(Some(Arc::new(loaded)))
    }

    /// Every table `index` has that the catalog handed over, loaded. A table
    /// the catalog lists but would not hand over (Unity Catalog lists its
    /// `information_schema` views, which are not Iceberg tables) is not
    /// served, so it is not listed; a query naming it reports why.
    fn load_all_tables(&self, index: &DatastoreIndex) -> Result<Vec<Arc<LoadedTable>>> {
        let mut tables = Vec::new();
        for (schema, schema_entry) in &index.schemas {
            for (table, entry) in &schema_entry.tables {
                if entry.is_err() {
                    continue;
                }
                let name = SchemaQualifiedTableName::new(schema.clone(), table.clone());
                if let Some(loaded) = self.load_table(index, &name)? {
                    tables.push(loaded);
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
            staged: StagedCommit::new(self.clone()),
            datastore: self,
            index,
            loaded_tables: Mutex::new(HashMap::new()),
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

    /// Refresh the index every `refresh_interval`, on the ambient runtime. A
    /// refresh that fails is logged and the index kept, so an
    /// unreachable catalog degrades to stale tables rather than none.
    fn start(self: Arc<Self>) {
        let datastore = Arc::clone(&self);
        let interval = self.refresh_interval;
        let task = tokio::spawn(async move {
            let mut tick = tokio::time::interval(interval);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            // The catalog was just fetched at open, so skip the interval's
            // immediate first tick and refresh one full interval from now.
            tick.tick().await;
            loop {
                tick.tick().await;
                // The refresh blocks on catalog calls, so it runs off the
                // reactor.
                let datastore = Arc::clone(&datastore);
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

/// One query's view of the catalog: the index as it was when the query began,
/// which every schema check and bind of the query resolves against, so a
/// refresh landing mid-query moves nothing under it. A table is loaded the
/// first time the query touches it and kept for the query's lifetime, so a
/// plan's scan and its late materialize see the same load, and a table the
/// query names twice is loaded once. The writes its statements stage wait
/// on `staged` for its commit.
pub struct IcebergTransaction {
    datastore: Arc<IcebergDatastore>,
    index: Arc<DatastoreIndex>,
    loaded_tables: Mutex<HashMap<SchemaQualifiedTableName, Option<Arc<LoadedTable>>>>,
    staged: StagedCommit<IcebergDatastore>,
}

impl std::fmt::Debug for IcebergTransaction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IcebergTransaction")
            .field("datastore", &self.datastore)
            .finish_non_exhaustive()
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
            self.staged.queue(),
        ))))
    }

    fn table_revision(
        &self,
        name: &SchemaQualifiedTableName,
    ) -> CatalogResult<Option<TableRevision>> {
        Ok(self.resolve_table(name)?.map(|table| table.revision()))
    }

    fn tables(&self) -> CatalogResult<Vec<DatastoreTableMetadata>> {
        let tables = self.datastore.load_all_tables(&self.index)?;
        let mut memo = self.loaded_tables.lock().unwrap();
        let described = tables
            .into_iter()
            .map(|table| {
                memo.insert(table.name.clone(), Some(table.clone()));
                crate::describe::describe_table(&table)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(described)
    }

    async fn commit(&self) -> CatalogResult<()> {
        self.staged.commit().await
    }

    fn rollback(&self) {
        self.staged.rollback();
    }
}
