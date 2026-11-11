//! One commit to a catalog table as the spec crate's transaction sees its
//! catalog: the REST catalog, watched for how the commit ends, with every
//! object the commit writes recorded.
//!
//! A commit is built again each time another writer commits first, and each
//! attempt writes a manifest and a manifest list of its own. An attempt the
//! catalog refused is known not to have succeeded, and its objects are deleted;
//! nothing else sweeps a catalog table. An attempt the catalog answered
//! without saying whether it applied the commit (a server error, a gateway
//! timeout, a connection that broke) may have succeeded, and nothing it wrote
//! is deleted.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use iceberg::table::Table;
use iceberg::{
    Catalog, ErrorKind, Namespace, NamespaceIdent, Runtime, TableCommit, TableCreation, TableIdent,
};
use iceberg_catalog_rest::RestCatalog;
use iceberg_tables::TableStore;
use iceberg_tables::storage::WrittenObjects;

/// How the catalog answered the last update a commit asked it for.
#[derive(Clone, Copy, PartialEq, Eq)]
enum UpdateOutcome {
    /// The catalog refused the update: the commit did not succeed.
    Refused,
    /// The catalog did not say whether it applied the update.
    Unknown,
}

/// The REST catalog for the length of one commit to one table.
pub(crate) struct TrackedCommit {
    catalog: Arc<RestCatalog>,
    /// The store the commit's manifests are written through.
    store: TableStore,
    file_io: iceberg::io::FileIO,
    /// The objects written by the attempt in progress.
    written: WrittenObjects,
    last_update: Mutex<Option<UpdateOutcome>>,
}

impl std::fmt::Debug for TrackedCommit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrackedCommit")
            .field("catalog", &self.catalog)
            .finish_non_exhaustive()
    }
}

impl TrackedCommit {
    /// A commit through `catalog` whose manifests are written through
    /// `store`.
    pub(crate) fn new(catalog: Arc<RestCatalog>, store: &TableStore) -> Self {
        let (file_io, written) = store.build_recording_file_io();
        Self {
            catalog,
            store: store.clone(),
            file_io,
            written,
            last_update: Mutex::new(None),
        }
    }

    /// Whether the commit, which ended in an error, may have succeeded: the
    /// catalog was asked to apply it and did not say whether it had. A commit
    /// that failed before the catalog was asked, or that the catalog refused,
    /// did not succeed.
    pub(crate) fn may_have_committed(&self) -> bool {
        *self.last_update.lock().unwrap() == Some(UpdateOutcome::Unknown)
    }

    /// Delete the objects of an attempt that did not succeed.
    pub(crate) fn withdraw_written_objects(&self) {
        self.written.withdraw(&self.store);
    }
}

#[async_trait]
impl Catalog for TrackedCommit {
    /// The table as the catalog has it, writing through this commit's
    /// recording file IO.
    async fn load_table(&self, table: &TableIdent) -> iceberg::Result<Table> {
        let loaded = self.catalog.load_table(table).await?;
        let builder = Table::builder()
            .metadata(loaded.metadata_ref())
            .identifier(loaded.identifier().clone())
            .file_io(self.file_io.clone())
            .runtime(Runtime::try_current()?)
            // A commit reads the manifests it rewrites once.
            .disable_cache();
        match loaded.metadata_location() {
            Some(location) => builder.metadata_location(location),
            None => builder,
        }
        .build()
    }

    /// Ask the catalog to apply the commit, and keep how it answered. A
    /// conflict or a table that is gone is a refusal, and the attempt's
    /// objects are deleted; any other error leaves the outcome unknown,
    /// which covers every answer the catalog gives without saying whether it
    /// applied the commit and every failure after it did.
    async fn update_table(&self, commit: TableCommit) -> iceberg::Result<Table> {
        let updated = self.catalog.update_table(commit).await;
        let outcome = match &updated {
            Ok(_) => None,
            Err(error)
                if matches!(
                    error.kind(),
                    ErrorKind::CatalogCommitConflicts
                        | ErrorKind::TableNotFound
                        | ErrorKind::NamespaceNotFound
                ) =>
            {
                self.withdraw_written_objects();
                Some(UpdateOutcome::Refused)
            }
            Err(_) => Some(UpdateOutcome::Unknown),
        };
        *self.last_update.lock().unwrap() = outcome;
        updated
    }

    async fn list_namespaces(
        &self,
        parent: Option<&NamespaceIdent>,
    ) -> iceberg::Result<Vec<NamespaceIdent>> {
        self.catalog.list_namespaces(parent).await
    }

    async fn create_namespace(
        &self,
        namespace: &NamespaceIdent,
        properties: HashMap<String, String>,
    ) -> iceberg::Result<Namespace> {
        self.catalog.create_namespace(namespace, properties).await
    }

    async fn get_namespace(&self, namespace: &NamespaceIdent) -> iceberg::Result<Namespace> {
        self.catalog.get_namespace(namespace).await
    }

    async fn namespace_exists(&self, namespace: &NamespaceIdent) -> iceberg::Result<bool> {
        self.catalog.namespace_exists(namespace).await
    }

    async fn update_namespace(
        &self,
        namespace: &NamespaceIdent,
        properties: HashMap<String, String>,
    ) -> iceberg::Result<()> {
        self.catalog.update_namespace(namespace, properties).await
    }

    async fn drop_namespace(&self, namespace: &NamespaceIdent) -> iceberg::Result<()> {
        self.catalog.drop_namespace(namespace).await
    }

    async fn list_tables(&self, namespace: &NamespaceIdent) -> iceberg::Result<Vec<TableIdent>> {
        self.catalog.list_tables(namespace).await
    }

    async fn create_table(
        &self,
        namespace: &NamespaceIdent,
        creation: TableCreation,
    ) -> iceberg::Result<Table> {
        self.catalog.create_table(namespace, creation).await
    }

    async fn drop_table(&self, table: &TableIdent) -> iceberg::Result<()> {
        self.catalog.drop_table(table).await
    }

    async fn purge_table(&self, table: &TableIdent) -> iceberg::Result<()> {
        self.catalog.purge_table(table).await
    }

    async fn table_exists(&self, table: &TableIdent) -> iceberg::Result<bool> {
        self.catalog.table_exists(table).await
    }

    async fn rename_table(&self, src: &TableIdent, dest: &TableIdent) -> iceberg::Result<()> {
        self.catalog.rename_table(src, dest).await
    }

    async fn register_table(
        &self,
        table: &TableIdent,
        metadata_location: String,
    ) -> iceberg::Result<Table> {
        self.catalog.register_table(table, metadata_location).await
    }
}
