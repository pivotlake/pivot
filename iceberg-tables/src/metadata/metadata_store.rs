//! The seam between a table's content and wherever its metadata versions
//! are kept, and what passes through it: a table's metadata file
//! ([`MetadataFile`]), which is one version of the table.
//!
//! Building a new version is the same everywhere: the spec crate's
//! transaction writes the manifests and the manifest list of the snapshot it
//! adds and describes them as changes to the metadata document. Making that
//! version current is what differs. A table in a directory numbers its
//! metadata files and makes a version current by creating the next number,
//! which exactly one writer can do. A table in a catalog service makes a
//! version current by asking the service to commit it.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use iceberg::spec::TableMetadata;
use iceberg::table::Table as SpecTable;
use iceberg::{
    Catalog, ErrorKind, Namespace, NamespaceIdent, TableCommit, TableCreation, TableIdent,
};
use planner::catalog::SchemaQualifiedTableName;

use crate::{Error, Result};

/// One metadata file of a table, which is one version of it: where the file
/// is, as the next version's metadata log records it, and the metadata it
/// holds. A table is loaded from one, and every commit from that version is
/// written on top of it.
#[derive(Clone)]
pub struct MetadataFile {
    pub location: String,
    pub metadata: Arc<TableMetadata>,
}

/// The metadata file a table of the spec crate was loaded from or committed
/// as, with the metadata it holds. A table only staged in a catalog has no
/// metadata location yet and is refused as [`Error::TableMetadataMissing`],
/// under its identifier.
impl TryFrom<&SpecTable> for MetadataFile {
    type Error = Error;

    fn try_from(table: &SpecTable) -> Result<Self> {
        let location = table
            .metadata_location()
            .ok_or_else(|| Error::TableMetadataMissing {
                table: table.identifier().to_string(),
            })?;
        Ok(Self {
            location: location.to_string(),
            metadata: table.metadata_ref(),
        })
    }
}

/// The identifier the spec crate knows the table `name` by, in a catalog
/// and on the table a commit is built on: its namespace is the schema, its
/// name the table.
pub fn build_table_ident(name: &SchemaQualifiedTableName) -> TableIdent {
    TableIdent::new(NamespaceIdent::new(name.schema.clone()), name.table.clone())
}

/// Where a table's metadata versions are found, read and written: one
/// table's own directory, or a catalog service that keeps every table's.
#[async_trait]
pub trait MetadataStore: Send + Sync {
    /// The latest metadata of the table `name` and where it is.
    fn load_latest(&self, name: &SchemaQualifiedTableName) -> Result<MetadataFile>;

    /// Make `commit`, which a transaction built on `base`, the table's
    /// current version as the successor of `base`, and answer that version.
    /// A commit conflict is the error when another writer committed a
    /// successor first, and only for a commit that was not applied: the
    /// transaction deletes the manifests and the manifest list of a commit
    /// answered with one, and keeps them after any other error. An error
    /// whose source is
    /// [`Error::CommitOutcomeUnknown`]
    /// tells that the commit may have been applied. A store that writes the
    /// table's metadata files drops `expired_metadata_files` from the
    /// successor's metadata log, and no other entry. A catalog service
    /// writes them and keeps their log itself, so a store over one ignores
    /// `expired_metadata_files`.
    async fn commit_successor(
        &self,
        base: &SpecTable,
        commit: TableCommit,
        expired_metadata_files: &[String],
    ) -> iceberg::Result<SpecTable>;
}

/// One commit through a table's metadata store, as the spec crate's
/// transaction sees its catalog: the version the commit is built on, and the
/// one update that makes its successor current. Every other call of a
/// catalog is refused.
pub(crate) struct CommitCatalog {
    pub(crate) metadata_store: Arc<dyn MetadataStore>,
    /// The version the commit is built on.
    pub(crate) base: SpecTable,
    /// The previous metadata files the successor's metadata log drops.
    pub(crate) expired_metadata_files: Vec<String>,
}

const NOT_A_COMMIT: &str = "a commit to a table asks its catalog for the commit only";

impl fmt::Debug for CommitCatalog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommitCatalog")
            .field("table", self.base.identifier())
            .field("base", &self.base.metadata_location())
            .finish()
    }
}

#[async_trait]
impl Catalog for CommitCatalog {
    /// The version the commit is built on, as it was loaded: whether it is
    /// still current is settled by the update alone.
    async fn load_table(&self, _table: &TableIdent) -> iceberg::Result<SpecTable> {
        Ok(self.base.clone())
    }

    async fn update_table(&self, commit: TableCommit) -> iceberg::Result<SpecTable> {
        self.metadata_store
            .commit_successor(&self.base, commit, &self.expired_metadata_files)
            .await
    }

    async fn list_namespaces(
        &self,
        _parent: Option<&NamespaceIdent>,
    ) -> iceberg::Result<Vec<NamespaceIdent>> {
        Err(iceberg::Error::new(
            ErrorKind::FeatureUnsupported,
            NOT_A_COMMIT,
        ))
    }

    async fn create_namespace(
        &self,
        _namespace: &NamespaceIdent,
        _properties: HashMap<String, String>,
    ) -> iceberg::Result<Namespace> {
        Err(iceberg::Error::new(
            ErrorKind::FeatureUnsupported,
            NOT_A_COMMIT,
        ))
    }

    async fn get_namespace(&self, _namespace: &NamespaceIdent) -> iceberg::Result<Namespace> {
        Err(iceberg::Error::new(
            ErrorKind::FeatureUnsupported,
            NOT_A_COMMIT,
        ))
    }

    async fn namespace_exists(&self, _namespace: &NamespaceIdent) -> iceberg::Result<bool> {
        Err(iceberg::Error::new(
            ErrorKind::FeatureUnsupported,
            NOT_A_COMMIT,
        ))
    }

    async fn update_namespace(
        &self,
        _namespace: &NamespaceIdent,
        _properties: HashMap<String, String>,
    ) -> iceberg::Result<()> {
        Err(iceberg::Error::new(
            ErrorKind::FeatureUnsupported,
            NOT_A_COMMIT,
        ))
    }

    async fn drop_namespace(&self, _namespace: &NamespaceIdent) -> iceberg::Result<()> {
        Err(iceberg::Error::new(
            ErrorKind::FeatureUnsupported,
            NOT_A_COMMIT,
        ))
    }

    async fn list_tables(&self, _namespace: &NamespaceIdent) -> iceberg::Result<Vec<TableIdent>> {
        Err(iceberg::Error::new(
            ErrorKind::FeatureUnsupported,
            NOT_A_COMMIT,
        ))
    }

    async fn create_table(
        &self,
        _namespace: &NamespaceIdent,
        _creation: TableCreation,
    ) -> iceberg::Result<SpecTable> {
        Err(iceberg::Error::new(
            ErrorKind::FeatureUnsupported,
            NOT_A_COMMIT,
        ))
    }

    async fn drop_table(&self, _table: &TableIdent) -> iceberg::Result<()> {
        Err(iceberg::Error::new(
            ErrorKind::FeatureUnsupported,
            NOT_A_COMMIT,
        ))
    }

    async fn purge_table(&self, _table: &TableIdent) -> iceberg::Result<()> {
        Err(iceberg::Error::new(
            ErrorKind::FeatureUnsupported,
            NOT_A_COMMIT,
        ))
    }

    async fn table_exists(&self, _table: &TableIdent) -> iceberg::Result<bool> {
        Err(iceberg::Error::new(
            ErrorKind::FeatureUnsupported,
            NOT_A_COMMIT,
        ))
    }

    async fn rename_table(&self, _src: &TableIdent, _dest: &TableIdent) -> iceberg::Result<()> {
        Err(iceberg::Error::new(
            ErrorKind::FeatureUnsupported,
            NOT_A_COMMIT,
        ))
    }

    async fn register_table(
        &self,
        _table: &TableIdent,
        _metadata_location: String,
    ) -> iceberg::Result<SpecTable> {
        Err(iceberg::Error::new(
            ErrorKind::FeatureUnsupported,
            NOT_A_COMMIT,
        ))
    }
}
