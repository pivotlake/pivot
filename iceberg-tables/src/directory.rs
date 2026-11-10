//! A table whose current version is kept in its own directory: numbered
//! metadata files under `<directory>/metadata/`, each version landed with a
//! conditional write that admits exactly one writer, so writers in several
//! processes share a table with no catalog service. Readable by any Iceberg
//! reader from the table's path.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use iceberg::Runtime;
use iceberg::spec::TableMetadata;
use iceberg::table::Table as SpecTable;
use iceberg::{
    Catalog, ErrorKind, Namespace, NamespaceIdent, TableCommit, TableCreation, TableIdent,
};
use object_storage::StoreError;

use crate::catalog::TableCatalog;
use crate::commit::{now_unix_ms, retry_io};
use crate::layout::{
    TableLocation, VERSION_HINT_FILE, hint_content, metadata_file_name, parse_hint,
    parse_metadata_file_version,
};
use crate::retention::snapshot_retention;
use crate::state::CurrentMetadata;
use crate::{Error, Result};

/// The numbered metadata files of the table at one location.
#[derive(Clone)]
pub struct DirectoryCatalog {
    location: TableLocation,
}

impl DirectoryCatalog {
    pub fn new(location: TableLocation) -> Self {
        Self { location }
    }

    /// The version the table's hint names, or `None` when there is no hint,
    /// which only costs the reader a longer listing. A hint that does not
    /// parse is an error.
    pub fn read_version_hint(&self) -> Result<Option<u64>> {
        let location = &self.location;
        match location
            .store()
            .get(&location.metadata_key(VERSION_HINT_FILE))?
        {
            Some(bytes) => Ok(Some(parse_hint(&bytes)?)),
            None => {
                tracing::warn!(
                    table = %location.table_uri(),
                    "no version hint; listing the table's metadata from its first version"
                );
                Ok(None)
            }
        }
    }

    /// Write the version hint naming `version`. Best effort by the caller:
    /// the hint is for readers that open the table by its path and is never
    /// trusted by this crate, so a failed write costs nothing but a warning.
    pub fn write_version_hint(&self, version: u64) -> Result<()> {
        let location = &self.location;
        location
            .store()
            .put(
                &location.metadata_key(VERSION_HINT_FILE),
                hint_content(version).as_bytes(),
            )
            .map_err(Error::from)
    }

    /// The table's current version: the highest metadata file listed from
    /// `from` up, or `None` when the table has no metadata file at all.
    /// `from` names a version known to have existed (the hint's, or the one
    /// a reader holds); a listing that no longer shows it means it was
    /// deleted, which the caller decides what to make of.
    pub fn find_current_version(&self, from: u64) -> Result<Option<u64>> {
        let location = &self.location;
        // The padded names sort in version order, so the name of the version
        // before `from` is a cursor that admits `from` itself and everything
        // newer.
        let after = match from {
            0 => None,
            from => Some(metadata_file_name(from - 1)),
        };
        let listed =
            location
                .store()
                .list_children(&location.metadata_dir(), "v", after.as_deref())?;
        let mut versions: Vec<u64> = listed
            .objects
            .iter()
            .filter_map(|object| parse_metadata_file_version(object.file.path.as_str()))
            .collect();
        versions.sort_unstable();
        if from > 0 && !versions.contains(&from) {
            return Err(Error::MetadataVersionMissing {
                table: location.table_uri(),
                version: from,
            });
        }
        Ok(versions.last().copied())
    }

    /// Read and parse metadata `version`.
    pub fn read_metadata(&self, version: u64) -> Result<CurrentMetadata> {
        let location = &self.location;
        let name = metadata_file_name(version);
        let bytes = location
            .store()
            .get(&location.metadata_key(&name))?
            .ok_or_else(|| Error::MetadataVersionMissing {
                table: location.table_uri(),
                version,
            })?;
        let metadata: TableMetadata = serde_json::from_slice(&bytes)?;
        Ok(CurrentMetadata {
            metadata_location: location.metadata_uri(&name),
            metadata: Arc::new(metadata),
        })
    }

    /// The current version, found from the hint: a cold open.
    fn find_version_from_hint(&self) -> Result<Option<u64>> {
        let hinted = self.read_version_hint()?.unwrap_or(0);
        match self.find_current_version(hinted) {
            // The hinted version was deleted from under a hint a sweep had not
            // caught up with yet: the whole listing is still exact.
            Err(Error::MetadataVersionMissing { .. }) if hinted > 0 => {
                tracing::warn!(
                    table = %self.location.table_uri(),
                    hinted,
                    "the hinted metadata version no longer exists; listing from the first version"
                );
                self.find_current_version(0)
            }
            found => found,
        }
    }

    /// Whether `base` was committed longer ago than its retention window.
    fn is_stale_base(base: &CurrentMetadata) -> bool {
        let retention_ms = snapshot_retention(&base.metadata)
            .map(|window| i64::try_from(window.as_millis()).unwrap_or(i64::MAX))
            .unwrap_or(0);
        base.metadata.last_updated_ms().saturating_add(retention_ms) < now_unix_ms()
    }

    /// Make `metadata` the table's current version, as the successor of
    /// `base` (`None` for a table's first version): create
    /// `v{N+1}.metadata.json` only if nothing has that name yet.
    /// Every writer that loaded N derives the same name, so exactly one
    /// lands, and `None` tells the others they lost. The hint is rewritten
    /// afterwards, best effort.
    pub fn publish(
        &self,
        base: Option<&CurrentMetadata>,
        metadata: Arc<TableMetadata>,
    ) -> Result<Option<CurrentMetadata>> {
        let location = &self.location;
        let store = location.store();
        let version = match base {
            Some(base) => metadata_location_version(&base.metadata_location)? + 1,
            None => 1,
        };
        let metadata_bytes = serde_json::to_vec(metadata.as_ref())?;
        let metadata_name = metadata_file_name(version);
        let metadata_key = location.metadata_key(&metadata_name);
        let landed = retry_io(
            || match store.put_if_absent(&metadata_key, &metadata_bytes) {
                Ok(()) => Ok(true),
                Err(StoreError::VersionConflict { .. }) => Ok(false),
                Err(error) => Err(error),
            },
        )?;
        if !landed {
            return Ok(None);
        }
        // A writer that loaded its base longer ago than the retention window
        // may be landing a version a sweep already deleted the successors
        // of: the name was free only because its file was swept. Such a
        // commit would be invisible, so it withdraws itself.
        if base.is_some_and(Self::is_stale_base) {
            let listed =
                store.list_children(&location.metadata_dir(), "v", Some(&metadata_name))?;
            if !listed.objects.is_empty() {
                let _ = store.delete(&metadata_key);
                return Ok(None);
            }
        }
        if let Err(error) = self.write_version_hint(version) {
            tracing::warn!(
                table = %location.table_uri(),
                version,
                error = %error,
                "writing the version hint failed; path readers see the previous hint"
            );
        }
        Ok(Some(CurrentMetadata {
            metadata_location: location.metadata_uri(&metadata_name),
            metadata,
        }))
    }
}

impl TableCatalog for DirectoryCatalog {
    fn load(&self) -> Result<CurrentMetadata> {
        let version =
            self.find_version_from_hint()?
                .ok_or_else(|| Error::TableMetadataMissing {
                    table: self.location.table_uri(),
                    location: self.location.table_uri(),
                })?;
        self.read_metadata(version)
    }

    /// One short listing from the held version; a held version that was
    /// swept (this copy is older than the retention window) is reloaded cold.
    fn refresh(&self, held: &CurrentMetadata) -> Result<Option<CurrentMetadata>> {
        let held_version = metadata_location_version(&held.metadata_location)?;
        let version = match self.find_current_version(held_version) {
            Ok(Some(version)) => version,
            Ok(None) => {
                return Err(Error::TableMetadataMissing {
                    table: self.location.table_uri(),
                    location: self.location.table_uri(),
                });
            }
            Err(Error::MetadataVersionMissing { .. }) => return self.load().map(Some),
            Err(error) => return Err(error),
        };
        if version == held_version {
            return Ok(None);
        }
        self.read_metadata(version).map(Some)
    }

    fn commit_catalog(&self, base: SpecTable) -> Result<Arc<dyn Catalog>> {
        Ok(Arc::new(DirectoryCommit {
            directory: self.clone(),
            base,
        }))
    }
}

/// One commit to a directory table, as the spec crate's transaction sees its
/// catalog: the version the commit is built on, and the one conditional write
/// that lands its successor. A directory holds one table and no namespaces,
/// so every other call of a catalog is refused.
struct DirectoryCommit {
    directory: DirectoryCatalog,
    base: SpecTable,
}

impl fmt::Debug for DirectoryCommit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DirectoryCommit")
            .field("table", &self.directory.location.table_uri())
            .field("base", &self.base.metadata_location())
            .finish()
    }
}

/// The error of a catalog call a single table's directory has no answer to.
fn not_a_catalog_service(call: &str) -> iceberg::Error {
    iceberg::Error::new(
        ErrorKind::FeatureUnsupported,
        format!("a table's directory serves commits to that table only, not `{call}`"),
    )
}

#[async_trait]
impl Catalog for DirectoryCommit {
    /// The version the commit is built on, as it was loaded: whether it is
    /// still current is settled by the conditional write alone.
    async fn load_table(&self, _table: &TableIdent) -> iceberg::Result<SpecTable> {
        Ok(self.base.clone())
    }

    /// Apply the commit to the base's metadata and publish the result as the
    /// base's successor, or as the table's first version when the base was
    /// never published (a table being created). A commit conflict means
    /// another writer published that version first.
    async fn update_table(&self, mut commit: TableCommit) -> iceberg::Result<SpecTable> {
        let base_metadata = self.base.metadata_ref();
        let base = self
            .base
            .metadata_location()
            .map(|location| CurrentMetadata {
                metadata_location: location.to_string(),
                metadata: Arc::clone(&base_metadata),
            });
        for requirement in commit.take_requirements() {
            requirement.check(Some(&base_metadata))?;
        }
        let mut builder = base_metadata.as_ref().clone().into_builder(
            base.as_ref()
                .map(|published| published.metadata_location.clone()),
        );
        for update in commit.take_updates() {
            builder = update.apply(builder)?;
        }
        let metadata = Arc::new(builder.build()?.metadata);
        let directory = self.directory.clone();
        // The store's calls block on the network, and the runtime's workers
        // must not.
        let published =
            tokio::task::spawn_blocking(move || directory.publish(base.as_ref(), metadata))
                .await
                .map_err(|panicked| {
                    iceberg::Error::new(ErrorKind::Unexpected, "publishing a version panicked")
                        .with_source(panicked)
                })?
                .map_err(|error| {
                    iceberg::Error::new(
                        ErrorKind::Unexpected,
                        format!("publishing a version: {error}"),
                    )
                })?;
        let Some(current) = published else {
            return Err(iceberg::Error::new(
                ErrorKind::CatalogCommitConflicts,
                "another writer published the table's next version first",
            ));
        };
        SpecTable::builder()
            .metadata(current.metadata)
            .metadata_location(current.metadata_location)
            .identifier(self.base.identifier().clone())
            .file_io(self.base.file_io().clone())
            .runtime(Runtime::try_current()?)
            .build()
    }

    async fn list_namespaces(
        &self,
        _parent: Option<&NamespaceIdent>,
    ) -> iceberg::Result<Vec<NamespaceIdent>> {
        Err(not_a_catalog_service("list_namespaces"))
    }

    async fn create_namespace(
        &self,
        _namespace: &NamespaceIdent,
        _properties: HashMap<String, String>,
    ) -> iceberg::Result<Namespace> {
        Err(not_a_catalog_service("create_namespace"))
    }

    async fn get_namespace(&self, _namespace: &NamespaceIdent) -> iceberg::Result<Namespace> {
        Err(not_a_catalog_service("get_namespace"))
    }

    async fn namespace_exists(&self, _namespace: &NamespaceIdent) -> iceberg::Result<bool> {
        Err(not_a_catalog_service("namespace_exists"))
    }

    async fn update_namespace(
        &self,
        _namespace: &NamespaceIdent,
        _properties: HashMap<String, String>,
    ) -> iceberg::Result<()> {
        Err(not_a_catalog_service("update_namespace"))
    }

    async fn drop_namespace(&self, _namespace: &NamespaceIdent) -> iceberg::Result<()> {
        Err(not_a_catalog_service("drop_namespace"))
    }

    async fn list_tables(&self, _namespace: &NamespaceIdent) -> iceberg::Result<Vec<TableIdent>> {
        Err(not_a_catalog_service("list_tables"))
    }

    async fn create_table(
        &self,
        _namespace: &NamespaceIdent,
        _creation: TableCreation,
    ) -> iceberg::Result<SpecTable> {
        Err(not_a_catalog_service("create_table"))
    }

    async fn drop_table(&self, _table: &TableIdent) -> iceberg::Result<()> {
        Err(not_a_catalog_service("drop_table"))
    }

    async fn purge_table(&self, _table: &TableIdent) -> iceberg::Result<()> {
        Err(not_a_catalog_service("purge_table"))
    }

    async fn table_exists(&self, _table: &TableIdent) -> iceberg::Result<bool> {
        Err(not_a_catalog_service("table_exists"))
    }

    async fn rename_table(&self, _src: &TableIdent, _dest: &TableIdent) -> iceberg::Result<()> {
        Err(not_a_catalog_service("rename_table"))
    }

    async fn register_table(
        &self,
        _table: &TableIdent,
        _metadata_location: String,
    ) -> iceberg::Result<SpecTable> {
        Err(not_a_catalog_service("register_table"))
    }
}

/// The version number a directory table's metadata location carries.
pub fn metadata_location_version(metadata_location: &str) -> Result<u64> {
    let name = metadata_location
        .rsplit('/')
        .next()
        .unwrap_or(metadata_location);
    parse_metadata_file_version(name).ok_or_else(|| Error::UnnumberedMetadataLocation {
        location: metadata_location.to_string(),
    })
}
