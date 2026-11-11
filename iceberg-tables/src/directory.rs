//! A table whose current version is kept in its own directory: numbered
//! metadata files under `<directory>/metadata/`, each version committed with a
//! conditional write that admits exactly one writer, so writers in several
//! processes share a table with no catalog service. Readable by any Iceberg
//! reader from the table's path.
//!
//! The metadata log of a version names every earlier metadata file except
//! the ones the commit that wrote it was told had expired
//! ([`Table::expired_metadata_files`](crate::Table::expired_metadata_files)):
//! `write.metadata.previous-versions-max` does not bound it, so a file leaves
//! the log, and is deleted, only once the file after it is older than the
//! retention window.

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

use crate::commit::{now_unix_ms, retry_io};
use crate::layout::{
    TableLocation, VERSION_HINT_FILE, hint_content, metadata_file_name, parse_hint,
    parse_metadata_file_version,
};
use crate::metadata_pointer::MetadataPointer;
use crate::retention::snapshot_retention;
use crate::state::CurrentMetadata;
use crate::{Error, Result};

/// The numbered metadata files of the table at one location.
#[derive(Clone)]
pub struct DirectoryMetadataPointer {
    location: TableLocation,
}

impl DirectoryMetadataPointer {
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
                .list_filtered(&location.metadata_dir(), "v", after.as_deref())?;
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

    /// Whether `base` was committed at least its retention window ago: the
    /// age from which the metadata file after it may have been deleted,
    /// which takes the file after that one to be as old as the window.
    fn is_stale_base(base: &CurrentMetadata) -> bool {
        let retention_ms = snapshot_retention(&base.metadata)
            .map(|window| i64::try_from(window.as_millis()).unwrap_or(i64::MAX))
            .unwrap_or(0);
        base.metadata.last_updated_ms().saturating_add(retention_ms) <= now_unix_ms()
    }

    /// Make `metadata` the table's current version, as the successor of
    /// `base` (`None` for a table's first version): create
    /// `v{N+1}.metadata.json` only if nothing has that name yet.
    /// Every writer that loaded N derives the same name, so exactly one
    /// succeeds, and `None` tells the others they lost. The hint is rewritten
    /// afterwards, best effort.
    ///
    /// A write that fails without saying whether the store applied it leaves
    /// the outcome open: the object under the name is then read back, and is
    /// this commit's exactly when it holds these bytes. An outcome that
    /// cannot be settled is [`Error::CommitOutcomeUnknown`], never a loss.
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
        // Whether some attempt at the write failed without saying whether
        // the store applied it.
        let mut is_put_ambiguous = false;
        let put = retry_io(
            || match store.put_if_absent(&metadata_key, &metadata_bytes) {
                Ok(()) => Ok(true),
                Err(StoreError::VersionConflict { .. }) => Ok(false),
                Err(error) => {
                    is_put_ambiguous = true;
                    Err(error)
                }
            },
        );
        let is_committed = match put {
            Ok(true) => true,
            Ok(false) if !is_put_ambiguous => false,
            // The name is taken after a write of ours may have been applied.
            Ok(false) => self.is_published(&metadata_key, &metadata_bytes, None)?,
            Err(error) => self.is_published(&metadata_key, &metadata_bytes, Some(error))?,
        };
        if !is_committed {
            return Ok(None);
        }
        // A writer that loaded its base longer ago than the retention window
        // may be committing a version the table had and has since deleted: the
        // name was free only because the file under it expired. The table's
        // current file is never deleted, so a later version is then still
        // listed, and the commit, which no reader would see, withdraws
        // itself.
        if base.is_some_and(Self::is_stale_base) {
            // The version is written: a listing that fails leaves open
            // whether it is the table's current one.
            let listed = retry_io(|| {
                store.list_filtered(&location.metadata_dir(), "v", Some(&metadata_name))
            })
            .map_err(|error| Error::CommitOutcomeUnknown {
                source: Box::new(error),
            })?;
            // The version hint's name also starts with `v` and sorts after
            // every version's, so only a numbered metadata file is a
            // successor.
            let has_successor = listed
                .objects
                .iter()
                .any(|object| parse_metadata_file_version(object.file.path.as_str()).is_some());
            if has_successor {
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

impl DirectoryMetadataPointer {
    /// Whether the object at `metadata_key` is the version this commit wrote
    /// as `metadata_bytes`, after a write of it that may or may not have been
    /// applied (`put_error` is the failure the write ended with, if it ended
    /// with one). Another writer's version under the name means this commit
    /// lost. No object, or a read that fails, settles nothing.
    fn is_published(
        &self,
        metadata_key: &object_storage::ObjectPath,
        metadata_bytes: &[u8],
        put_error: Option<Error>,
    ) -> Result<bool> {
        let unknown = |error: Error| Error::CommitOutcomeUnknown {
            source: Box::new(error),
        };
        let store = self.location.store();
        match retry_io(|| store.get(metadata_key)) {
            Ok(Some(published)) => Ok(published == metadata_bytes),
            Ok(None) => Err(unknown(put_error.unwrap_or_else(|| {
                Error::MetadataVersionMissing {
                    table: self.location.table_uri(),
                    version: parse_metadata_file_version(metadata_key.as_str()).unwrap_or(0),
                }
            }))),
            Err(read_error) => Err(unknown(put_error.unwrap_or(read_error))),
        }
    }
}

impl MetadataPointer for DirectoryMetadataPointer {
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
    /// deleted (this copy is older than the retention window) is reloaded
    /// cold.
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

    fn commit_catalog(
        &self,
        base: SpecTable,
        expired_metadata_files: Vec<String>,
    ) -> Result<Arc<dyn Catalog>> {
        Ok(Arc::new(DirectoryCommit {
            directory: self.clone(),
            base,
            expired_metadata_files,
        }))
    }
}

/// One commit to a directory table, as the spec crate's transaction sees its
/// catalog: the version the commit is built on, and the one conditional write
/// that commits its successor. A directory holds one table and no namespaces,
/// so every other call of a catalog is refused.
struct DirectoryCommit {
    directory: DirectoryMetadataPointer,
    base: SpecTable,
    /// The previous metadata files the successor's metadata log drops. It
    /// keeps every other entry of the base's log and adds the base's file.
    expired_metadata_files: Vec<String>,
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
    /// never published (a table being created). The successor's metadata
    /// log loses the expired files this commit was given and nothing else,
    /// however many entries it holds. A commit conflict means another writer
    /// published that version first.
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
        let metadata = Arc::new(
            builder
                .retain_metadata_log_except(self.expired_metadata_files.iter().cloned())
                .build()?
                .metadata,
        );
        let directory = self.directory.clone();
        // The store's calls block on the network, and the runtime's workers
        // must not.
        let published =
            tokio::task::spawn_blocking(move || directory.publish(base.as_ref(), metadata))
                .await
                // A publish that panicked may have written the version.
                .unwrap_or_else(|panicked| {
                    Err(Error::CommitOutcomeUnknown {
                        source: Box::new(Error::spec(
                            iceberg::Error::new(
                                ErrorKind::Unexpected,
                                "publishing a version panicked",
                            )
                            .with_source(panicked),
                        )),
                    })
                })
                // The error rides along as the source, which is how the
                // table tells an unknown outcome from a failure.
                .map_err(|error| {
                    iceberg::Error::new(
                        ErrorKind::Unexpected,
                        format!("publishing a version: {error}"),
                    )
                    .with_source(error)
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
